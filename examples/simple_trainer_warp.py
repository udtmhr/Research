# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, RasterizeMode
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path: "interp", "ellipse", "spiral", or "raw" (use captured poses as-is)
    render_traj_path: str = "interp"

    # Dataset backend: "colmap" or "ncore"
    data_type: str = "colmap"
    # Path to the Mip-NeRF 360 dataset (colmap) or NCore v4 meta-JSON file (ncore)
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: CameraModel = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True

    # --- NCore-specific options (only used when data_type="ncore") ---
    # Camera sensor IDs to load (auto-detected from sequence if empty)
    ncore_camera_ids: List[str] = field(default_factory=list)
    # Point cloud source IDs to load -- accepts lidar, radar, or native point cloud
    # source IDs (auto-detected from sequence if empty). Field name kept for backward compat.
    ncore_lidar_ids: List[str] = field(default_factory=list)
    # Temporal seek offset in seconds
    ncore_seek_offset_sec: Optional[float] = None
    # Clip duration in seconds (None = full sequence)
    ncore_duration_sec: Optional[float] = None
    # Maximum number of lidar init points
    ncore_max_lidar_points: int = 500_000
    # Generic-data key for lidar point RGB colors (fallback to gray if unavailable)
    ncore_lidar_color_generic_data_name: str = "rgb"
    # NCore component group names
    ncore_poses_component_group: str = "default"
    ncore_intrinsics_component_group: str = "default"
    ncore_masks_component_group: str = "default"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable texture-warp view consistency loss. COLMAP + pinhole only in this trainer.
    use_warp_loss: bool = False
    # Weight for texture-warp view consistency loss
    lambda_warp: float = 0.05
    # Number of nearby training cameras used as warp references
    warp_num_neighbors: int = 3
    # Viewing-direction similarity cutoff in degrees
    warp_theta_max_deg: float = 30.0
    # Depth consistency temperature
    warp_tau_depth: float = 0.01
    # First iteration where warp loss is active
    warp_start_iter: int = 7000
    # Compute warp loss every N iterations
    warp_interval: int = 10
    # Number of manual virtual keyframes sampled per active iteration
    warp_num_virtual_views: int = 1
    # Minimum accumulated visibility confidence for warp supervision
    warp_min_gamma: float = 0.05
    # Downsample factor for warp renders and image sampling
    warp_downsample: int = 1
    # Use keyframes saved from the viewer as virtual camera poses.
    warp_manual_camera_path: Optional[str] = "camera_paths/virtual.json"
    # Scale ratio used by nerfview when saving/loading camera path JSON.
    warp_manual_camera_scale_ratio: float = 10.0
    # Error if no saved keyframes are available when warp_manual_camera_path is set.
    warp_require_manual_keyframes: bool = True
    # Save camera pose visualization/debug JSON for active warp iterations
    warp_save_camera_debug: bool = True
    # Save camera pose debug every N warp-active iterations
    warp_camera_debug_interval: int = 100
    # Show the latest warp virtual/reference camera set in the viewer
    warp_viewer_debug: bool = True
    # Use offline precomputed virtual C_ref supervision instead of online warping
    use_precomputed_virtual_refs: bool = False
    # Directory containing ref_*.npz, relative to data_dir or result_dir
    virtual_ref_dir: str = "virtual_refs"
    # Advance the selected precomputed virtual reference every N steps
    virtual_ref_interval: int = 10

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.warp_start_iter = int(self.warp_start_iter * factor)
        self.warp_interval = max(1, int(self.warp_interval * factor))
        self.warp_camera_debug_interval = max(
            1, int(self.warp_camera_debug_interval * factor)
        )

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm" or init_type == "lidar":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or lidar")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N,     features_dc im]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.data_type == "ncore":
            from datasets.ncore import NCoreDataset, NCoreParser

            self.parser = NCoreParser(
                meta_json_path=cfg.data_dir,
                factor=1.0 / cfg.data_factor if cfg.data_factor > 1 else 1.0,
                test_every=cfg.test_every,
                camera_ids=cfg.ncore_camera_ids or None,
                lidar_ids=cfg.ncore_lidar_ids or None,
                seek_offset_sec=cfg.ncore_seek_offset_sec,
                duration_sec=cfg.ncore_duration_sec,
                max_lidar_points=cfg.ncore_max_lidar_points,
                lidar_color_generic_data_name=cfg.ncore_lidar_color_generic_data_name,
                poses_component_group=cfg.ncore_poses_component_group,
                intrinsics_component_group=cfg.ncore_intrinsics_component_group,
                masks_component_group=cfg.ncore_masks_component_group,
                normalize_world_space=cfg.normalize_world_space,
            )
            self.trainset = NCoreDataset(self.parser, split="train")
            self.valset = NCoreDataset(self.parser, split="val")
            self.ncore_camera_data = [
                self.parser.camera_render_data[cam_id]
                for cam_id in self.parser.camera_ids
            ]
            if (
                any(d.camera_model == "ftheta" for d in self.ncore_camera_data)
                and not cfg.with_eval3d
            ):
                print(
                    "[NCore] Warning: FTheta cameras detected; pass --with-eval3d True for correct results."
                )
        else:
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                load_exposure=cfg.load_exposure,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
        self.warp_cache = None
        self._saved_warp_reference_sets = set()
        if cfg.use_warp_loss:
            if cfg.data_type != "colmap":
                raise ValueError("Warp loss is only implemented for data_type='colmap'.")
            if cfg.camera_model != "pinhole":
                raise ValueError("Warp loss v1 requires camera_model='pinhole'.")
            if cfg.batch_size != 1:
                raise ValueError("Warp loss v1 requires batch_size=1.")
            if cfg.patch_size is not None:
                raise ValueError(
                    "Warp loss v1 requires full-image training; patch_size must be None."
                )
            if cfg.warp_num_neighbors <= 0:
                raise ValueError("warp_num_neighbors must be positive.")
            if cfg.warp_interval <= 0:
                raise ValueError("warp_interval must be positive.")
            if cfg.warp_num_virtual_views <= 0:
                raise ValueError("warp_num_virtual_views must be positive.")
            if cfg.warp_downsample <= 0:
                raise ValueError("warp_downsample must be positive.")
            if cfg.warp_manual_camera_scale_ratio <= 0:
                raise ValueError("warp_manual_camera_scale_ratio must be positive.")
            if cfg.warp_camera_debug_interval <= 0:
                raise ValueError("warp_camera_debug_interval must be positive.")
            if cfg.virtual_ref_interval <= 0:
                raise ValueError("virtual_ref_interval must be positive.")
            if cfg.warp_tau_depth <= 0:
                raise ValueError("warp_tau_depth must be positive.")
            if not (0.0 < cfg.warp_theta_max_deg < 180.0):
                raise ValueError("warp_theta_max_deg must be in (0, 180).")
            if cfg.use_precomputed_virtual_refs:
                self.precomputed_virtual_refs = self._load_precomputed_virtual_refs()
                self.manual_warp_cameras = None
            else:
                self.precomputed_virtual_refs = None
                self.warp_cache = self._build_warp_cache()
                self.manual_warp_cameras = self._load_manual_warp_cameras()
        else:
            self.precomputed_virtual_refs = None
            self.manual_warp_cameras = None
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )
            self._warp_viewer_handles = []
            if (
                cfg.use_warp_loss
                and cfg.warp_viewer_debug
                and not cfg.use_precomputed_virtual_refs
            ):
                self._init_warp_viewer_debug()

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model: Optional[CameraModel] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        ftheta_coeffs = None
        radial_coeffs = None
        tangential_coeffs = None
        thin_prism_coeffs = None
        with_ut = self.cfg.with_ut

        if camera_idcs is not None and hasattr(self, "ncore_camera_data"):
            cam = self.ncore_camera_data[camera_idcs.item()]
            camera_model = cam.camera_model
            ftheta_coeffs = cam.ftheta_coeffs
            if cam.radial_coeffs is not None:
                radial_coeffs = (
                    torch.from_numpy(cam.radial_coeffs).to(means.device).unsqueeze(0)
                )
            if cam.tangential_coeffs is not None:
                tangential_coeffs = (
                    torch.from_numpy(cam.tangential_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )
            if cam.thin_prism_coeffs is not None:
                thin_prism_coeffs = (
                    torch.from_numpy(cam.thin_prism_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=with_ut,
            with_eval3d=self.cfg.with_eval3d,
            ftheta_coeffs=ftheta_coeffs,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def _build_warp_cache(self) -> Dict[str, object]:
        train_indices = np.asarray(self.trainset.indices)
        camtoworlds = torch.from_numpy(self.parser.camtoworlds[train_indices]).float()
        Ks = []
        heights = []
        widths = []
        camera_idcs = []
        image_names = []
        ref_images = []
        ref_depths = []
        for dataset_idx, parser_idx in enumerate(train_indices):
            camera_id = self.parser.camera_ids[int(parser_idx)]
            width, height = self.parser.imsize_dict[camera_id]
            Ks.append(torch.from_numpy(self.parser.Ks_dict[camera_id]).float())
            widths.append(width)
            heights.append(height)
            camera_idcs.append(self.parser.camera_indices[int(parser_idx)])
            image_names.append(Path(self.parser.image_names[int(parser_idx)]).name)
            target_height, target_width = self._warp_size(height, width)
            ref_images.append(
                self._load_single_warp_reference_image(
                    dataset_idx, target_height, target_width
                )
            )
            ref_depths.append(
                self._load_single_warp_reference_depth(
                    dataset_idx, target_height, target_width
                )
            )

        return {
            "camtoworlds": camtoworlds,
            "centers": camtoworlds[:, :3, 3],
            "Ks": torch.stack(Ks, dim=0),
            "heights": torch.tensor(heights, dtype=torch.long),
            "widths": torch.tensor(widths, dtype=torch.long),
            "camera_idcs": torch.tensor(camera_idcs, dtype=torch.long),
            "image_names": image_names,
            "ref_images": ref_images,
            "ref_depths": ref_depths,
        }

    def _warp_size(self, height: int, width: int) -> Tuple[int, int]:
        downsample = max(1, self.cfg.warp_downsample)
        return max(2, height // downsample), max(2, width // downsample)

    def _scale_Ks(
        self,
        Ks: Tensor,
        heights: Tensor,
        widths: Tensor,
        target_height: int,
        target_width: int,
    ) -> Tensor:
        scaled_Ks = Ks.clone()
        sx = target_width / widths.to(Ks).float()
        sy = target_height / heights.to(Ks).float()
        scaled_Ks[:, 0, :] *= sx[:, None]
        scaled_Ks[:, 1, :] *= sy[:, None]
        return scaled_Ks

    def _select_warp_neighbors(
        self,
        exclude_image_idx: Optional[int],
        camtoworld: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        assert self.warp_cache is not None
        target_height, target_width = self._warp_size(height, width)
        cache = self.warp_cache
        cache_heights = cache["heights"]
        cache_widths = cache["widths"]
        cache_heights_ds = torch.tensor(
            [
                self._warp_size(int(h), int(w))[0]
                for h, w in zip(cache_heights, cache_widths)
            ],
            dtype=torch.long,
        )
        cache_widths_ds = torch.tensor(
            [
                self._warp_size(int(h), int(w))[1]
                for h, w in zip(cache_heights, cache_widths)
            ],
            dtype=torch.long,
        )
        same_size = (cache_heights_ds == target_height) & (
            cache_widths_ds == target_width
        )
        valid = same_size
        if exclude_image_idx is not None:
            valid = valid & (torch.arange(len(cache_heights)) != exclude_image_idx)
        candidates = torch.nonzero(valid, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=camtoworld.device)

        centers = cache["centers"].to(camtoworld.device)
        current_center = camtoworld[0, :3, 3].detach()
        distances = torch.linalg.norm(
            centers[candidates.to(camtoworld.device)] - current_center, dim=-1
        )
        order = torch.argsort(distances)
        selected = candidates[order.cpu()[: self.cfg.warp_num_neighbors]]
        return selected.to(camtoworld.device)

    def _load_single_warp_reference_image(
        self, dataset_idx: int, target_height: int, target_width: int
    ) -> Tensor:
        ref_data = self.trainset[int(dataset_idx)]
        image = ref_data["image"].float()[..., :3] / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(
            image,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return image.squeeze(0).permute(1, 2, 0).contiguous()

    def _get_cached_warp_reference_images(
        self, neighbor_idcs: Tensor, device: torch.device
    ) -> Tensor:
        assert self.warp_cache is not None
        ref_images = self.warp_cache["ref_images"]
        images = [
            ref_images[int(idx)].to(device)
            for idx in neighbor_idcs.detach().cpu().tolist()
        ]
        return torch.stack(images, dim=0)

    def _write_warp_reference_images(
        self,
        virtual_camera_source: str,
        neighbor_idcs: Tensor,
        ref_images: Tensor,
    ) -> None:
        assert self.warp_cache is not None
        neighbor_indices = tuple(
            int(idx) for idx in neighbor_idcs.detach().cpu().tolist()
        )
        save_key = (virtual_camera_source, neighbor_indices)
        if save_key in self._saved_warp_reference_sets:
            return

        safe_source = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_" for c in virtual_camera_source
        )
        save_dir = Path(self.cfg.result_dir) / "warp_reference_images" / safe_source
        save_dir.mkdir(parents=True, exist_ok=True)

        image_names = self.warp_cache["image_names"]
        manifest = {
            "virtual_camera_source": virtual_camera_source,
            "neighbor_image_indices": list(neighbor_indices),
            "references": [],
        }
        for ref_order, dataset_idx in enumerate(neighbor_indices):
            source_name = image_names[dataset_idx]
            source_stem = Path(source_name).stem
            filename = f"ref_{ref_order:02d}_idx_{dataset_idx:06d}_{source_stem}.png"
            image = (
                ref_images[ref_order].detach().clamp(0.0, 1.0).mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )
            imageio.imwrite(save_dir / filename, image)
            manifest["references"].append(
                {
                    "order": ref_order,
                    "dataset_index": dataset_idx,
                    "source_image": source_name,
                    "saved_image": filename,
                }
            )

        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        self._saved_warp_reference_sets.add(save_key)

    def _resolve_warp_depth_path(self, dataset_idx: int) -> Path:
        parser_idx = int(self.trainset.indices[dataset_idx])
        image_name = Path(self.parser.image_names[parser_idx])
        image_path = Path(self.parser.image_paths[parser_idx])
        depth_dir = Path(self.cfg.data_dir) / "depths"
        candidates = [
            depth_dir / image_name.name,
            depth_dir / image_path.name,
        ]
        for stem in {image_name.stem, image_path.stem}:
            for suffix in [".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                candidates.append(depth_dir / f"{stem}{suffix}")
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No depth map found for {image_name.name} under {depth_dir}"
        )

    def _read_warp_depth(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            depth = np.load(path)
        elif path.suffix.lower() == ".npz":
            data = np.load(path)
            key = "depth" if "depth" in data else data.files[0]
            depth = data[key]
        else:
            depth = imageio.imread(path)
            if depth.ndim == 3:
                depth = depth[..., 0]
            if np.issubdtype(depth.dtype, np.integer):
                depth = depth.astype(np.float32) / np.iinfo(depth.dtype).max
        return np.asarray(depth, dtype=np.float32)

    def _load_single_warp_reference_depth(
        self, dataset_idx: int, target_height: int, target_width: int
    ) -> Tensor:
        parser_idx = int(self.trainset.indices[int(dataset_idx)])
        camera_id = self.parser.camera_ids[parser_idx]
        depth = self._read_warp_depth(self._resolve_warp_depth_path(int(dataset_idx)))
        params = self.parser.params_dict[camera_id]
        if len(params) > 0:
            import cv2

            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            depth = cv2.remap(depth, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            depth = depth[y : y + h, x : x + w]
        depth_tensor = torch.from_numpy(depth).float()[None, None]
        depth_tensor = F.interpolate(
            depth_tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return depth_tensor.squeeze(0).permute(1, 2, 0).contiguous()

    def _get_cached_warp_reference_depths(
        self, neighbor_idcs: Tensor, device: torch.device
    ) -> Tensor:
        assert self.warp_cache is not None
        ref_depths = self.warp_cache["ref_depths"]
        depths = [
            ref_depths[int(idx)].to(device)
            for idx in neighbor_idcs.detach().cpu().tolist()
        ]
        return torch.stack(depths, dim=0)

    def _load_manual_warp_cameras(self) -> Optional[Dict[str, Tensor]]:
        if self.cfg.warp_manual_camera_path is None:
            return None

        path_arg = self.cfg.warp_manual_camera_path
        if path_arg in {"latest", "auto"}:
            camera_path_dir = Path(self.cfg.result_dir) / "camera_paths"
            candidates = sorted(camera_path_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if len(candidates) == 0:
                raise FileNotFoundError(f"No camera path JSON found in {camera_path_dir}")
            path = candidates[-1]
        else:
            path = Path(path_arg)
            if not path.is_absolute():
                data_path = Path(self.cfg.data_dir) / path
                result_path = Path(self.cfg.result_dir) / path
                result_camera_path = Path(self.cfg.result_dir) / "camera_paths" / path
                if data_path.exists():
                    path = data_path
                elif result_path.exists():
                    path = result_path
                else:
                    path = result_camera_path
        if not path.exists():
            raise FileNotFoundError(f"Manual warp camera path does not exist: {path}")

        with open(path, "r") as f:
            data = json.load(f)
        keyframes = data.get("keyframes", [])
        if len(keyframes) == 0:
            if self.cfg.warp_require_manual_keyframes:
                raise ValueError(f"No keyframes found in manual warp camera path: {path}")
            return None

        camtoworlds = []
        fovs = []
        aspects = []
        opencv_from_viewer = np.eye(4, dtype=np.float32)
        opencv_from_viewer[1, 1] = -1.0
        opencv_from_viewer[2, 2] = -1.0
        for keyframe in keyframes:
            if "matrix" not in keyframe:
                raise ValueError(f"Keyframe is missing matrix in {path}")
            saved_camtoworld = np.array(keyframe["matrix"], dtype=np.float32).reshape(4, 4)
            # nerfview saves keyframes as:
            #   matrix = [viewer_rotation @ Rx(pi), viewer_position / scale_ratio]
            # The training viewer render path uses the original viewer pose, so invert
            # that save-time convention and use only the manually added keyframes.
            camtoworld = saved_camtoworld @ opencv_from_viewer
            camtoworld[:3, 3] = (
                saved_camtoworld[:3, 3] * self.cfg.warp_manual_camera_scale_ratio
            )
            camtoworlds.append(torch.from_numpy(camtoworld))
            fovs.append(float(keyframe.get("fov", data.get("default_fov", 60.0))))
            aspects.append(float(keyframe.get("aspect", 1.0)))

        print(f"[Warp] Loaded {len(camtoworlds)} manual virtual cameras from {path}")
        return {
            "camtoworlds": torch.stack(camtoworlds, dim=0),
            "fovs": torch.tensor(fovs, dtype=torch.float32),
            "aspects": torch.tensor(aspects, dtype=torch.float32),
        }

    def _load_precomputed_virtual_refs(self) -> List[Dict[str, Tensor]]:
        ref_dir = Path(self.cfg.virtual_ref_dir)
        if not ref_dir.is_absolute():
            data_ref_dir = Path(self.cfg.data_dir) / ref_dir
            result_ref_dir = Path(self.cfg.result_dir) / ref_dir
            ref_dir = data_ref_dir if data_ref_dir.exists() else result_ref_dir
        if not ref_dir.exists():
            raise FileNotFoundError(f"Precomputed virtual ref dir does not exist: {ref_dir}")

        ref_paths = sorted(ref_dir.glob("ref_*.npz"))
        ref_paths = [p for p in ref_paths if not p.name.endswith("_debug.npz")]
        if len(ref_paths) == 0:
            raise FileNotFoundError(f"No ref_*.npz found in {ref_dir}")

        refs = []
        for path in ref_paths:
            data = np.load(path)
            required = ["C_ref", "M_ref", "weight_sum", "depth_virtual", "camtoworld", "K"]
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path} is missing keys: {missing}")
            refs.append(
                {
                    "C_ref": torch.from_numpy(data["C_ref"].astype(np.float32)),
                    "M_ref": torch.from_numpy(data["M_ref"].astype(bool)),
                    "weight_sum": torch.from_numpy(data["weight_sum"].astype(np.float32)),
                    "depth_virtual": torch.from_numpy(data["depth_virtual"].astype(np.float32)),
                    "camtoworld": torch.from_numpy(data["camtoworld"].astype(np.float32)),
                    "K": torch.from_numpy(data["K"].astype(np.float32)),
                    "source_keyframe_index": torch.tensor(
                        int(data["source_keyframe_index"])
                        if "source_keyframe_index" in data
                        else len(refs),
                        dtype=torch.long,
                    ),
                }
            )
        print(f"[Warp] Loaded {len(refs)} precomputed virtual refs from {ref_dir}")
        return refs

    def _make_K_from_fov(
        self,
        fov_deg: Tensor,
        aspect: Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        fov_rad = torch.deg2rad(fov_deg.to(device))
        fy = (0.5 * height) / torch.tan(0.5 * fov_rad).clamp(min=1e-6)
        fx = fy
        K = torch.eye(3, device=device).unsqueeze(0)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = width * 0.5
        K[:, 1, 2] = height * 0.5
        return K

    def _get_virtual_camera(
        self,
        step: int,
        current_camtoworld: Tensor,
        current_K: Tensor,
        source_height: int,
        source_width: int,
        target_height: int,
        target_width: int,
        virtual_camera_index: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, str]:
        if self.manual_warp_cameras is None:
            raise RuntimeError(
                "Manual warp cameras are not loaded. "
                "Set --warp-manual-camera-path to a viewer keyframe JSON."
            )

        camtoworlds = self.manual_warp_cameras["camtoworlds"]
        if virtual_camera_index is None:
            index = (step - self.cfg.warp_start_iter) // self.cfg.warp_interval
            index = int(index % len(camtoworlds))
        else:
            index = int(virtual_camera_index % len(camtoworlds))
        virtual_c2w = camtoworlds[index : index + 1].to(current_camtoworld.device)
        K_virtual = self._make_K_from_fov(
            self.manual_warp_cameras["fovs"][index : index + 1],
            self.manual_warp_cameras["aspects"][index : index + 1],
            target_height,
            target_width,
            current_camtoworld.device,
        )
        return virtual_c2w, K_virtual, f"manual_keyframe_{index}"

    def _backproject_depth(self, depths: Tensor, camtoworld: Tensor, K: Tensor) -> Tensor:
        height, width = depths.shape[:2]
        ys, xs = torch.meshgrid(
            torch.arange(height, device=depths.device, dtype=depths.dtype),
            torch.arange(width, device=depths.device, dtype=depths.dtype),
            indexing="ij",
        )
        z = depths[..., 0]
        x = (xs - K[0, 0, 2]) / K[0, 0, 0] * z
        y = (ys - K[0, 1, 2]) / K[0, 1, 1] * z
        points_cam = torch.stack([x, y, z], dim=-1)
        points_world = (
            torch.matmul(camtoworld[0, :3, :3], points_cam.reshape(-1, 3).T)
            + camtoworld[0, :3, 3:4]
        ).T
        return points_world.reshape(height, width, 3).detach()

    def _draw_point(
        self, canvas: np.ndarray, point: np.ndarray, color: Tuple[float, float, float], radius: int
    ) -> None:
        height, width = canvas.shape[:2]
        x, y = int(round(point[0])), int(round(point[1]))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        canvas[y0:y1, x0:x1][mask] = color

    def _draw_line(
        self,
        canvas: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        color: Tuple[float, float, float],
        steps: int = 64,
    ) -> None:
        for t in np.linspace(0.0, 1.0, steps):
            point = start * (1.0 - t) + end * t
            self._draw_point(canvas, point, color, radius=1)

    def _rotation_matrix_to_wxyz(self, rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rotation[2, 1] - rotation[1, 2]) / s
            qy = (rotation[0, 2] - rotation[2, 0]) / s
            qz = (rotation[1, 0] - rotation[0, 1]) / s
        else:
            diag = np.diag(rotation)
            if diag[0] > diag[1] and diag[0] > diag[2]:
                s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                qw = (rotation[2, 1] - rotation[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation[0, 1] + rotation[1, 0]) / s
                qz = (rotation[0, 2] + rotation[2, 0]) / s
            elif diag[1] > diag[2]:
                s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                qw = (rotation[0, 2] - rotation[2, 0]) / s
                qx = (rotation[0, 1] + rotation[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation[1, 2] + rotation[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                qw = (rotation[1, 0] - rotation[0, 1]) / s
                qx = (rotation[0, 2] + rotation[2, 0]) / s
                qy = (rotation[1, 2] + rotation[2, 1]) / s
                qz = 0.25 * s
        quat = np.array([qw, qx, qy, qz], dtype=np.float64)
        return quat / np.linalg.norm(quat)

    def _camera_fov_aspect(self, K: Tensor, height: int, width: int) -> Tuple[float, float]:
        fy = float(K[1, 1].detach().cpu())
        fov = 2.0 * math.atan(height / (2.0 * max(fy, 1e-6)))
        return fov, width / height

    def _init_warp_viewer_debug(self) -> None:
        if self.warp_cache is None:
            return
        centers = self.warp_cache["centers"].detach().cpu().numpy()
        colors = np.full((len(centers), 3), 170, dtype=np.uint8)
        self.server.scene.add_point_cloud(
            "/warp_cameras/all_training_centers",
            points=centers,
            colors=colors,
            point_size=max(0.002 * self.scene_scale, 0.001),
            point_shape="circle",
        )

    def _update_warp_viewer_cameras(
        self,
        step: int,
        current_c2w: Tensor,
        virtual_c2w: Tensor,
        ref_c2ws: Tensor,
        K_virtual: Tensor,
        ref_Ks: Tensor,
        virtual_image: Tensor,
        ref_images: Tensor,
        height: int,
        width: int,
    ) -> None:
        if self.cfg.disable_viewer or not self.cfg.warp_viewer_debug:
            return
        if not hasattr(self, "server"):
            return
        self.server.scene.remove_by_name("/warp_cameras/latest")
        self._warp_viewer_handles = []
        scale = max(0.08 * self.scene_scale, 0.02)

        def add_frustum(
            name: str,
            c2w: Tensor,
            K: Tensor,
            color: Tuple[int, int, int],
            image: Optional[Tensor],
        ) -> None:
            c2w_np = c2w.detach().cpu().numpy()
            fov, aspect = self._camera_fov_aspect(K, height, width)
            image_np = None
            if image is not None:
                image_np = (
                    image.detach().clamp(0.0, 1.0).cpu().numpy() * 255
                ).astype(np.uint8)
            self._warp_viewer_handles.append(
                self.server.scene.add_camera_frustum(
                    name,
                    fov=fov,
                    aspect=aspect,
                    scale=scale,
                    line_width=2.0,
                    color=color,
                    image=image_np,
                    wxyz=self._rotation_matrix_to_wxyz(c2w_np[:3, :3]),
                    position=c2w_np[:3, 3],
                )
            )

        add_frustum(
            f"/warp_cameras/latest/current_step_{step}",
            current_c2w[0],
            K_virtual[0],
            (255, 140, 0),
            None,
        )
        add_frustum(
            f"/warp_cameras/latest/virtual_step_{step}",
            virtual_c2w[0],
            K_virtual[0],
            (255, 0, 0),
            virtual_image,
        )
        for i in range(ref_c2ws.shape[0]):
            add_frustum(
                f"/warp_cameras/latest/reference_{i}_step_{step}",
                ref_c2ws[i],
                ref_Ks[i],
                (40, 90, 255),
                ref_images[i],
            )

    def _make_warp_camera_viz(
        self,
        current_c2w: Tensor,
        virtual_c2w: Tensor,
        ref_c2ws: Tensor,
        neighbor_idcs: Tensor,
    ) -> Tensor:
        assert self.warp_cache is not None
        canvas_size = 512
        margin = 32
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)

        all_c2ws = self.warp_cache["camtoworlds"].detach().cpu()
        all_centers = all_c2ws[:, :3, 3].numpy()
        current_center = current_c2w[0, :3, 3].detach().cpu().numpy()
        virtual_center = virtual_c2w[0, :3, 3].detach().cpu().numpy()
        ref_centers = ref_c2ws[:, :3, 3].detach().cpu().numpy()
        centers_for_axes = np.concatenate(
            [all_centers, current_center[None], virtual_center[None], ref_centers],
            axis=0,
        )
        axes = np.argsort(np.var(centers_for_axes, axis=0))[-2:]
        xy = centers_for_axes[:, axes]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        scale = (canvas_size - margin * 2) / max(float((xy_max - xy_min).max()), 1e-6)

        def project(points: np.ndarray) -> np.ndarray:
            projected = (points[:, axes] - xy_min) * scale + margin
            projected[:, 1] = canvas_size - projected[:, 1]
            return projected

        def draw_cameras(c2ws: np.ndarray, color: Tuple[float, float, float], radius: int) -> None:
            centers = project(c2ws[:, :3, 3])
            forwards = c2ws[:, :3, 2]
            forward_points = project(c2ws[:, :3, 3] + forwards * (0.04 / scale) * canvas_size)
            for center, forward_point in zip(centers, forward_points):
                self._draw_point(canvas, center, color, radius)
                self._draw_line(canvas, center, forward_point, color)

        draw_cameras(all_c2ws.numpy(), (0.70, 0.70, 0.70), 2)
        draw_cameras(ref_c2ws.detach().cpu().numpy(), (0.10, 0.35, 1.00), 5)
        draw_cameras(current_c2w.detach().cpu().numpy(), (1.00, 0.55, 0.05), 6)
        draw_cameras(virtual_c2w.detach().cpu().numpy(), (1.00, 0.05, 0.05), 6)
        return torch.from_numpy(canvas)

    def _make_warp_camera_debug(
        self,
        step: int,
        current_image_idx: int,
        virtual_camera_source: str,
        neighbor_idcs: Tensor,
        current_c2w: Tensor,
        virtual_c2w: Tensor,
        ref_c2ws: Tensor,
        K_virtual: Tensor,
        ref_Ks: Tensor,
    ) -> Dict[str, object]:
        return {
            "step": step,
            "current_image_idx": current_image_idx,
            "virtual_camera_source": virtual_camera_source,
            "neighbor_image_indices": neighbor_idcs.detach().cpu().tolist(),
            "current_camtoworld": current_c2w[0].detach().cpu().tolist(),
            "virtual_camtoworld": virtual_c2w[0].detach().cpu().tolist(),
            "reference_camtoworlds": ref_c2ws.detach().cpu().tolist(),
            "virtual_K": K_virtual[0].detach().cpu().tolist(),
            "reference_Ks": ref_Ks.detach().cpu().tolist(),
            "color_legend": {
                "all_training_cameras": "gray",
                "selected_reference_cameras": "blue",
                "current_training_camera": "orange",
                "virtual_camera": "red",
            },
        }

    def _compute_single_warp_loss(
        self,
        step: int,
        current_camtoworld: Tensor,
        current_K: Tensor,
        current_image_id: Tensor,
        sh_degree: int,
        height: int,
        width: int,
        virtual_camera_index: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, object]]]:
        assert self.warp_cache is not None
        device = current_camtoworld.device
        current_idx = int(current_image_id.item())
        target_height, target_width = self._warp_size(height, width)
        virtual_c2w, K_virtual, virtual_camera_source = self._get_virtual_camera(
            step,
            current_camtoworld,
            current_K,
            height,
            width,
            target_height,
            target_width,
            virtual_camera_index,
        )
        neighbor_idcs = self._select_warp_neighbors(
            None, virtual_c2w, height, width
        )
        if neighbor_idcs.numel() == 0:
            return torch.zeros((), device=device), None

        virtual_renders, _, _ = self.rasterize_splats(
            camtoworlds=virtual_c2w,
            Ks=K_virtual,
            width=target_width,
            height=target_height,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            image_ids=current_image_id,
            render_mode="RGB+ED",
            frame_idcs=None,
            camera_idcs=None,
            exposure=None,
        )
        virtual_colors = virtual_renders[..., :3]
        virtual_depths = virtual_renders[..., 3:4].detach()
        points_world = self._backproject_depth(
            virtual_depths[0], virtual_c2w, K_virtual
        )
        valid_virtual_depth = torch.isfinite(virtual_depths[0]) & (
            virtual_depths[0] > 0
        )

        cache = self.warp_cache
        neighbor_cpu = neighbor_idcs.detach().cpu()
        ref_c2ws = cache["camtoworlds"][neighbor_cpu].to(device)
        ref_heights = cache["heights"][neighbor_cpu].to(device)
        ref_widths = cache["widths"][neighbor_cpu].to(device)
        ref_Ks = self._scale_Ks(
            cache["Ks"][neighbor_cpu].to(device),
            ref_heights,
            ref_widths,
            target_height,
            target_width,
        )
        if self.cfg.pose_noise:
            ref_c2ws = self.pose_perturb(ref_c2ws, neighbor_idcs)
        if self.cfg.pose_opt:
            ref_c2ws = self.pose_adjust(ref_c2ws, neighbor_idcs)

        ref_depths = self._get_cached_warp_reference_depths(neighbor_idcs, device)
        ref_images = self._get_cached_warp_reference_images(neighbor_idcs, device)
        self._write_warp_reference_images(
            virtual_camera_source, neighbor_idcs, ref_images
        )
        self._update_warp_viewer_cameras(
            step,
            current_camtoworld,
            virtual_c2w,
            ref_c2ws,
            K_virtual,
            ref_Ks,
            virtual_colors[0].detach(),
            ref_images,
            target_height,
            target_width,
        )

        num_refs = int(neighbor_idcs.numel())
        points_flat = points_world.reshape(-1, 3)
        world_to_refs = torch.linalg.inv(ref_c2ws)
        points_cam = (
            torch.matmul(world_to_refs[:, :3, :3], points_flat.T)
            + world_to_refs[:, :3, 3:4]
        ).transpose(1, 2)
        z_ref = points_cam[..., 2].reshape(num_refs, target_height, target_width, 1)
        z_safe = z_ref.clamp(min=1e-6)
        u_ref = (
            ref_Ks[:, None, None, 0, 0]
            * points_cam[..., 0].reshape(num_refs, target_height, target_width)
            / z_safe[..., 0]
            + ref_Ks[:, None, None, 0, 2]
        )
        v_ref = (
            ref_Ks[:, None, None, 1, 1]
            * points_cam[..., 1].reshape(num_refs, target_height, target_width)
            / z_safe[..., 0]
            + ref_Ks[:, None, None, 1, 2]
        )
        inside = (
            (z_ref[..., 0] > 1e-6)
            & valid_virtual_depth[..., 0][None]
            & (u_ref >= 0)
            & (u_ref <= target_width - 1)
            & (v_ref >= 0)
            & (v_ref <= target_height - 1)
        )
        sample_grid = torch.stack(
            [
                u_ref / (target_width - 1) * 2 - 1,
                v_ref / (target_height - 1) * 2 - 1,
            ],
            dim=-1,
        )
        sampled_depths = F.grid_sample(
            ref_depths.permute(0, 3, 1, 2),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        sampled_colors = F.grid_sample(
            ref_images.permute(0, 3, 1, 2),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).permute(0, 2, 3, 1)

        depth_weight = torch.exp(
            -torch.abs(sampled_depths - z_ref) / self.cfg.warp_tau_depth
        )
        weights = inside[..., None].float() * depth_weight
        weight_sum = weights.sum(dim=0)
        ref_colors = (weights * sampled_colors).sum(dim=0) / (weight_sum + 1e-6)
        gamma_vis = torch.clamp(weight_sum, 0.0, 1.0)

        virtual_center = virtual_c2w[0, :3, 3]
        ref_centers = ref_c2ws[:, :3, 3]
        virtual_dirs = F.normalize(
            virtual_center[None, None, :] - points_world, dim=-1
        )
        ref_dirs = F.normalize(
            ref_centers[:, None, None, :] - points_world[None], dim=-1
        )
        d_max = (virtual_dirs[None] * ref_dirs).sum(dim=-1).max(dim=0).values
        cos_theta_max = math.cos(math.radians(self.cfg.warp_theta_max_deg))
        q_cam = torch.clamp(
            (d_max[..., None] - cos_theta_max) / (1.0 - cos_theta_max),
            0.0,
            1.0,
        )
        warp_weight = (1.0 - q_cam) * gamma_vis
        warp_weight = torch.where(
            gamma_vis >= self.cfg.warp_min_gamma,
            warp_weight,
            torch.zeros_like(warp_weight),
        )
        robust = F.smooth_l1_loss(
            virtual_colors[0], ref_colors.detach(), reduction="none"
        ).mean(dim=-1, keepdim=True)
        warp_loss = (robust * warp_weight.detach()).mean()
        #warp_loss = robust.mean() #一時的にwarp_weightの効果を切る
        debug = {
            "I_virtual": virtual_colors[0].detach(),
            "C_ref": ref_colors.detach(),
            "gamma_vis": gamma_vis.detach(),
            "q_cam": q_cam.detach(),
            "final_weight": warp_weight.detach(),
            "virtual_camera_source": virtual_camera_source,
        }
        if (
            self.cfg.warp_save_camera_debug
            and step % self.cfg.warp_camera_debug_interval == 0
        ):
            debug["camera_viewpoints"] = self._make_warp_camera_viz(
                current_camtoworld, virtual_c2w, ref_c2ws, neighbor_idcs
            )
            debug["camera_debug"] = self._make_warp_camera_debug(
                step,
                current_idx,
                virtual_camera_source,
                neighbor_idcs,
                current_camtoworld,
                virtual_c2w,
                ref_c2ws,
                K_virtual,
                ref_Ks,
            )
        return warp_loss, debug

    def _compute_warp_loss(
        self,
        step: int,
        current_camtoworld: Tensor,
        current_K: Tensor,
        current_image_id: Tensor,
        sh_degree: int,
        height: int,
        width: int,
    ) -> Tuple[Tensor, Optional[Dict[str, object]]]:
        losses = []
        debug = None
        if self.manual_warp_cameras is None:
            num_virtual_views = self.cfg.warp_num_virtual_views
        else:
            num_virtual_views = len(self.manual_warp_cameras["camtoworlds"])
        for virtual_camera_index in range(num_virtual_views):
            warp_loss, warp_debug = self._compute_single_warp_loss(
                step,
                current_camtoworld,
                current_K,
                current_image_id,
                sh_degree,
                height,
                width,
                virtual_camera_index,
            )
            losses.append(warp_loss)
            if debug is None and warp_debug is not None:
                debug = warp_debug
        return torch.stack(losses).mean(), debug

    def _compute_precomputed_warp_loss(
        self,
        step: int,
        current_image_id: Tensor,
        sh_degree: int,
    ) -> Tuple[Tensor, Optional[Dict[str, object]]]:
        if self.precomputed_virtual_refs is None:
            raise RuntimeError("Precomputed virtual refs are not loaded.")
        device = self.device
        ref_index = (step - self.cfg.warp_start_iter) // self.cfg.virtual_ref_interval
        ref_index = int(ref_index % len(self.precomputed_virtual_refs))
        ref = self.precomputed_virtual_refs[ref_index]

        c_ref = ref["C_ref"].to(device)
        mask = ref["M_ref"].to(device)
        weight_sum = ref["weight_sum"].to(device)
        camtoworld = ref["camtoworld"].to(device).unsqueeze(0)
        K = ref["K"].to(device).unsqueeze(0)
        height, width = c_ref.shape[:2]

        renders, _, _ = self.rasterize_splats(
            camtoworlds=camtoworld,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            image_ids=current_image_id,
            render_mode="RGB",
            frame_idcs=None,
            camera_idcs=None,
            exposure=None,
        )
        virtual_colors = renders[..., :3]
        robust = F.smooth_l1_loss(
            virtual_colors[0], c_ref.detach(), reduction="none"
        ).mean(dim=-1)
        mask_f = mask.float()
        warp_loss = (robust * mask_f.detach()).sum() / (mask_f.sum() + 1e-6)
        source_idx = int(ref["source_keyframe_index"].item())
        debug = {
            "I_virtual": virtual_colors[0].detach(),
            "C_ref": c_ref.detach(),
            "gamma_vis": weight_sum.detach()[..., None].clamp(0.0, 1.0),
            "q_cam": torch.zeros_like(weight_sum.detach()[..., None]),
            "final_weight": mask_f.detach()[..., None],
            "virtual_camera_source": f"precomputed_ref_{source_idx}",
        }
        return warp_loss, debug

    def _write_warp_debug_images(
        self, warp_debug: Dict[str, object], step: int
    ) -> None:
        for name, image in warp_debug.items():
            if not isinstance(image, Tensor):
                continue
            image = image.detach().clamp(0.0, 1.0).cpu()
            if image.shape[-1] == 1:
                image = image.repeat(1, 1, 3)
            self.writer.add_image(f"warp/{name}", image, step, dataformats="HWC")

    def _write_warp_virtual_images(
        self, warp_debug: Dict[str, object], step: int
    ) -> None:
        virtual_camera_source = str(warp_debug.get("virtual_camera_source", "unknown"))
        safe_source = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_"
            for c in virtual_camera_source
        )
        save_dir = Path(self.cfg.result_dir) / "warp_virtual_images" / safe_source
        save_dir.mkdir(parents=True, exist_ok=True)

        for name in ("I_virtual", "C_ref"):
            image = warp_debug.get(name)
            if not isinstance(image, Tensor):
                continue
            image = image.detach().clamp(0.0, 1.0)
            if image.shape[-1] == 1:
                image = image.repeat(1, 1, 3)
            image = image.mul(255.0).byte().cpu().numpy()
            imageio.imwrite(save_dir / f"step_{step:06d}_{name}.png", image)

    def _write_warp_camera_debug_json(
        self, warp_debug: Dict[str, object], step: int
    ) -> None:
        camera_debug = warp_debug.get("camera_debug")
        if camera_debug is None:
            return
        debug_dir = Path(self.cfg.result_dir) / "warp_camera_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / f"step_{step:06d}_rank{self.world_rank}.json", "w") as f:
            json.dump(camera_debug, f, indent=2)

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3 or 4]
            # RGBA画像の場合、アルファチャンネルを分離してRGBのみにする
            alpha_gt = None
            if pixels.shape[-1] == 4:
                alpha_gt = pixels[..., 3:4]
                pixels = pixels[..., :3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
                if alpha_gt is not None:
                    pixels = pixels * alpha_gt + bkgd * (1.0 - alpha_gt)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            if masks is not None:
                # Exclude masked pixels (e.g. ego vehicle) from L1.
                # For SSIM (patch-based), zero out both sides at masked locations
                # so masked patches don't pull colors toward an arbitrary value.
                l1loss = F.l1_loss(colors[masks], pixels[masks])
                colors_ssim = colors * masks[..., None]
                pixels_ssim = pixels * masks[..., None]
            else:
                l1loss = F.l1_loss(colors, pixels)
                colors_ssim = colors
                pixels_ssim = pixels
            ssimloss = 1.0 - fused_ssim(
                colors_ssim.permute(0, 3, 1, 2),
                pixels_ssim.permute(0, 3, 1, 2),
                padding="valid",
            )
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            warp_loss = None
            warp_debug = None
            if (
                cfg.use_warp_loss
                and step >= cfg.warp_start_iter
                and step % cfg.warp_interval == 0
            ):
                if cfg.use_precomputed_virtual_refs:
                    warp_loss, warp_debug = self._compute_precomputed_warp_loss(
                        step=step,
                        current_image_id=image_ids,
                        sh_degree=sh_degree_to_use,
                    )
                else:
                    warp_loss, warp_debug = self._compute_warp_loss(
                        step=step,
                        current_camtoworld=camtoworlds,
                        current_K=Ks,
                        current_image_id=image_ids,
                        sh_degree=sh_degree_to_use,
                        height=height,
                        width=width,
                    )
                loss += cfg.lambda_warp * warp_loss
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if warp_loss is not None:
                desc += f"warp loss={warp_loss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)
            if (
                world_rank == 0
                and cfg.warp_save_camera_debug
                and warp_debug is not None
            ):
                self._write_warp_camera_debug_json(warp_debug, step)
                self._write_warp_virtual_images(warp_debug, step)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if warp_loss is not None:
                    self.writer.add_scalar("train/warp_loss", warp_loss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step, dataformats="HWC")
                    if warp_debug is not None:
                        self._write_warp_debug_images(warp_debug, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            # RGBA画像の場合、評価でもアルファチャンネルを分離してRGBのみにする
            if pixels.shape[-1] == 4:
                pixels = pixels[..., :3]
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "raw":
            # Use captured poses as-is
            camtoworlds_all = camtoworlds_all[:, :3, :]  # [N, 3, 4]
        elif cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut and cfg.with_eval3d:
        print(
            "[Trainer] Note: with_ut=True + with_eval3d=True (full 3DGUT mode). "
            "DefaultStrategy is incompatible with eval3d; use MCMCStrategy (the `mcmc` subcommand)."
        )

    cli(main, cfg, verbose=True)
