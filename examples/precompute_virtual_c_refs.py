#!/usr/bin/env python
"""Precompute virtual-view reference colors from all COLMAP training images."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from datasets.colmap import Dataset, Parser


def load_manual_keyframes(path: Path, scale_ratio: float) -> Tuple[List[np.ndarray], List[float], List[float]]:
    with path.open("r") as f:
        data = json.load(f)
    keyframes = data.get("keyframes", [])
    if len(keyframes) == 0:
        raise ValueError(f"No keyframes found in {path}")

    opencv_from_viewer = np.eye(4, dtype=np.float32)
    opencv_from_viewer[1, 1] = -1.0
    opencv_from_viewer[2, 2] = -1.0
    camtoworlds = []
    fovs = []
    aspects = []
    for keyframe in keyframes:
        saved_camtoworld = np.array(keyframe["matrix"], dtype=np.float32).reshape(4, 4)
        camtoworld = saved_camtoworld @ opencv_from_viewer
        camtoworld[:3, 3] = saved_camtoworld[:3, 3] * scale_ratio
        camtoworlds.append(camtoworld.astype(np.float32))
        fovs.append(float(keyframe.get("fov", data.get("default_fov", 60.0))))
        aspects.append(float(keyframe.get("aspect", 1.0)))
    return camtoworlds, fovs, aspects


def make_k_from_fov(fov_deg: float, height: int, width: int) -> np.ndarray:
    fov_rad = np.deg2rad(fov_deg)
    fy = (0.5 * height) / max(np.tan(0.5 * fov_rad), 1e-6)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fy
    K[1, 1] = fy
    K[0, 2] = width * 0.5
    K[1, 2] = height * 0.5
    return K


def project_points_to_depth(
    points_world: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    splat_radius: int,
    sigma_s: float,
    tau: float,
) -> np.ndarray:
    worldtocam = np.linalg.inv(camtoworld)
    points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
    z = points_cam[:, 2]
    valid = z > 0.0
    points_cam = points_cam[valid]
    z = z[valid]
    proj = (K @ points_cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)

    numerator = np.zeros((height, width), dtype=np.float32)
    denominator = np.zeros((height, width), dtype=np.float32)
    min_depth = np.full((height, width), np.inf, dtype=np.float32)
    radius = max(0, int(splat_radius))
    sigma_s = max(float(sigma_s), 1e-6)
    tau = max(float(tau), 1e-6)
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ] or [(0, 0)]

    for dy, dx in offsets:
        xx = x + dx
        yy = y + dy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        np.minimum.at(min_depth, (yy[inside], xx[inside]), z[inside].astype(np.float32))

    for dy, dx in offsets:
        xx = x + dx
        yy = y + dy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        if not np.any(inside):
            continue
        dist2 = (xx[inside].astype(np.float32) - uv[inside, 0]) ** 2 + (
            yy[inside].astype(np.float32) - uv[inside, 1]
        ) ** 2
        spatial_weight = np.exp(-dist2 / (2.0 * sigma_s * sigma_s)).astype(np.float32)
        # Subtracting the per-pixel minimum z keeps the soft z-buffer numerically stable.
        z_inside = z[inside].astype(np.float32)
        z_weight = np.exp(
            -(z_inside - min_depth[yy[inside], xx[inside]]) / tau
        ).astype(np.float32)
        weight = spatial_weight * z_weight
        np.add.at(numerator, (yy[inside], xx[inside]), weight * z_inside)
        np.add.at(denominator, (yy[inside], xx[inside]), weight)

    depth = numerator / (denominator + 1e-6)
    depth[denominator <= 0.0] = 0.0
    return depth


def backproject_depth(depth: np.ndarray, camtoworld: np.ndarray, K: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    z = depth.astype(np.float32)
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    points_world = (camtoworld[:3, :3] @ points_cam.T + camtoworld[:3, 3:4]).T
    return points_world.reshape(height, width, 3).astype(np.float32)


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.clip(u0 + 1, 0, width - 1)
    v1 = np.clip(v0 + 1, 0, height - 1)
    u0 = np.clip(u0, 0, width - 1)
    v0 = np.clip(v0, 0, height - 1)
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)
    wa = (1.0 - du) * (1.0 - dv)
    wb = du * (1.0 - dv)
    wc = (1.0 - du) * dv
    wd = du * dv
    if image.ndim == 2:
        return (
            wa * image[v0, u0]
            + wb * image[v0, u1]
            + wc * image[v1, u0]
            + wd * image[v1, u1]
        )
    return (
        wa[:, None] * image[v0, u0]
        + wb[:, None] * image[v0, u1]
        + wc[:, None] * image[v1, u0]
        + wd[:, None] * image[v1, u1]
    )


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        depth = data["depth"] if "depth" in data else data[data.files[0]]
    else:
        depth = imageio.imread(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if np.issubdtype(depth.dtype, np.integer):
            depth = depth.astype(np.float32) / np.iinfo(depth.dtype).max
    return np.asarray(depth, dtype=np.float32)


def resolve_depth_path(data_dir: Path, image_name: str, image_path: str) -> Path:
    depth_dir = data_dir / "depths"
    image_name_path = Path(image_name)
    image_path_path = Path(image_path)
    candidates = [depth_dir / image_name_path.name, depth_dir / image_path_path.name]
    for stem in {image_name_path.stem, image_path_path.stem}:
        for suffix in [".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidates.append(depth_dir / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No depth found for {image_name} under {depth_dir}")


def resize_tensor_hwc(array: np.ndarray, height: int, width: int, mode: str = "bilinear") -> np.ndarray:
    tensor = torch.from_numpy(array).float()
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    else:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    if array.ndim == 2:
        return resized[0, 0].numpy()
    return resized[0].permute(1, 2, 0).numpy()


def downsample_hwc(array: np.ndarray, height: int, width: int, mode: str = "area") -> np.ndarray:
    tensor = torch.from_numpy(array).float()
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    else:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode=mode)
    if array.ndim == 2:
        return resized[0, 0].numpy()
    return resized[0].permute(1, 2, 0).numpy()


def save_debug_images(debug_dir: Path, c_ref: np.ndarray, mask: np.ndarray, weight_sum: np.ndarray, depth: np.ndarray) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(debug_dir / "C_ref.png", np.clip(c_ref * 255.0, 0, 255).astype(np.uint8))
    imageio.imwrite(debug_dir / "M_ref.png", np.clip(mask * 255.0, 0, 255).astype(np.uint8))
    weight_norm = weight_sum / max(float(np.percentile(weight_sum, 99.0)), 1e-6)
    imageio.imwrite(debug_dir / "weight_sum.png", np.clip(weight_norm * 255.0, 0, 255).astype(np.uint8))
    valid = depth > 0.0
    depth_vis = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        lo = float(np.percentile(depth[valid], 1.0))
        hi = float(np.percentile(depth[valid], 99.0))
        depth_vis[valid] = (depth[valid] - lo) / max(hi - lo, 1e-6)
    imageio.imwrite(debug_dir / "depth_virtual.png", np.clip(depth_vis * 255.0, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--normalize-world-space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--virtual-camera-path", type=Path, default=Path("camera_paths/virtual.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warp-downsample", type=int, default=2)
    parser.add_argument(
        "--virtual-supersample",
        type=int,
        default=1,
        help="Compute virtual refs at Nx resolution and area-downsample to reduce jagged silhouettes.",
    )
    parser.add_argument("--tau-z", type=float, default=0.01)
    parser.add_argument("--view-gamma", type=float, default=4.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-weight-sum", type=float, default=0.05)
    parser.add_argument("--splat-radius", type=int, default=2)
    parser.add_argument(
        "--soft-z-sigma-s",
        type=float,
        default=None,
        help="Pixel-space Gaussian sigma for soft z-buffer splatting. Defaults to splat_radius / 2.",
    )
    parser.add_argument(
        "--soft-z-tau",
        type=float,
        default=0.05,
        help="Depth temperature for soft z-buffer. Smaller values prioritize nearer points.",
    )
    parser.add_argument("--manual-camera-scale-ratio", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    virtual_ref_dir = output_dir / "virtual_refs"
    if virtual_ref_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{virtual_ref_dir} exists. Pass --overwrite.")
        shutil.rmtree(virtual_ref_dir)
    virtual_ref_dir.mkdir(parents=True, exist_ok=True)

    virtual_camera_path = args.virtual_camera_path
    if not virtual_camera_path.is_absolute():
        virtual_camera_path = data_dir / virtual_camera_path
    camtoworlds_virtual, fovs, _ = load_manual_keyframes(
        virtual_camera_path, args.manual_camera_scale_ratio
    )

    parser_obj = Parser(
        data_dir=str(data_dir),
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
        load_exposure=False,
    )
    trainset = Dataset(parser_obj, split="train")
    train_indices = np.asarray(trainset.indices)
    first_camera_id = parser_obj.camera_ids[int(train_indices[0])]
    source_width, source_height = parser_obj.imsize_dict[first_camera_id]
    out_height = max(2, int(source_height) // max(1, args.warp_downsample))
    out_width = max(2, int(source_width) // max(1, args.warp_downsample))
    supersample = max(1, int(args.virtual_supersample))
    height = out_height * supersample
    width = out_width * supersample
    soft_z_sigma_s = args.soft_z_sigma_s
    if soft_z_sigma_s is None:
        soft_z_sigma_s = max(float(args.splat_radius) * 0.5, 1.0)

    ref_c2ws = parser_obj.camtoworlds[train_indices].astype(np.float32)
    ref_centers = ref_c2ws[:, :3, 3]
    ref_Ks = []
    ref_images = []
    ref_depths = []
    for dataset_idx, parser_idx in enumerate(tqdm.tqdm(train_indices, desc="Load refs")):
        camera_id = parser_obj.camera_ids[int(parser_idx)]
        orig_width, orig_height = parser_obj.imsize_dict[camera_id]
        K = parser_obj.Ks_dict[camera_id].copy().astype(np.float32)
        K[0, :] *= width / float(orig_width)
        K[1, :] *= height / float(orig_height)
        ref_Ks.append(K)

        data = trainset[dataset_idx]
        image = (data["image"].numpy()[..., :3] / 255.0).astype(np.float32)
        ref_images.append(resize_tensor_hwc(image, height, width))

        depth_path = resolve_depth_path(
            data_dir,
            parser_obj.image_names[int(parser_idx)],
            parser_obj.image_paths[int(parser_idx)],
        )
        ref_depths.append(resize_tensor_hwc(load_depth(depth_path), height, width))
    ref_Ks = np.stack(ref_Ks, axis=0)

    points_world = parser_obj.points.astype(np.float32)
    num_pixels = height * width
    topk = max(1, min(int(args.topk), len(train_indices)))

    for virtual_idx, virtual_c2w in enumerate(tqdm.tqdm(camtoworlds_virtual, desc="Virtual refs")):
        K_virtual = make_k_from_fov(fovs[virtual_idx], height, width)
        depth_virtual = project_points_to_depth(
            points_world,
            virtual_c2w,
            K_virtual,
            height,
            width,
            args.splat_radius,
            soft_z_sigma_s,
            args.soft_z_tau,
        )
        valid_virtual = depth_virtual.reshape(-1) > 0.0
        points = backproject_depth(depth_virtual, virtual_c2w, K_virtual).reshape(-1, 3)

        top_weights = np.zeros((topk, num_pixels), dtype=np.float32)
        top_colors = np.zeros((topk, num_pixels, 3), dtype=np.float32)
        virtual_center = virtual_c2w[:3, 3]
        v_virtual = virtual_center[None, :] - points
        v_virtual /= np.linalg.norm(v_virtual, axis=-1, keepdims=True).clip(min=1e-6)

        for ref_idx in tqdm.trange(len(train_indices), desc=f"Fuse {virtual_idx:04d}", leave=False):
            worldtocam = np.linalg.inv(ref_c2ws[ref_idx])
            points_cam = (worldtocam[:3, :3] @ points.T + worldtocam[:3, 3:4]).T
            z_ref = points_cam[:, 2]
            z_safe = np.clip(z_ref, 1e-6, None)
            u = ref_Ks[ref_idx, 0, 0] * points_cam[:, 0] / z_safe + ref_Ks[ref_idx, 0, 2]
            v = ref_Ks[ref_idx, 1, 1] * points_cam[:, 1] / z_safe + ref_Ks[ref_idx, 1, 2]
            inside = (
                valid_virtual
                & (z_ref > 1e-6)
                & (u >= 0.0)
                & (u <= width - 1)
                & (v >= 0.0)
                & (v <= height - 1)
            )
            sampled_depth = bilinear_sample(ref_depths[ref_idx], u, v)
            sampled_color = bilinear_sample(ref_images[ref_idx], u, v)
            depth_valid = sampled_depth > 0.0
            m_vis = inside.astype(np.float32) * depth_valid.astype(np.float32)
            m_vis *= np.exp(-np.abs(sampled_depth - z_ref) / args.tau_z).astype(np.float32)

            v_ref = ref_centers[ref_idx][None, :] - points
            v_ref /= np.linalg.norm(v_ref, axis=-1, keepdims=True).clip(min=1e-6)
            s_view = np.maximum(0.0, np.sum(v_virtual * v_ref, axis=-1)) ** args.view_gamma
            weights = (m_vis * s_view).astype(np.float32)

            combined_weights = np.concatenate([top_weights, weights[None]], axis=0)
            combined_colors = np.concatenate([top_colors, sampled_color[None].astype(np.float32)], axis=0)
            keep = np.argpartition(combined_weights, -topk, axis=0)[-topk:]
            cols = np.arange(num_pixels)[None, :]
            top_weights = combined_weights[keep, cols]
            top_colors = combined_colors[keep, cols]

        weight_sum = top_weights.sum(axis=0)
        c_ref = (top_weights[..., None] * top_colors).sum(axis=0) / (weight_sum[:, None] + 1e-6)
        mask = (weight_sum >= args.min_weight_sum).astype(np.float32)
        c_ref = c_ref.reshape(height, width, 3).astype(np.float32)
        mask = mask.reshape(height, width).astype(np.float32)
        weight_sum = weight_sum.reshape(height, width).astype(np.float32)
        if supersample > 1:
            c_ref_premult = c_ref * mask[..., None]
            c_ref_premult = downsample_hwc(c_ref_premult, out_height, out_width)
            mask = downsample_hwc(mask, out_height, out_width).astype(np.float32)
            c_ref = c_ref_premult / np.clip(mask[..., None], 1e-6, None)
            c_ref = np.where(mask[..., None] > 1e-6, c_ref, 0.0).astype(np.float32)
            weight_sum = downsample_hwc(weight_sum, out_height, out_width).astype(np.float32)
            depth_virtual = downsample_hwc(depth_virtual, out_height, out_width).astype(np.float32)

        out_path = virtual_ref_dir / f"ref_{virtual_idx:04d}.npz"
        np.savez_compressed(
            out_path,
            C_ref=c_ref,
            M_ref=mask,
            weight_sum=weight_sum,
            depth_virtual=depth_virtual.astype(np.float32),
            camtoworld=virtual_c2w.astype(np.float32),
            K=K_virtual.astype(np.float32),
            source_keyframe_index=np.array(virtual_idx, dtype=np.int32),
        )
        save_debug_images(
            virtual_ref_dir / f"ref_{virtual_idx:04d}_debug",
            c_ref,
            mask,
            weight_sum,
            depth_virtual,
        )
        print(
            f"{out_path.name}: mask_mean={mask.mean():.4f}, "
            f"weight_max={weight_sum.max():.4f}, C_ref_mean={c_ref.mean():.4f}"
        )


if __name__ == "__main__":
    main()
