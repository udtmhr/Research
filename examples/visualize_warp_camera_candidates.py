#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import viser

from datasets.colmap import Dataset, Parser


Color = Tuple[int, int, int]


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
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


def normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), eps)


def camera_fov_aspect(K: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    fy = float(K[1, 1])
    fov = 2.0 * math.atan(height / (2.0 * max(fy, 1e-6)))
    return fov, width / height


def look_at_opencv(eye: np.ndarray, target: np.ndarray, source_c2w: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    up_hint = normalize(-source_c2w[:3, 1])
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = source_c2w[:3, 0]
    right = normalize(right)
    down = normalize(np.cross(forward, right))

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def offset_local(c2w: np.ndarray, offset: np.ndarray) -> np.ndarray:
    out = c2w.copy()
    out[:3, 3] = c2w[:3, 3] + c2w[:3, :3] @ offset
    return out


def select_neighbors(camtoworlds: np.ndarray, source: int, num_neighbors: int) -> np.ndarray:
    centers = camtoworlds[:, :3, 3]
    distances = np.linalg.norm(centers - centers[source], axis=-1)
    distances[source] = np.inf
    return np.argsort(distances)[:num_neighbors]


def build_candidates(
    source_c2w: np.ndarray,
    neighbor_c2ws: np.ndarray,
    K: np.ndarray,
    width: int,
    scene_scale: float,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    source_center = source_c2w[:3, 3]
    neighbor_centers = neighbor_c2ws[:, :3, 3]
    neighbor_distances = np.linalg.norm(neighbor_centers - source_center[None], axis=-1)
    median_neighbor_distance = float(np.median(neighbor_distances))
    baseline = max(median_neighbor_distance, 0.03 * scene_scale)
    focus_distance = max(median_neighbor_distance * 3.0, 0.1 * scene_scale)
    focus = source_center + source_c2w[:3, 2] * focus_distance

    candidates: List[Dict[str, object]] = []

    for i, scale in enumerate([0.25, 0.5, 1.0]):
        direction = normalize(rng.normal(size=2))
        pose = offset_local(source_c2w, np.array([direction[0], direction[1], 0.0]) * baseline * scale)
        candidates.append(
            {
                "method": "local_jitter",
                "name": f"jitter_{i}",
                "camtoworld": pose,
                "params": {"baseline_scale": scale},
            }
        )

    for i, neighbor_c2w in enumerate(neighbor_c2ws[:3]):
        for t in [0.25, 0.5, 0.75]:
            eye = source_center * (1.0 - t) + neighbor_c2w[:3, 3] * t
            pose = look_at_opencv(eye, focus, source_c2w)
            candidates.append(
                {
                    "method": "neighbor_interp",
                    "name": f"neighbor_{i}_t_{t:.2f}",
                    "camtoworld": pose,
                    "params": {"neighbor_rank": i, "t": t},
                }
            )

    for i, neighbor_c2w in enumerate(neighbor_c2ws[:3]):
        direction = source_center - neighbor_c2w[:3, 3]
        for alpha in [0.25, 0.5]:
            eye = source_center + direction * alpha
            pose = look_at_opencv(eye, focus, source_c2w)
            candidates.append(
                {
                    "method": "neighbor_extrap",
                    "name": f"neighbor_{i}_alpha_{alpha:.2f}",
                    "camtoworld": pose,
                    "params": {"neighbor_rank": i, "alpha": alpha},
                }
            )

    for yaw_deg, pitch_deg in [(-15.0, 0.0), (15.0, 0.0), (0.0, -10.0), (0.0, 10.0)]:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        offset = (
            source_c2w[:3, 0] * math.sin(yaw) * baseline
            + source_c2w[:3, 1] * math.sin(pitch) * baseline
        )
        eye = source_center + offset
        pose = look_at_opencv(eye, focus, source_c2w)
        candidates.append(
            {
                "method": "focus_orbit",
                "name": f"yaw_{yaw_deg:.0f}_pitch_{pitch_deg:.0f}",
                "camtoworld": pose,
                "params": {"yaw_deg": yaw_deg, "pitch_deg": pitch_deg},
            }
        )

    fx = max(float(K[0, 0]), 1e-6)
    for parallax_px in [16.0, 32.0, 64.0]:
        shift = parallax_px / fx * focus_distance
        for axis_name, offset in [
            ("x", np.array([shift, 0.0, 0.0])),
            ("y", np.array([0.0, shift, 0.0])),
        ]:
            pose = offset_local(source_c2w, offset)
            pose = look_at_opencv(pose[:3, 3], focus, source_c2w)
            candidates.append(
                {
                    "method": "parallax_shift",
                    "name": f"{axis_name}_{parallax_px:.0f}px",
                    "camtoworld": pose,
                    "params": {"axis": axis_name, "parallax_px": parallax_px},
                }
            )

    return candidates


def add_frustum(
    server: viser.ViserServer,
    name: str,
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    scale: float,
    color: Color,
) -> None:
    fov, aspect = camera_fov_aspect(K, width, height)
    server.scene.add_camera_frustum(
        name,
        fov=fov,
        aspect=aspect,
        scale=scale,
        line_width=2.0,
        color=color,
        wxyz=rotation_matrix_to_wxyz(c2w[:3, :3]),
        position=c2w[:3, 3],
    )


def save_metadata(
    output_dir: Path,
    source_rank: int,
    source_parser_index: int,
    neighbor_ranks: np.ndarray,
    neighbor_parser_indices: np.ndarray,
    source_c2w: np.ndarray,
    candidates: List[Dict[str, object]],
) -> None:
    debug_dir = output_dir / "warp_camera_candidate_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_rank": int(source_rank),
        "source_parser_index": int(source_parser_index),
        "neighbor_ranks": neighbor_ranks.astype(int).tolist(),
        "neighbor_parser_indices": neighbor_parser_indices.astype(int).tolist(),
        "source_camtoworld": source_c2w.tolist(),
        "candidates": [
            {
                "method": str(candidate["method"]),
                "name": str(candidate["name"]),
                "params": candidate["params"],
                "camtoworld": np.asarray(candidate["camtoworld"]).tolist(),
            }
            for candidate in candidates
        ],
    }
    with open(debug_dir / f"source_{source_rank:04d}.json", "w") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize candidate virtual camera poses for warp-loss experiments."
    )
    parser.add_argument("--data-dir", "--data_dir", type=str, required=True)
    parser.add_argument("--data-factor", "--data_factor", type=int, default=1)
    parser.add_argument("--test-every", "--test_every", type=int, default=8)
    parser.add_argument(
        "--normalize-world-space",
        "--normalize_world_space",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", "--output_dir", type=str, default="results")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--source-index", "--source_index", type=int, default=0)
    parser.add_argument("--num-neighbors", "--num_neighbors", type=int, default=3)
    parser.add_argument("--num-random-sources", "--num_random_sources", type=int, default=0)
    parser.add_argument(
        "--show-all-train",
        "--show_all_train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--frustum-scale", "--frustum_scale", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
        load_exposure=False,
    )
    trainset = Dataset(parser, split="train")
    train_indices = np.asarray(trainset.indices)
    if len(train_indices) < 2:
        raise ValueError("Need at least two training cameras to build candidates.")
    if not (0 <= args.source_index < len(train_indices)):
        raise ValueError(
            f"--source-index must be in [0, {len(train_indices) - 1}], got {args.source_index}."
        )
    if args.num_neighbors <= 0:
        raise ValueError("--num-neighbors must be positive.")

    source_ranks = [args.source_index]
    if args.num_random_sources > 0:
        count = min(args.num_random_sources, len(train_indices))
        random_ranks = rng.choice(len(train_indices), size=count, replace=False).tolist()
        source_ranks = sorted(set(source_ranks + random_ranks))

    train_c2ws = parser.camtoworlds[train_indices].astype(np.float64)
    train_centers = train_c2ws[:, :3, 3]
    first_camera_id = parser.camera_ids[int(train_indices[args.source_index])]
    first_width, first_height = parser.imsize_dict[first_camera_id]
    first_K = parser.Ks_dict[first_camera_id]

    server = viser.ViserServer(port=args.port, verbose=False)
    server.gui.set_panel_label("warp camera candidates")

    if args.show_all_train:
        colors = np.full((len(train_centers), 3), 170, dtype=np.uint8)
        server.scene.add_point_cloud(
            "/training/all_centers",
            points=train_centers,
            colors=colors,
            point_size=max(0.002 * parser.scene_scale, 0.001),
            point_shape="circle",
        )

    method_colors: Dict[str, Color] = {
        "local_jitter": (255, 70, 70),
        "neighbor_interp": (255, 120, 40),
        "neighbor_extrap": (220, 20, 140),
        "focus_orbit": (180, 40, 255),
        "parallax_shift": (255, 200, 30),
    }
    output_dir = Path(args.output_dir)
    frustum_scale = max(args.frustum_scale * parser.scene_scale, 0.01)

    for source_rank in source_ranks:
        source_parser_index = int(train_indices[source_rank])
        source_c2w = train_c2ws[source_rank]
        camera_id = parser.camera_ids[source_parser_index]
        K = parser.Ks_dict[camera_id]
        width, height = parser.imsize_dict[camera_id]
        neighbor_ranks = select_neighbors(
            train_c2ws, source_rank, min(args.num_neighbors, len(train_indices) - 1)
        )
        neighbor_parser_indices = train_indices[neighbor_ranks]
        neighbor_c2ws = train_c2ws[neighbor_ranks]
        candidates = build_candidates(
            source_c2w,
            neighbor_c2ws,
            K,
            width,
            parser.scene_scale,
            rng,
        )

        prefix = f"/source_{source_rank:04d}"
        add_frustum(
            server,
            f"{prefix}/source",
            source_c2w,
            K,
            width,
            height,
            frustum_scale,
            (255, 140, 0),
        )
        for neighbor_order, neighbor_rank in enumerate(neighbor_ranks):
            neighbor_parser_index = int(train_indices[neighbor_rank])
            neighbor_camera_id = parser.camera_ids[neighbor_parser_index]
            neighbor_K = parser.Ks_dict[neighbor_camera_id]
            neighbor_width, neighbor_height = parser.imsize_dict[neighbor_camera_id]
            add_frustum(
                server,
                f"{prefix}/neighbors/neighbor_{neighbor_order}_rank_{neighbor_rank}",
                train_c2ws[neighbor_rank],
                neighbor_K,
                neighbor_width,
                neighbor_height,
                frustum_scale,
                (40, 90, 255),
            )

        method_counts: Dict[str, int] = {}
        for candidate in candidates:
            method = str(candidate["method"])
            method_counts[method] = method_counts.get(method, 0) + 1
            name = str(candidate["name"])
            add_frustum(
                server,
                f"{prefix}/candidates/{method}/{method_counts[method]:02d}_{name}",
                np.asarray(candidate["camtoworld"]),
                K,
                width,
                height,
                frustum_scale * 0.8,
                method_colors[method],
            )

        save_metadata(
            output_dir,
            source_rank,
            source_parser_index,
            neighbor_ranks,
            neighbor_parser_indices,
            source_c2w,
            candidates,
        )

    print(f"Viewer running at http://localhost:{args.port}")
    print(f"Saved metadata to {output_dir / 'warp_camera_candidate_debug'}")
    print("Color legend: gray=train centers, orange=source, blue=neighbors, red/pink/yellow=virtual candidates.")
    while True:
        time.sleep(100000)


if __name__ == "__main__":
    main()
