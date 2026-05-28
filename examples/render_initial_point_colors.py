#!/usr/bin/env python
"""Render initial COLMAP point colors from virtual cameras."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import tqdm

from datasets.colmap import Parser


def make_K_from_fov(fov_deg: float, height: int, width: int) -> np.ndarray:
    fov_rad = math.radians(fov_deg)
    focal = (0.5 * height) / max(math.tan(0.5 * fov_rad), 1e-6)
    return np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def viewer_saved_to_opencv_c2w(saved_c2w: np.ndarray, scale_ratio: float) -> np.ndarray:
    opencv_from_viewer = np.eye(4, dtype=np.float64)
    opencv_from_viewer[1, 1] = -1.0
    opencv_from_viewer[2, 2] = -1.0

    c2w = saved_c2w @ opencv_from_viewer
    c2w[:3, 3] = saved_c2w[:3, 3] * scale_ratio
    return c2w


def load_virtual_cameras(
    camera_path: Path,
    source: str,
    scale_ratio: float,
    fallback_width: int,
    fallback_height: int,
) -> Tuple[List[Dict[str, object]], int, int]:
    with camera_path.open("r") as f:
        data = json.load(f)

    width = int(data.get("render_width", fallback_width))
    height = int(data.get("render_height", fallback_height))
    default_fov = float(data.get("default_fov", 60.0))

    if source == "auto":
        entries = data.get("camera_path") or data.get("keyframes") or []
    elif source == "camera_path":
        entries = data.get("camera_path", [])
    elif source == "keyframes":
        entries = data.get("keyframes", [])
    else:
        raise ValueError(f"Unknown camera source: {source}")

    cameras = []
    for idx, entry in enumerate(entries):
        matrix = entry.get("camera_to_world", entry.get("matrix"))
        if matrix is None:
            raise ValueError(f"Camera entry {idx} in {camera_path} has no matrix")

        saved_c2w = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        c2w = viewer_saved_to_opencv_c2w(saved_c2w, scale_ratio)
        fov = float(entry.get("fov", default_fov))
        cameras.append({"camtoworld": c2w, "K": make_K_from_fov(fov, height, width)})

    if len(cameras) == 0:
        raise ValueError(f"No virtual cameras found in {camera_path}")

    return cameras, width, height


def project_point_colors(
    points_world: np.ndarray,
    colors: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    splat_radius: int,
    background: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    worldtocam = np.linalg.inv(camtoworld)
    points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
    z = points_cam[:, 2]
    valid = z > 0.0
    points_cam = points_cam[valid]
    point_colors = colors[valid]
    z = z[valid]

    proj = (K @ points_cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)

    order = np.argsort(z)[::-1]
    x = x[order]
    y = y[order]
    z = z[order]
    point_colors = point_colors[order]

    image = np.full((height, width, 3), background, dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    radius = max(0, int(splat_radius))
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    if len(offsets) == 0:
        offsets = [(0, 0)]

    for dy, dx in offsets:
        xx = x + dx
        yy = y + dy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        image[yy[inside], xx[inside]] = point_colors[inside]
        depth[yy[inside], xx[inside]] = z[inside].astype(np.float32)

    return image, depth


def parse_background(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("background must be R,G,B")
    try:
        rgb = tuple(int(v) for v in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("background must be R,G,B") from exc
    if any(v < 0 or v > 255 for v in rgb):
        raise argparse.ArgumentTypeError("background values must be in [0, 255]")
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project initial COLMAP point colors into virtual cameras and save PNGs."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument(
        "--normalize-world-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match simple_trainer normalize_world_space behavior.",
    )
    parser.add_argument(
        "--camera-source",
        choices=["auto", "camera_path", "keyframes"],
        default="auto",
        help="Which entries to render from the viewer camera-path JSON.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--viewer-scale-ratio",
        type=float,
        default=1.0,
        help="Scale for viewer-saved camera positions.",
    )
    parser.add_argument(
        "--splat-radius",
        type=int,
        default=2,
        help="Pixel radius for drawing each sparse point.",
    )
    parser.add_argument(
        "--background",
        type=parse_background,
        default=(0, 0, 0),
        help="Background RGB as R,G,B.",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Also save z-buffer depth as .npy per rendered image.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    camera_path = args.camera_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    colmap_parser = Parser(
        data_dir=str(data_dir),
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
        load_exposure=False,
    )

    cameras, width, height = load_virtual_cameras(
        camera_path=camera_path,
        source=args.camera_source,
        scale_ratio=args.viewer_scale_ratio,
        fallback_width=args.width,
        fallback_height=args.height,
    )

    points_world = colmap_parser.points.astype(np.float64)
    colors = colmap_parser.points_rgb.astype(np.uint8)
    coverage = []
    for idx, camera in enumerate(tqdm.tqdm(cameras, desc="Rendering point colors")):
        image, depth = project_point_colors(
            points_world=points_world,
            colors=colors,
            camtoworld=np.asarray(camera["camtoworld"], dtype=np.float64),
            K=np.asarray(camera["K"], dtype=np.float64),
            height=height,
            width=width,
            splat_radius=args.splat_radius,
            background=args.background,
        )
        covered = int((depth > 0.0).sum())
        coverage.append(covered)
        imageio.imwrite(output_dir / f"point_colors_{idx:04d}.png", image)
        if args.save_depth:
            np.save(output_dir / f"point_colors_{idx:04d}_depth.npy", depth)

    coverage_arr = np.asarray(coverage)
    print(f"Wrote {len(cameras)} images to {output_dir}")
    print(
        "Covered pixels: "
        f"min={coverage_arr.min()}, median={int(np.median(coverage_arr))}, "
        f"max={coverage_arr.max()}"
    )


if __name__ == "__main__":
    main()
