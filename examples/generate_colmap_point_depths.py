#!/usr/bin/env python
"""Generate metric COLMAP point-cloud depth maps for warp supervision."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import tqdm

from datasets.colmap import Parser


def project_points_to_depth(
    points_world: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    splat_radius: int,
) -> np.ndarray:
    worldtocam = np.linalg.inv(camtoworld)
    points_cam = (
        worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]
    ).T
    z = points_cam[:, 2]
    valid_z = z > 0.0
    points_cam = points_cam[valid_z]
    z = z[valid_z]

    proj = (K @ points_cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)

    depth = np.full((height, width), np.inf, dtype=np.float32)
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
        np.minimum.at(depth, (yy[inside], xx[inside]), z[inside].astype(np.float32))

    depth[~np.isfinite(depth)] = 0.0
    return depth


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project the initial COLMAP point cloud into each camera and save "
            "float32 depth maps under <data-dir>/depths."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument(
        "--normalize-world-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match simple_trainer normalize_world_space behavior.",
    )
    parser.add_argument(
        "--splat-radius",
        type=int,
        default=2,
        help="Pixel radius used to splat sparse points into the z-buffer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate <data-dir>/depths if it already exists.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    depth_dir = data_dir / "depths"
    if depth_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{depth_dir} already exists. Pass --overwrite to recreate it."
            )
        if depth_dir.resolve().parent != data_dir:
            raise RuntimeError(f"Refusing to delete unexpected path: {depth_dir}")
        shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)

    colmap_parser = Parser(
        data_dir=str(data_dir),
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
        load_exposure=False,
    )

    points_world = colmap_parser.points.astype(np.float32)
    nonzero_counts = []
    for idx, image_name in enumerate(tqdm.tqdm(colmap_parser.image_names)):
        camera_id = colmap_parser.camera_ids[idx]
        width, height = colmap_parser.imsize_dict[camera_id]
        depth = project_points_to_depth(
            points_world=points_world,
            camtoworld=colmap_parser.camtoworlds[idx],
            K=colmap_parser.Ks_dict[camera_id],
            height=height,
            width=width,
            splat_radius=args.splat_radius,
        )
        nonzero_counts.append(int((depth > 0.0).sum()))
        out_path = depth_dir / f"{Path(image_name).stem}.npy"
        np.save(out_path, depth.astype(np.float32))

    nonzero = np.asarray(nonzero_counts)
    print(f"Wrote {len(colmap_parser.image_names)} depth maps to {depth_dir}")
    print(
        "Nonzero depth pixels per image: "
        f"min={nonzero.min()}, median={int(np.median(nonzero))}, max={nonzero.max()}"
    )


if __name__ == "__main__":
    main()
