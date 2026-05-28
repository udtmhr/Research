#!/usr/bin/env python
"""View a COLMAP points3D.bin/points.bin file in a local viser server."""

import argparse
import struct
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import viser


def read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Unexpected EOF while reading {num_bytes} bytes")
    return struct.unpack("<" + fmt, data)


def read_points3d_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xyzs = []
    rgbs = []
    with path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 8 + 8 * 3 + 3 + 8 + 8, "QdddBBBdQ")
            xyzs.append(props[1:4])
            rgbs.append(props[4:7])
            track_length = props[8]
            fid.read(8 * 2 * track_length)
    return np.asarray(xyzs, dtype=np.float32), np.asarray(rgbs, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a COLMAP points3D.bin/points.bin point cloud on localhost."
    )
    parser.add_argument(
        "--points-bin",
        type=Path,
        required=True,
        help="Path to COLMAP points3D.bin or points.bin.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Randomly subsample to this many points. 0 means no subsampling.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.points_bin.exists():
        raise FileNotFoundError(args.points_bin)

    points, colors = read_points3d_binary(args.points_bin)
    if args.max_points > 0 and len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    server = viser.ViserServer(port=args.port, verbose=False)
    server.gui.set_panel_label("points.bin viewer")
    server.scene.add_point_cloud(
        "/points",
        points=points,
        colors=colors,
        point_size=args.point_size,
        point_shape="circle",
    )

    center = points.mean(axis=0)
    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    server.scene.add_frame(
        "/center",
        position=center,
        axes_length=max(extent * 0.03, 1e-3),
        axes_radius=max(extent * 0.002, 1e-4),
    )
    server.gui.add_markdown(
        f"Loaded `{args.points_bin}`\n\n"
        f"- points: `{len(points)}`\n"
        f"- point size: `{args.point_size}`"
    )

    print(f"Loaded {len(points)} points from {args.points_bin}")
    print(f"Viewer: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
