#!/usr/bin/env python
"""View visual hull points/voxels in a local viser server."""

import argparse
import struct
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import viser


TURBO_ANCHORS = np.asarray(
    [
        [48, 18, 59],
        [50, 101, 186],
        [24, 177, 162],
        [237, 209, 73],
        [216, 70, 39],
        [122, 4, 3],
    ],
    dtype=np.float32,
)


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


def read_voxels_npz(
    path: Path,
    coordinate: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    data = np.load(path)
    key = "points_world" if coordinate == "world" else "points_normalized"
    if key not in data:
        raise KeyError(f"{path} does not contain `{key}`")

    points = data[key].astype(np.float32)
    if "colors" in data:
        colors = data["colors"].astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)

    grid_extent = float(data["grid_extent"]) if "grid_extent" in data else None
    return points, colors, grid_extent


def subsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors

    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=max_points, replace=False)
    return points[keep], colors[keep]


def parse_depth_range(value: str) -> Tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("depth range must be min,max")
    try:
        near, far = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depth range must be min,max") from exc
    if far <= near:
        raise argparse.ArgumentTypeError("depth range requires max > min")
    return near, far


def depth_values(points: np.ndarray, axis: str) -> np.ndarray:
    if axis == "x":
        return points[:, 0]
    if axis == "y":
        return points[:, 1]
    if axis == "z":
        return points[:, 2]
    if axis == "radius":
        return np.linalg.norm(points, axis=1)
    raise ValueError(f"Unknown depth axis: {axis}")


def apply_depth_colormap(values: np.ndarray, colormap: str) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    if colormap == "gray":
        gray = (values * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    scaled = values * (len(TURBO_ANCHORS) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.clip(lower + 1, 0, len(TURBO_ANCHORS) - 1)
    t = (scaled - lower)[:, None]
    colors = TURBO_ANCHORS[lower] * (1.0 - t) + TURBO_ANCHORS[upper] * t
    return np.clip(colors, 0, 255).astype(np.uint8)


def colorize_by_depth(
    points: np.ndarray,
    axis: str,
    depth_range: Optional[Tuple[float, float]],
    colormap: str,
    invert: bool,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    values = depth_values(points, axis)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.uint8), (0.0, 0.0)

    if depth_range is None:
        near = float(np.min(values))
        far = float(np.max(values))
    else:
        near, far = depth_range

    denom = max(far - near, 1e-8)
    normalized = (values - near) / denom
    if invert:
        normalized = 1.0 - normalized
    return apply_depth_colormap(normalized, colormap), (near, far)


def add_point_cloud(
    server: viser.ViserServer,
    name: str,
    points: np.ndarray,
    colors: np.ndarray,
    point_size: float,
) -> None:
    if len(points) == 0:
        server.gui.add_markdown(f"`{name}` is empty.")
        return

    server.scene.add_point_cloud(
        name,
        points=points,
        colors=colors,
        point_size=point_size,
        point_shape="circle",
    )


def add_scene_frame(server: viser.ViserServer, points: np.ndarray) -> None:
    if len(points) == 0:
        return

    center = points.mean(axis=0)
    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    server.scene.add_frame(
        "/center",
        position=center,
        axes_length=max(extent * 0.03, 1e-3),
        axes_radius=max(extent * 0.002, 1e-4),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve visual hull points3D.bin and/or visual_hull_voxels.npz on "
            "localhost with viser."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing points3D.bin and visual_hull_voxels.npz.",
    )
    parser.add_argument("--points-bin", type=Path, default=None)
    parser.add_argument("--voxels-npz", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=["auto", "points", "voxels", "both"],
        default="auto",
        help="Which data to visualize. auto prefers voxels, then points.",
    )
    parser.add_argument(
        "--voxel-coordinate",
        choices=["world", "normalized"],
        default="world",
        help="Coordinate array used when loading visual_hull_voxels.npz.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--voxel-point-size", type=float, default=None)
    parser.add_argument(
        "--color-mode",
        choices=["rgb", "depth"],
        default="rgb",
        help="Use stored RGB colors or depth colormap.",
    )
    parser.add_argument(
        "--depth-axis",
        choices=["x", "y", "z", "radius"],
        default="z",
        help="Point coordinate used as depth when --color-mode depth.",
    )
    parser.add_argument(
        "--depth-range",
        type=parse_depth_range,
        default=None,
        help="Depth normalization range as min,max. Auto range if omitted.",
    )
    parser.add_argument(
        "--depth-colormap",
        choices=["turbo", "gray"],
        default="turbo",
    )
    parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Invert depth colors after normalization.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1_000_000,
        help="Randomly subsample each cloud to this many points. 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def resolve_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path]]:
    points_bin = args.points_bin
    voxels_npz = args.voxels_npz

    if args.data_dir is not None:
        data_dir = args.data_dir.resolve()
        if points_bin is None:
            points_bin = data_dir / "points3D.bin"
        if voxels_npz is None:
            voxels_npz = data_dir / "visual_hull_voxels.npz"

    return points_bin, voxels_npz


def main() -> None:
    args = build_arg_parser().parse_args()
    points_bin, voxels_npz = resolve_paths(args)
    voxel_point_size = (
        args.voxel_point_size
        if args.voxel_point_size is not None
        else args.point_size
    )

    load_points = args.source in {"points", "both"}
    load_voxels = args.source in {"voxels", "both"}
    if args.source == "auto":
        load_voxels = voxels_npz is not None and voxels_npz.exists()
        load_points = not load_voxels and points_bin is not None and points_bin.exists()

    if load_points and (points_bin is None or not points_bin.exists()):
        raise FileNotFoundError(f"points3D.bin not found: {points_bin}")
    if load_voxels and (voxels_npz is None or not voxels_npz.exists()):
        raise FileNotFoundError(f"voxel npz not found: {voxels_npz}")
    if not load_points and not load_voxels:
        raise FileNotFoundError(
            "No input found. Pass --data-dir, --points-bin, or --voxels-npz."
        )

    clouds = []
    depth_ranges = []
    if load_points:
        points, colors = read_points3d_binary(points_bin)
        points, colors = subsample_points(points, colors, args.max_points, args.seed)
        if args.color_mode == "depth":
            colors, depth_range = colorize_by_depth(
                points,
                args.depth_axis,
                args.depth_range,
                args.depth_colormap,
                args.invert_depth,
            )
            depth_ranges.append(("/points3D", depth_range))
        clouds.append(("/points3D", points, colors, args.point_size, points_bin))

    if load_voxels:
        points, colors, grid_extent = read_voxels_npz(
            voxels_npz, args.voxel_coordinate
        )
        points, colors = subsample_points(points, colors, args.max_points, args.seed)
        if args.color_mode == "depth":
            colors, depth_range = colorize_by_depth(
                points,
                args.depth_axis,
                args.depth_range,
                args.depth_colormap,
                args.invert_depth,
            )
            depth_ranges.append(("/voxels", depth_range))
        clouds.append(("/voxels", points, colors, voxel_point_size, voxels_npz))
    else:
        grid_extent = None

    server = viser.ViserServer(port=args.port, verbose=False)
    server.gui.set_panel_label("visual hull viewer")

    all_points = []
    markdown_lines = ["### Loaded visual hull"]
    for name, points, colors, point_size, path in clouds:
        add_point_cloud(server, name, points, colors, point_size)
        all_points.append(points)
        markdown_lines.append(
            f"- `{name}`: `{len(points)}` points, size `{point_size}`, `{path}`"
        )

    if len(all_points) > 0:
        nonempty = [points for points in all_points if len(points) > 0]
        if len(nonempty) > 0:
            add_scene_frame(server, np.concatenate(nonempty, axis=0))

    if grid_extent is not None:
        markdown_lines.append(f"- voxel grid extent: `{grid_extent}`")
    markdown_lines.append(f"- voxel coordinate: `{args.voxel_coordinate}`")
    markdown_lines.append(f"- color mode: `{args.color_mode}`")
    if args.color_mode == "depth":
        markdown_lines.append(f"- depth axis: `{args.depth_axis}`")
        for name, (near, far) in depth_ranges:
            markdown_lines.append(f"- `{name}` depth range: `{near:.6g}` to `{far:.6g}`")
    server.gui.add_markdown("\n".join(markdown_lines))

    print("Loaded:")
    for name, points, _, point_size, path in clouds:
        print(f"  {name}: {len(points)} points, point_size={point_size}, path={path}")
    print(f"Viewer: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
