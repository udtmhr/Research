#!/usr/bin/env python
"""Create colorized COLMAP points3D.bin from masks by visual hull carving."""

import argparse
import os
import struct
from pathlib import Path
from typing import Dict, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from skimage import measure
from tqdm import tqdm


CAMERA_MODEL_PARAM_COUNTS = {
    0: 3,  # SIMPLE_PINHOLE
    1: 4,  # PINHOLE
    2: 4,  # SIMPLE_RADIAL
    3: 5,  # RADIAL
    4: 8,  # OPENCV
    5: 8,  # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,  # FOV
    8: 4,  # SIMPLE_RADIAL_FISHEYE
    9: 5,  # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_binary(path: Path) -> Dict[int, dict]:
    cameras = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            param_count = CAMERA_MODEL_PARAM_COUNTS.get(model_id)
            if param_count is None:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            params = read_next_bytes(fid, 8 * param_count, "d" * param_count)
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": np.asarray(params, dtype=np.float64),
            }
    return cameras


def read_images_binary(path: Path) -> Dict[int, dict]:
    images = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            name = ""
            char = read_next_bytes(fid, 1, "c")[0]
            while char != b"\x00":
                name += char.decode("utf-8")
                char = read_next_bytes(fid, 1, "c")[0]

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2d * 24)
            images[props[0]] = {
                "qvec": np.asarray(props[1:5], dtype=np.float64),
                "tvec": np.asarray(props[5:8], dtype=np.float64),
                "camera_id": props[8],
                "name": name,
            }
    return images


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[3] * qvec[0],
                2 * qvec[1] * qvec[3] + 2 * qvec[2] * qvec[0],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[3] * qvec[0],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[1] * qvec[0],
            ],
            [
                2 * qvec[1] * qvec[3] - 2 * qvec[2] * qvec[0],
                2 * qvec[2] * qvec[3] + 2 * qvec[1] * qvec[0],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


@torch.no_grad()
def qvec2rotmat_torch(qvec: torch.Tensor) -> torch.Tensor:
    qvec = qvec / (torch.norm(qvec, dim=-1, keepdim=True) + 1e-9)
    w, x, y, z = qvec.unbind(-1)
    return torch.stack(
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y,
        ],
        dim=-1,
    ).reshape(qvec.shape[:-1] + (3, 3))


@torch.no_grad()
def project_points_colmap(
    points_3d: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points_cam = points_3d @ R.T + t
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    inv_z = 1.0 / (z + 1e-8)
    u = fx * (x * inv_z) + cx
    v = fy * (y * inv_z) + cy
    return u, v, z


def get_normalization_transform_from_cameras(images: Dict[int, dict]) -> Tuple[np.ndarray, float]:
    cam_centers = []
    for image in images.values():
        R = qvec2rotmat(image["qvec"])
        t = image["tvec"]
        cam_centers.append(-R.T @ t)

    cam_centers = np.asarray(cam_centers, dtype=np.float64)
    min_bound = cam_centers.min(axis=0)
    max_bound = cam_centers.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    max_dist = np.max(np.abs(cam_centers - center))
    scale = 1.0 if max_dist < 1e-6 else 1.0 / max_dist
    return center.astype(np.float32), float(scale)


def camera_intrinsics(params: np.ndarray, model_id: int) -> Tuple[float, float, float, float]:
    if model_id in {1, 4, 5, 6, 10}:
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    return float(params[0]), float(params[0]), float(params[1]), float(params[2])


def write_points3d_binary_with_color(points_3d: np.ndarray, colors: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {points_3d.shape[0]} colorized points to {path}...")
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", points_3d.shape[0]))
        for i in tqdm(range(points_3d.shape[0]), desc="Saving points3D.bin"):
            x, y, z = points_3d[i]
            r, g, b = colors[i]
            fid.write(
                struct.pack(
                    "<QdddBBBdQ",
                    i + 1,
                    float(x),
                    float(y),
                    float(z),
                    int(r),
                    int(g),
                    int(b),
                    0.0,
                    0,
                )
            )


def write_mesh_ply(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fid:
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write(f"element vertex {vertices.shape[0]}\n")
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write(f"element face {faces.shape[0]}\n")
        fid.write("property list uchar int vertex_indices\n")
        fid.write("end_header\n")
        for x, y, z in vertices:
            fid.write(f"{float(x)} {float(y)} {float(z)}\n")
        for i, j, k in faces:
            fid.write(f"3 {int(i)} {int(j)} {int(k)}\n")


def voxel_index_vertices_to_normalized(
    vertices: np.ndarray,
    grid_center: np.ndarray,
    grid_extent: float,
    resolution: int,
) -> np.ndarray:
    half = grid_extent * 0.5
    min_bound = grid_center.astype(np.float32) - half
    if resolution <= 1:
        return np.repeat(min_bound[None, :], vertices.shape[0], axis=0)
    step = grid_extent / float(resolution - 1)
    return (min_bound[None, :] + vertices.astype(np.float32) * step).astype(np.float32)


def laplacian_smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    steps: int,
    lambd: float,
) -> np.ndarray:
    if steps <= 0 or vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices.astype(np.float32, copy=True)

    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges.astype(np.int64), axis=1)
    edges = np.unique(edges, axis=0)
    smoothed = vertices.astype(np.float32, copy=True)
    lambd = float(lambd)

    for _ in tqdm(range(steps), desc="Smoothing mesh"):
        neighbor_sum = np.zeros_like(smoothed)
        neighbor_count = np.zeros((smoothed.shape[0], 1), dtype=np.float32)
        np.add.at(neighbor_sum, edges[:, 0], smoothed[edges[:, 1]])
        np.add.at(neighbor_sum, edges[:, 1], smoothed[edges[:, 0]])
        np.add.at(neighbor_count, edges[:, 0], 1.0)
        np.add.at(neighbor_count, edges[:, 1], 1.0)

        has_neighbors = neighbor_count[:, 0] > 0
        averaged = smoothed.copy()
        averaged[has_neighbors] = (
            neighbor_sum[has_neighbors] / neighbor_count[has_neighbors]
        )
        smoothed[has_neighbors] += lambd * (
            averaged[has_neighbors] - smoothed[has_neighbors]
        )

    return smoothed.astype(np.float32)


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    if sample_count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-12
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)

    triangles = triangles[valid]
    areas = areas[valid]
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(
        triangles.shape[0],
        size=sample_count,
        replace=True,
        p=areas / areas.sum(),
    )
    selected = triangles[face_indices]

    r1 = np.sqrt(rng.random(sample_count, dtype=np.float32))
    r2 = rng.random(sample_count, dtype=np.float32)
    samples = (
        (1.0 - r1)[:, None] * selected[:, 0]
        + (r1 * (1.0 - r2))[:, None] * selected[:, 1]
        + (r1 * r2)[:, None] * selected[:, 2]
    )
    return samples.astype(np.float32)


def extract_smoothed_mesh(
    occupancy: np.ndarray,
    grid_center: np.ndarray,
    grid_extent: float,
    resolution: int,
    smoothing_steps: int,
    smoothing_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not occupancy.any() or occupancy.all():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    padded = np.pad(occupancy.astype(np.float32), 1, mode="constant")
    vertices_index, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
    )
    vertices_index -= 1.0
    vertices = voxel_index_vertices_to_normalized(
        vertices_index,
        grid_center=grid_center,
        grid_extent=grid_extent,
        resolution=resolution,
    )
    vertices = laplacian_smooth_mesh(
        vertices,
        faces.astype(np.int32),
        steps=smoothing_steps,
        lambd=smoothing_lambda,
    )
    return vertices, faces.astype(np.int32)


def points_normalized_to_world(
    points_normalized: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
) -> np.ndarray:
    return ((points_normalized / norm_scale) + norm_center).astype(np.float64)


def colorize_points(
    points_normalized: np.ndarray,
    normalized_images,
    masks_t,
    cameras: Dict[int, dict],
    chunk: int,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    num_points = points_normalized.shape[0]
    if num_points == 0:
        return np.empty((0, 3), dtype=np.uint8), 0

    points_t = torch.from_numpy(points_normalized.astype(np.float32)).to(device)
    color_sum = torch.zeros((num_points, 3), dtype=torch.float32, device=device)
    color_count = torch.zeros((num_points, 1), dtype=torch.float32, device=device)
    valid_chunk_count = 0

    for i, item in enumerate(tqdm(normalized_images, desc="Colorizing")):
        if not item["image_path"].exists():
            continue

        image = imageio.imread(item["image_path"])
        image_t = torch.from_numpy(image).to(device).float()[..., :3]
        height_image, width_image = image_t.shape[:2]

        camera = cameras[item["camera_id"]]
        fx, fy, cx, cy = camera_intrinsics(camera["params"], camera["model_id"])
        width_colmap, height_colmap = camera["width"], camera["height"]

        R = qvec2rotmat_torch(
            torch.tensor(item["qvec"], dtype=torch.float32, device=device)
        )
        t = torch.tensor(item["tvec"], dtype=torch.float32, device=device)
        mask = masks_t[i]
        height_mask, width_mask = mask.shape
        if height_mask != height_colmap or width_mask != width_colmap:
            sx = width_mask / width_colmap
            sy = height_mask / height_colmap
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy

        if i == 0:
            print(
                "[DEBUG] Image 0: "
                f"Mask_Size={width_mask}x{height_mask}, "
                f"Image_Size={width_image}x{height_image}"
            )

        for start in range(0, num_points, chunk):
            points = points_t[start : start + chunk]
            u, v, z = project_points_colmap(points, R, t, fx, fy, cx, cy)
            ui = (u + 0.5).to(torch.int64)
            vi = (v + 0.5).to(torch.int64)
            valid = (
                (z > 0)
                & (ui >= 0)
                & (ui < width_mask)
                & (vi >= 0)
                & (vi < height_mask)
            )

            if valid.any():
                valid_chunk_count += 1
                in_mask = mask[vi[valid], ui[valid]] > 0
                valid_indices = torch.where(valid)[0]
                final_valid_indices = valid_indices[in_mask]
                if final_valid_indices.numel() == 0:
                    continue

                v_mask = vi[final_valid_indices].float()
                u_mask = ui[final_valid_indices].float()
                image_vi = (v_mask * height_image / height_mask).long().clamp(
                    0, height_image - 1
                )
                image_ui = (u_mask * width_image / width_mask).long().clamp(
                    0, width_image - 1
                )
                color_sum[start : start + chunk][final_valid_indices] += image_t[
                    image_vi, image_ui
                ]
                color_count[start : start + chunk][final_valid_indices] += 1

    final_colors = (color_sum / (color_count + 1e-8)).cpu().numpy()
    final_colors[color_count.cpu().numpy().flatten() == 0] = [128, 128, 128]
    return np.clip(final_colors, 0, 255).astype(np.uint8), valid_chunk_count


def parse_grid_center(value: str) -> np.ndarray:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("grid center must be x,y,z")
    try:
        center = np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid center must be x,y,z") from exc
    return center


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Carve a visual hull from <data-dir>/masks and save colorized "
            "COLMAP points3D.bin plus voxel metadata."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-points", type=Path, default=None)
    parser.add_argument("--output-voxels", type=Path, default=None)
    parser.add_argument("--output-mesh", type=Path, default=None)
    parser.add_argument(
        "--point-generation",
        choices=["mesh", "voxels"],
        default="mesh",
        help="Generate points from a smoothed marching-cubes mesh or voxel centers.",
    )
    parser.add_argument("--mesh-smoothing-steps", type=int, default=10)
    parser.add_argument("--mesh-smoothing-lambda", type=float, default=0.5)
    parser.add_argument(
        "--mesh-sample-count",
        type=int,
        default=None,
        help="Number of mesh surface samples. Defaults to carved voxel count.",
    )
    parser.add_argument("--mesh-sample-seed", type=int, default=0)
    parser.add_argument(
        "--grid-center", type=parse_grid_center, default=parse_grid_center("0,0,0")
    )
    parser.add_argument("--grid-extent", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=700)
    parser.add_argument("--chunk", type=int, default=200_000)
    parser.add_argument("--pass-ratio", type=float, default=0.90)
    parser.add_argument("--min-views", type=int, default=8)
    parser.add_argument("--mask-ext", type=str, default=".png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir = args.data_dir.resolve()
    colmap_dir = data_dir / "sparse"
    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    output_points = (
        args.output_points.resolve()
        if args.output_points
        else data_dir / "points3D.bin"
    )
    output_voxels = (
        args.output_voxels.resolve()
        if args.output_voxels
        else data_dir / "visual_hull_voxels.npz"
    )
    output_mesh = (
        args.output_mesh.resolve()
        if args.output_mesh
        else data_dir / "visual_hull_mesh.ply"
    )

    cameras_path = colmap_dir / "cameras.bin"
    images_path = colmap_dir / "images.bin"
    if not cameras_path.exists() or not images_path.exists():
        raise FileNotFoundError(
            f"Expected COLMAP binaries at {cameras_path} and {images_path}"
        )
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("Loading COLMAP data...")
    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    norm_center, norm_scale = get_normalization_transform_from_cameras(images)

    normalized_images = []
    for image in images.values():
        basename = os.path.basename(image["name"])
        mask_path = mask_dir / (Path(basename).stem + args.mask_ext)
        image_path = image_dir / image["name"]
        if not mask_path.exists():
            continue

        R = qvec2rotmat(image["qvec"])
        cam_center = -R.T @ image["tvec"]
        cam_center_normalized = (cam_center - norm_center) * norm_scale
        t_normalized = -R @ cam_center_normalized
        normalized_images.append(
            {
                "qvec": image["qvec"],
                "tvec": t_normalized,
                "camera_id": image["camera_id"],
                "mask_path": mask_path,
                "image_path": image_path,
            }
        )

    if len(normalized_images) == 0:
        raise RuntimeError(f"No COLMAP images with masks found in {mask_dir}")
    print(f"Using {len(normalized_images)} masked images.")

    masks_t = []
    for item in tqdm(normalized_images, desc="Loading masks"):
        mask = imageio.imread(item["mask_path"])
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 127).astype(np.uint8)
        masks_t.append(torch.from_numpy(mask).to(device))

    half = args.grid_extent * 0.5
    grid_center = args.grid_center.astype(np.float32)
    mn = (grid_center - half).astype(np.float32)
    mx = (grid_center + half).astype(np.float32)
    xs = np.linspace(mn[0], mx[0], args.resolution, dtype=np.float32)
    ys = np.linspace(mn[1], mx[1], args.resolution, dtype=np.float32)
    zs = np.linspace(mn[2], mx[2], args.resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    voxels = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    voxels_t = torch.from_numpy(voxels).to(device)
    num_voxels = voxels.shape[0]
    print(f"Voxel grid: {args.resolution}^3 = {num_voxels} points")

    seen_count = torch.zeros((num_voxels,), dtype=torch.int16, device=device)
    match_count = torch.zeros((num_voxels,), dtype=torch.int16, device=device)

    print("Pass 1: Carving...")
    for i, item in enumerate(tqdm(normalized_images, desc="Carving")):
        camera = cameras[item["camera_id"]]
        fx, fy, cx, cy = camera_intrinsics(camera["params"], camera["model_id"])
        width_colmap, height_colmap = camera["width"], camera["height"]

        R = qvec2rotmat_torch(
            torch.tensor(item["qvec"], dtype=torch.float32, device=device)
        )
        t = torch.tensor(item["tvec"], dtype=torch.float32, device=device)
        mask = masks_t[i]
        height_mask, width_mask = mask.shape
        if height_mask != height_colmap or width_mask != width_colmap:
            sx = width_mask / width_colmap
            sy = height_mask / height_colmap
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy

        for start in range(0, num_voxels, args.chunk):
            points = voxels_t[start : start + args.chunk]
            u, v, z = project_points_colmap(points, R, t, fx, fy, cx, cy)
            ui = (u + 0.5).to(torch.int64)
            vi = (v + 0.5).to(torch.int64)
            valid = (
                (z > 0)
                & (ui >= 0)
                & (ui < width_mask)
                & (vi >= 0)
                & (vi < height_mask)
            )
            if valid.any():
                seen_count[start : start + args.chunk] += valid.to(torch.int16)
                is_matched = torch.zeros_like(valid, dtype=torch.int16)
                is_matched[valid] = mask[vi[valid], ui[valid]].to(torch.int16)
                match_count[start : start + args.chunk] += is_matched

    ratios = match_count.float() / (seen_count.float() + 1e-8)
    keep_mask = (seen_count >= args.min_views) & (ratios >= args.pass_ratio)
    carved_indices_t = torch.where(keep_mask)[0]
    carved_points_t = voxels_t[carved_indices_t]
    num_final = carved_points_t.shape[0]
    print(f"Remaining carved points: {num_final}")

    if num_final == 0:
        output_voxels.parent.mkdir(parents=True, exist_ok=True)
        write_points3d_binary_with_color(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
            output_points,
        )
        write_mesh_ply(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            output_mesh,
        )
        np.savez_compressed(
            output_voxels,
            points_normalized=np.empty((0, 3), dtype=np.float32),
            points_world=np.empty((0, 3), dtype=np.float64),
            colors=np.empty((0, 3), dtype=np.uint8),
            carved_indices=np.empty((0,), dtype=np.int64),
            seen_count=np.empty((0,), dtype=np.int16),
            match_count=np.empty((0,), dtype=np.int16),
            mesh_vertices_normalized=np.empty((0, 3), dtype=np.float32),
            mesh_vertices_world=np.empty((0, 3), dtype=np.float64),
            mesh_faces=np.empty((0, 3), dtype=np.int32),
            sampled_points_normalized=np.empty((0, 3), dtype=np.float32),
            sampled_points_world=np.empty((0, 3), dtype=np.float64),
            sampled_colors=np.empty((0, 3), dtype=np.uint8),
            grid_center=grid_center,
            grid_extent=np.float32(args.grid_extent),
            resolution=np.int32(args.resolution),
            norm_center=norm_center,
            norm_scale=np.float32(norm_scale),
            pass_ratio=np.float32(args.pass_ratio),
            min_views=np.int32(args.min_views),
            point_generation=args.point_generation,
            mesh_smoothing_steps=np.int32(args.mesh_smoothing_steps),
            mesh_smoothing_lambda=np.float32(args.mesh_smoothing_lambda),
            mesh_sample_count=np.int32(0),
        )
        print(
            "No points remaining. "
            f"Saved empty outputs to {output_points}, {output_voxels}, and {output_mesh}"
        )
        return

    carved_normalized = carved_points_t.cpu().numpy().astype(np.float32)
    carved_world = points_normalized_to_world(carved_normalized, norm_center, norm_scale)
    carved_indices = carved_indices_t.cpu().numpy().astype(np.int64)
    carved_seen_count = seen_count[carved_indices_t].cpu().numpy()
    carved_match_count = match_count[carved_indices_t].cpu().numpy()

    occupancy = keep_mask.reshape(args.resolution, args.resolution, args.resolution)
    occupancy_np = occupancy.cpu().numpy().astype(bool)
    print("Pass 2: Extracting smoothed mesh...")
    mesh_vertices_normalized, mesh_faces = extract_smoothed_mesh(
        occupancy_np,
        grid_center=grid_center,
        grid_extent=args.grid_extent,
        resolution=args.resolution,
        smoothing_steps=args.mesh_smoothing_steps,
        smoothing_lambda=args.mesh_smoothing_lambda,
    )
    mesh_vertices_world = points_normalized_to_world(
        mesh_vertices_normalized,
        norm_center,
        norm_scale,
    )
    write_mesh_ply(mesh_vertices_world, mesh_faces, output_mesh)
    print(
        f"Saved smoothed mesh: {output_mesh} "
        f"({len(mesh_vertices_world)} vertices, {len(mesh_faces)} faces)"
    )

    if args.point_generation == "mesh":
        sample_count = args.mesh_sample_count
        if sample_count is None:
            sample_count = int(num_final)
        print(f"Pass 3: Sampling {sample_count} points from mesh...")
        sampled_normalized = sample_mesh_surface(
            mesh_vertices_normalized,
            mesh_faces,
            sample_count=sample_count,
            seed=args.mesh_sample_seed,
        )
        if sampled_normalized.shape[0] == 0:
            print("Mesh sampling produced no points; falling back to voxel centers.")
            sampled_normalized = carved_normalized
    else:
        sampled_normalized = carved_normalized

    sampled_world = points_normalized_to_world(sampled_normalized, norm_center, norm_scale)
    print(f"Pass 4: Colorizing {sampled_normalized.shape[0]} output points...")
    sampled_colors, valid_chunk_count = colorize_points(
        sampled_normalized,
        normalized_images,
        masks_t,
        cameras,
        args.chunk,
        device,
    )
    if sampled_normalized.shape[0] == carved_normalized.shape[0] and np.array_equal(
        sampled_normalized,
        carved_normalized,
    ):
        carved_colors = sampled_colors
    else:
        print(f"Pass 5: Colorizing {carved_normalized.shape[0]} carved voxels...")
        carved_colors, _ = colorize_points(
            carved_normalized,
            normalized_images,
            masks_t,
            cameras,
            args.chunk,
            device,
        )

    write_points3d_binary_with_color(sampled_world, sampled_colors, output_points)
    output_voxels.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_voxels,
        points_normalized=carved_normalized,
        points_world=carved_world,
        colors=carved_colors,
        carved_indices=carved_indices,
        seen_count=carved_seen_count,
        match_count=carved_match_count,
        mesh_vertices_normalized=mesh_vertices_normalized,
        mesh_vertices_world=mesh_vertices_world,
        mesh_faces=mesh_faces,
        sampled_points_normalized=sampled_normalized,
        sampled_points_world=sampled_world,
        sampled_colors=sampled_colors,
        grid_center=grid_center,
        grid_extent=np.float32(args.grid_extent),
        resolution=np.int32(args.resolution),
        norm_center=norm_center,
        norm_scale=np.float32(norm_scale),
        pass_ratio=np.float32(args.pass_ratio),
        min_views=np.int32(args.min_views),
        point_generation=args.point_generation,
        mesh_smoothing_steps=np.int32(args.mesh_smoothing_steps),
        mesh_smoothing_lambda=np.float32(args.mesh_smoothing_lambda),
        mesh_sample_count=np.int32(sampled_normalized.shape[0]),
    )
    print(f"Saved colorized COLMAP points3D: {output_points}")
    print(f"Saved carved voxel metadata: {output_voxels}")
    print(f"Point generation: {args.point_generation}")
    print(f"Colorized valid chunks: {valid_chunk_count}")


if __name__ == "__main__":
    main()
