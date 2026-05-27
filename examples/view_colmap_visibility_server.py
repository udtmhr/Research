#!/usr/bin/env python3
import argparse
import os
import struct
import numpy as np

from dash import Dash, dcc, html
import plotly.graph_objects as go


# -------------------------
# COLMAP binary readers
# -------------------------

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}

    camera_models = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE",
    }

    num_params = {
        "SIMPLE_PINHOLE": 3,
        "PINHOLE": 4,
        "SIMPLE_RADIAL": 4,
        "RADIAL": 5,
        "OPENCV": 8,
        "OPENCV_FISHEYE": 8,
        "FULL_OPENCV": 12,
        "FOV": 5,
        "SIMPLE_RADIAL_FISHEYE": 4,
        "RADIAL_FISHEYE": 5,
        "THIN_PRISM_FISHEYE": 12,
    }

    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(
                fid, 24, "iiQQ"
            )
            model = camera_models[model_id]
            n = num_params[model]
            params = read_next_bytes(fid, 8 * n, "d" * n)

            cameras[camera_id] = {
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": np.array(params, dtype=np.float64),
            }

    return cameras


def read_images_binary(path):
    images = {}

    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]

            name = b""
            while True:
                ch = fid.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * 24)

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }

    return images


def read_points3D_binary_xyz_rgb(path):
    xyzs = []
    rgbs = []

    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):
            props = read_next_bytes(
                fid,
                8 + 8 * 3 + 1 * 3 + 8 + 8,
                "QdddBBBdQ",
            )

            xyz = props[1:4]
            rgb = props[4:7]
            track_length = props[8]

            # track が空でも，存在しても読み飛ばす
            fid.read(8 * 2 * track_length)

            xyzs.append(xyz)
            rgbs.append(rgb)

    return (
        np.asarray(xyzs, dtype=np.float64),
        np.asarray(rgbs, dtype=np.uint8),
    )


# -------------------------
# Geometry
# -------------------------

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec

    return np.array([
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qw * qz,
            2 * qz * qx + 2 * qw * qy,
        ],
        [
            2 * qx * qy + 2 * qw * qz,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qw * qx,
        ],
        [
            2 * qz * qx - 2 * qw * qy,
            2 * qy * qz + 2 * qw * qx,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ], dtype=np.float64)


def project_points(xyz_world, image, camera):
    R = qvec2rotmat(image["qvec"])
    t = image["tvec"]

    xyz_cam = xyz_world @ R.T + t[None, :]

    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    z = xyz_cam[:, 2]

    valid_depth = z > 1e-6

    xn = x / z
    yn = y / z

    model = camera["model"]
    params = camera["params"]

    # 基本的なモデルをサポート
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        u = f * xn + cx
        v = f * yn + cy

    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        u = fx * xn + cx
        v = fy * yn + cy

    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k = params[:4]
        r2 = xn * xn + yn * yn
        radial = 1 + k * r2
        u = f * radial * xn + cx
        v = f * radial * yn + cy

    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2
        u = f * radial * xn + cx
        v = f * radial * yn + cy

    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x_dist = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        y_dist = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        u = fx * x_dist + cx
        v = fy * y_dist + cy

    else:
        # 未対応モデルは歪みなしPINHOLE近似
        if len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        else:
            f, cx, cy = params[:3]
            fx, fy = f, f
        u = fx * xn + cx
        v = fy * yn + cy

    inside = (
        valid_depth
        & (u >= 0)
        & (u < camera["width"])
        & (v >= 0)
        & (v < camera["height"])
    )

    return inside


def compute_projected_visibility(xyzs, images, cameras, chunk_size=200000):
    counts = np.zeros(len(xyzs), dtype=np.int32)

    image_list = list(images.values())

    for start in range(0, len(xyzs), chunk_size):
        end = min(start + chunk_size, len(xyzs))
        pts = xyzs[start:end]
        local_counts = np.zeros(len(pts), dtype=np.int32)

        for img in image_list:
            cam = cameras[img["camera_id"]]
            inside = project_points(pts, img, cam)
            local_counts += inside.astype(np.int32)

        counts[start:end] = local_counts
        print(f"processed {end}/{len(xyzs)}")

    return counts


def downsample(xyzs, rgbs, counts, max_points):
    if len(xyzs) <= max_points:
        return xyzs, rgbs, counts

    idx = np.random.choice(len(xyzs), max_points, replace=False)
    return xyzs[idx], rgbs[idx], counts[idx]


# -------------------------
# Dash viewer
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--chunk_size", type=int, default=200000)
    args = parser.parse_args()

    cameras_path = os.path.join(args.model_path, "cameras.bin")
    images_path = os.path.join(args.model_path, "images.bin")
    points_path = os.path.join(args.model_path, "points3D.bin")

    if not os.path.exists(cameras_path):
        raise FileNotFoundError(cameras_path)
    if not os.path.exists(images_path):
        raise FileNotFoundError(images_path)
    if not os.path.exists(points_path):
        raise FileNotFoundError(points_path)

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    xyzs, rgbs = read_points3D_binary_xyz_rgb(points_path)

    print(f"num cameras: {len(cameras)}")
    print(f"num images: {len(images)}")
    print(f"num points: {len(xyzs)}")

    counts = compute_projected_visibility(
        xyzs,
        images,
        cameras,
        chunk_size=args.chunk_size,
    )

    print("visibility min:", counts.min())
    print("visibility max:", counts.max())
    print("visibility mean:", counts.mean())
    print("visibility median:", np.median(counts))

    xyzs_v, rgbs_v, counts_v = downsample(
        xyzs,
        rgbs,
        counts,
        args.max_points,
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyzs_v[:, 0],
                y=xyzs_v[:, 1],
                z=xyzs_v[:, 2],
                mode="markers",
                marker=dict(
                    size=args.point_size,
                    color=counts_v,
                    colorscale="Turbo",
                    colorbar=dict(title="Projected cameras"),
                    opacity=0.9,
                ),
                text=[
                    f"projected cameras: {int(c)}"
                    for c in counts_v
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="Projected Visibility Count from COLMAP Cameras",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3("COLMAP Projected Visibility Count"),
            html.P(
                f"points={len(xyzs)}, images={len(images)}, "
                f"visibility min={counts.min()}, max={counts.max()}, "
                f"mean={counts.mean():.2f}, median={np.median(counts):.2f}"
            ),
            dcc.Graph(
                figure=fig,
                style={"width": "100vw", "height": "90vh"},
            ),
        ]
    )

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
