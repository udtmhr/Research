import argparse
import json
import numpy as np
from plyfile import PlyData


C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussian_ply(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    names = v.data.dtype.names

    xyz = np.stack(
        [
            np.asarray(v["x"]),
            np.asarray(v["y"]),
            np.asarray(v["z"]),
        ],
        axis=1,
    ).astype(np.float64)

    f_dc = np.stack(
        [
            np.asarray(v["f_dc_0"]),
            np.asarray(v["f_dc_1"]),
            np.asarray(v["f_dc_2"]),
        ],
        axis=1,
    ).astype(np.float64)

    rgb = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)

    if "opacity" in names:
        opacity_logit = np.asarray(v["opacity"]).astype(np.float64)
        alpha = sigmoid(opacity_logit)
    else:
        alpha = np.ones(len(xyz), dtype=np.float64)

    return xyz, rgb, alpha


def random_downsample(xyz, rgb, alpha, max_points, seed=0):
    n = len(xyz)
    if max_points is None or n <= max_points:
        return xyz, rgb, alpha

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx], rgb[idx], alpha[idx]


def to_list_float(arr):
    return arr.astype(float).tolist()


def make_html(xyz, rgb, alpha, dark_threshold, opacity_threshold, grid_size):
    brightness = rgb.mean(axis=1)
    dark_mask = (brightness < dark_threshold) & (alpha > opacity_threshold)

    rgb255 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    data = {
        "x": to_list_float(xyz[:, 0]),
        "y": to_list_float(xyz[:, 1]),
        "z": to_list_float(xyz[:, 2]),
        "r": rgb255[:, 0].astype(int).tolist(),
        "g": rgb255[:, 1].astype(int).tolist(),
        "b": rgb255[:, 2].astype(int).tolist(),
        "alpha": to_list_float(alpha),
        "brightness": to_list_float(brightness),
        "dark": dark_mask.astype(int).tolist(),
        "darkThreshold": dark_threshold,
        "opacityThreshold": opacity_threshold,
        "gridSize": grid_size,
    }

    data_json = json.dumps(data)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mouse Interactive Front Dark Gaussian Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: sans-serif;
      background: white;
    }}
    #controls {{
      padding: 10px 14px;
      border-bottom: 1px solid #ddd;
      background: #f7f7f7;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    button {{
      padding: 6px 10px;
      margin-right: 6px;
      border: 1px solid #aaa;
      background: white;
      cursor: pointer;
    }}
    button.active {{
      background: #e6f0ff;
      border-color: #4a7fd1;
    }}
    #info {{
      margin-top: 6px;
      font-size: 13px;
      color: #333;
    }}
    #plot {{
      width: 100vw;
      height: calc(100vh - 82px);
    }}
  </style>
</head>

<body>
  <div id="controls">
    <button id="btnAll">All</button>
    <button id="btnDark">Dark Only</button>
    <button id="btnAllFrontDark" class="active">All + Front Dark</button>
    <button id="btnFrontDark">Front Dark Only</button>
    <button id="btnUpdate">Update Front Dark</button>
    <div id="info"></div>
  </div>

  <div id="plot"></div>

<script>
const DATA = {data_json};
const N = DATA.x.length;

let mode = "all_front_dark";

let currentCamera = {{
  eye: {{x: 1.5, y: 1.5, z: 1.0}},
  center: {{x: 0, y: 0, z: 0}},
  up: {{x: 0, y: 0, z: 1}}
}};

const colors = new Array(N);
for (let i = 0; i < N; i++) {{
  colors[i] = `rgb(${{DATA.r[i]}},${{DATA.g[i]}},${{DATA.b[i]}})`;
}}

const darkIndices = [];
for (let i = 0; i < N; i++) {{
  if (DATA.dark[i] === 1) darkIndices.push(i);
}}

function mean(arr) {{
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}}

const cx = mean(DATA.x);
const cy = mean(DATA.y);
const cz = mean(DATA.z);

const centered = new Array(N);
for (let i = 0; i < N; i++) {{
  centered[i] = [
    DATA.x[i] - cx,
    DATA.y[i] - cy,
    DATA.z[i] - cz
  ];
}}

function normalize(v) {{
  const n = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) + 1e-12;
  return [v[0]/n, v[1]/n, v[2]/n];
}}

function cross(a, b) {{
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
}}

function dot(a, b) {{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}}

function percentile(values, p) {{
  const arr = Array.from(values).sort((a, b) => a - b);
  const idx = Math.floor((p / 100.0) * (arr.length - 1));
  return arr[idx];
}}

function getViewDirFromCamera(camera) {{
  const eye = camera.eye || {{x: 1.5, y: 1.5, z: 1.0}};
  return normalize([eye.x, eye.y, eye.z]);
}}

function makeBasisFromViewDir(viewDir) {{
  const w = normalize(viewDir);

  let up = [0, 0, 1];
  if (Math.abs(dot(w, up)) > 0.95) {{
    up = [0, 1, 0];
  }}

  const u = normalize(cross(up, w));
  const v = normalize(cross(w, u));

  return [u, v, w];
}}

function projectForFront(camera) {{
  const viewDir = getViewDirFromCamera(camera);
  const [u, v, w] = makeBasisFromViewDir(viewDir);

  const x2d = new Float64Array(N);
  const y2d = new Float64Array(N);
  const depth = new Float64Array(N);

  for (let i = 0; i < N; i++) {{
    const p = centered[i];
    x2d[i] = dot(p, u);
    y2d[i] = dot(p, v);

    // 大きいほどカメラ側，つまり手前
    depth[i] = dot(p, w);
  }}

  return {{x2d, y2d, depth, viewDir}};
}}

function computeFrontMost(x2d, y2d, depth, gridSize) {{
  const xMin = percentile(x2d, 1);
  const xMax = percentile(x2d, 99);
  const yMin = percentile(y2d, 1);
  const yMax = percentile(y2d, 99);

  const best = new Map();
  const eps = 1e-12;

  for (let i = 0; i < N; i++) {{
    const gx = Math.floor((x2d[i] - xMin) / (xMax - xMin + eps) * gridSize);
    const gy = Math.floor((y2d[i] - yMin) / (yMax - yMin + eps) * gridSize);

    if (gx < 0 || gx >= gridSize || gy < 0 || gy >= gridSize) continue;

    const cid = gy * gridSize + gx;

    if (!best.has(cid)) {{
      best.set(cid, i);
    }} else {{
      const j = best.get(cid);
      if (depth[i] > depth[j]) {{
        best.set(cid, i);
      }}
    }}
  }}

  return Array.from(best.values());
}}

function select3D(indices) {{
  const xs = new Array(indices.length);
  const ys = new Array(indices.length);
  const zs = new Array(indices.length);
  const cs = new Array(indices.length);

  for (let k = 0; k < indices.length; k++) {{
    const i = indices[k];
    xs[k] = DATA.x[i];
    ys[k] = DATA.y[i];
    zs[k] = DATA.z[i];
    cs[k] = colors[i];
  }}

  return {{xs, ys, zs, cs}};
}}

function allIndices() {{
  const idx = new Array(N);
  for (let i = 0; i < N; i++) idx[i] = i;
  return idx;
}}

const allIdx = allIndices();

let frontDarkIdx = [];
let frontIdx = [];

function updateFrontDark() {{
  const projected = projectForFront(currentCamera);
  frontIdx = computeFrontMost(
    projected.x2d,
    projected.y2d,
    projected.depth,
    DATA.gridSize
  );

  frontDarkIdx = [];
  for (const i of frontIdx) {{
    if (DATA.dark[i] === 1) frontDarkIdx.push(i);
  }}

  const ratio = frontDarkIdx.length / Math.max(frontIdx.length, 1) * 100.0;
  const vd = projected.viewDir;

  document.getElementById("info").textContent =
    `view_dir=(${{vd[0].toFixed(3)}}, ${{vd[1].toFixed(3)}}, ${{vd[2].toFixed(3)}}), ` +
    `front-most=${{frontIdx.length}}, ` +
    `front-most dark=${{frontDarkIdx.length}} ` +
    `(${{ratio.toFixed(2)}}%), ` +
    `dark total=${{darkIndices.length}} / ${{N}}`;
}}

function updateButtons() {{
  const map = [
    ["btnAll", "all"],
    ["btnDark", "dark"],
    ["btnAllFrontDark", "all_front_dark"],
    ["btnFrontDark", "front_dark"],
  ];

  for (const [id, m] of map) {{
    const b = document.getElementById(id);
    if (mode === m) b.classList.add("active");
    else b.classList.remove("active");
  }}
}}

function makeTraces() {{
  const traces = [];

  if (mode === "all") {{
    const a = select3D(allIdx);
    traces.push({{
      x: a.xs,
      y: a.ys,
      z: a.zs,
      mode: "markers",
      type: "scatter3d",
      name: "All Gaussians",
      marker: {{
        size: 2,
        color: a.cs,
        opacity: 1.0
      }}
    }});
  }}

  if (mode === "dark") {{
    const d = select3D(darkIndices);
    traces.push({{
      x: d.xs,
      y: d.ys,
      z: d.zs,
      mode: "markers",
      type: "scatter3d",
      name: "Dark Gaussians",
      marker: {{
        size: 4,
        color: d.cs,
        opacity: 1.0
      }}
    }});
  }}

  if (mode === "all_front_dark") {{
    const a = select3D(allIdx);
    traces.push({{
      x: a.xs,
      y: a.ys,
      z: a.zs,
      mode: "markers",
      type: "scatter3d",
      name: "All Gaussians",
      marker: {{
        size: 2,
        color: a.cs,
        opacity: 0.22
      }}
    }});

    const d = select3D(frontDarkIdx);
    traces.push({{
      x: d.xs,
      y: d.ys,
      z: d.zs,
      mode: "markers",
      type: "scatter3d",
      name: "Front-most Dark",
      marker: {{
        size: 6,
        color: "red",
        opacity: 1.0
      }}
    }});
  }}

  if (mode === "front_dark") {{
    const d = select3D(frontDarkIdx);
    traces.push({{
      x: d.xs,
      y: d.ys,
      z: d.zs,
      mode: "markers",
      type: "scatter3d",
      name: "Front-most Dark",
      marker: {{
        size: 7,
        color: "red",
        opacity: 1.0
      }}
    }});
  }}

  return traces;
}}

function render() {{
  updateButtons();

  const layout = {{
    title: "Mouse Interactive Front-most Dark Gaussian Diagnostic",
    margin: {{l: 0, r: 0, t: 40, b: 0}},
    showlegend: true,
    scene: {{
      aspectmode: "data",
      camera: currentCamera,
      xaxis: {{title: "x"}},
      yaxis: {{title: "y"}},
      zaxis: {{title: "z"}}
    }}
  }};

  Plotly.react("plot", makeTraces(), layout, {{responsive: true}});
}}

function recomputeAndRender() {{
  updateFrontDark();
  render();
}}

document.getElementById("btnAll").addEventListener("click", () => {{
  mode = "all";
  render();
}});

document.getElementById("btnDark").addEventListener("click", () => {{
  mode = "dark";
  render();
}});

document.getElementById("btnAllFrontDark").addEventListener("click", () => {{
  mode = "all_front_dark";
  render();
}});

document.getElementById("btnFrontDark").addEventListener("click", () => {{
  mode = "front_dark";
  render();
}});

document.getElementById("btnUpdate").addEventListener("click", () => {{
  recomputeAndRender();
}});

const plotDiv = document.getElementById("plot");

plotDiv.on("plotly_relayout", function(eventData) {{
  if (eventData["scene.camera"]) {{
    currentCamera = eventData["scene.camera"];
  }}
}});

recomputeAndRender();

</script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", default="mouse_front_dark.html")

    parser.add_argument("--max_points", type=int, default=150000)
    parser.add_argument("--dark_threshold", type=float, default=0.25)
    parser.add_argument("--opacity_threshold", type=float, default=0.3)
    parser.add_argument("--grid_size", type=int, default=500)

    args = parser.parse_args()

    xyz, rgb, alpha = load_gaussian_ply(args.ply)

    xyz, rgb, alpha = random_downsample(
        xyz,
        rgb,
        alpha,
        max_points=args.max_points,
        seed=0,
    )

    print(f"Visualized Gaussians: {len(xyz)}")
    print(f"dark_threshold: {args.dark_threshold}")
    print(f"opacity_threshold: {args.opacity_threshold}")
    print(f"grid_size: {args.grid_size}")

    html = make_html(
        xyz=xyz,
        rgb=rgb,
        alpha=alpha,
        dark_threshold=args.dark_threshold,
        opacity_threshold=args.opacity_threshold,
        grid_size=args.grid_size,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()