import torch
import tyro
from gsplat import export_splats
from pathlib import Path
from typing import Optional

def main(
    ckpt_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
):
    ckpt_path = Path(ckpt_path)
    
    # 出力先が指定されていない場合、ckptsの親フォルダ下のplyフォルダに保存する
    if output_path is None:
        # results/<project>/ckpts/ckpt_XXXX.pt -> results/<project>/ply/ckpt_XXXX.ply
        project_dir = ckpt_path.parent.parent
        ply_dir = project_dir / "ply"
        output_path = ply_dir / f"{ckpt_path.stem}.ply"
    else:
        output_path = Path(output_path)

    # ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # チェックポイントの読み込み
    ckpt = torch.load(ckpt_path, map_location=device)
    splats = ckpt["splats"]
    
    # データの調整
    means = splats["means"]
    scales = splats["scales"]
    quats = splats["quats"]
    opacities = splats["opacities"]
    sh0 = splats["sh0"]
    shN = splats["shN"]

    # PLYとして出力
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=output_path,
    )
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    tyro.cli(main)
