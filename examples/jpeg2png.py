import cv2
import os
import glob

def convert_jpg_to_png_cv2(folder_path):
    # globを使用してjpegファイルを一括取得（大文字小文字対応）
    extension_list = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    files = []
    for ext in extension_list:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        print("指定されたフォルダにJPEGファイルが見つかりませんでした。")
        return

    for img_path in files:
        # 画像の読み込み
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"失敗: {img_path} を読み込めませんでした。")
            continue

        # 拡張子を除いたファイル名を取得し、.pngを付与
        file_root = os.path.splitext(img_path)[0]
        new_file_path = f"{file_root}.png"

        # PNG形式で保存
        # cv2.imwriteは拡張子を自動判別して適切なエンコーダを使用します
        success = cv2.imwrite(new_file_path, img)
        
        if success:
            print(f"変換成功: {os.path.basename(img_path)} -> {os.path.basename(new_file_path)}")
        else:
            print(f"失敗: {os.path.basename(img_path)} の保存に失敗しました。")

# ----------------------------------------------
# 設定：変換したいフォルダのパスを指定
# ----------------------------------------------
target_folder = "data/Nordtank2018_512_colmap/images"

if __name__ == "__main__":
    if os.path.exists(target_folder):
        convert_jpg_to_png_cv2(target_folder)
        print("処理が完了しました。")
    else:
        print(f"エラー: フォルダ '{target_folder}' が存在しません。")