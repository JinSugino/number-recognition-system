import cv2
import os
import preprocessing
import number_reader

INPUT_DIR = "input"
DATA_DIR = "data"

def crop_border(img, crop_ratio=0.02):
    """画像の端を crop_ratio だけ削る"""
    h, w = img.shape[:2]
    top = int(h * crop_ratio)
    bottom = int(h * (1 - crop_ratio))
    left = int(w * crop_ratio)
    right = int(w * (1 - crop_ratio))
    return img[top:bottom, left:right]

def main():
    """メイン処理：画像の前処理、二値化・分割、数字認識を順次実行"""
    print("=== 数字認識システム開始 ===")
    
    # 1. ディレクトリの作成
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("\n=== ステップ1: 画像の前処理（枠削り） ===")
    # 2. 画像の前処理（枠削り）
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(INPUT_DIR, file)
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️ 読み込み失敗: {path}")
                continue

            cropped = crop_border(img, crop_ratio=0.02)
            save_path = os.path.join(DATA_DIR, file)
            cv2.imwrite(save_path, cropped)
            print(f"{file}: 枠削り後、data に保存完了")
    
    print("\n=== ステップ2: 画像の二値化と分割 ===")
    # 3. 画像の二値化と分割
    preprocessing.main()
    
    print("\n=== ステップ3: 数字認識とCSV出力 ===")
    # 4. 数字認識とCSV出力
    number_reader.main()
    
    print(f"\n=== 処理完了 ===")
    print(f"結果は result.csv に保存されました")

if __name__ == "__main__":
    main()
