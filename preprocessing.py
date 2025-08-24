import cv2
import os
import numpy as np

INPUT_DIR = "data"
DEBUG_DIR = "debug"

def preprocess_image(img):
    """平滑化＋ノイズ除去"""
    # 中央値フィルタでガウスノイズを除去
    img_blur = cv2.medianBlur(img, 3)
    # 軽いガウシアン平滑化で小さなノイズを平滑化
    img_smooth = cv2.GaussianBlur(img_blur, (3, 3), 0)
    return img_smooth

def binarize(image_path):
    """画像を前処理して適応二値化"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 読み込み失敗: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)

    # ブロックサイズを画像幅に応じて決定（奇数にする必要あり）
    block_size = max(15, (gray.shape[1] // 20) | 1)  # 画像幅の1/20以上、奇数
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 2
    )

    return binary

def segment_digits(binary_img, base_name):
    """8文字想定で縦分割して保存"""
    h, w = binary_img.shape
    slice_width = w // 8
    debug_subdir = os.path.join(DEBUG_DIR, base_name)
    os.makedirs(debug_subdir, exist_ok=True)

    digits = []
    for i in range(8):
        x_start = i * slice_width
        x_end = w if i == 7 else (i + 1) * slice_width
        digit = binary_img[:, x_start:x_end]
        digit_resized = cv2.resize(digit, (28, 28))
        save_path = os.path.join(debug_subdir, f"{base_name}_digit_{i}.png")
        cv2.imwrite(save_path, digit_resized)
        digits.append(digit_resized)

    print(f"{base_name}: 8個に分割して保存完了")
    return digits

def main():
    """画像の二値化と分割処理のメイン関数"""
    os.makedirs(DEBUG_DIR, exist_ok=True)

    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(INPUT_DIR, file)
            base_name = os.path.splitext(file)[0]
            binary = binarize(path)
            if binary is not None:
                segment_digits(binary, base_name)

if __name__ == "__main__":
    main()
