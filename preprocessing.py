# -*- coding: utf-8 -*-
"""
画像二値化・分割モジュール
=========================

このモジュールは、前処理済み画像の二値化と数字分割を担当します。

主な機能:
- 適応的二値化による文字抽出
- 8桁の数字の自動分割
- デバッグ画像の保存

処理フロー:
1. data/フォルダから前処理済み画像を読み込み
2. グレースケール変換とノイズ除去
3. 適応的二値化で文字を抽出
4. 8個の数字領域に分割
5. 各数字を28x28にリサイズ
6. debug/フォルダに分割結果を保存

この処理により、数字認識の精度向上を図ります。
"""

import cv2
import os
import numpy as np
from image_utils import safe_imread, safe_imwrite, preprocess_image
from config import (
    DirectoryConfig, 
    ImageProcessingConfig, 
    DigitSegmentationConfig,
    DebugConfig
)

INPUT_DIR = "data"
DEBUG_DIR = "debug"

def preprocess_image(img):
    """平滑化＋ノイズ除去"""
    # 中央値フィルタでガウスノイズを除去
    img_blur = cv2.medianBlur(img, 3)
    # 軽いガウシアン平滑化で小さなノイズを平滑化
    img_smooth = cv2.GaussianBlur(img_blur, (3, 3), 0)
    return img_smooth

import cv2
import numpy as np
from image_utils import safe_imread, preprocess_image
from config import ImageProcessingConfig

def binarize(image_path):
    """
    画像を前処理して完全二値化（薄い文字対応版）
    
    Args:
        image_path: 入力画像のパス
    
    Returns:
        二値化された画像（numpy.ndarray, 0 or 255）
        None if エラー
    """
    img = safe_imread(image_path)
    if img is None:
        print(f"⚠️ 読み込み失敗: {image_path}")
        return None

    # --- グレースケール化 + 平滑化 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)

    # --- 適応二値化 ---
    block_size = max(
        ImageProcessingConfig.MIN_BLOCK_SIZE, 
        (gray.shape[1] // ImageProcessingConfig.ADAPTIVE_BLOCK_SIZE_RATIO) | 1
    )
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        ImageProcessingConfig.ADAPTIVE_C
    )

    # --- モルフォロジーで文字をくっきり ---
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 文字の穴埋め
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # 小さなノイズ除去

    # --- 完全二値化確認 ---
    binary[binary > 0] = 255
    binary[binary <= 0] = 0

    return binary

def segment_digits(binary_img, base_name):
    """
    二値化画像を垂直投影プロファイルで数字領域に分割（2段階閾値判定）
    + MNISTモデルにフィットする正規化処理
      （正方形パディング → 重心正規化 → 合同拡大 → 保存）

    Args:
        binary_img: 二値化された画像 (numpy.ndarray)
        base_name: ベースファイル名

    Returns:
        digits: 分割された数字画像のリスト（MNIST互換28x28）
    """
    import matplotlib.pyplot as plt

    h, w = binary_img.shape
    debug_subdir = os.path.join(DirectoryConfig.DEBUG_DIR, base_name)
    os.makedirs(debug_subdir, exist_ok=True)

    # --- 垂直投影プロファイル算出 ---
    proj = np.sum(binary_img == 255, axis=0)

    threshold1 = DigitSegmentationConfig.PROJECTION_THRESHOLD1
    threshold2 = DigitSegmentationConfig.PROJECTION_THRESHOLD2
    min_width = DigitSegmentationConfig.MIN_WIDTH

    # --- プロファイル可視化 ---
    plt.figure(figsize=(10, 4))
    plt.plot(proj, label="Vertical Projection")
    plt.axhline(y=threshold1, color="red", linestyle="--", label="Threshold1")
    plt.axhline(y=threshold2, color="blue", linestyle="--", label="Threshold2")
    plt.title(f"Projection Profile: {base_name}")
    plt.xlabel("X-axis (columns)")
    plt.ylabel("White Pixel Count")
    plt.legend()
    profile_path = os.path.join(debug_subdir, f"{base_name}_projection.png")
    plt.savefig(profile_path)
    plt.close()

    digits = []
    in_region = False
    x_start = 0
    peak_in_region = False

    def process_digit(digit, idx):
        # ① 正方形パディング
        digit_square = pad_to_square(digit)
        # ② 重心正規化
        digit_centered = center_of_mass_normalization(digit_square)
        # ③ 枠いっぱいに拡大（最終28x28）
        digit_resized = scale_to_max_size(digit_centered, ImageProcessingConfig.TARGET_DIGIT_SIZE)
        # デバッグ保存
        save_path = os.path.join(debug_subdir, f"{base_name}_digit_{idx}.png")
        if safe_imwrite(save_path, digit_resized):
            return digit_resized
        else:
            print(f"⚠️ 数字画像保存失敗: {save_path}")
            return None

    for x in range(w):
        if not in_region and proj[x] > threshold1:
            in_region = True
            x_start = x
            peak_in_region = proj[x] > threshold2
        elif in_region:
            if proj[x] > threshold2:
                peak_in_region = True
            if proj[x] <= threshold1:
                in_region = False
                x_end = x
                width = x_end - x_start
                if width < min_width or not peak_in_region:
                    continue
                digit = binary_img[:, x_start:x_end]
                if digit.shape[1] > 0:
                    res = process_digit(digit, len(digits))
                    if res is not None:
                        digits.append(res)

    # 最後まで閾値1を超えていた場合
    if in_region:
        x_end = w
        width = x_end - x_start
        if width >= min_width and peak_in_region:
            digit = binary_img[:, x_start:x_end]
            if digit.shape[1] > 0:
                res = process_digit(digit, len(digits))
                if res is not None:
                    digits.append(res)

    if DebugConfig.VERBOSE_LOGGING:
        print(f"{base_name}: {len(digits)}個の数字領域を抽出して保存完了")

    return digits


def scale_to_max_size(img, target_size, threshold=128):
    """
    画像をターゲットサイズに枠いっぱいに収める最大化処理
    - 内部で二値化 (閾値threshold)
    - 正方形パディング + 枠いっぱいに拡大
    - MNIST互換28x28サイズ向け

    Args:
        img: 入力グレースケール画像 (numpy.ndarray)
        target_size: 出力画像サイズ（例: 28）
        threshold: 二値化の閾値 (0-255)

    Returns:
        final_img: target_size x target_size の二値化画像
    """
    import numpy as np
    import cv2

    if img is None or img.size == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    # --- 1. 閾値二値化 ---
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # --- 2. 非ゼロ部分の外接矩形で切り取り ---
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    digit = binary[y:y+h, x:x+w]

    # --- 3. 正方形パディング ---
    size = max(w, h)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    digit_square = cv2.copyMakeBorder(digit, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)

    # --- 4. 枠いっぱいに拡大 ---
    final_img = cv2.resize(digit_square, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return final_img


def center_of_mass_normalization(img):
    """
    重心を画像中心に移動する（MNISTの重心正規化）
    """
    import numpy as np
    from scipy.ndimage import center_of_mass

    cy, cx = center_of_mass(img)
    h, w = img.shape
    shift_x = int(w / 2 - cx)
    shift_y = int(h / 2 - cy)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img_shifted = cv2.warpAffine(img, M, (w, h), borderValue=0)
    return img_shifted

def pad_to_square(img):
    """
    入力画像を上下左右にパディングして正方形にする。
    """
    h, w = img.shape
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=0)  # 黒で埋める

def main():
    """
    画像の二値化と分割処理のメイン関数
    
    処理内容:
    1. data/フォルダから前処理済み画像を読み込み
    2. 各画像を二値化
    3. 8個の数字領域に分割
    4. デバッグフォルダに保存
    """
    os.makedirs(DirectoryConfig.DEBUG_DIR, exist_ok=True)
    processed_count = 0

    for file in os.listdir(DirectoryConfig.DATA_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                path = os.path.join(DirectoryConfig.DATA_DIR, file)
                base_name = os.path.splitext(file)[0]
                binary = binarize(path)
                if binary is not None:
                    segment_digits(binary, base_name)
                    processed_count += 1
            except Exception as e:
                print(f"⚠️ 処理エラー: {file} - {e}")
                continue
    
    print(f"前処理完了: {processed_count}個の画像を処理しました")

if __name__ == "__main__":
    main()