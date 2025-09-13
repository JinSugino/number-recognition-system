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

def binarize(image_path):
    """
    画像を前処理して適応二値化
    
    処理内容:
    1. 画像の読み込み
    2. グレースケール変換
    3. ノイズ除去
    4. 適応的二値化
    
    Args:
        image_path: 入力画像のパス
    
    Returns:
        二値化された画像（None if エラー）
    """
    img = safe_imread(image_path)
    if img is None:
        print(f"⚠️ 読み込み失敗: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)

    # ブロックサイズを画像幅に応じて決定（奇数にする必要あり）
    block_size = max(
        ImageProcessingConfig.MIN_BLOCK_SIZE, 
        (gray.shape[1] // ImageProcessingConfig.ADAPTIVE_BLOCK_SIZE_RATIO) | 1
    )
    
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, ImageProcessingConfig.ADAPTIVE_C
    )

    return binary

def segment_digits(binary_img, base_name):
    """
    二値化画像を垂直投影プロファイルで数字領域に分割（2段階閾値判定）
    + MNISTモデルにフィットする正規化処理（合同拡大 + 重心正規化）

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

    # --- 1. 垂直投影プロファイル算出 ---
    proj = np.sum(binary_img == 255, axis=0)  # 白画素の数をカウント

    threshold1 = DigitSegmentationConfig.PROJECTION_THRESHOLD1
    threshold2 = DigitSegmentationConfig.PROJECTION_THRESHOLD2
    min_width = DigitSegmentationConfig.MIN_WIDTH

    # --- 2. プロファイル可視化 ---
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

    # --- 3. 区間抽出 & 正規化 ---
    digits = []
    in_region = False
    x_start = 0
    peak_in_region = False

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
                    # ① 正方形パディング
                    digit_square = pad_to_square(digit)
                    # ② 合同拡大正規化
                    digit_normalized = scale_to_max_size(digit_square, ImageProcessingConfig.TARGET_DIGIT_SIZE)
                    # ③ 重心正規化
                    digit_centered = center_of_mass_normalization(digit_normalized)
                    # ④ MNISTサイズにリサイズ
                    digit_resized = cv2.resize(
                        digit_centered,
                        (ImageProcessingConfig.TARGET_DIGIT_SIZE, ImageProcessingConfig.TARGET_DIGIT_SIZE)
                    )
                    # デバッグ保存
                    save_path = os.path.join(debug_subdir, f"{base_name}_digit_{len(digits)}.png")
                    if safe_imwrite(save_path, digit_resized):
                        digits.append(digit_resized)
                    else:
                        print(f"⚠️ 数字画像保存失敗: {save_path}")

    # 最後まで閾値1を超えていた場合
    if in_region:
        x_end = w
        width = x_end - x_start
        if width >= min_width and peak_in_region:
            digit = binary_img[:, x_start:x_end]
            if digit.shape[1] > 0:
                digit_square = pad_to_square(digit)
                digit_normalized = scale_to_max_size(digit_square, ImageProcessingConfig.TARGET_DIGIT_SIZE)
                digit_centered = center_of_mass_normalization(digit_normalized)
                digit_resized = cv2.resize(
                    digit_centered,
                    (ImageProcessingConfig.TARGET_DIGIT_SIZE, ImageProcessingConfig.TARGET_DIGIT_SIZE)
                )
                save_path = os.path.join(debug_subdir, f"{base_name}_digit_{len(digits)}.png")
                if safe_imwrite(save_path, digit_resized):
                    digits.append(digit_resized)
                else:
                    print(f"⚠️ 数字画像保存失敗: {save_path}")

    if DebugConfig.VERBOSE_LOGGING:
        print(f"{base_name}: {len(digits)}個の数字領域を抽出して保存完了")

    return digits


def scale_to_max_size(img, target_size):
    """
    画像を最大辺に合わせて合同拡大（縦横比維持）
    """
    h, w = img.shape
    scale = target_size / max(h, w)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


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