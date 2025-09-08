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
    二値化画像を8個の数字領域に分割
    
    処理内容:
    1. 画像を8等分に分割
    2. 各領域を28x28にリサイズ
    3. デバッグフォルダに保存
    
    Args:
        binary_img: 二値化された画像
        base_name: ベースファイル名
    
    Returns:
        分割された数字画像のリスト
    """
    h, w = binary_img.shape
    slice_width = w // DigitSegmentationConfig.EXPECTED_DIGITS
    debug_subdir = os.path.join(DirectoryConfig.DEBUG_DIR, base_name)
    os.makedirs(debug_subdir, exist_ok=True)

    digits = []
    for i in range(DigitSegmentationConfig.EXPECTED_DIGITS):
        x_start = i * slice_width
        x_end = w if i == DigitSegmentationConfig.EXPECTED_DIGITS - 1 else (i + 1) * slice_width
        digit = binary_img[:, x_start:x_end]
        digit_resized = cv2.resize(digit, (ImageProcessingConfig.TARGET_DIGIT_SIZE, ImageProcessingConfig.TARGET_DIGIT_SIZE))
        save_path = os.path.join(debug_subdir, f"{base_name}_digit_{i}.png")
        if safe_imwrite(save_path, digit_resized):
            digits.append(digit_resized)
        else:
            print(f"⚠️ 数字画像保存失敗: {save_path}")

    if DebugConfig.VERBOSE_LOGGING:
        print(f"{base_name}: {DigitSegmentationConfig.EXPECTED_DIGITS}個に分割して保存完了")
    return digits

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