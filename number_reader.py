# -*- coding: utf-8 -*-
"""
数字認識モジュール
=================

このモジュールは、分割された数字画像の認識を担当します。

主な機能:
- 既存の学習済みモデルの読み込み
- 数字画像の前処理（正規化、中心化、リサイズ）
- 数字の予測と信頼度計算
- 結果のCSV出力

処理フロー:
1. 既存の学習済みモデルを読み込み
2. debug/フォルダから分割された数字画像を読み込み
3. 各数字画像を前処理（正規化、中心化、リサイズ）
4. 学習済みモデルで数字を予測
5. 結果をCSVファイルに出力

前処理パラメータの最適化により、認識精度が向上します。
"""

import os
import csv
import numpy as np
import cv2
import pickle
from image_utils import safe_imread_gray, safe_imwrite
from config import (
    DirectoryConfig,
    DigitRecognitionConfig,
    OutputConfig,
    DebugConfig
)

def setup_model():
    """
    既存の学習済みモデルを読み込み
    
    Returns:
        学習済みモデル
    """
    if not os.path.exists(DirectoryConfig.MODEL_PATH):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {DirectoryConfig.MODEL_PATH}")
    
    print("既存のモデルを読み込み中...")
    with open(DirectoryConfig.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("モデル読み込み完了")
    return model

def predict_digit(image_path, model):
    """画像ファイルから数字を予測"""
    img = safe_imread_gray(image_path)
    if img is None:
        return "?"

    # 保存フォルダ（元画像と同じディレクトリ）
    save_dir = os.path.dirname(image_path)

    # 01: 元画像保存
    safe_imwrite(os.path.join(save_dir, "step_01_original.png"), img)

    # 反転の自動判定（背景が白なら反転）
    if np.mean(img) > 127:
        img = 255 - img
    safe_imwrite(os.path.join(save_dir, "step_02_inverted.png"), img)

    img = img.astype("float32") / 255.0  # 正規化（0〜1）

    # 画像サイズを正方形に調整（必要なら）
    h, w = img.shape
    size = max(h, w)
    square_img = np.zeros((size, size), dtype=np.float32)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
    safe_imwrite(os.path.join(save_dir, "step_03_square.png"), (square_img * 255).astype(np.uint8))

    # 中心化（重心で移動）
    moments = cv2.moments((square_img * 255).astype(np.uint8))
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = size // 2, size // 2
    shift_x = (size // 2) - cx
    shift_y = (size // 2) - cy
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(square_img, M, (size, size))
    safe_imwrite(os.path.join(save_dir, "step_04_centered.png"), (centered * 255).astype(np.uint8))

    # リサイズ to 28x28
    resized = cv2.resize(centered, (28, 28), interpolation=cv2.INTER_AREA)
    safe_imwrite(os.path.join(save_dir, "step_05_resized.png"), (resized * 255).astype(np.uint8))

    # 推論用形状に変換（784次元の1次元配列）
    input_img = resized.flatten()

    # 予測
    digit = model.predict([input_img])[0]
    return str(digit)

def main():
    """数字認識とCSV出力のメイン関数"""
    # モデルのセットアップ
    model = setup_model()
    
    # CSV出力（UTF-8 BOM付きで日本語対応）
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ファイル名", "認識結果"])

        processed_count = 0
        for folder_name in os.listdir(DEBUG_DIR):
            folder_path = os.path.join(DEBUG_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue

            digits = []
            for i in range(8):
                digit_path = os.path.join(folder_path, f"{folder_name}_digit_{i}.png")
                digits.append(predict_digit(digit_path, model))

            result = "".join(digits)
            writer.writerow([folder_name, result])
            print(f"{folder_name}: {result}")
            processed_count += 1
        
        print(f"数字認識完了: {processed_count}個の画像を処理しました")

if __name__ == "__main__":
    main()