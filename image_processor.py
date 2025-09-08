# -*- coding: utf-8 -*-
"""
画像前処理モジュール
===================

このモジュールは、入力画像の前処理を担当します。

主な機能:
- 入力画像の読み込み
- 画像の枠削り処理
- 前処理済み画像の保存

処理フロー:
1. input/フォルダから画像ファイルを読み込み
2. 各画像の端を指定した比率で削除（黒枠除去）
3. 前処理済み画像をdata/フォルダに保存

この処理により、数字認識の精度向上を図ります。
"""

import os
from image_utils import safe_imread, safe_imwrite, crop_border
from config import DirectoryConfig, ImageProcessingConfig, DebugConfig

def process_images():
    """
    画像の前処理（枠削り）を実行
    
    処理内容:
    1. input/フォルダから画像ファイルを検索
    2. 各画像の端を削除（黒枠やノイズ除去）
    3. 前処理済み画像をdata/フォルダに保存
    
    Returns:
        int: 処理した画像の数
    """
    print("\n=== ステップ1: 画像の前処理（枠削り） ===")
    
    # ディレクトリの作成
    os.makedirs(DirectoryConfig.DATA_DIR, exist_ok=True)
    
    processed_count = 0
    for file in os.listdir(DirectoryConfig.INPUT_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                path = os.path.join(DirectoryConfig.INPUT_DIR, file)
                img = safe_imread(path)
                if img is None:
                    print(f"⚠️ 読み込み失敗: {file}")
                    continue

                # コンフィグから枠削り比率を取得
                cropped = crop_border(img, ImageProcessingConfig.CROP_RATIO)
                save_path = os.path.join(DirectoryConfig.DATA_DIR, file)
                
                if safe_imwrite(save_path, cropped):
                    if DebugConfig.VERBOSE_LOGGING:
                        print(f"{file}: 枠削り後、data に保存完了")
                    processed_count += 1
                else:
                    print(f"⚠️ 保存失敗: {file}")
            except Exception as e:
                print(f"⚠️ 処理エラー: {file} - {e}")
                continue
    
    print(f"処理完了: {processed_count}個の画像を処理しました")
    return processed_count

if __name__ == "__main__":
    process_images()
