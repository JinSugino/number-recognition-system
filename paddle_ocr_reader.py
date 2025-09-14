# -*- coding: utf-8 -*-
"""
PaddleOCR数字認識モジュール
=========================

PaddleOCRを使用した高精度な数字認識を実装します。
"""

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from image_utils import safe_imread
from config import DirectoryConfig, DigitRecognitionConfig

class PaddleOCRReader:
    """PaddleOCRを使用した数字認識クラス"""
    
    def __init__(self):
        """PaddleOCRインスタンスを初期化"""
        # 日本語と英語に対応したPaddleOCRを初期化
        self.ocr = PaddleOCR(
            lang='japan'  # 日本語対応
        )
        print("✅ PaddleOCR初期化完了")
    
    def recognize_digits_from_image(self, image_path):
        """
        画像から数字を認識
        
        Args:
            image_path: 入力画像のパス
            
        Returns:
            list: 認識された数字のリスト
        """
        try:
            # 画像を読み込み
            img = safe_imread(image_path)
            if img is None:
                print(f"⚠️ 画像読み込み失敗: {image_path}")
                return []
            
            # PaddleOCRで文字認識を実行
            results = self.ocr.ocr(img)
            
            digits = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]  # 認識されたテキスト
                        confidence = line[1][1]  # 信頼度
                        
                        # 数字のみを抽出
                        digit_text = ''.join(filter(str.isdigit, text))
                        
                        if digit_text and confidence > 0.5:  # 信頼度チェック
                            # 各桁を個別に追加
                            for digit in digit_text:
                                digits.append(digit)
            
            return digits
            
        except Exception as e:
            print(f"⚠️ PaddleOCR認識エラー: {image_path} - {e}")
            return []
    
    def recognize_digits_from_region(self, digit_region):
        """
        数字領域画像から数字を認識
        
        Args:
            digit_region: 数字領域の画像（numpy.ndarray）
            
        Returns:
            str: 認識された数字（1文字）
        """
        try:
            # 画像を28x28にリサイズ（PaddleOCR用）
            if digit_region.shape != (28, 28):
                digit_region = cv2.resize(digit_region, (28, 28), interpolation=cv2.INTER_AREA)
            
            # グレースケールに変換
            if len(digit_region.shape) == 3:
                digit_region = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
            
            # 白背景に黒文字に変換（PaddleOCR用）
            digit_region = cv2.bitwise_not(digit_region)
            
            # PaddleOCRで認識（新しいバージョンではclsパラメータを削除）
            results = self.ocr.ocr(digit_region)
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        # 数字のみを抽出
                        digit_text = ''.join(filter(str.isdigit, text))
                        
                        if digit_text and confidence > DigitRecognitionConfig.MIN_CONFIDENCE:
                            return digit_text[0]  # 最初の数字を返す
            
            return "?"  # 認識失敗
            
        except Exception as e:
            print(f"⚠️ 数字認識エラー: {e}")
            return "?"
    
    def process_debug_folder(self):
        """
        debugフォルダ内の画像を処理して数字認識を実行
        
        Returns:
            dict: 画像名をキー、認識結果を値とする辞書
        """
        results = {}
        processed_count = 0
        
        print("\n=== PaddleOCR数字認識開始 ===")
        
        # debugフォルダ内の各画像フォルダを処理
        for folder_name in os.listdir(DirectoryConfig.DEBUG_DIR):
            folder_path = os.path.join(DirectoryConfig.DEBUG_DIR, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # 数字画像ファイルを取得（digit_*.png）
            digit_files = []
            for file in os.listdir(folder_path):
                if file.startswith(f"{folder_name}_digit_") and file.endswith(".png"):
                    digit_files.append(file)
            
            # ファイル名でソート
            digit_files.sort()
            
            # 各数字を認識
            recognized_digits = []
            for digit_file in digit_files:
                digit_path = os.path.join(folder_path, digit_file)
                digit_img = safe_imread(digit_path)
                
                if digit_img is not None:
                    digit = self.recognize_digits_from_region(digit_img)
                    recognized_digits.append(digit)
                else:
                    recognized_digits.append("?")
            
            # 結果を保存
            if recognized_digits:
                results[folder_name] = ''.join(recognized_digits)
                processed_count += 1
                print(f"{folder_name}: {''.join(recognized_digits)}")
        
        print(f"\n✅ PaddleOCR認識完了: {processed_count}個の画像を処理しました")
        return results

def main():
    """メイン関数"""
    reader = PaddleOCRReader()
    results = reader.process_debug_folder()
    
    # 結果をCSVに保存
    output_file = "result_paddleocr.csv"
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write("画像名,認識結果\n")
        for image_name, digits in results.items():
            f.write(f"{image_name},{digits}\n")
    
    print(f"結果は {output_file} に保存されました")

if __name__ == "__main__":
    main()
