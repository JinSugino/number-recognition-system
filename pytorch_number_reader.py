# -*- coding: utf-8 -*-
"""
PyTorch数字認識モジュール
======================

PyTorchベースのCNNモデルを使用した数字認識システム
多数決方式による精度向上を実装
"""

import os
import csv
import numpy as np
import cv2
from image_utils import safe_imread_gray, safe_imwrite
from config import (
    DirectoryConfig,
    DigitRecognitionConfig,
    OutputConfig,
    DebugConfig
)
from pytorch_model import PyTorchDigitRecognizer

def setup_pytorch_model():
    """
    PyTorch学習済みモデルを読み込み
    
    Returns:
        PyTorchDigitRecognizer: 学習済みモデル
    """
    model_path = "pytorch_digit_model.pth"
    
    if not os.path.exists(model_path):
        print("PyTorchモデルが見つかりません。MNISTデータセットで訓練を開始します...")
        recognizer = PyTorchDigitRecognizer()
        
        # MNISTデータセットで訓練
        from pytorch_model import create_mnist_dataset
        train_data, val_data = create_mnist_dataset()
        recognizer.train_model(train_data, val_data, epochs=20)
        
        # モデル保存
        recognizer.save_model(model_path)
    else:
        print("PyTorchモデルを読み込み中...")
        recognizer = PyTorchDigitRecognizer(model_path)
    
    print("PyTorchモデル読み込み完了")
    return recognizer

def predict_digit_with_pytorch(image_path, recognizer):
    """PyTorchモデルで画像ファイルから数字を予測（多数決方式）"""
    if not os.path.exists(image_path):
        return "?", 0.0, [], []
    
    # 保存フォルダ（元画像と同じディレクトリ）
    save_dir = os.path.dirname(image_path)
    
    # デバッグ用：元画像保存
    if DebugConfig.SAVE_ORIGINAL:
        img = safe_imread_gray(image_path)
        if img is not None:
            safe_imwrite(os.path.join(save_dir, "pytorch_01_original.png"), img)
    
    # PyTorchモデルで予測（多数決方式）
    digit, confidence, predictions, confidences = recognizer.predict_digit(image_path)
    
    # デバッグ情報を保存
    if DebugConfig.ENABLE_DEBUG_OUTPUT:
        debug_info = {
            'predictions': predictions,
            'confidences': confidences,
            'final_digit': digit,
            'final_confidence': confidence
        }
        
        # デバッグ情報をテキストファイルに保存
        debug_file = os.path.join(save_dir, f"pytorch_debug_{os.path.basename(image_path)}.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"予測結果: {digit}\n")
            f.write(f"信頼度: {confidence:.4f}\n")
            f.write(f"個別予測: {predictions}\n")
            f.write(f"個別信頼度: {[f'{c:.4f}' for c in confidences]}\n")
    
    return digit, confidence, predictions, confidences

def main():
    """PyTorch数字認識とCSV出力のメイン関数"""
    print("=== PyTorch数字認識システム開始 ===")
    
    # PyTorchモデルのセットアップ
    recognizer = setup_pytorch_model()
    
    # CSV出力（UTF-8 BOM付きで日本語対応）
    output_file = "result_pytorch.csv"
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ファイル名", "認識結果", "信頼度", "個別予測", "個別信頼度"])

        processed_count = 0
        total_confidence = 0.0
        
        for folder_name in os.listdir(DirectoryConfig.DEBUG_DIR):
            folder_path = os.path.join(DirectoryConfig.DEBUG_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue

            digits = []
            confidences = []
            all_predictions = []
            all_individual_confidences = []
            
            for i in range(8):
                digit_path = os.path.join(folder_path, f"{folder_name}_digit_{i}.png")
                digit, confidence, predictions, individual_confidences = predict_digit_with_pytorch(digit_path, recognizer)
                
                digits.append(digit)
                confidences.append(confidence)
                all_predictions.append(predictions)
                all_individual_confidences.append(individual_confidences)

            result = "".join(digits)
            avg_confidence = np.mean(confidences)
            total_confidence += avg_confidence
            
            # 個別予測と信頼度を文字列として保存
            predictions_str = "|".join([str(p) for p in all_predictions])
            confidences_str = "|".join([str(c) for c in all_individual_confidences])
            
            writer.writerow([folder_name, result, f"{avg_confidence:.4f}", predictions_str, confidences_str])
            print(f"{folder_name}: {result} (信頼度: {avg_confidence:.4f})")
            processed_count += 1
        
        avg_total_confidence = total_confidence / processed_count if processed_count > 0 else 0.0
        print(f"\nPyTorch数字認識完了: {processed_count}個の画像を処理しました")
        print(f"平均信頼度: {avg_total_confidence:.4f}")
        print(f"結果は {output_file} に保存されました")

def compare_with_sklearn():
    """scikit-learnモデルとの比較"""
    print("\n=== scikit-learnモデルとの比較 ===")
    
    # PyTorchモデル
    pytorch_recognizer = setup_pytorch_model()
    
    # scikit-learnモデル
    import pickle
    with open(DirectoryConfig.MODEL_PATH, 'rb') as f:
        sklearn_model = pickle.load(f)
    
    comparison_results = []
    
    for folder_name in os.listdir(DirectoryConfig.DEBUG_DIR):
        folder_path = os.path.join(DirectoryConfig.DEBUG_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        pytorch_result = ""
        sklearn_result = ""
        
        for i in range(8):
            digit_path = os.path.join(folder_path, f"{folder_name}_digit_{i}.png")
            
            # PyTorch予測
            pytorch_digit, _, _, _ = predict_digit_with_pytorch(digit_path, pytorch_recognizer)
            pytorch_result += pytorch_digit
            
            # scikit-learn予測
            img = safe_imread_gray(digit_path)
            if img is not None:
                # 前処理（元のnumber_reader.pyと同じ）
                if np.mean(img) > 127:
                    img = 255 - img
                img = img.astype("float32") / 255.0
                
                h, w = img.shape
                size = max(h, w)
                square_img = np.zeros((size, size), dtype=np.float32)
                x_offset = (size - w) // 2
                y_offset = (size - h) // 2
                square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
                
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
                
                resized = cv2.resize(centered, (28, 28), interpolation=cv2.INTER_AREA)
                input_img = resized.flatten()
                
                sklearn_digit = sklearn_model.predict([input_img])[0]
                sklearn_result += str(sklearn_digit)
            else:
                sklearn_result += "?"
        
        comparison_results.append({
            'filename': folder_name,
            'pytorch': pytorch_result,
            'sklearn': sklearn_result,
            'match': pytorch_result == sklearn_result
        })
    
    # 比較結果をCSVに保存
    with open("model_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ファイル名", "PyTorch結果", "scikit-learn結果", "一致"])
        
        matches = 0
        for result in comparison_results:
            writer.writerow([result['filename'], result['pytorch'], result['sklearn'], result['match']])
            if result['match']:
                matches += 1
        
        accuracy = matches / len(comparison_results) if comparison_results else 0
        print(f"モデル一致率: {accuracy:.2%} ({matches}/{len(comparison_results)})")
        print("比較結果は model_comparison.csv に保存されました")

if __name__ == "__main__":
    main()
    compare_with_sklearn()
