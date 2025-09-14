# -*- coding: utf-8 -*-
"""
数字認識システム - PyTorch版メインハブ
====================================

画像の前処理、二値化・分割、PyTorch数字認識を順次実行
多数決方式による精度向上を実装
"""
import sys

# 日本語ファイル名対応のための設定
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'Japanese_Japan.932')
    except:
        pass
    # コンソール出力のUTF-8化（PowerShell/Terminalでの文字化け対策）
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from image_processor import process_images
import preprocessing
import number_reader

def main():
    """メイン処理：各モジュールを順次実行"""
    print("=== PyTorch数字認識システム開始 ===")
    
    # 1. 画像の前処理（枠削り）
    processed_count = process_images()
    
    if processed_count == 0:
        print("⚠️ 処理可能な画像が見つかりませんでした")
        return
    
    # 2. 画像の二値化と分割
    print("\n=== ステップ2: 画像の二値化と分割 ===")
    preprocessing.main()
    
    # 3. PyTorch数字認識とCSV出力
    print("\n=== ステップ3: PyTorch数字認識とCSV出力 ===")
    number_reader.main()
    
    print(f"\n=== 処理完了 ===")
    print(f"認識結果は result_pytorch.csv に保存されました")

if __name__ == "__main__":
    main()
