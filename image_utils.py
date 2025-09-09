# -*- coding: utf-8 -*-
"""
画像処理ユーティリティモジュール
=====================================

このモジュールは、数字認識システム全体で使用される画像処理の共通機能を提供します。

主な機能:
- 日本語ファイル名対応の画像読み込み・保存
- 画像の前処理（枠削り、ノイズ除去）
- エラー耐性のある画像処理

使用される場面:
- 入力画像の読み込み
- 前処理済み画像の保存
- デバッグ画像の出力
- 数字認識用画像の準備
"""

import cv2
import numpy as np
import os
import sys
from config import ImageProcessingConfig, DebugConfig

# 日本語ファイル名対応のための設定
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'Japanese_Japan.932')
    except:
        pass
    # デバッグ出力のUTF-8化（文字化け対策）
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def safe_imread(filepath):
    """日本語ファイル名対応の画像読み込み"""
    try:
        # 方法1: 通常の読み込み
        img = cv2.imread(filepath)
        if img is not None:
            return img
        
        # 方法2: numpy配列として読み込み
        with open(filepath, 'rb') as f:
            data = f.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"⚠️ 画像読み込みエラー: {filepath} - {e}")
        return None

def safe_imread_gray(filepath):
    """日本語ファイル名対応のグレースケール画像読み込み"""
    try:
        # 方法1: 通常の読み込み
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        
        # 方法2: numpy配列として読み込み
        with open(filepath, 'rb') as f:
            data = f.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"⚠️ 画像読み込みエラー: {filepath} - {e}")
        return None

def safe_imwrite(filepath, img):
    """日本語ファイル名対応の画像保存"""
    try:
        # 方法1: 通常の保存
        success = cv2.imwrite(filepath, img)
        if success:
            return True
        
        # 方法2: エンコードして保存
        ext = os.path.splitext(filepath)[1]
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            with open(filepath, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        return False
    except Exception as e:
        print(f"⚠️ 画像保存エラー: {filepath} - {e}")
        return False

def crop_border(img, crop_ratio=None):
    """
    画像の端を指定した比率で削る
    
    Args:
        img: 入力画像
        crop_ratio: 削る比率（Noneの場合はコンフィグから取得）
    
    Returns:
        枠削り後の画像
    """
    if crop_ratio is None:
        crop_ratio = ImageProcessingConfig.CROP_RATIO
    
    h, w = img.shape[:2]
    top = int(h * crop_ratio)
    bottom = int(h * (1 - crop_ratio))
    left = int(w * crop_ratio)
    right = int(w * (1 - crop_ratio))
    return img[top:bottom, left:right]

def preprocess_image(img):
    """
    画像の平滑化とノイズ除去
    
    処理内容:
    1. 中央値フィルタでガウスノイズを除去
    2. ガウシアンフィルタで小さなノイズを平滑化
    
    Args:
        img: 入力画像（グレースケール）
    
    Returns:
        前処理済み画像
    """
    # 中央値フィルタでガウスノイズを除去
    img_blur = cv2.medianBlur(img, ImageProcessingConfig.MEDIAN_BLUR_KERNEL)
    # 軽いガウシアン平滑化で小さなノイズを平滑化
    img_smooth = cv2.GaussianBlur(img_blur, ImageProcessingConfig.GAUSSIAN_BLUR_KERNEL, 0)
    return img_smooth
