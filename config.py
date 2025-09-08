# -*- coding: utf-8 -*-
"""
数字認識システム - 設定ファイル
各モジュールのパラメータを一元管理し、パフォーマンス調整を容易にする
"""

# =============================================================================
# ディレクトリ設定
# =============================================================================
class DirectoryConfig:
    """ディレクトリパスの設定"""
    INPUT_DIR = "input"          # 入力画像フォルダ
    DATA_DIR = "data"            # 前処理済み画像フォルダ
    DEBUG_DIR = "debug"          # デバッグ画像フォルダ
    OUTPUT_CSV = "result.csv"    # 結果出力CSVファイル
    MODEL_PATH = "sklearn_mnist_model.pkl"  # 学習済みモデルファイル

# =============================================================================
# 画像前処理設定
# =============================================================================
class ImageProcessingConfig:
    """画像前処理のパラメータ設定"""
    
    # 枠削り設定
    CROP_RATIO = 0.03
    
    # ノイズ除去設定
    MEDIAN_BLUR_KERNEL = 5
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    
    # 適応的二値化設定
    ADAPTIVE_BLOCK_SIZE_RATIO = 15
    ADAPTIVE_C = 3
    
    # 画像サイズ設定
    TARGET_DIGIT_SIZE = 28        # 数字認識用の画像サイズ (28x28)
    MIN_BLOCK_SIZE = 15           # 最小ブロックサイズ

# =============================================================================
# 数字分割設定
# =============================================================================
class DigitSegmentationConfig:
    """数字分割のパラメータ設定"""
    
    EXPECTED_DIGITS = 8           # 期待する数字の桁数
    SEGMENTATION_METHOD = "uniform"  # 分割方法: "uniform", "adaptive"
    
    # 適応的分割のパラメータ（SEGMENTATION_METHOD = "adaptive"の場合）
    MIN_DIGIT_WIDTH = 20          # 最小数字幅
    MAX_DIGIT_WIDTH = 100         # 最大数字幅
    WHITE_SPACE_THRESHOLD = 0.1   # 空白判定の閾値

# =============================================================================
# モデル設定
# =============================================================================
class ModelConfig:
    """モデル関連の設定"""
    
    # モデルファイルパス
    MODEL_PATH = "sklearn_mnist_model.pkl"  # 学習済みモデルファイル

# =============================================================================
# 数字認識設定
# =============================================================================
class DigitRecognitionConfig:
    """数字認識のパラメータ設定"""
    
    # 画像正規化設定
    NORMALIZE_IMAGE = True        # 画像の正規化 (0-1範囲)
    AUTO_INVERT = True            # 自動反転 (背景が白の場合)
    
    # 中心化設定
    CENTER_IMAGE = True           # 画像の中心化
    CENTER_METHOD = "centroid"    # 中心化方法: "centroid", "geometric"
    
    # リサイズ設定
    RESIZE_METHOD = "inter_area"  # リサイズ方法: "inter_area", "inter_cubic", "inter_linear"
    
    # 信頼度設定
    MIN_CONFIDENCE = 0.5
    ENABLE_CONFIDENCE_CHECK = True

# =============================================================================
# デバッグ・ログ設定
# =============================================================================
class DebugConfig:
    """デバッグとログの設定"""
    
    ENABLE_DEBUG_OUTPUT = True    # デバッグ画像の出力
    SAVE_INTERMEDIATE_STEPS = True  # 中間ステップの保存
    VERBOSE_LOGGING = True        # 詳細ログの出力
    
    # デバッグ画像の保存設定
    SAVE_ORIGINAL = True          # 元画像の保存
    SAVE_INVERTED = True          # 反転画像の保存
    SAVE_SQUARE = True            # 正方形化画像の保存
    SAVE_CENTERED = True          # 中心化画像の保存
    SAVE_RESIZED = True           # リサイズ画像の保存

# =============================================================================
# パフォーマンス設定
# =============================================================================
class PerformanceConfig:
    """パフォーマンス最適化の設定"""
    
    # 並列処理設定
    N_JOBS = -1                   # 並列処理数 (-1: 全CPU使用, 1: シングルスレッド)

# =============================================================================
# 出力設定
# =============================================================================
class OutputConfig:
    """出力形式の設定"""
    
    # CSV出力設定
    CSV_ENCODING = "utf-8-sig"    # CSVファイルのエンコーディング
    CSV_HEADER_JP = True          # 日本語ヘッダーの使用
    
    # 結果表示設定
    SHOW_CONFIDENCE = True        # 信頼度の表示
    SHOW_PROCESSING_TIME = True   # 処理時間の表示
    SHOW_PROGRESS = True          # 進捗の表示

# =============================================================================
# パフォーマンス向上のための推奨設定
# =============================================================================
class PerformanceOptimization:
    """パフォーマンス向上のための推奨設定"""
    
    @staticmethod
    def get_high_accuracy_config():
        """高精度設定 (90%以上を目指す)"""
        return {
            'ImageProcessingConfig': {
                'CROP_RATIO': 0.03,
                'MEDIAN_BLUR_KERNEL': 5,
                'GAUSSIAN_BLUR_KERNEL': (5, 5),
                'ADAPTIVE_BLOCK_SIZE_RATIO': 15,
                'ADAPTIVE_C': 3,
            },
            'DigitRecognitionConfig': {
                'MIN_CONFIDENCE': 0.5,
                'ENABLE_CONFIDENCE_CHECK': True,
            }
        }
    
    @staticmethod
    def get_balanced_config():
        """バランス設定 (精度と速度のバランス)"""
        return {
            'ImageProcessingConfig': {
                'CROP_RATIO': 0.02,
                'MEDIAN_BLUR_KERNEL': 3,
                'ADAPTIVE_BLOCK_SIZE_RATIO': 20,
            }
        }
    
    @staticmethod
    def get_fast_config():
        """高速設定 (速度重視)"""
        return {
            'ImageProcessingConfig': {
                'CROP_RATIO': 0.01,
                'MEDIAN_BLUR_KERNEL': 3,
                'ADAPTIVE_BLOCK_SIZE_RATIO': 25,
            }
        }
