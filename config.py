# -*- coding: utf-8 -*-
"""
数字認識システム - 設定ファイル
各モジュールのパラメータを一元管理し、パフォーマンス調整を容易にする
"""

import os

# =============================================================================
# ディレクトリ設定
# =============================================================================
class DirectoryConfig:
    """ディレクトリパスの設定"""
    INPUT_DIR = "input"          # 元の入力画像を置くフォルダ（処理の起点）
    DATA_DIR = "data"            # 枠削りなど前処理後の画像を書き出すフォルダ（中間成果物）
    DEBUG_DIR = "debug"          # 分割結果や各ステップのデバッグ画像を書き出すフォルダ
    OUTPUT_CSV = "result.csv"    # 認識結果を保存するCSV（プロジェクト直下に出力）
    MODEL_PATH = "sklearn_mnist_model.pkl"  # 推論に使う学習済みモデルのファイルパス

# =============================================================================
# 画像前処理設定
# =============================================================================
class ImageProcessingConfig:
    """画像前処理のパラメータ設定"""
    
    # 枠削り設定
    CROP_RATIO = 0.03           # 画像の四辺から切り落とす比率（0.02=各辺2%を削る）
    
    # ノイズ除去設定
    MEDIAN_BLUR_KERNEL = 5        # 中央値フィルタのカーネルサイズ（大きいほど平滑化が強い）
    GAUSSIAN_BLUR_KERNEL = (5, 5) # ガウシアン平滑化のカーネル（奇数×奇数で指定）
    
    # 適応的二値化設定
    ADAPTIVE_BLOCK_SIZE_RATIO = 15 # 適応的二値化の近傍ウィンドウサイズを画像サイズから決める比率
    ADAPTIVE_C = 3                # 適応的二値化の閾値補正（大きいほど黒寄り、輪郭強調）
    
    # 画像サイズ設定
    TARGET_DIGIT_SIZE = 28        # 最終的に推論へ渡す数字画像の一辺サイズ（MNIST互換）(固定値)
    MIN_BLOCK_SIZE = 15           # 近傍ウィンドウの下限サイズ（過小分割の抑制）

# =============================================================================
# 数字分割設定
# =============================================================================
class DigitSegmentationConfig:
    """数字分割のパラメータ設定"""
    
    EXPECTED_DIGITS = 8           # 1枚あたり桁数（固定値）
    SEGMENTATION_METHOD = "adaptive"  # 分割方法（uniform=等幅、adaptive=空白/輪郭ベース）
    
    # 適応的分割のパラメータ（SEGMENTATION_METHOD = "adaptive"の場合）
    MIN_DIGIT_WIDTH = 40          # 1桁領域として許容する最小幅（小さすぎる誤検出を抑制）
    MAX_DIGIT_WIDTH = 100         # 1桁領域として許容する最大幅（広すぎる結合を抑制）
    WHITE_SPACE_THRESHOLD = 0.6  # 列方向の白画素率がこの値以下なら区切りとみなす（0〜1）

# =============================================================================
# モデル設定
# =============================================================================
class ModelConfig:
    """モデル関連の設定"""
    
    # モデルファイルパス
    MODEL_PATH = "sklearn_mnist_model.pkl"  # 学習時と互換のライブラリ版で保存されたモデル

# =============================================================================
# 数字認識設定
# =============================================================================
class DigitRecognitionConfig:
    """数字認識のパラメータ設定"""
    
    # 画像正規化設定
    NORMALIZE_IMAGE = True        # 画素値を0〜1へ正規化（学習前処理と揃える）
    AUTO_INVERT = True            # 背景が白っぽい場合に自動反転（黒背景・白文字へ）
    
    # 中心化設定
    CENTER_IMAGE = True           # 重心が中央に来るよう平行移動（位置ズレの影響を軽減）
    CENTER_METHOD = "centroid"    # 中心化方法（centroid=重心、geometric=幾何中心）
    
    # リサイズ設定
    RESIZE_METHOD = "inter_area"  # 縮小に強いinter_area／高品質はinter_cubic／汎用はinter_linear
    
    # 信頼度設定
    MIN_CONFIDENCE = 0.5          # 予測の最小信頼度（運用でフィルタリング指標に使用）
    ENABLE_CONFIDENCE_CHECK = True # 信頼度チェックを有効化（ログ・閾値判定に利用）

# =============================================================================
# デバッグ・ログ設定
# =============================================================================
class DebugConfig:
    """デバッグとログの設定"""
    
    ENABLE_DEBUG_OUTPUT = True    # デバッグ出力の大元スイッチ（画像保存や詳細ログを制御）
    SAVE_INTERMEDIATE_STEPS = True  # 各段階の中間画像を保存（原因調査・精度検証向け）
    VERBOSE_LOGGING = True        # 進捗や詳細情報をコンソールに出力
    
    # デバッグ画像の保存設定
    SAVE_ORIGINAL = True          # 元画像（分割前）を保存
    SAVE_INVERTED = True          # 反転後の画像（背景白→黒）を保存
    SAVE_SQUARE = True            # 正方形キャンバスへ整形後を保存
    SAVE_CENTERED = True          # 重心合わせ（中心化）後を保存
    SAVE_RESIZED = True           # 最終リサイズ（28×28）後を保存

# =============================================================================
# パフォーマンス設定
# =============================================================================
class PerformanceConfig:
    """パフォーマンス最適化の設定"""
    
    # 並列処理設定
    N_JOBS = -1                   # 並列処理数（-1=全CPU、1=単一、N=指定スレッド数）

# =============================================================================
# 出力設定
# =============================================================================
class OutputConfig:
    """出力形式の設定"""
    
    # CSV出力設定
    CSV_ENCODING = "utf-8-sig"    # CSVの文字コード（Excel互換のUTF-8 BOM付き）
    CSV_HEADER_JP = True          # 日本語ヘッダーを使用（英語にする場合はFalse）
    
    # 結果表示設定
    SHOW_CONFIDENCE = True        # 予測時の信頼度を表示
    SHOW_PROCESSING_TIME = True   # 各処理の所要時間を表示
    SHOW_PROGRESS = True          # 全体の進捗（件数など）を表示

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
