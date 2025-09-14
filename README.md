# 数字認識システム

このプロジェクトは、画像から8桁の数字を自動認識するシステムです。scikit-learnのRandom ForestモデルとPyTorchのCNNモデルの両方をサポートし、多数決方式による精度向上を実装しています。

## 機能

* 画像の前処理（枠削り、ノイズ除去）
* 適応的二値化による文字抽出
* 8桁の数字の自動分割
* **PyTorch CNNモデルによる高精度数字認識**
* **データ拡張（角度・スケール変更）による多数決方式**
* **scikit-learnモデルとの比較機能**
* 結果のCSV出力（日本語対応）

## セットアップ

### 必要な環境

* Python 3.8以上
* OpenCV
* scikit-learn
* NumPy
* **PyTorch (2.0以上)**
* **torchvision**
* **Pillow**
* **matplotlib**

### インストール

1. リポジトリをクローン

```bash
git clone <repository-url>
cd number-recognition-system
```

2. 仮想環境を作成（推奨）

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

3. 依存関係をインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 入力画像の準備

`input/` フォルダに認識したい数字画像を配置してください。

* 対応形式: PNG, JPG, JPEG
* 推奨: 8桁の数字が横一列に並んでいる画像
* 日本語ファイル名対応

### 2. 実行方法

#### 従来のscikit-learnモデルを使用する場合

```bash
python main.py
```

#### 新しいPyTorchモデルを使用する場合（推奨）

```bash
# 1. PyTorchモデルの訓練（初回のみ）
python train_pytorch_model.py

# 2. PyTorchモデルで数字認識
python main_pytorch.py
```

### 3. 結果の確認

処理が完了すると、以下のファイル・フォルダが生成されます：

* `data/`: 前処理済み画像
* `debug/`: 各ステップのデバッグ画像
* `result.csv`: scikit-learnモデルの認識結果
* `result_pytorch.csv`: **PyTorchモデルの認識結果（信頼度付き）**
* `model_comparison.csv`: **両モデルの比較結果**
* `pytorch_digit_model.pth`: **PyTorch学習済みモデル**
* `sklearn_mnist_model.pkl`: scikit-learn学習済みモデル

## 処理フロー

1. **画像の前処理**  
   * 画像の端を2%削除（画像端にある黒枠が原因でモデルが正しく読み取ってくれません）  
   * ノイズ除去と平滑化（背景にフケのようなものが現れてしまうのを防ぎます）
2. **二値化と分割**  
   * 適応的二値化で文字を抽出（画像の各点に対して近傍の平均値を超えていれば十分黒であると判定する）  
   * 適切な区分を認識して文字数分の画像に分離（縦にみて黒成分が多ければ文字の中腹、少なければ字間として最適な7点を選ぶ）
3. **数字認識**  
   * 各数字画像を28x28にリサイズ（モデルにフィットさせる）  
   * **PyTorch CNNモデルで数字を予測（多数決方式）**
   * **データ拡張：角度±15度、スケール0.8-1.2倍の3枚の画像を生成**
   * **3枚の予測結果から多数決で最終判定**
   * 結果をCSVに出力（信頼度付き）

## 出力形式

### scikit-learnモデルの結果 (`result.csv`)

```
ファイル名,認識結果
image1,12345678
image2,87654321
...
```

### PyTorchモデルの結果 (`result_pytorch.csv`)

```
ファイル名,認識結果,信頼度,個別予測,個別信頼度
image1,12345678,0.9234,"[1,1,1]|[2,2,2]|[3,3,3]|[4,4,4]|[5,5,5]|[6,6,6]|[7,7,7]|[8,8,8]","[0.95,0.92,0.91]|[0.89,0.87,0.85]|..."
image2,87654321,0.8756,"[8,8,8]|[7,7,7]|[6,6,6]|[5,5,5]|[4,4,4]|[3,3,3]|[2,2,2]|[1,1,1]","[0.93,0.91,0.88]|[0.87,0.85,0.82]|..."
...
```

### モデル比較結果 (`model_comparison.csv`)

```
ファイル名,PyTorch結果,scikit-learn結果,一致
image1,12345678,12345678,True
image2,87654321,87654320,False
...
```

## ファイル構成

```
number-recognition-system/
├── main.py                    # メインハブ（scikit-learn版）
├── main_pytorch.py           # メインハブ（PyTorch版）
├── config.py                  # 設定ファイル（パラメータ管理）
├── image_processor.py         # 画像前処理（枠削り）
├── preprocessing.py           # 画像の二値化と分割処理
├── number_reader.py           # 数字認識とCSV出力処理（scikit-learn版）
├── pytorch_number_reader.py   # 数字認識とCSV出力処理（PyTorch版）
├── pytorch_model.py           # PyTorch CNNモデル定義
├── train_pytorch_model.py     # PyTorchモデル訓練スクリプト
├── image_utils.py             # 画像処理ユーティリティ
├── requirements.txt           # 依存関係
├── README.md                 # このファイル
├── input/                    # 入力画像フォルダ
├── data/                     # 前処理済み画像
├── debug/                    # デバッグ画像
├── result.csv                # scikit-learn認識結果
├── result_pytorch.csv        # PyTorch認識結果
├── model_comparison.csv      # モデル比較結果
├── sklearn_mnist_model.pkl   # scikit-learn学習済みモデル
└── pytorch_digit_model.pth   # PyTorch学習済みモデル
```

## モジュール構成

### main.py / main_pytorch.py
* 全体の処理フローを制御するハブ機能
* 各モジュールの順次実行

### image_processor.py
* 画像の前処理（枠削り）
* 日本語ファイル名対応

### preprocessing.py
* 画像の二値化処理
* 8桁の数字分割
* デバッグ画像の保存

### number_reader.py / pytorch_number_reader.py
* 学習済みモデルの読み込み
* 数字認識処理
* CSV結果出力（日本語対応）
* **PyTorch版では多数決方式と信頼度計算を実装**

### pytorch_model.py
* **PyTorch CNNモデルの定義**
* **データ拡張機能**
* **多数決による最終判定**
* **MNISTデータセットでの訓練機能**

### train_pytorch_model.py
* **PyTorchモデルの訓練スクリプト**
* **学習曲線の可視化**
* **モデル評価機能**

### image_utils.py
* 画像読み込み・保存のユーティリティ
* 日本語ファイル名対応
* 画像前処理関数

## 個別実行

各モジュールは独立して実行することも可能です：

```bash
# 画像の前処理のみ
python image_processor.py

# 画像の二値化と分割のみ
python preprocessing.py

# scikit-learn数字認識のみ
python number_reader.py

# PyTorch数字認識のみ
python pytorch_number_reader.py

# PyTorchモデル訓練のみ
python train_pytorch_model.py
```

## パフォーマンス最適化

### 前処理パラメータの調整

OCR精度向上のため、`config.py`ファイルで以下のパラメータを調整できます：

- **画像前処理**: 枠削り比率、ノイズ除去強度
- **二値化**: 適応的ブロックサイズ、閾値定数
- **認識**: 信頼度閾値、画像正規化設定

### 推奨設定

高精度を目指す場合の推奨パラメータ：

```python
# 高精度設定
CROP_RATIO = 0.03
MEDIAN_BLUR_KERNEL = 5
ADAPTIVE_BLOCK_SIZE_RATIO = 15
ADAPTIVE_C = 3
MIN_CONFIDENCE = 0.5
```

## 特徴

* **日本語ファイル名対応**: 日本語のファイル名でも正常に処理
* **高精度**: PyTorch CNNモデルによる高精度な数字認識
* **多数決方式**: データ拡張による3枚の画像から多数決で最終判定
* **信頼度表示**: 各予測の信頼度を表示
* **モデル比較**: scikit-learnとPyTorchモデルの性能比較
* **モジュラー設計**: 各機能が独立したモジュールとして分離
* **エラー耐性**: ファイル読み込みエラーに対する代替処理
* **パラメータ調整**: 前処理パラメータの簡単な調整
* **詳細コメント**: 各モジュールの役割と処理内容を明確化

## PyTorchモデルの利点

* **CNNアーキテクチャ**: 畳み込みニューラルネットワークによる高精度認識
* **データ拡張**: 角度・スケール変更による汎化性能向上
* **多数決方式**: 3枚の拡張画像から多数決で最終判定
* **信頼度計算**: 各予測の信頼度を数値で表示
* **GPU対応**: CUDA対応GPUがあれば高速処理
* **学習曲線**: 訓練過程の可視化