# 数字認識システム

このプロジェクトは、画像から8桁の数字を自動認識するシステムです。PyTorchのCNNモデルを使用し、データ拡張による多数決方式で高精度な数字認識を実現しています。

## 機能

* 画像の前処理（枠削り、ノイズ除去）
* 適応的二値化による文字抽出
* 8桁の数字の自動分割
* **PyTorch CNNモデルによる高精度数字認識**
* **データ拡張（角度・スケール変更）による多数決方式**
* **信頼度表示機能**
* 結果のCSV出力（日本語対応）

## セットアップ

### 必要な環境

* Python 3.8以上
* OpenCV
* NumPy
* **PyTorch (2.0以上)**
* **torchvision**
* **Pillow**

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

### 2. 実行

```bash
python main.py
```

### 3. 結果の確認

処理が完了すると、以下のファイルが生成されます：

* `result_pytorch.csv`: **PyTorchモデルの認識結果（信頼度付き）**
* `model_comparison_detailed.csv`: **詳細な比較結果**

### 4. 精度確認

```bash
python check_result.py compare
```

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

### PyTorchモデルの結果 (`result_pytorch.csv`)

```
ファイル名,認識結果,信頼度,個別予測,個別信頼度
image1,12345678,0.9234,"[1,1,1]|[2,2,2]|[3,3,3]|[4,4,4]|[5,5,5]|[6,6,6]|[7,7,7]|[8,8,8]","[0.95,0.92,0.91]|[0.89,0.87,0.85]|..."
image2,87654321,0.8756,"[8,8,8]|[7,7,7]|[6,6,6]|[5,5,5]|[4,4,4]|[3,3,3]|[2,2,2]|[1,1,1]","[0.93,0.91,0.88]|[0.87,0.85,0.82]|..."
...
```

### 詳細比較結果 (`model_comparison_detailed.csv`)

```
filename,accurate,pytorch_result,pytorch_matches
image1,12345678,12345678,8
image2,87654321,87654320,7
...
pytorch_accuracy,0.915
```

## ファイル構成

```
number-recognition-system/
├── main.py                    # メイン実行ファイル
├── number_reader.py           # PyTorch数字認識モジュール
├── pytorch_model.py           # PyTorch CNNモデル定義
├── image_processor.py         # 画像前処理（枠削り）
├── preprocessing.py           # 画像の二値化と分割処理
├── image_utils.py             # 画像処理ユーティリティ
├── config.py                  # 設定ファイル（パラメータ管理）
├── check_result.py            # 精度確認スクリプト
├── pytorch_digit_model.pth    # 学習済みPyTorchモデル
├── requirements.txt           # 依存関係
├── README.md                 # このファイル
├── input/                    # 入力画像フォルダ（50枚）
├── result_accurate.csv        # 正解データ
├── result_pytorch.csv         # PyTorch認識結果
└── model_comparison_detailed.csv # 詳細比較結果
```

## モジュール構成

### main.py
* 全体の処理フローを制御するハブ機能
* 各モジュールの順次実行

### image_processor.py
* 画像の前処理（枠削り）
* 日本語ファイル名対応

### preprocessing.py
* 画像の二値化処理
* 8桁の数字分割

### number_reader.py
* PyTorch学習済みモデルの読み込み
* 数字認識処理（多数決方式）
* 信頼度計算
* CSV結果出力（日本語対応）

### pytorch_model.py
* **PyTorch CNNモデルの定義**
* **データ拡張機能**
* **多数決による最終判定**

### check_result.py
* **精度確認スクリプト**
* **詳細比較機能**

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

# PyTorch数字認識のみ
python number_reader.py

# 精度確認のみ
python check_result.py compare
```

## パフォーマンス

### 認識精度

* **PyTorchモデル**: 91.5%（50枚のテストデータで検証）
* **多数決方式**: データ拡張による精度向上
* **信頼度表示**: 各予測の信頼度を数値で確認可能

### 処理速度

* **GPU最適化**: Apple Silicon MPS対応
* **軽量化モデル**: 効率的なCNNアーキテクチャ
* **バッチ処理**: 高速な画像処理

## 特徴

* **日本語ファイル名対応**: 日本語のファイル名でも正常に処理
* **高精度**: PyTorch CNNモデルによる91.5%の高精度な数字認識
* **多数決方式**: データ拡張による3枚の画像から多数決で最終判定
* **信頼度表示**: 各予測の信頼度を数値で表示
* **GPU最適化**: Apple Silicon MPS対応による高速処理
* **モジュラー設計**: 各機能が独立したモジュールとして分離
* **エラー耐性**: ファイル読み込みエラーに対する代替処理
* **詳細コメント**: 各モジュールの役割と処理内容を明確化

## PyTorchモデルの利点

* **CNNアーキテクチャ**: 畳み込みニューラルネットワークによる高精度認識
* **データ拡張**: 角度・スケール変更による汎化性能向上
* **多数決方式**: 3枚の拡張画像から多数決で最終判定
* **信頼度計算**: 各予測の信頼度を数値で表示
* **GPU対応**: Apple Silicon MPSによる高速処理
* **軽量化**: 効率的なモデルアーキテクチャ