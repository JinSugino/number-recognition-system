# 数字認識システム

このプロジェクトは、画像から8桁の数字を自動認識するシステムです。scikit-learnのRandom Forestモデルを使用して、画像内の数字を高精度で識別します。

## 機能

* 画像の前処理（枠削り、ノイズ除去）
* 適応的二値化による文字抽出
* 8桁の数字の自動分割
* Random Forestモデルによる数字認識
* 結果のCSV出力（日本語対応）

## セットアップ

### 必要な環境

* Python 3.8以上
* OpenCV
* scikit-learn
* NumPy

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

処理が完了すると、以下のファイル・フォルダが生成されます：

* `data/`: 前処理済み画像
* `debug/`: 各ステップのデバッグ画像
* `result.csv`: 認識結果（日本語ヘッダー）
* `sklearn_mnist_model.pkl`: 学習済みモデル（初回実行時に生成）

## 処理フロー

1. **画像の前処理**  
   * 画像の端を2%削除（画像端にある黒枠が原因でモデルが正しく読み取ってくれません）  
   * ノイズ除去と平滑化（背景にフケのようなものが現れてしまうのを防ぎます）
2. **二値化と分割**  
   * 適応的二値化で文字を抽出（画像の各点に対して近傍の平均値を超えていれば十分黒であると判定する）  
   * 適切な区分を認識して文字数分の画像に分離（縦にみて黒成分が多ければ文字の中腹、少なければ字間として最適な7点を選ぶ）
3. **数字認識**  
   * 各数字画像を28x28にリサイズ（モデルにフィットさせる）  
   * Random Forestモデルで数字を予測  
   * 結果をCSVに出力

## 出力形式

`result.csv` には以下の形式で結果が保存されます：

```
ファイル名,認識結果
image1,12345678
image2,87654321
...
```

## ファイル構成

```
number-recognition-system/
├── main.py                    # メインハブ（全体の制御）
├── config.py                  # 設定ファイル（パラメータ管理）
├── image_processor.py         # 画像前処理（枠削り）
├── preprocessing.py           # 画像の二値化と分割処理
├── number_reader.py           # 数字認識とCSV出力処理
├── image_utils.py             # 画像処理ユーティリティ
├── requirements.txt           # 依存関係
├── README.md                 # このファイル
├── input/                    # 入力画像フォルダ
├── data/                     # 前処理済み画像
├── debug/                    # デバッグ画像
├── result.csv                # 認識結果
└── sklearn_mnist_model.pkl   # 学習済みモデル
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
* デバッグ画像の保存

### number_reader.py
* 既存の学習済みモデルの読み込み
* 数字認識処理
* CSV結果出力（日本語対応）

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

# 数字認識のみ
python number_reader.py
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
* **高精度**: 学習済みモデルによる高精度な数字認識
* **軽量**: 既存の学習済みモデルを使用
* **モジュラー設計**: 各機能が独立したモジュールとして分離
* **エラー耐性**: ファイル読み込みエラーに対する代替処理
* **パラメータ調整**: 前処理パラメータの簡単な調整
* **詳細コメント**: 各モジュールの役割と処理内容を明確化