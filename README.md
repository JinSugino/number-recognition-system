# 数字認識システム

このプロジェクトは、画像から8桁の数字を自動認識するシステムです。MNISTデータセットで学習したCNNモデルを使用して、画像内の数字を高精度で識別します。

## 機能

- 画像の前処理（枠削り、ノイズ除去）
- 適応的二値化による文字抽出
- 8桁の数字の自動分割
- CNNモデルによる数字認識
- 結果のCSV出力

## セットアップ

### 必要な環境

- Python 3.8以上
- OpenCV
- TensorFlow
- NumPy

### インストール

1. リポジトリをクローン
```bash
git clone <repository-url>
cd number
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
- 対応形式: PNG, JPG, JPEG
- 推奨: 8桁の数字が横一列に並んでいる画像

### 2. 実行

```bash
python main.py
```

### 3. 結果の確認

処理が完了すると、以下のファイル・フォルダが生成されます：

- `data/`: 前処理済み画像
- `debug/`: 各ステップのデバッグ画像
- `result.csv`: 認識結果
- `cnn_mnist_model.h5`: 学習済みモデル（初回実行時に生成）

## 処理フロー

1. **画像の前処理**
   - 画像の端を2%削除（画像端にある黒枠が原因でモデルが正しく読み取ってくれません）
   - ノイズ除去と平滑化（背景にフケのようなものが現れてしまうのを防ぎます）

2. **二値化と分割**
   - 適応的二値化で文字を抽出（画像の各点に対して近傍の平均値を超えていれば十分黒であると判定する）
   - 適切な区分を認識して文字数分の画像に分離（縦にみて黒成分が多ければ文字の中腹、少なければ字間として最適な7点を選ぶ）

3. **数字認識**
   - 各数字画像を28x28にリサイズ（モデルにフィットさせる）
   - CNNモデルで数字を予測
   - 結果をCSVに出力

## 出力形式

`result.csv` には以下の形式で結果が保存されます：

```csv
file_name,digits
image1,12345678
image2,87654321
...
```

## ファイル構成

```
number/
├── main.py              # メイン処理ファイル（全体の制御）
├── preprocessing.py     # 画像の二値化と分割処理
├── number_reader.py     # 数字認識とCSV出力処理
├── requirements.txt     # 依存関係
├── README.md           # このファイル
├── input/              # 入力画像フォルダ
├── data/               # 前処理済み画像
├── debug/              # デバッグ画像
├── result.csv          # 認識結果
└── cnn_mnist_model.h5  # 学習済みモデル
```

## モジュール構成

### main.py
- 全体の処理フローを制御
- 画像の前処理（枠削り）
- 各モジュールの順次実行

### preprocessing.py
- 画像の二値化処理
- 8桁の数字分割
- デバッグ画像の保存

### number_reader.py
- CNNモデルの作成・読み込み
- 数字認識処理
- CSV結果出力

## 個別実行

- 各モジュールは独立して実行することも可能

```bash
# 画像の前処理のみ
python preprocessing.py

# 数字認識のみ
python number_reader.py
```


