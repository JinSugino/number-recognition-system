import os
import csv
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

DEBUG_DIR = "debug"
OUTPUT_CSV = "result.csv"
MODEL_PATH = "cnn_mnist_model.h5"

def setup_model():
    """モデルの作成・読み込み"""
    if not os.path.exists(MODEL_PATH):
        print("モデルを作成中...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
        model.save(MODEL_PATH)
        print("モデル作成完了")
    else:
        print("既存のモデルを読み込み中...")
        model = load_model(MODEL_PATH)
    return model

def predict_digit(image_path, model):
    """画像ファイルから数字を予測"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "?"

    # 保存フォルダ（元画像と同じディレクトリ）
    save_dir = os.path.dirname(image_path)

    # 01: 元画像保存
    cv2.imwrite(os.path.join(save_dir, "step_01_original.png"), img)

    # 反転の自動判定（背景が白なら反転）
    if np.mean(img) > 127:
        img = 255 - img
    cv2.imwrite(os.path.join(save_dir, "step_02_inverted.png"), img)

    img = img.astype("float32") / 255.0  # 正規化（0〜1）

    # 画像サイズを正方形に調整（必要なら）
    h, w = img.shape
    size = max(h, w)
    square_img = np.zeros((size, size), dtype=np.float32)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
    cv2.imwrite(os.path.join(save_dir, "step_03_square.png"), (square_img * 255).astype(np.uint8))

    # 中心化（重心で移動）
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
    cv2.imwrite(os.path.join(save_dir, "step_04_centered.png"), (centered * 255).astype(np.uint8))

    # リサイズ to 28x28
    resized = cv2.resize(centered, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(save_dir, "step_05_resized.png"), (resized * 255).astype(np.uint8))

    # 推論用形状に変換
    input_img = np.expand_dims(resized, axis=(0, -1))  # (1,28,28,1)

    pred = model.predict(input_img, verbose=0)
    digit = np.argmax(pred)
    confidence = np.max(pred)
    return str(digit)

def main():
    """数字認識とCSV出力のメイン関数"""
    # モデルのセットアップ
    model = setup_model()
    
    # CSV出力
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "digits"])

        for folder_name in os.listdir(DEBUG_DIR):
            folder_path = os.path.join(DEBUG_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue

            digits = []
            for i in range(8):
                digit_path = os.path.join(folder_path, f"{folder_name}_digit_{i}.png")
                digits.append(predict_digit(digit_path, model))

            writer.writerow([folder_name, "".join(digits)])
            print(f"{folder_name}: {''.join(digits)}")

if __name__ == "__main__":
    main()
