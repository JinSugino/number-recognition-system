# -*- coding: utf-8 -*-
"""
PyTorch CNNモデル
================

数字認識用のCNNモデルとデータ拡張、多数決機能を実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import pickle
from PIL import Image
import random
from collections import Counter
from config import DirectoryConfig, DigitRecognitionConfig

class DigitCNN(nn.Module):
    """数字認識用CNNモデル（軽量版）"""
    
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        
        # 畳み込み層（軽量化）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # プーリング層
        self.pool = nn.MaxPool2d(2, 2)
        
        # ドロップアウト
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # 全結合層（軽量化）
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 畳み込み層1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # 畳み込み層2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # 畳み込み層3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # フラット化
        x = x.view(-1, 64 * 3 * 3)
        
        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class DigitDataset(Dataset):
    """数字画像データセット"""
    
    def __init__(self, image_paths, labels=None, transform=None, is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # 画像読み込み
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # 画像が読み込めない場合は黒画像を返す
            img = np.zeros((28, 28), dtype=np.uint8)
        
        # 前処理
        img = self.preprocess_image(img)
        
        # PIL Imageに変換
        img = Image.fromarray(img)
        
        # 変換適用
        if self.transform:
            img = self.transform(img)
        
        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img
    
    def preprocess_image(self, img):
        """画像の前処理"""
        # 反転の自動判定
        if np.mean(img) > 127:
            img = 255 - img
        
        # 正規化
        img = img.astype(np.float32) / 255.0
        
        # 正方形に調整
        h, w = img.shape
        size = max(h, w)
        square_img = np.zeros((size, size), dtype=np.float32)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
        
        # 中心化
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
        
        # リサイズ
        resized = cv2.resize(centered, (28, 28), interpolation=cv2.INTER_AREA)
        
        return (resized * 255).astype(np.uint8)

class DataAugmentation:
    """データ拡張クラス"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),  # ±15度の回転
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # スケール変更
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 正規化
        ])
    
    def augment_image(self, image_path):
        """1枚の画像から3枚の拡張画像を生成"""
        augmented_images = []
        
        # 元画像を読み込み
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return [np.zeros((28, 28), dtype=np.uint8) for _ in range(3)]
        
        # 前処理
        img = self.preprocess_image(img)
        img_pil = Image.fromarray(img)
        
        # 3回の拡張を適用
        for _ in range(3):
            augmented = self.transform(img_pil)
            augmented_images.append(augmented)
        
        return augmented_images
    
    def preprocess_image(self, img):
        """画像の前処理（DigitDatasetと同じ）"""
        if np.mean(img) > 127:
            img = 255 - img
        
        img = img.astype(np.float32) / 255.0
        
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
        
        return (resized * 255).astype(np.uint8)

class MajorityVoting:
    """多数決による最終判定"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.augmentation = DataAugmentation()
    
    def predict_with_voting(self, image_path):
        """多数決による予測"""
        # 3枚の拡張画像を生成
        augmented_images = self.augmentation.augment_image(image_path)
        
        predictions = []
        confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for img_tensor in augmented_images:
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predictions.append(predicted.item())
                confidences.append(confidence.item())
        
        # 多数決
        prediction_counts = Counter(predictions)
        most_common = prediction_counts.most_common(1)[0]
        
        # 信頼度は平均を取る
        avg_confidence = np.mean(confidences)
        
        return most_common[0], avg_confidence, predictions, confidences

class PyTorchDigitRecognizer:
    """PyTorchベースの数字認識器"""
    
    def __init__(self, model_path=None):
        # MPS (Apple Silicon GPU) を優先的に使用
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = DigitCNN().to(self.device)
        self.voting = MajorityVoting(self.model, self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train_model(self, train_data, val_data=None, epochs=50, batch_size=32, learning_rate=0.001):
        """モデルの訓練"""
        print(f"デバイス: {self.device}")
        
        # データローダー（MPS使用時はnum_workers=0に設定）
        num_workers = 0 if self.device.type == 'mps' else 2
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_data else None
        
        # 最適化と損失関数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # 訓練
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # 検証
            val_acc = 0.0
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                val_acc = 100. * val_correct / val_total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model("best_pytorch_model.pth")
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        print(f'最高検証精度: {best_val_acc:.2f}%')
    
    def save_model(self, model_path):
        """モデルを保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': self.model.__class__.__name__
        }, model_path)
        print(f"モデルを保存しました: {model_path}")
    
    def load_model(self, model_path):
        """モデルを読み込み"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"モデルを読み込みました: {model_path}")
    
    def predict_digit(self, image_path):
        """数字を予測（多数決方式）"""
        digit, confidence, predictions, confidences = self.voting.predict_with_voting(image_path)
        return str(digit), confidence, predictions, confidences

def create_mnist_dataset():
    """MNISTデータセットを作成（訓練用）"""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # MNISTデータセットをダウンロード
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    # テスト用のコード
    recognizer = PyTorchDigitRecognizer()
    
    # MNISTデータセットで訓練
    train_data, val_data = create_mnist_dataset()
    recognizer.train_model(train_data, val_data, epochs=10)
    
    # モデル保存
    recognizer.save_model("pytorch_digit_model.pth")
