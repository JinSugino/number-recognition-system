# -*- coding: utf-8 -*-
"""
PyTorchモデル訓練スクリプト
=========================

MNISTデータセットを使用してPyTorch CNNモデルを訓練
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from pytorch_model import DigitCNN, DigitDataset, PyTorchDigitRecognizer
import os
import time

def train_model():
    """PyTorchモデルの訓練"""
    print("=== PyTorchモデル訓練開始 ===")
    
    # デバイス設定（MPS優先）
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"使用デバイス: {device}")
    
    # データ変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # MNISTデータセットの読み込み
    print("MNISTデータセットを読み込み中...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # データローダー（MPS使用時はnum_workers=0に設定、バッチサイズも調整）
    num_workers = 0 if device.type == 'mps' else 2
    batch_size = 128 if device.type == 'mps' else 64  # MPS使用時はバッチサイズを大きく
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"訓練データ: {len(train_dataset)}枚")
    print(f"テストデータ: {len(test_dataset)}枚")
    
    # モデル初期化
    model = DigitCNN().to(device)
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失関数と最適化（MPS使用時は学習率を上げて高速化）
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.002 if device.type == 'mps' else 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # より頻繁に学習率を下げる
    
    # 訓練ループ（MPS使用時はエポック数を減らして高速化）
    num_epochs = 15 if device.type == 'mps' else 30
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:  # より頻繁に進捗表示
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # テストフェーズ
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        # 記録
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # 最高精度のモデルを保存
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'train_accuracy': train_acc
            }, 'best_pytorch_model.pth')
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} (所要時間: {epoch_time:.1f}秒):')
        print(f'  訓練損失: {avg_train_loss:.4f}, 訓練精度: {train_acc:.2f}%')
        print(f'  テスト精度: {test_acc:.2f}%')
        print(f'  最高テスト精度: {best_test_acc:.2f}%')
        print(f'  累計時間: {total_time:.1f}秒')
        print('-' * 50)
    
    # 最終モデルを保存
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_acc,
        'train_accuracy': train_acc
    }, 'pytorch_digit_model.pth')
    
    total_training_time = time.time() - start_time
    print(f"\n訓練完了!")
    print(f"総訓練時間: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分)")
    print(f"最高テスト精度: {best_test_acc:.2f}%")
    print(f"最終テスト精度: {test_acc:.2f}%")
    print(f"モデルを保存しました: pytorch_digit_model.pth")
    
    # 学習曲線をプロット
    plot_training_curves(train_losses, train_accuracies, test_accuracies)
    
    return model, best_test_acc

def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    """学習曲線をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 損失曲線
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 精度曲線
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("学習曲線を training_curves.png に保存しました")

def evaluate_model(model_path='pytorch_digit_model.pth'):
    """訓練済みモデルの評価"""
    print(f"\n=== モデル評価: {model_path} ===")
    
    # デバイス設定（MPS優先）
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # モデル読み込み
    model = DigitCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # テストデータで評価
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # データローダー（MPS使用時はnum_workers=0に設定）
    num_workers = 0 if device.type == 'mps' else 2
    batch_size = 128 if device.type == 'mps' else 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # クラス別精度
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_accuracy = 100. * correct / total
    print(f"全体精度: {overall_accuracy:.2f}%")
    
    print("\nクラス別精度:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100. * class_correct[i] / class_total[i]
            print(f"数字 {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return overall_accuracy

if __name__ == "__main__":
    # モデル訓練
    model, best_acc = train_model()
    
    # モデル評価
    evaluate_model('pytorch_digit_model.pth')
    
    print("\n=== 訓練完了 ===")
    print("次のコマンドでPyTorchモデルを使用できます:")
    print("python main_pytorch.py")
