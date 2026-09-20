import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

DATA_DIR = Path("/mnt/workspace/Attention-of-EEG/data/processed/nback")
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 5e-2
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGDataset(Dataset):
    def __init__(self, windows, labels, augment=False):
        self.windows = windows  # (N, 19, 500)
        self.labels = labels
        self.augment = augment
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        x = self.windows[idx].copy()
        y = self.labels[idx]
        if self.augment:
            # 时间窗口随机偏移±10个采样点
            shift = np.random.randint(-10, 11)
            x = np.roll(x, shift, axis=1)
            # 加高斯噪声
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.FloatTensor(x), torch.LongTensor([y])[0]

class SmallCNN(nn.Module):
    def __init__(self, n_channels=19, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=15, stride=5, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=3, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(32 * 8, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total

def main():
    train_w = np.load(DATA_DIR / "train_windows.npy")
    train_l = np.load(DATA_DIR / "train_labels.npy")
    val_w = np.load(DATA_DIR / "val_windows.npy")
    val_l = np.load(DATA_DIR / "val_labels.npy")
    test_w = np.load(DATA_DIR / "test_windows.npy")
    test_l = np.load(DATA_DIR / "test_labels.npy")

    # 按通道做z-score归一化（用训练集统计量）
    mean = train_w.mean(axis=(0, 2), keepdims=True)  # (1, 19, 1)
    std = train_w.std(axis=(0, 2), keepdims=True) + 1e-8
    train_w = (train_w - mean) / std
    val_w = (val_w - mean) / std
    test_w = (test_w - mean) / std

    print(f"Train: {train_w.shape}, Val: {val_w.shape}, Test: {test_w.shape}")
    print(f"Device: {DEVICE}")
    print(f"Train mean: {train_w.mean():.4f}, std: {train_w.std():.4f}")

    train_loader = DataLoader(EEGDataset(train_w, train_l, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EEGDataset(val_w, val_l), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EEGDataset(test_w, test_l), batch_size=BATCH_SIZE)

    model = SmallCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0
    patience_counter = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "best_cnn_v2.pt")
        else:
            patience_counter += 1
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if patience_counter >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(DATA_DIR / "best_cnn_v2.pt"))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
