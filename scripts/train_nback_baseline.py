import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("/mnt/workspace/Attention-of-EEG/data/processed/nback")
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, n_channels=19, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc1 = nn.Linear(128 * 16, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
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
    print(f"Train: {train_w.shape}, Val: {val_w.shape}, Test: {test_w.shape}")
    print(f"Device: {DEVICE}")
    train_loader = DataLoader(EEGDataset(train_w, train_l), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EEGDataset(val_w, val_l), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EEGDataset(test_w, test_l), batch_size=BATCH_SIZE)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "best_cnn.pt")
        else:
            patience_counter += 1
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if patience_counter >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break
    model.load_state_dict(torch.load(DATA_DIR / "best_cnn.pt"))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
