"""Shared Teacher B model and subject-group data utilities."""
import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cv_spectral_cnn import load_development, normalize_windows

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


class TeacherB(nn.Module):
    def __init__(self, d_model=96, n_heads=4, n_layers=2, dropout=0.2, n_channels=8, n_time=500):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_channels, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.GELU())
            for k in (7, 15, 31)
        ])
        self.temporal_projection = nn.Conv1d(96, d_model, 1)
        self.pool = nn.AvgPool1d(kernel_size=10, stride=10)
        self.position = nn.Parameter(torch.zeros(1, n_time // 10, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.spectral = nn.Sequential(
            nn.LayerNorm(n_channels * len(BANDS)),
            nn.Linear(n_channels * len(BANDS), 64), nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model + 64), nn.Linear(d_model + 64, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 2),
        )
        frequency = torch.fft.rfftfreq(n_time, d=1.0 / 250)
        self.register_buffer("total_mask", (frequency >= 1) & (frequency <= 45))
        self.register_buffer("band_masks", torch.stack([(frequency >= lo) & (frequency < hi) for lo, hi in BANDS]))

    def forward(self, x, return_features=False):
        temporal = torch.cat([branch(x) for branch in self.temporal_branches], dim=1)
        temporal = self.temporal_projection(temporal)
        temporal = self.pool(temporal).transpose(1, 2)
        temporal = temporal + self.position[:, :temporal.size(1)]
        temporal = self.transformer(temporal).mean(dim=1)
        power = torch.fft.rfft(x, dim=-1).abs().square()
        total = power[..., self.total_mask].mean(dim=-1, keepdim=True) + 1e-8
        bands = [power[..., mask].mean(dim=-1) / total.squeeze(-1) for mask in self.band_masks]
        spectral = self.spectral(torch.log(torch.cat(bands, dim=1) + 1e-8))
        features = torch.cat([temporal, spectral], dim=1)
        logits = self.classifier(features)
        return {"logits": logits, "features": features} if return_features else {"logits": logits}


def make_loaders(x, y, train_idx, val_idx, batch_size):
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])), batch_size=batch_size)
    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval(); probabilities, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probabilities.append(torch.softmax(model(x.to(device))["logits"], dim=1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    predictions = probabilities >= 0.5
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def train_fold(x, y, train_idx, val_idx, params, seed, epochs, device, return_model=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TeacherB(**params).to(device)
    train_loader, val_loader = make_loaders(x, y, train_idx, val_idx, params["batch_size"])
    counts = np.bincount(y[train_idx], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_state, best_score, stale = None, -np.inf, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device))["logits"], yb.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        metrics = evaluate(model, val_loader, device)
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= 4: break
    model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, device)
    return (metrics, model) if return_model else metrics
