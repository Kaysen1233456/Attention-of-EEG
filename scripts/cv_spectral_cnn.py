"""Subject-disjoint CV for a controlled temporal-CNN + spectral branch model."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def load_development(root):
    x = np.concatenate([np.load(root / f"{s}_waveforms.npy") for s in ("train", "val")])
    y = np.concatenate([np.load(root / f"{s}_labels.npy") for s in ("train", "val")])
    starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    runs = list(zip(starts[:-1], starts[1:]))
    if len(runs) % 2:
        raise ValueError("Expected paired rest/focus runs per subject")
    groups = np.empty(len(y), dtype=np.int64)
    for subject, index in enumerate(range(0, len(runs), 2)):
        groups[runs[index][0]:runs[index + 1][1]] = subject
    return x.astype(np.float32), y.astype(np.int64), groups


def normalize_windows(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


class SpectralCNN(nn.Module):
    def __init__(self, n_channels=8, sampling_rate=250, n_time=500):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_time = n_time
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spectral = nn.Sequential(
            nn.LayerNorm(n_channels * len(BANDS)),
            nn.Linear(n_channels * len(BANDS), 32), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )
        frequency = torch.fft.rfftfreq(n_time, d=1.0 / sampling_rate)
        masks = []
        total = (frequency >= 1) & (frequency <= 45)
        for low, high in BANDS:
            masks.append((frequency >= low) & (frequency < high))
        self.register_buffer("total_mask", total)
        self.register_buffer("band_masks", torch.stack(masks))

    def forward(self, x):
        temporal = self.temporal(x).squeeze(-1)
        spectrum = torch.fft.rfft(x, dim=-1).abs().square()
        total = spectrum[..., self.total_mask].mean(dim=-1, keepdim=True) + 1e-8
        bands = [spectrum[..., mask].mean(dim=-1) / total.squeeze(-1) for mask in self.band_masks]
        spectral = self.spectral(torch.log(torch.cat(bands, dim=1) + 1e-8))
        return self.classifier(torch.cat([temporal, spectral], dim=1))


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    pred = probs >= 0.5
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="artifacts/spectral_cnn_cv.json")
    args = parser.parse_args()
    torch.manual_seed(42)
    x, y, groups = load_development(Path(args.data))
    x = normalize_windows(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(args.splits).split(x, y, groups)):
        model = SpectralCNN().to(device)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])), batch_size=args.batch_size)
        counts = np.bincount(y[train_idx], minlength=2)
        weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=weights)
        best_state, best_score = None, -np.inf
        stale = 0
        for _ in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb.to(device)), yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            metrics = evaluate(model, val_loader, device)
            if metrics["balanced_accuracy"] > best_score:
                best_score = metrics["balanced_accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= 4:
                    break
        model.load_state_dict(best_state)
        metrics = evaluate(model, val_loader, device)
        rows.append({"fold": fold, "validation_subjects": sorted(np.unique(groups[val_idx]).tolist()), **metrics})
        print(rows[-1], flush=True)
    keys = ("balanced_accuracy", "macro_f1", "roc_auc")
    result = {
        "protocol": "GroupKFold",
        "model": "window-zscore temporal CNN + relative spectral power",
        "folds": rows,
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "std": {key: float(np.std([row[key] for row in rows])) for key in keys},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
