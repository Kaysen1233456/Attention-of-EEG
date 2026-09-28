"""Stage A teacher: larger multi-scale CNN plus relative spectral branch."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cv_spectral_cnn import load_development, normalize_windows


BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


class TeacherA(nn.Module):
    def __init__(self, n_channels=8, sampling_rate=250, n_time=500):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_channels, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.GELU())
            for k in (7, 15, 31)
        ])
        self.temporal = nn.Sequential(
            nn.Conv1d(96, 128, 7, padding=3), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spectral = nn.Sequential(
            nn.LayerNorm(n_channels * len(BANDS)),
            nn.Linear(n_channels * len(BANDS), 64), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        frequency = torch.fft.rfftfreq(n_time, d=1.0 / sampling_rate)
        self.register_buffer("total_mask", (frequency >= 1) & (frequency <= 45))
        self.register_buffer("band_masks", torch.stack([(frequency >= lo) & (frequency < hi) for lo, hi in BANDS]))

    def forward(self, x):
        temporal = self.temporal(torch.cat([branch(x) for branch in self.branches], dim=1)).squeeze(-1)
        power = torch.fft.rfft(x, dim=-1).abs().square()
        total = power[..., self.total_mask].mean(dim=-1, keepdim=True) + 1e-8
        bands = [power[..., mask].mean(dim=-1) / total.squeeze(-1) for mask in self.band_masks]
        spectral = self.spectral(torch.log(torch.cat(bands, dim=1) + 1e-8))
        return self.classifier(torch.cat([temporal, spectral], dim=1))


def evaluate(model, loader, device):
    model.eval(); probs, labels = [], []
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
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/teacher_a_cv")
    args = parser.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    x, y, groups = load_development(Path(args.data)); x = normalize_windows(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(8).split(x, y, groups)):
        np.random.seed(args.seed + fold); torch.manual_seed(args.seed + fold)
        model = TeacherA().to(device)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])), batch_size=args.batch_size)
        counts = np.bincount(y[train_idx], minlength=2)
        weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=weights)
        best_state, best_score, stale = None, -np.inf, 0
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb.to(device)), yb.to(device))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            metrics = evaluate(model, val_loader, device)
            if metrics["balanced_accuracy"] > best_score:
                best_score = metrics["balanced_accuracy"]; best_state = copy.deepcopy(model.state_dict()); stale = 0
            else:
                stale += 1
                if stale >= 5: break
        model.load_state_dict(best_state)
        metrics = evaluate(model, val_loader, device)
        fold_dir = Path(args.output) / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "model.pt")
        rows.append({"fold": fold, "seed": args.seed, "validation_subjects": sorted(np.unique(groups[val_idx]).tolist()), **metrics})
        print(rows[-1], flush=True)
    keys = ("balanced_accuracy", "macro_f1", "roc_auc")
    result = {"stage": "A", "model": "TeacherA multi-scale CNN + relative spectral power", "folds": rows,
              "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
              "std": {key: float(np.std([row[key] for row in rows])) for key in keys}}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
