"""Train the fixed Gen5 model on development data and evaluate test once."""
import argparse
import copy
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cv_spectral_cnn import SpectralCNN, normalize_windows


def load_split(root, split):
    x = np.load(root / f"{split}_waveforms.npy").astype(np.float32)
    y = np.load(root / f"{split}_labels.npy").astype(np.int64)
    return normalize_windows(x), y


def load_subjects(root, split):
    path = root / f"{split}_subjects.npy"
    return np.load(path) if path.exists() else None


def collect(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(labels, probabilities, threshold=0.5):
    predictions = probabilities >= threshold
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "n_samples": int(labels.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/gen5_final")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.data)
    train_x, train_y = load_split(root, "train")
    val_x, val_y = load_split(root, "val")
    test_x, test_y = load_split(root, "test")
    test_subjects = load_subjects(root, "test")
    development_x = np.concatenate([train_x, val_x])
    development_y = np.concatenate([train_y, val_y])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralCNN().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(development_x), torch.from_numpy(development_y)), batch_size=args.batch_size, shuffle=True)
    counts = np.bincount(development_y, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch + 1:02d}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)

    # This is the only test evaluation in this final protocol.
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)), batch_size=args.batch_size)
    test_prob, test_labels = collect(model, test_loader, device)
    result = {
        "protocol": "fixed Gen5; train+val fit; one final test evaluation",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "test_labels_sha256": hashlib.sha256(test_y.tobytes()).hexdigest(),
        "normalization": "window_local_zscore_clip_8",
        "model": "Gen5 temporal CNN + relative spectral power",
        "test_metrics": metrics(test_labels, test_prob),
    }
    if test_subjects is not None:
        result["test_by_subject"] = {
            str(int(subject)): metrics(
                test_labels[test_subjects == subject],
                test_prob[test_subjects == subject],
            )
            for subject in np.unique(test_subjects)
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
