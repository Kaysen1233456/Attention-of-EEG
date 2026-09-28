"""Fine-tune Gen4 with consistent normalization and evaluation."""
import argparse
import copy
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from attention_model.data import EEGDataset
from attention_model.models import MiniNeurIPTClassifier
from attention_model.training.trainer import set_seed


CHANNEL_POSITIONS = [
    [0.0, 1.0, 0.3], [-0.4, 0.8, 0.5], [0.4, 0.8, 0.5],
    [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.3],
    [-0.4, -0.8, 0.5], [0.4, -0.8, 0.5],
]


def collect(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["waveform"].to(device))
            probabilities.append(torch.softmax(output["logits"], dim=1)[:, 1].cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/finetune_freq.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train statistics are shared by train, validation, and test inputs.
    dataset = EEGDataset(
        args.data,
        normalize=True,
        use_train_stats=True,
        normalization_mode="train_subjects_only",
        clip_std=8.0,
        include_test=True,
    )
    loaders = dataset.get_dataloaders(batch_size=args.batch_size, num_workers=0)

    model = MiniNeurIPTClassifier(
        d_model=48, n_heads=6, d_ff=384, n_layers=2,
        n_channels=8, n_classes=2,
        channel_positions=CHANNEL_POSITIONS,
        left_indices=[0, 1, 2, 3, 4], right_indices=[5, 6, 7],
        dropout=0.395, pretrained_path=args.pretrained,
        freeze_encoder=True, temporal_pool=4,
        iilp_pooling="attention", use_iilp_pooling=True,
        use_pmoe=True, classifier_type="swiglu",
        use_difference_feature=True, use_product_feature=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.23e-5, weight_decay=5.29e-6,
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.13, 2.55], device=device))

    best_val_ba = -np.inf
    best_epoch = -1
    best_state = None
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loaders["train"]:
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(waveform)["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_prob, val_labels = collect(model, loaders["val"], device)
        val_ba = balanced_accuracy_score(val_labels, val_prob >= 0.5)
        print(f"Epoch {epoch + 1:02d} | Val BA={val_ba:.4f} AUC={roc_auc_score(val_labels, val_prob):.4f}", flush=True)
        if val_ba > best_val_ba:
            best_val_ba = float(val_ba)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    if best_state is None:
        raise RuntimeError("No best validation checkpoint was selected")
    model.load_state_dict(best_state)
    val_prob, val_labels = collect(model, loaders["val"], device)
    test_prob, test_labels = collect(model, loaders["test"], device)

    # Threshold is selected from the restored best checkpoint on validation only.
    threshold, threshold_ba = 0.5, -np.inf
    for candidate in np.linspace(0.05, 0.95, 181):
        score = balanced_accuracy_score(val_labels, val_prob >= candidate)
        if score > threshold_ba:
            threshold, threshold_ba = float(candidate), float(score)

    def metrics(labels, probabilities):
        predictions = (probabilities >= threshold).astype(np.int64)
        return {
            "accuracy": float((predictions == labels).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "n_samples": int(labels.size),
        }

    result = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "normalization": "train_subjects_only",
        "checkpoint_selection": "validation_balanced_accuracy_at_0.5",
        "validation_threshold": threshold,
        "validation_threshold_balanced_accuracy": threshold_ba,
        "validation_metrics": metrics(val_labels, val_prob),
        "test_metrics": metrics(test_labels, test_prob),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
