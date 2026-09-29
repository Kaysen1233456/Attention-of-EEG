"""Full 8-fold validation for a fixed Teacher B configuration."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from teacher_b_common import load_development, normalize_windows, train_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--params-json", help="best_params.json produced by the selection script")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    x, y, groups = load_development(Path(args.data)); x = normalize_windows(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {key: getattr(args, key.replace("-", "_")) for key in ("d_model", "n_heads", "n_layers", "dropout", "learning_rate", "weight_decay", "batch_size")}
    if args.params_json:
        loaded = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        missing = sorted(set(params) - set(loaded))
        if missing:
            raise ValueError(f"Missing parameters in {args.params_json}: {missing}")
        params = {key: loaded[key] for key in params}
    rows = []
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(8).split(x, y, groups)):
        metrics = train_fold(x, y, train_idx, val_idx, params, args.seed + fold, args.epochs, device)
        rows.append({"fold": fold, "validation_subjects": sorted(np.unique(groups[val_idx]).tolist()), **metrics})
        print(rows[-1], flush=True)
    keys = ("balanced_accuracy", "macro_f1", "roc_auc")
    result = {"stage": "B", "model": f"CNN + {params['n_layers']}-layer Transformer + relative spectral power", "params": params, "folds": rows,
              "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
              "std": {key: float(np.std([row[key] for row in rows])) for key in keys}}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
