"""Plot available validation-only training and ablation results."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_runs(root: Path):
    runs = []
    for result_dir in sorted(root.glob("B*")):
        history_path = result_dir / "history.json"
        results_path = result_dir / "results.json"
        if not history_path.exists() or not results_path.exists():
            continue
        history = json.loads(history_path.read_text())
        results = json.loads(results_path.read_text())
        metrics = results.get("best_val_metrics") or results.get("final_val_metrics") or {}
        runs.append({"name": result_dir.name, "history": history, "results": results, "metrics": metrics})
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/ablation")
    parser.add_argument("--output", default="artifacts/ablation_training_results.png")
    args = parser.parse_args()

    runs = load_runs(Path(args.root))
    if not runs:
        raise SystemExit(f"No completed runs found under {args.root}")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, run in enumerate(runs):
        name = run["name"]
        history = run["history"]
        epochs = range(1, len(history["val_loss"]) + 1)
        color = colors[index % len(colors)]
        best_epoch = run["results"].get("best_epoch")
        axes[0, 0].plot(epochs, history["train_loss"], "--", color=color, alpha=0.7, label=f"{name} train")
        axes[0, 0].plot(epochs, history["val_loss"], color=color, label=f"{name} val")
        axes[0, 1].plot(epochs, history["val_balanced_acc"], color=color, label=name)
        axes[1, 0].plot(epochs, history["val_macro_f1"], color=color, label=name)
        if best_epoch:
            best_metrics = run["metrics"]
            axes[0, 1].scatter([best_epoch], [best_metrics["balanced_accuracy"]], color=color, s=45)
            axes[1, 0].scatter([best_epoch], [best_metrics["macro_f1"]], color=color, s=45)

    labels = [run["name"] for run in runs]
    metric_names = [("balanced_accuracy", "BA"), ("macro_f1", "Macro-F1"), ("roc_auc", "ROC-AUC")]
    x = list(range(len(labels)))
    width = 0.24
    for offset, (key, label) in enumerate(metric_names):
        values = [run["metrics"].get(key, 0.0) for run in runs]
        axes[1, 1].bar([v + (offset - 1) * width for v in x], values, width, label=label)
        for position, value in zip([v + (offset - 1) * width for v in x], values):
            axes[1, 1].text(position, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)

    axes[0, 0].set_title("Train / validation loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("Validation balanced accuracy")
    axes[0, 1].set_ylabel("Balanced accuracy")
    axes[0, 1].axhline(0.5, color="#888888", linestyle=":")
    axes[1, 0].set_title("Validation macro-F1")
    axes[1, 0].set_ylabel("Macro-F1")
    axes[1, 1].set_title("Best validation metrics")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_xticks(x, labels)
    for axis in axes.flat:
        axis.set_xlabel("Epoch" if axis is not axes[1, 1] else "Variant")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Ear-SAAD training results (validation only)")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
