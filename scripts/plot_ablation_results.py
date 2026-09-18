"""Plot validation ablation results from an explicit JSON manifest."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="artifacts/ablation_comparison.png")
    args = parser.parse_args()
    entries = json.loads(Path(args.manifest).read_text())
    labels, values, targets = [], [], []
    for entry in entries:
        result = json.loads((Path(entry["result_dir"]) / "results.json").read_text())
        metrics = result.get("best_val_metrics") or result.get("final_val_metrics") or {}
        labels.append(entry["label"])
        values.append(float(metrics["balanced_accuracy"]))
        targets.append(float(entry.get("minimum_balanced_accuracy", 0.0)))
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(labels, values, color="#4472C4")
    axis.plot(labels, targets, "o--", color="#C00000", label="minimum target")
    axis.axhline(0.5, color="#777777", linestyle=":", label="chance level")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Validation balanced accuracy")
    axis.set_title("Ear-SAAD supervised architecture ablation")
    axis.legend()
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
