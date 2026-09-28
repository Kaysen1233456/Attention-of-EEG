"""Select the best Teacher B configuration across completed search methods."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", help="Search summary.json files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = []
    for source in args.summaries:
        path = Path(source)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("test_accessed") is not False:
            raise ValueError(f"{path} does not prove test_accessed=false")
        candidates.append({
            "source": str(path), "method": result["method"],
            "objective": float(result["best_objective"]),
            "params": result["best_params"], "metrics": result["best_metrics"],
        })
    best = max(candidates, key=lambda item: item["objective"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "best_params.json").write_text(json.dumps(best["params"], indent=2), encoding="utf-8")
    selection = {"criterion": "max(mean_BA - 0.25 * std_BA)", "selected": best, "candidates": candidates}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
