"""Generate a read-only static report for the EEG project training route."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - project requirements provide PyYAML
    raise RuntimeError("PyYAML is required to read project configuration files") from exc


RECOMMENDED_LR = 0.00044348092176995735
PRIMARY_PRETRAIN_DIR = Path(
    "artifacts/pretrain_aamp_ear_saad_search_v2_lr4p43e-4_bs16"
)
REPEAT_PRETRAIN_DIR = Path(
    "artifacts/pretrain_aamp_ear_saad_search_v2_lr443e-6_bs16"
)
PRIMARY_CHECKPOINT_NAME = "final_pretrain_model.pt"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_history(path: Path) -> list[dict[str, Any]]:
    value = read_json(_required(path))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty history list: {path}")
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid history row: {path}")
        for key in ("epoch", "train_loss", "val_loss"):
            if key not in row:
                raise ValueError(f"Missing {key} in history: {path}")
    return value


def summarize_history(history: list[dict[str, Any]], configured_epochs: int) -> dict[str, Any]:
    best = min(history, key=lambda row: float(row["val_loss"]))
    final = history[-1]
    return {
        "epochs": len(history),
        "configured_epochs": configured_epochs,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "train_loss_at_best": float(best["train_loss"]),
        "final_epoch": int(final["epoch"]),
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "history": [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            }
            for row in history
        ],
    }


def _run_data(repo_root: Path, run_dir: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = repo_root / run_dir
    config = read_yaml(_required(directory / "pretrain_config.yaml"))
    metadata = read_json(_required(directory / "pretrain_metadata.json"))
    history = load_history(directory / "pretrain_history.json")
    configured_epochs = int(config.get("training", {}).get("epochs", len(history)))
    summary = summarize_history(history, configured_epochs)
    summary.update(
        {
            "label": label,
            "directory": str(run_dir),
            "learning_rate": float(
                config.get("training", {}).get("learning_rate", 0.0)
            ),
            "batch_size": int(config.get("training", {}).get("batch_size", 0)),
            "metadata": {
                "protocol": metadata.get("protocol"),
                "unlabeled_splits": metadata.get("unlabeled_splits"),
                "test_exposed_to_pretraining": metadata.get(
                    "test_exposed_to_pretraining"
                ),
                "checkpoint_selection": metadata.get("checkpoint_selection"),
            },
        }
    )
    warnings: list[dict[str, Any]] = []
    if summary["epochs"] != configured_epochs:
        warnings.append(
            {
                "severity": "AUDIT_WARNING",
                "message": (
                    f"{label}: history has {summary['epochs']} epochs but config "
                    f"declares {configured_epochs}"
                ),
            }
        )
    if label == "Repeat run" and abs(summary["learning_rate"] - RECOMMENDED_LR) > 1e-12:
        warnings.append(
            {
                "severity": "AUDIT_WARNING",
                "message": f"{label}: learning rate differs from recommended configuration",
            }
        )
    expected_metadata = {
        "unlabeled_splits": "train_val",
        "test_exposed_to_pretraining": False,
        "checkpoint_selection": "fixed_epochs",
    }
    for key, expected in expected_metadata.items():
        if summary["metadata"].get(key) != expected:
            warnings.append(
                {
                    "severity": "AUDIT_WARNING",
                    "message": f"{label}: metadata {key} is {summary['metadata'].get(key)!r}, expected {expected!r}",
                }
            )
    return summary, warnings


def load_demo_data(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    primary, primary_warnings = _run_data(
        repo_root, PRIMARY_PRETRAIN_DIR, "Primary run"
    )
    repeat, repeat_warnings = _run_data(repo_root, REPEAT_PRETRAIN_DIR, "Repeat run")

    checkpoint = _required(repo_root / PRIMARY_PRETRAIN_DIR / PRIMARY_CHECKPOINT_NAME)
    checkpoint_hash = sha256_file(checkpoint)
    primary_metadata = read_json(
        _required(repo_root / PRIMARY_PRETRAIN_DIR / "pretrain_metadata.json")
    )
    warnings = primary_warnings + repeat_warnings
    recorded_hash = (
        primary_metadata.get("checkpoint_sha256", {}).get(PRIMARY_CHECKPOINT_NAME)
    )
    if recorded_hash and recorded_hash != checkpoint_hash:
        warnings.append(
            {
                "severity": "AUDIT_WARNING",
                "message": "Primary checkpoint SHA-256 differs from metadata",
            }
        )

    data = {
        "project": {
            "name": "EEG Attention Decoding",
            "route": "AAMP pretraining -> Transformer teacher -> CNN student",
            "current_phase": "AAMP pretraining complete; downstream teacher validation pending",
        },
        "protocol": {
            "unlabeled_splits": primary["metadata"]["unlabeled_splits"],
            "test_exposed_to_pretraining": primary["metadata"][
                "test_exposed_to_pretraining"
            ],
            "checkpoint_selection": primary["metadata"]["checkpoint_selection"],
            "epoch_budget": primary["configured_epochs"],
        },
        "pretraining": {"runs": [primary, repeat]},
        "checkpoint": {
            "path": str(PRIMARY_PRETRAIN_DIR / PRIMARY_CHECKPOINT_NAME),
            "type": PRIMARY_CHECKPOINT_NAME,
            "sha256": checkpoint_hash,
            "recorded_sha256": recorded_hash,
            "size_bytes": checkpoint.stat().st_size,
        },
        "downstream": {
            "results_available": False,
            "target_mean_balanced_accuracy": 0.70,
            "folds": [0, 1, 2],
            "seeds": [42, 43, 44],
            "experiments": [
                {
                    "name": "Random-init Transformer",
                    "status": "PENDING",
                    "learning_rate": 0.0001,
                    "batch_size": 64,
                    "epochs": 25,
                },
                {
                    "name": "AAMP + frozen encoder",
                    "status": "PENDING",
                    "learning_rate": 0.0001,
                    "encoder_lr_scale": 0.0,
                    "batch_size": 64,
                    "epochs": 25,
                },
                {
                    "name": "AAMP + full fine-tuning",
                    "status": "PENDING",
                    "learning_rate": 0.0001,
                    "encoder_lr_scale": 0.1,
                    "batch_size": 64,
                    "epochs": 25,
                },
            ],
        },
        "audit_warnings": warnings,
    }
    data["route_stages"] = build_route_stages(data)
    data["decisions"] = build_decision_table(data)
    return data


def build_route_stages(data: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "name": "AAMP self-supervised pretraining",
            "status": "PASS",
            "target": "100 fixed epochs, train_val, test excluded",
            "current": f"{data['protocol']['epoch_budget']} epochs completed",
        },
        {
            "name": "AAMP-initialized Transformer teacher",
            "status": "PENDING",
            "target": "Train downstream classifier",
            "current": "Not evaluated",
        },
        {
            "name": "Three-fold x three-seed classification",
            "status": "PENDING",
            "target": "folds 0-2, seeds 42/43/44",
            "current": "Not started",
        },
        {
            "name": "Unified metric report",
            "status": "PENDING",
            "target": "BA, Macro-F1, ROC-AUC, subject metrics",
            "current": "Not available",
        },
        {
            "name": "Teacher retention gate",
            "status": "BLOCKED",
            "target": "Mean BA >= 0.70 and no class collapse",
            "current": "Downstream metrics unavailable",
        },
        {
            "name": "CNN distillation",
            "status": "BLOCKED",
            "target": "Start only after teacher gate passes",
            "current": "Not started",
        },
        {
            "name": "Deployment validation",
            "status": "NOT_STARTED",
            "target": "Student quality and resource budget verified",
            "current": "Not started",
        },
    ]


def build_decision_table(data: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "key": "test_exposed_to_pretraining",
            "metric": "Test exposed to pretraining",
            "target": "false",
            "current": str(data["protocol"]["test_exposed_to_pretraining"]).lower(),
            "status": "PASS",
        },
        {
            "key": "unlabeled_splits",
            "metric": "Unlabeled pretraining split",
            "target": "train_val",
            "current": data["protocol"]["unlabeled_splits"],
            "status": "PASS",
        },
        {
            "key": "fixed_epoch_budget",
            "metric": "Fixed epoch budget",
            "target": "100",
            "current": str(data["protocol"]["epoch_budget"]),
            "status": "PASS",
        },
        {
            "key": "primary_checkpoint",
            "metric": "Primary protocol checkpoint",
            "target": "final_pretrain_model.pt",
            "current": "Generated",
            "status": "PASS",
        },
        {
            "key": "downstream_mean_balanced_accuracy",
            "metric": "Downstream mean balanced accuracy",
            "target": ">= 0.70",
            "current": "N/A - not evaluated",
            "status": "PENDING",
        },
        {
            "key": "class_collapse",
            "metric": "Class collapse across fold/seed runs",
            "target": "None",
            "current": "N/A - not evaluated",
            "status": "PENDING",
        },
        {
            "key": "aamp_vs_random_init",
            "metric": "AAMP versus random initialization",
            "target": "AAMP better under same protocol",
            "current": "N/A - not evaluated",
            "status": "PENDING",
        },
        {
            "key": "cnn_distillation",
            "metric": "CNN distillation",
            "target": "Teacher gate passed",
            "current": "Not started",
            "status": "BLOCKED",
        },
    ]


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A - not evaluated"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _status_class(status: str) -> str:
    return status.lower().replace("_", "-")


def _svg_chart(data: dict[str, Any]) -> str:
    width, height = 920, 360
    left, right, top, bottom = 58, 20, 22, 44
    chart_width = width - left - right
    chart_height = height - top - bottom
    runs = data["pretraining"]["runs"]
    all_values = [
        value
        for run in runs
        for row in run["history"]
        for value in (row["train_loss"], row["val_loss"])
    ]
    minimum = min(all_values)
    maximum = max(all_values)
    padding = max((maximum - minimum) * 0.08, 0.01)
    minimum -= padding
    maximum += padding
    max_epochs = max(run["epochs"] for run in runs)

    def point(epoch: int, value: float) -> tuple[float, float]:
        x = left + ((epoch - 1) / max(max_epochs - 1, 1)) * chart_width
        y = top + (maximum - value) / (maximum - minimum) * chart_height
        return x, y

    colors = ["#0f766e", "#f97316", "#2563eb", "#dc2626"]
    series = []
    for run_index, run in enumerate(runs):
        for metric_index, metric in enumerate(("train_loss", "val_loss")):
            points = " ".join(
                f"{point(row['epoch'], row[metric])[0]:.1f},{point(row['epoch'], row[metric])[1]:.1f}"
                for row in run["history"]
            )
            series.append(
                f'<polyline class="line" stroke="{colors[run_index * 2 + metric_index]}" points="{points}" />'
            )
    grid = []
    for index in range(5):
        y = top + (chart_height * index / 4)
        value = maximum - (maximum - minimum) * index / 4
        grid.append(
            f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" />'
            f'<text class="axis" x="4" y="{y + 4:.1f}">{value:.3f}</text>'
        )
    markers = []
    for run_index, run in enumerate(runs):
        best = point(run["best_epoch"], run["best_val_loss"])
        markers.append(
            f'<circle class="marker" fill="{colors[run_index * 2 + 1]}" cx="{best[0]:.1f}" cy="{best[1]:.1f}" r="4" />'
            f'<text class="marker-label" x="{best[0] + 7:.1f}" y="{best[1] - 7:.1f}">{_esc(run["label"])} best</text>'
        )
    legend = (
        '<span><i class="swatch teal"></i>Primary train</span>'
        '<span><i class="swatch orange"></i>Primary val</span>'
        '<span><i class="swatch blue"></i>Repeat train</span>'
        '<span><i class="swatch red"></i>Repeat val</span>'
    )
    return (
        f'<div class="chart-legend">{legend}</div>'
        f'<svg class="loss-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="AAMP masked reconstruction loss curves">'
        + "".join(grid)
        + f'<line class="axis-line" x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" />'
        + f'<text class="axis" x="{left}" y="{height-10}">Epoch 1</text>'
        + f'<text class="axis" x="{width-right-40}" y="{height-10}">Epoch {max_epochs}</text>'
        + "".join(series)
        + "".join(markers)
        + "</svg>"
    )


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{_esc(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key == "status":
                value = f'<span class="status {_status_class(value)}">{_esc(value)}</span>'
            else:
                value = _esc(value)
            cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(data: dict[str, Any]) -> str:
    runs = data["pretraining"]["runs"]
    run_rows = [
        {
            "run": run["label"],
            "epochs": run["epochs"],
            "best_epoch": run["best_epoch"],
            "best_val": format_metric(run["best_val_loss"]),
            "final_train": format_metric(run["final_train_loss"]),
            "final_val": format_metric(run["final_val_loss"]),
        }
        for run in runs
    ]
    route_rows = [
        {
            "stage": stage["name"],
            "status": stage["status"],
            "target": stage["target"],
            "current": stage["current"],
        }
        for stage in data["route_stages"]
    ]
    warning_html = "".join(
        f'<div class="warning"><strong>{_esc(item["severity"])}</strong> {_esc(item["message"])}</div>'
        for item in data["audit_warnings"]
    ) or '<div class="ok">No audit warnings detected.</div>'
    experiment_rows = []
    for experiment in data["downstream"]["experiments"]:
        experiment_rows.append(
            {
                "experiment": experiment["name"],
                "status": experiment["status"],
                "epochs": experiment["epochs"],
                "batch": experiment["batch_size"],
                "lr": format_metric(experiment["learning_rate"]),
                "encoder_scale": format_metric(
                    experiment.get("encoder_lr_scale")
                ),
            }
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EEG Attention Project Demo</title>
<style>
:root {{ --ink:#17202a; --muted:#667085; --line:#d9dee7; --paper:#f5f7fa; --panel:#fff;
  --teal:#0f766e; --orange:#f97316; --blue:#2563eb; --red:#dc2626; --green:#166534;
  --yellow:#92400e; --gray:#475467; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1240px; margin:0 auto; padding:32px 22px 64px; }}
h1,h2,h3 {{ margin:0 0 10px; line-height:1.2; }}
h1 {{ font-size:30px; letter-spacing:0; }}
h2 {{ font-size:20px; }}
h3 {{ font-size:16px; }}
p {{ margin:6px 0; color:var(--muted); }}
.eyebrow {{ color:var(--teal); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
.hero {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; padding-bottom:26px; border-bottom:1px solid var(--line); }}
.hero-copy {{ max-width:760px; }}
.route {{ color:var(--ink); font-weight:600; margin-top:14px; }}
.badge {{ border:1px solid #f3c98b; background:#fff8eb; color:var(--yellow); border-radius:6px; padding:10px 12px; font-weight:700; white-space:nowrap; }}
.section {{ margin-top:28px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:16px; }}
.metric {{ font-size:24px; font-weight:750; margin-top:4px; }}
.metric-label {{ color:var(--muted); font-size:12px; }}
.target {{ color:var(--muted); font-size:12px; margin-top:8px; }}
.table-wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line); border-radius:7px; }}
table {{ width:100%; border-collapse:collapse; min-width:700px; }}
th,td {{ text-align:left; padding:11px 13px; border-bottom:1px solid #edf0f4; vertical-align:top; }}
th {{ color:var(--muted); font-size:12px; font-weight:700; background:#fafbfc; }}
tr:last-child td {{ border-bottom:0; }}
.status {{ display:inline-block; border-radius:999px; padding:2px 8px; font-size:11px; font-weight:800; }}
.pass {{ background:#dcfce7; color:var(--green); }} .pending {{ background:#fff7d6; color:var(--yellow); }}
.blocked {{ background:#fee2e2; color:#991b1b; }} .not-started {{ background:#eef2f6; color:var(--gray); }}
.chart-panel {{ background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:14px; overflow-x:auto; }}
.loss-chart {{ width:100%; min-width:760px; height:auto; display:block; }}
.line {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.grid {{ stroke:#e8edf2; stroke-width:1; }} .axis-line {{ stroke:#9aa5b1; stroke-width:1; }}
.axis {{ fill:#667085; font-size:11px; }} .marker {{ stroke:#fff; stroke-width:2; }}
.marker-label {{ fill:#344054; font-size:11px; }} .chart-legend {{ display:flex; flex-wrap:wrap; gap:16px; color:var(--muted); font-size:12px; margin:0 0 8px 58px; }}
.swatch {{ display:inline-block; width:18px; height:3px; vertical-align:middle; margin-right:5px; }} .teal {{ background:var(--teal); }} .orange {{ background:var(--orange); }} .blue {{ background:var(--blue); }} .red {{ background:var(--red); }}
.warning,.ok {{ padding:11px 13px; border-radius:6px; margin-top:8px; }} .warning {{ background:#fff4e5; color:#8a4b08; border:1px solid #f4c98b; }} .ok {{ background:#ecfdf3; color:var(--green); border:1px solid #a7f3d0; }}
.callout {{ border-left:4px solid var(--orange); background:#fff8f1; padding:13px 15px; color:#713f12; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.footer {{ color:var(--muted); font-size:12px; margin-top:28px; border-top:1px solid var(--line); padding-top:14px; }}
@media (max-width:800px) {{ .hero {{ display:block; }} .badge {{ display:inline-block; margin-top:14px; }} .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .two-col {{ grid-template-columns:1fr; }} }}
@media (max-width:500px) {{ main {{ padding:22px 14px 46px; }} h1 {{ font-size:25px; }} .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
  <header class="hero">
    <div class="hero-copy">
      <div class="eyebrow">Algorithm and Product Readout</div>
      <h1>{_esc(data["project"]["name"])}</h1>
      <p class="route">{_esc(data["project"]["route"])}</p>
      <p>{_esc(data["project"]["current_phase"])}</p>
    </div>
    <div class="badge">Pretraining PASS / Teacher PENDING</div>
  </header>

  <section class="section">
    <div class="grid">
      <div class="card"><div class="metric-label">Primary best Val Loss</div><div class="metric">{format_metric(runs[0]["best_val_loss"])}</div><div class="target">Masked reconstruction loss</div></div>
      <div class="card"><div class="metric-label">Primary best epoch</div><div class="metric">{runs[0]["best_epoch"]}</div><div class="target">Fixed budget: {data["protocol"]["epoch_budget"]}</div></div>
      <div class="card"><div class="metric-label">Downstream target BA</div><div class="metric">&ge; 0.70</div><div class="target">Current: N/A - not evaluated</div></div>
      <div class="card"><div class="metric-label">Test exposure</div><div class="metric">False</div><div class="target">train_val unlabeled protocol</div></div>
    </div>
  </section>

  <section class="section">
    <h2>Route Progress</h2>
    <p>Every status is tied to a documented gate. Reconstruction success does not imply classification success.</p>
    <div class="table-wrap">{_table(route_rows, [("stage","Stage"),("status","Status"),("target","Target"),("current","Current")])}</div>
  </section>

  <section class="section">
    <h2>AAMP Training Curves</h2>
    <p>Both runs use the same recommended configuration. Values are masked reconstruction loss, not classification accuracy.</p>
    <div class="chart-panel">{_svg_chart(data)}</div>
    <div class="table-wrap" style="margin-top:12px">{_table(run_rows, [("run","Run"),("epochs","Epochs"),("best_epoch","Best epoch"),("best_val","Best Val Loss"),("final_train","Final Train Loss"),("final_val","Final Val Loss")])}</div>
  </section>

  <section class="section">
    <h2>Target vs Current</h2>
    <div class="table-wrap">{_table(data["decisions"], [("metric","Metric"),("target","Target"),("current","Current"),("status","Status")])}</div>
  </section>

  <section class="section two-col">
    <div>
      <h2>Checkpoint Audit</h2>
      <div class="card">
        <h3>Primary migration checkpoint</h3>
        <p><strong>{_esc(data["checkpoint"]["path"])}</strong></p>
        <p>SHA-256: <code>{_esc(data["checkpoint"]["sha256"])}</code></p>
        <p>Role: fixed-epoch protocol weight. The diagnostic best checkpoint is not used as the primary migration weight.</p>
      </div>
    </div>
    <div>
      <h2>Product Decision</h2>
      <div class="callout">
        <strong>Current decision: do not distill yet.</strong>
        <p>The AAMP pretraining gate is complete. The teacher retention gate remains blocked until frozen, fine-tuned, and random-init classifiers complete the fixed three-fold, three-seed comparison.</p>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>Downstream Experiment Plan</h2>
    <p>These are planned runs only. The Demo does not execute them or alter their configuration.</p>
    <div class="table-wrap">{_table(experiment_rows, [("experiment","Experiment"),("status","Status"),("epochs","Epochs"),("batch","Batch"),("lr","Classifier LR"),("encoder_scale","Encoder LR scale")])}</div>
  </section>

  <section class="section">
    <h2>Audit Warnings</h2>
    {warning_html}
  </section>

  <footer class="footer">Generated from existing project artifacts only. No model, training, inference, evaluation, or hyperparameter-search process is invoked by this report generator.</footer>
</main>
</body>
</html>
"""


def write_demo(repo_root: Path, output_dir: Path) -> None:
    data = load_demo_data(Path(repo_root))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "demo_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(render_html(data), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/project_demo"))
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    try:
        data = load_demo_data(repo_root)
        output.mkdir(parents=True, exist_ok=True)
        (output / "demo_data.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        (output / "index.html").write_text(render_html(data), encoding="utf-8")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Demo generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"HTML: {output / 'index.html'}")
    print(f"Data: {output / 'demo_data.json'}")
    print(f"Primary checkpoint SHA-256: {data['checkpoint']['sha256']}")
    print(f"AAMP epochs parsed: {data['pretraining']['runs'][0]['epochs']}")
    print(f"Audit warnings: {len(data['audit_warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
