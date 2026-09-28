import hashlib
import json
from pathlib import Path

import pytest

from scripts.generate_project_demo import (
    build_decision_table,
    load_demo_data,
    main,
    render_html,
    sha256_file,
    write_demo,
)


PRIMARY_DIR = Path("artifacts/pretrain_aamp_ear_saad_search_v2_lr4p43e-4_bs16")
REPEAT_DIR = Path("artifacts/pretrain_aamp_ear_saad_search_v2_lr443e-6_bs16")


def _write_run(root: Path, directory: Path, learning_rate: float) -> None:
    run_dir = root / directory
    run_dir.mkdir(parents=True)
    (run_dir / "pretrain_config.yaml").write_text(
        f"training:\n  epochs: 2\n  learning_rate: {learning_rate}\n",
        encoding="utf-8",
    )
    (run_dir / "pretrain_history.json").write_text(
        json.dumps(
            [
                {"epoch": 1, "train_loss": 0.6, "val_loss": 0.5},
                {"epoch": 2, "train_loss": 0.4, "val_loss": 0.3},
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "pretrain_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "development_train_val_unlabeled",
                "unlabeled_splits": "train_val",
                "test_exposed_to_pretraining": False,
                "validation_is_used_for_checkpoint_selection": False,
                "checkpoint_selection": "fixed_epochs",
                "checkpoint_sha256": {},
            }
        ),
        encoding="utf-8",
    )


def _make_fixture(root: Path) -> Path:
    _write_run(root, PRIMARY_DIR, 0.00044348092176995735)
    _write_run(root, REPEAT_DIR, 0.00044348092176995735)
    checkpoint = root / PRIMARY_DIR / "final_pretrain_model.pt"
    checkpoint.write_bytes(b"demo checkpoint")
    metadata_path = root / PRIMARY_DIR / "pretrain_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = {
        "final_pretrain_model.pt": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _decision(rows, key):
    return next(row for row in rows if row["key"] == key)


def test_load_demo_data_reads_histories_and_checkpoint_hash(tmp_path):
    root = _make_fixture(tmp_path)
    data = load_demo_data(root)

    assert data["pretraining"]["runs"][0]["epochs"] == 2
    assert data["checkpoint"]["sha256"] == sha256_file(
        root / PRIMARY_DIR / "final_pretrain_model.pt"
    )


def test_missing_required_history_fails_with_path(tmp_path):
    root = _make_fixture(tmp_path)
    (root / PRIMARY_DIR / "pretrain_history.json").unlink()

    with pytest.raises(FileNotFoundError, match="pretrain_history.json"):
        load_demo_data(root)


def test_pretraining_decisions_are_pass_and_downstream_is_pending(tmp_path):
    data = load_demo_data(_make_fixture(tmp_path))
    decisions = build_decision_table(data)

    assert _decision(decisions, "test_exposed_to_pretraining")["status"] == "PASS"
    assert _decision(decisions, "downstream_mean_balanced_accuracy")["status"] == "PENDING"
    assert _decision(decisions, "cnn_distillation")["status"] == "BLOCKED"


def test_render_html_contains_actual_metrics_and_route_status(tmp_path):
    data = load_demo_data(_make_fixture(tmp_path))
    html = render_html(data)

    assert "0.3000" in html
    assert "N/A - not evaluated" in html
    assert "&gt;= 0.70" in html
    assert "PASS" in html
    assert "PENDING" in html
    assert "BLOCKED" in html
    assert "<svg" in html


def test_write_demo_only_writes_demo_output(tmp_path):
    root = _make_fixture(tmp_path)
    output = root / "artifacts" / "project_demo"

    write_demo(root, output)

    assert (output / "index.html").exists()
    assert (output / "demo_data.json").exists()
    assert not (output / "final_pretrain_model.pt").exists()


def test_cli_generates_demo_without_training(tmp_path):
    root = _make_fixture(tmp_path)
    output = root / "artifacts" / "project_demo"

    assert main(["--repo-root", str(root), "--output", str(output)]) == 0
    assert (output / "index.html").exists()


def test_generation_does_not_modify_source_checkpoint(tmp_path):
    root = _make_fixture(tmp_path)
    checkpoint = root / PRIMARY_DIR / "final_pretrain_model.pt"
    before = sha256_file(checkpoint)

    write_demo(root, root / "artifacts" / "project_demo")

    assert sha256_file(checkpoint) == before
