"""Audit study artifacts, baseline preservation and final CSV-derived metrics."""

import argparse
import csv
import json

import numpy as np
import torch

from common import BASELINE_ROOT, OUTPUT_ROOT, sha256_file, write_json
from evaluate_frozen import metrics


def audit_baseline():
    """Require all original experiment bytes to match the pre-study snapshot."""
    before = json.loads((OUTPUT_ROOT / "baseline-integrity-before.json").read_text())
    for name, record in before.items():
        path = BASELINE_ROOT / name
        assert path.stat().st_size == record["size"] and sha256_file(path) == record["sha256"], path
    profile = OUTPUT_ROOT / "baseline-integrity-profile.json"
    expected = json.loads(profile.read_text())["file_count"] if profile.exists() else 107
    assert len([p for p in BASELINE_ROOT.rglob("*") if p.is_file()]) == len(before) == expected
    return len(before)


def audit_training(directory):
    """Check full split membership, exact initialization, steps and finite weights."""
    config = json.loads((directory / "config.json").read_text())
    history = json.loads((directory / "training.json").read_text())
    summary = json.loads((directory / "training-summary.json").read_text())
    manifest = (directory / "splits.json").read_bytes()
    reference = BASELINE_ROOT / f"{config['dataset']}-seed{config['seed']}"
    assert manifest == (reference / "splits.json").read_bytes()
    assert config["epochs"] == 100
    assert summary["mode"] == "train" and summary["status"] == "completed"
    actual_epochs = config.get("stop_after") or 100
    assert len(history) == summary["epochs_completed"] == actual_epochs
    steps_per_epoch = 61 if config["dataset"] == "eth80" else 98
    assert summary["steps_per_epoch"] == steps_per_epoch
    for index, row in enumerate(history, 1):
        assert row["epoch"] == index and row["steps"] == index * steps_per_epoch
        assert all(np.isfinite(row[k]) for k in ("loss", "invariance", "sigreg", "learning_rate"))
    initial = torch.load(directory / "random-encoder.pt", weights_only=True, map_location="cpu")
    baseline = torch.load(reference / "random-encoder.pt", weights_only=True, map_location="cpu")
    assert set(initial) == set(baseline)
    assert all(torch.equal(initial[k], baseline[k]) for k in initial)
    checkpoints = []
    for path in sorted(directory.glob("checkpoint-epoch*.pt")):
        saved = torch.load(path, weights_only=False, map_location="cpu")
        assert saved["steps"] == saved["epoch"] * steps_per_epoch
        assert saved["config"] == config
        assert saved["model"]["sigreg.global_step"].item() == saved["steps"]
        assert all(torch.isfinite(value).all() for value in saved["model"].values())
        changed = sum(not torch.equal(initial[k.removeprefix("backbone.")], v) for k, v in saved["model"].items() if k.startswith("backbone."))
        assert changed > 0
        checkpoints.append({"epoch": saved["epoch"], "sha256": sha256_file(path), "changed_encoder_tensors": changed})
    return {"config": config, "steps": summary["steps"], "checkpoints": checkpoints, "initial_encoder_exactly_matches_baseline": True}


def audit_final():
    """Recompute all published final metrics from per-image prediction rows."""
    lock_path = OUTPUT_ROOT / "selection-lock.json"
    lock = json.loads(lock_path.read_text())
    assert lock["protocol_lock_sha256"] == sha256_file(OUTPUT_ROOT / "protocol-lock.json")
    report = json.loads((OUTPUT_ROOT / "final-report.json").read_text())
    assert report["status"] == "passed" and report["selection_lock_sha256"] == sha256_file(lock_path)
    count = 0
    for path in sorted((OUTPUT_ROOT / "final").rglob("results.json")):
        result = json.loads(path.read_text())
        assert result["selection_lock_sha256"] == sha256_file(lock_path)
        with (path.parent / "predictions.csv").open() as source:
            rows = list(csv.DictReader(source))
        reference = json.loads((BASELINE_ROOT / f"{result['dataset']}-seed{result['seed']}/splits.json").read_text())["test"]
        assert [(r["image_id"], int(r["label"])) for r in rows] == [(r["image_id"], r["label"]) for r in reference]
        actual = metrics(np.array([int(r["label"]) for r in rows]), np.array([int(r["prediction"]) for r in rows]), np.arange(8 if result["dataset"] == "eth80" else 25))
        for key, value in actual.items():
            assert value == result[key], (path, key)
        count += 1
    assert count == len(report["runs"])
    return count


def main():
    """Audit completed portions, or require a complete locked final report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    records = {"baseline_files_unchanged": audit_baseline(), "training": {}}
    for path in sorted((OUTPUT_ROOT / "training").rglob("training-summary.json")):
        records["training"][str(path.parent.relative_to(OUTPUT_ROOT))] = audit_training(path.parent)
    if args.final:
        records["final_evaluations_verified"] = audit_final()
    records["status"] = "passed"
    write_json(OUTPUT_ROOT / ("final-audit.json" if args.final else "interim-audit.json"), records)
    print(json.dumps({"baseline_files_unchanged": records["baseline_files_unchanged"], "completed_training_runs": len(records["training"]), "status": "passed"}))


if __name__ == "__main__":
    main()
