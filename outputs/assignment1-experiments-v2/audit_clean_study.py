"""Audit the pixel-clean study without fitting models or changing selections."""

import csv
import json

import numpy as np

from audit_run import audit_baseline, audit_training
from common import BASELINE_ROOT, OUTPUT_ROOT, sha256_file, write_json
from evaluate_frozen import load_probe, metrics, predict_probe
from pixel_clean import POLICY_PATH, ensure_policy, test_sensitivity_masks


def main():
    """Verify every run and replay saved numeric-head predictions."""
    ensure_policy()
    lock_path = OUTPUT_ROOT / "clean-selection-lock.json"
    lock = json.loads(lock_path.read_text())
    assert lock["pixel_clean_policy_sha256"] == sha256_file(POLICY_PATH)
    assert lock["protocol_lock_sha256"] == sha256_file(OUTPUT_ROOT / "protocol-lock.json")
    final = json.loads((OUTPUT_ROOT / "clean-final-report.json").read_text())
    assert final["status"] == "passed" and final["selection_lock_sha256"] == sha256_file(lock_path)
    training = {str(p.parent.relative_to(OUTPUT_ROOT)): audit_training(p.parent) for p in sorted((OUTPUT_ROOT / "training").rglob("training-summary.json"))}
    reports = []
    for path in sorted((OUTPUT_ROOT / "clean-final").rglob("results.json")):
        result = json.loads(path.read_text())
        assert result["selection_lock_sha256"] == sha256_file(lock_path)
        assert not result["head_refitted"]
        assert result["correction"]["pixel_clean_policy_sha256"] == sha256_file(POLICY_PATH)
        assert result["correction"]["split_counts"]["validation"] == (656 if result["dataset"] == "eth80" else 785)
        with (path.parent / "predictions.csv").open() as source:
            rows = list(csv.DictReader(source))
        expected = json.loads((BASELINE_ROOT / f"{result['dataset']}-seed{result['seed']}/splits.json").read_text())["test"]
        assert [(r["image_id"], int(r["label"])) for r in rows] == [(r["image_id"], r["label"]) for r in expected]
        labels = np.asarray([int(r["label"]) for r in rows])
        predictions = np.asarray([int(r["prediction"]) for r in rows])
        classes = np.arange(8 if result["dataset"] == "eth80" else 25)
        official = metrics(labels, predictions, classes)
        for key, value in official.items():
            assert result[key] == value
        assert result["official_test"] == official
        masks = test_sensitivity_masks(result["dataset"], [r["image_id"] for r in rows])
        for name, mask in masks.items():
            assert result["pixel_sensitivity"][name] == metrics(labels[mask], predictions[mask], classes)
        if result["candidate"] == "B0":
            vr = OUTPUT_ROOT / f"clean-B0-validation/{result['dataset']}-seed{result['seed']}-{result['method']}/report.json"
        elif result["method"] == "random" and result["image_size"] == 64:
            vr = OUTPUT_ROOT / f"clean-B0-validation/{result['dataset']}-seed{result['seed']}-random/report.json"
        elif result["method"] == "random":
            vr = OUTPUT_ROOT / f"clean-validation/{result['dataset']}/random{result['image_size']}/seed{result['seed']}/report.json"
        else:
            vr = OUTPUT_ROOT / f"clean-validation/{result['dataset']}/{result['candidate']}/seed{result['seed']}/epoch{result['epoch']:03d}/report.json"
        validation = json.loads(vr.read_text())
        assert sha256_file(validation["probe_path"]) == result["probe_sha256"]
        head = load_probe(validation["probe_path"])
        with np.load(result["feature_cache"]["npz"], allow_pickle=False) as source:
            assert np.array_equal(predict_probe(head, source["features"]), predictions)
        reports.append({"path": str(path.relative_to(OUTPUT_ROOT)), "correct": official["correct"], "count": official["count"], "prediction_replay": True})
    assert len(reports) == len(final["runs"])
    report = {"status": "passed", "original_baseline_files_unchanged": audit_baseline(), "training": training, "final_evaluations": reports, "selection_lock_sha256": sha256_file(lock_path), "pixel_clean_policy_sha256": sha256_file(POLICY_PATH)}
    write_json(OUTPUT_ROOT / "clean-final-audit.json", report)
    print(json.dumps({"status": "passed", "completed_training_runs": len(training), "replayed_final_evaluations": len(reports), "original_files_unchanged": report["original_baseline_files_unchanged"]}))


if __name__ == "__main__":
    main()
