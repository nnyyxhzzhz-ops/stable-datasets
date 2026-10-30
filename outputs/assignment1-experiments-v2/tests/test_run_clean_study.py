"""Verify corrected-study lock and sensitivity safeguards without model training."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
SPEC = importlib.util.spec_from_file_location("run_clean_study", SOURCE_DIR / "run_clean_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_sensitivity_uses_fixed_order_and_retains_official_predictions(monkeypatch):
    import pixel_clean

    ids = ["a", "b", "c", "d", "e", "f"]
    masks = {
        "official": np.ones(6, dtype=bool),
        "unseen_pixels": np.array([1, 1, 1, 1, 0, 0], dtype=bool),
        "unique_unseen_pixels": np.array([1, 1, 0, 1, 0, 0], dtype=bool),
    }

    def fixed_masks(dataset, ordered_ids):
        assert dataset == "freiburg" and ordered_ids == ids
        return masks

    monkeypatch.setattr(pixel_clean, "test_sensitivity_masks", fixed_masks)
    labels = np.array([0, 1, 0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1, 1, 1])
    results = study.sensitivity_metrics("freiburg", ids, labels, predictions, np.arange(2))
    assert results["official"]["count"] == 6 and results["official"]["correct"] == 4
    assert results["unseen_pixels"]["count"] == 4 and results["unseen_pixels"]["correct"] == 3
    assert results["unique_unseen_pixels"]["count"] == 3 and results["unique_unseen_pixels"]["correct"] == 3
    assert predictions.tolist() == [0, 1, 1, 1, 1, 1]


def test_changed_protocol_lock_blocks_confirmation_before_training(tmp_path, monkeypatch):
    lock = tmp_path / "selection.json"
    lock.write_text(json.dumps({"protocol_lock_sha256": "incorrect"}))
    (tmp_path / "protocol-lock.json").write_text("{}")
    monkeypatch.setattr(study, "CLEAN_LOCK", lock)
    monkeypatch.setattr(study, "OUT", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid selection must not launch training or test extraction")

    monkeypatch.setattr(study, "train", forbidden)
    monkeypatch.setattr(study, "extract_features", forbidden)
    with pytest.raises(AssertionError):
        study.confirm(1)


def test_changed_head_blocks_test_before_feature_extraction(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"trusted local checkpoint fixture")
    head = tmp_path / "probe.npz"
    head.write_bytes(b"modified head")
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": study.sha256_file(checkpoint),
        "probe_path": str(head), "probe_sha256": "incorrect",
    }
    monkeypatch.setattr(study, "clean_splits", lambda dataset: {})

    def forbidden(*args, **kwargs):
        pytest.fail("Changed head must not reach encoder loading or test extraction")

    monkeypatch.setattr(study, "baseline_encoder", forbidden)
    monkeypatch.setattr(study, "extract_features", forbidden)
    destination = tmp_path / "final"
    with pytest.raises(AssertionError):
        study.evaluate_test("eth80", 42, "B0", 20, "lejepa", 64, report, destination, "lock")
    assert not destination.exists()


def test_old_validation_membership_is_not_silently_reused(tmp_path, monkeypatch):
    destination = tmp_path / "cached"
    destination.mkdir()
    (destination / "report.json").write_text(json.dumps({"correction": {"validation": 786}}))
    monkeypatch.setattr(study, "clean_splits", lambda dataset: {})
    monkeypatch.setattr(study, "correction_provenance", lambda dataset, splits: {"validation": 785})
    with pytest.raises(AssertionError):
        study.fit_condition("freiburg", 42, "lejepa", 64, tmp_path / "checkpoint.pt", destination)
