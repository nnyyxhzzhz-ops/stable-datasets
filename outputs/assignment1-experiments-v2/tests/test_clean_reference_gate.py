"""Offline checks for the clean pretrained reference's final-test gate."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pretrained_reference_clean as reference
from common import sha256_file
from evaluate_frozen import EVALUATOR


def test_final_test_requires_a_separate_lock():
    args = SimpleNamespace(lock=None, lock_sha256=None)
    with pytest.raises(ValueError, match="separately saved"):
        reference.verify_test_lock(args, {})


def test_changed_lock_file_is_rejected(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        reference.verify_test_lock(SimpleNamespace(lock=path, lock_sha256="0" * 64), {})


def test_selected_head_is_bound_to_the_clean_policy(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    policy.write_text('{"excluded_validation_ids": ["SPICES/SPICES0052.png"]}')
    monkeypatch.setattr(reference, "POLICY_PATH", policy)
    selection = {
        "probe_sha256": "1" * 64,
        "pixel_clean_policy_sha256": sha256_file(policy),
        "clean_wrapper_sha256": sha256_file(reference.__file__),
    }
    lock = {
        "policy_sha256": sha256_file(policy),
        "weights_sha256": "2" * 64,
        "evaluation": EVALUATOR,
        "original_test_scores_already_known": True,
        "reference_selections": {
            "freiburg": {"selection_sha256": "3" * 64, "probe_sha256": "1" * 64}
        },
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    args = SimpleNamespace(
        lock=path,
        lock_sha256=sha256_file(path),
        weights_sha256="2" * 64,
        dataset="freiburg",
        selection_sha256="3" * 64,
    )
    assert reference.verify_test_lock(args, selection)
    selection["probe_sha256"] = "4" * 64
    with pytest.raises(AssertionError):
        reference.verify_test_lock(args, selection)
