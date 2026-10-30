"""Pixel-audited pretrained reference with a separately locked final test phase.

Historical 786-image Freiburg validation results remain untouched. This study
removes the one validation image whose RGB pixels also occur in training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import initialize_runtime, sha256_file, write_json
from evaluate_frozen import EVALUATOR, metrics
from evaluate_pretrained_reference import load_encoder, test_selected, validate
from pixel_clean import (
    POLICY_PATH,
    ensure_policy,
    load_clean_splits,
    test_sensitivity_masks,
)


def verify_test_lock(args, selection):
    """Require exact policy, weight and selected-head hashes before test access."""
    if args.lock is None or args.lock_sha256 is None:
        raise ValueError(
            "Final test requires a separately saved root selection lock and its SHA-256."
        )
    if sha256_file(args.lock) != args.lock_sha256:
        raise ValueError("Clean study selection-lock checksum does not match.")
    lock = json.loads(args.lock.read_text())
    assert lock["policy_sha256"] == sha256_file(POLICY_PATH)
    assert lock["weights_sha256"] == args.weights_sha256
    assert lock["evaluation"] == json.loads(json.dumps(EVALUATOR))
    assert lock["original_test_scores_already_known"] is True
    entry = lock["reference_selections"][args.dataset]
    assert entry["selection_sha256"] == args.selection_sha256
    assert entry["probe_sha256"] == selection["probe_sha256"]
    assert selection["pixel_clean_policy_sha256"] == sha256_file(POLICY_PATH)
    assert selection["clean_wrapper_sha256"] == sha256_file(__file__)
    return lock


def validation_phase(args, encoder, weights_sha256, splits):
    """Select heads using the clean validation membership without test pixels."""
    result = validate(args, encoder, weights_sha256, splits)
    path = args.output / "selection.json"
    selection = json.loads(path.read_text())
    selection.update(
        pixel_clean_policy_sha256=sha256_file(POLICY_PATH),
        clean_wrapper_sha256=sha256_file(__file__),
        validation_protocol="pixel_clean_v1",
        historical_validation_count=786 if args.dataset == "freiburg" else 656,
        clean_validation_count=len(splits["validation"]),
        historical_validation_results_overwritten=False,
        training_membership_changed=False,
        official_test_membership_changed=False,
        original_test_scores_already_known=True,
    )
    write_json(path, selection)
    result.update(
        validation_protocol="pixel_clean_v1",
        clean_validation_count=len(splits["validation"]),
        selection_sha256=sha256_file(path),
    )
    return result


def test_phase(args, encoder, weights_sha256, splits):
    """Evaluate locked heads and report the official and source-pixel subsets."""
    if sha256_file(args.selection) != args.selection_sha256:
        raise ValueError("Clean validation selection checksum does not match.")
    selection = json.loads(args.selection.read_text())
    verify_test_lock(args, selection)
    result = test_selected(args, encoder, weights_sha256, splits)
    with (args.output / "test-predictions.csv").open() as source:
        rows = list(csv.DictReader(source))
    image_ids = [row["image_id"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows])
    predictions = np.asarray([int(row["prediction"]) for row in rows])
    masks = test_sensitivity_masks(args.dataset, image_ids)
    classes = splits["train"].features["label"].num_classes
    result["sensitivity"] = {
        name: metrics(labels[mask], predictions[mask], np.arange(classes))
        for name, mask in masks.items()
    }
    assert result["sensitivity"]["official"]["count"] == result["count"]
    assert result["sensitivity"]["official"]["correct"] == result["correct"]
    result.update(
        pixel_clean_policy_sha256=sha256_file(POLICY_PATH),
        clean_selection_lock_sha256=args.lock_sha256,
        clean_wrapper_sha256=sha256_file(__file__),
        validation_protocol="pixel_clean_v1",
        original_test_scores_already_known=True,
        sensitivity_subsets_selected_without_test_labels=True,
        head_refitted=False,
    )
    write_json(args.output / "test-results.json", result)
    return result


def main():
    """Run clean validation, or test only after an external selection lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--dataset", choices=("eth80", "freiburg"), required=True)
    parser.add_argument("--phase", choices=("validate", "test"), required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--lock-sha256")
    args = parser.parse_args()
    if args.phase == "test" and not all(
        (args.selection, args.selection_sha256, args.lock, args.lock_sha256)
    ):
        parser.error(
            "test requires the clean validation selection and a separate final lock, both with SHA-256"
        )
    initialize_runtime(threads=4)
    torch.use_deterministic_algorithms(True)
    ensure_policy()
    # Validate the external lock before loading test images or writing outputs.
    if args.phase == "test":
        if sha256_file(args.selection) != args.selection_sha256:
            parser.error("clean validation selection checksum does not match")
        verify_test_lock(args, json.loads(args.selection.read_text()))
    encoder, weights_sha256 = load_encoder(args.weights, args.weights_sha256)
    splits = load_clean_splits(args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    result = (
        validation_phase(args, encoder, weights_sha256, splits)
        if args.phase == "validate"
        else test_phase(args, encoder, weights_sha256, splits)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
