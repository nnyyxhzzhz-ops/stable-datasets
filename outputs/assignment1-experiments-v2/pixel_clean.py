"""Pixel-derived validation exclusion and fixed official-test sensitivity masks."""

import json

import common
import numpy as np


AUDIT_PATH = common.OUTPUT_ROOT / "exact-pixel-duplicate-audit.json"
AUDIT_SHA256 = "2f465626c89220bfeeb0e142ee6679ad0d692ed67119e45f1865824a5fb092d9"
POLICY_PATH = common.OUTPUT_ROOT / "pixel-clean-protocol.json"
ORIGINAL_PROTOCOL_PATH = common.OUTPUT_ROOT / "protocol-lock.json"


def _manifest(dataset):
    """Read the original experimental metadata without loading image pixels."""
    if dataset not in ("eth80", "freiburg"):
        raise ValueError(f"Unknown dataset: {dataset}")
    return json.loads((common.BASELINE_ROOT / f"{dataset}-seed42/splits.json").read_text())


def _derived_policy():
    """Derive exclusions entirely from the pinned source-pixel audit."""
    if common.sha256_file(AUDIT_PATH) != AUDIT_SHA256:
        raise RuntimeError("Source-pixel audit changed after its exclusions were defined.")
    audit = json.loads(AUDIT_PATH.read_text())
    datasets = {}
    for dataset in ("eth80", "freiburg"):
        source = audit["datasets"][dataset]
        assert source["conflicting_label_groups"] == 0
        original = _manifest(dataset)
        validation_excluded, test_seen, test_repeated = set(), set(), set()
        for group in source["duplicate_details"]:
            images = group["images"]
            splits = {row["split"] for row in images}
            if "train" in splits:
                validation_excluded.update(row["image_id"] for row in images if row["split"] == "validation")
            if splits & {"train", "validation"}:
                test_seen.update(row["image_id"] for row in images if row["split"] == "test")
            tests = sorted(row["image_id"] for row in images if row["split"] == "test")
            if len(tests) > 1:
                test_repeated.update(tests[1:])
        test_ids = {row["image_id"] for row in original["test"]}
        assert test_seen <= test_ids and test_repeated <= test_ids
        datasets[dataset] = {
            "validation_excluded_ids": sorted(validation_excluded),
            "test_seen_in_train_or_original_validation_ids": sorted(test_seen),
            "test_noncanonical_duplicate_ids": sorted(test_repeated),
            "test_unique_unseen_excluded_ids": sorted(test_seen | test_repeated),
            "counts": {
                "train": len(original["train"]),
                "original_validation": len(original["validation"]),
                "clean_validation": len(original["validation"]) - len(validation_excluded),
                "official_test": len(test_ids),
                "unseen_pixels_test": len(test_ids - test_seen),
                "unique_unseen_pixels_test": len(test_ids - test_seen - test_repeated),
            },
            "original_split_manifest_sha256": source["split_manifest_sha256"],
        }
    assert datasets["freiburg"]["validation_excluded_ids"] == ["SPICES/SPICES0052.png"]
    assert datasets["freiburg"]["counts"] == {
        "train": 3143,
        "original_validation": 786,
        "clean_validation": 785,
        "official_test": 1018,
        "unseen_pixels_test": 1010,
        "unique_unseen_pixels_test": 1008,
    }
    assert datasets["eth80"]["validation_excluded_ids"] == []
    return {
        "source_pixel_audit_sha256": AUDIT_SHA256,
        "original_protocol_sha256": common.sha256_file(ORIGINAL_PROTOCOL_PATH),
        "decision_basis": "Decoded source pixels and original split metadata only; no classifier predictions or accuracy were used.",
        "validation_rule": "Exclude validation IDs whose exact RGB array also occurs in the training split; keep all training and official test rows.",
        "test_rules": {
            "official": "Retain all author test rows in their original order.",
            "unseen_pixels": "Exclude test IDs with an exact RGB match in train or the original validation split.",
            "unique_unseen_pixels": "Additionally retain only the lexicographically first test ID per identical-pixel group.",
        },
        "loaders_and_official_membership_modified": False,
        "datasets": datasets,
    }


def ensure_policy():
    """Create once, or verify, the pixel-only policy before evaluation."""
    policy = _derived_policy()
    if POLICY_PATH.exists():
        saved = json.loads(POLICY_PATH.read_text())
        if saved != policy:
            raise RuntimeError("Existing pixel-clean policy differs; refusing silent reselection.")
    else:
        POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with POLICY_PATH.open("x") as target:
            target.write(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    return policy


def load_clean_splits(dataset):
    """Preserve train/test and remove the one exact train/validation duplicate."""
    policy = ensure_policy()["datasets"][dataset]
    splits = common.load_splits(dataset)
    excluded = set(policy["validation_excluded_ids"])
    if not excluded:
        return splits
    validation = splits["validation"]
    ids = [row["image_id"] for row in validation.remove_columns("image").with_format("raw")]
    assert excluded <= set(ids)
    indices = [index for index, image_id in enumerate(ids) if image_id not in excluded]
    result = {**splits, "validation": validation.select(indices)}
    assert len(result["validation"]) == policy["counts"]["clean_validation"]
    assert result["train"] is splits["train"] and result["test"] is splits["test"]
    return result


def test_sensitivity_masks(dataset, ordered_ids):
    """Return aligned boolean masks without looking at labels or predictions."""
    policy = ensure_policy()["datasets"][dataset]
    expected = [row["image_id"] for row in _manifest(dataset)["test"]]
    if list(ordered_ids) != expected:
        raise ValueError("Sensitivity masks require the full original ordered test IDs.")
    seen = set(policy["test_seen_in_train_or_original_validation_ids"])
    unique_excluded = set(policy["test_unique_unseen_excluded_ids"])
    masks = {
        "official": np.ones(len(expected), dtype=bool),
        "unseen_pixels": np.asarray([image_id not in seen for image_id in expected], dtype=bool),
        "unique_unseen_pixels": np.asarray([image_id not in unique_excluded for image_id in expected], dtype=bool),
    }
    assert int(masks["official"].sum()) == policy["counts"]["official_test"]
    assert int(masks["unseen_pixels"].sum()) == policy["counts"]["unseen_pixels_test"]
    assert int(masks["unique_unseen_pixels"].sum()) == policy["counts"]["unique_unseen_pixels_test"]
    return masks


if __name__ == "__main__":
    print(json.dumps(ensure_policy(), indent=2, sort_keys=True))
