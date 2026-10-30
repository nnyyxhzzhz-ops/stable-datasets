"""Independently check the supervised reference's locked heads and predictions."""

import csv
import json

import numpy as np

from common import BASELINE_ROOT, OUTPUT_ROOT, sha256_file, write_json
from evaluate_frozen import load_probe, metrics, predict_probe
from pixel_clean import POLICY_PATH, ensure_policy, test_sensitivity_masks


def main():
    """Recompute predictions from numeric cached features and selected heads."""
    root = OUTPUT_ROOT / "pretrained-reference"
    ensure_policy()
    policy = json.loads(POLICY_PATH.read_text())
    lock_path = root / "clean-selection-lock.json"
    lock = json.loads(lock_path.read_text())
    assert lock["policy_sha256"] == sha256_file(POLICY_PATH)
    report = {"pretraining": "ImageNet-1K supervised; not LeJEPA", "datasets": {}}
    for dataset in ("eth80", "freiburg"):
        selection_path = root / dataset / "clean-validation/selection.json"
        selection = json.loads(selection_path.read_text())
        assert sha256_file(selection_path) == lock["reference_selections"][dataset]["selection_sha256"]
        assert not selection["test_images_accessed"]
        probe_path = selection["probe_path"]
        assert sha256_file(probe_path) == selection["probe_sha256"]
        head = load_probe(probe_path)
        manifest = json.loads((BASELINE_ROOT / f"{dataset}-seed42/splits.json").read_text())
        excluded = set(policy["datasets"][dataset]["validation_excluded_ids"])
        manifest["validation"] = [row for row in manifest["validation"] if row["image_id"] not in excluded]
        for split in ("train", "validation"):
            metadata = selection["feature_cache"][split]["metadata"]
            assert metadata["ordered_image_ids"] == [r["image_id"] for r in manifest[split]]
            assert metadata["ordered_labels"] == [r["label"] for r in manifest[split]]
        with np.load(selection["feature_cache"]["train"]["npz"], allow_pickle=False) as source:
            training = np.asarray(source["features"], dtype=np.float64)
        assert np.array_equal(head["mean"], training.mean(0))
        assert np.array_equal(head["std"], training.std(0, ddof=1).clip(min=1e-6))
        result = json.loads((root / dataset / "clean-test/test-results.json").read_text())
        assert result["selection_sha256"] == sha256_file(selection_path)
        assert result["weights_sha256"] == lock["weights_sha256"]
        assert result["clean_selection_lock_sha256"] == sha256_file(lock_path)
        assert result["pretraining"]["method"] == "supervised" and not result["pretraining"]["is_LeJEPA"]
        assert not result["head_refitted"]
        with np.load(result["feature_cache"]["npz"], allow_pickle=False) as source:
            features, labels = source["features"], source["labels"]
        predicted = predict_probe(head, features)
        with (root / dataset / "clean-test/test-predictions.csv").open() as source:
            rows = list(csv.DictReader(source))
        assert [(r["image_id"], int(r["label"])) for r in rows] == [(r["image_id"], r["label"]) for r in manifest["test"]]
        assert np.array_equal(predicted, [int(r["prediction"]) for r in rows])
        assert np.array_equal(labels, [r["label"] for r in manifest["test"]])
        actual = metrics(labels, predicted, head["classes"])
        for key, value in actual.items():
            assert result[key] == value, (dataset, key)
        masks = test_sensitivity_masks(dataset, [row["image_id"] for row in rows])
        for name, mask in masks.items():
            assert result["sensitivity"][name] == metrics(labels[mask], predicted[mask], head["classes"])
        report["datasets"][dataset] = {"validation_accuracy": selection["evaluation"]["selected"]["validation_accuracy"], "selected_C": selection["evaluation"]["selected"]["C"], **actual, "sensitivity": result["sensitivity"], "clean_validation_count": len(manifest["validation"]), "saved_head_predictions_replayed": True, "train_only_standardization_verified": True}
    report["status"] = "passed"
    write_json(root / "clean-audit.json", report)
    lines = ["# Frozen supervised-pretraining reference", "", "Official torchvision ResNet-18 ImageNet-1K supervised weights were frozen. The dataset loaders and SSL training membership were retained. One Freiburg validation image with identical pixels in training was excluded before head selection (785 validation images). Only a multinomial logistic-regression head was fitted on training features; C was selected on validation data before each test. This is an external-supervised-data transfer reference and is not a LeJEPA result or a same-architecture causal comparison.", "", "| Dataset | Validation (%) | Selected C | Correct/official test | Official test (%) | Balanced (%) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in report["datasets"].items():
        lines.append(f"| {name} | {100*r['validation_accuracy']:.2f} | {r['selected_C']:g} | {r['correct']}/{r['count']} | {100*r['accuracy']:.2f} | {100*r['balanced_accuracy']:.2f} |")
    lines += ["", "## Source-pixel sensitivity", "", "The official Freiburg test split contains eight images with exact RGB matches in train or the original validation split. The subsets were fixed from a source-pixel audit before test predictions were computed. The unique-unseen subset additionally removes two within-test duplicate copies. Loader membership is unchanged.", "", "| Dataset | Test subset | Correct/count | Accuracy (%) | Balanced (%) |", "| --- | --- | ---: | ---: | ---: |"]
    for name, r in report["datasets"].items():
        for subset, m in r["sensitivity"].items():
            lines.append(f"| {name} | {subset} | {m['correct']}/{m['count']} | {100*m['accuracy']:.2f} | {100*m['balanced_accuracy']:.2f} |")
    lines += ["", "The encoder and solver are fixed and deterministic; these are single evaluations, not three independent training seeds. All head searches, warnings, iterations, official preprocessing, checksums and per-image predictions are saved. The audit replayed the numeric head predictions and verified train-only normalization and exact test IDs/labels. Original LeJEPA experiments are unchanged. Pretraining-source overlap with these datasets has not been independently audited.", "", "Weights were obtained from the official PyTorch distribution URL. Official filename hash prefix f37072fd was checked and the full SHA-256 recorded in weights-download.json. Test heads were locked by clean-selection-lock.json before test feature extraction.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
