"""Select and evaluate LeJEPA candidates with pixel-clean validation holdouts.

SSL training samples are unchanged. The corrected Freiburg validation set drops
one image whose exact RGB pixels occur in training. Official test predictions
are accompanied by prespecified exact-pixel overlap sensitivity analyses.
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common import ROOT, environment, initialize_runtime, sha256_file, split_manifest, write_json
from evaluate_frozen import EVALUATOR, baseline_encoder, extract_features, fit_probe, load_probe, metrics, predict_probe, save_probe
from run_study import encoder_from_checkpoint, train


OUT = ROOT / "outputs/experiments-v2"
CACHE = OUT / "feature-cache"
BASELINE = OUT / "clean-B0-validation"
CLEAN_LOCK = OUT / "clean-selection-lock.json"
AUDIT = OUT / "exact-pixel-duplicate-audit.json"


def clean_splits(dataset):
    """Get cache-only corrected holdouts from the shared pixel-audit module."""
    from pixel_clean import load_clean_splits

    return load_clean_splits(dataset)


def correction_provenance(dataset, splits):
    """Bind corrected ordered split membership and duplicate findings."""
    from evaluate_frozen import digest_json
    from pixel_clean import POLICY_PATH, ensure_policy

    ensure_policy()
    manifest = split_manifest(splits)
    return {
        "dataset": dataset,
        "pixel_duplicate_audit_sha256": sha256_file(AUDIT),
        "pixel_clean_source_sha256": sha256_file(Path(__file__).with_name("pixel_clean.py")),
        "pixel_clean_policy_sha256": sha256_file(POLICY_PATH),
        "ordered_split_manifest_sha256": digest_json(manifest),
        "split_counts": {name: len(part) for name, part in splits.items()},
        "ssl_training_samples_unchanged": True,
        "test_images_accessed": False,
    }


def fit_condition(dataset, seed, method, image_size, checkpoint, directory):
    """Fit one validation-only head with corrected sample membership."""
    splits = clean_splits(dataset)
    provenance = correction_provenance(dataset, splits)
    if (directory / "report.json").is_file():
        report = json.loads((directory / "report.json").read_text())
        assert report["correction"] == provenance
        assert report["checkpoint_sha256"] == sha256_file(checkpoint)
        assert report["probe_sha256"] == sha256_file(report["probe_path"])
        return report
    directory.mkdir(parents=True, exist_ok=False)
    if method == "random" or checkpoint.name == "checkpoint.pt":
        encoder, source = baseline_encoder(ROOT / "outputs/experiments" / f"{dataset}-seed{seed}", seed, method)
        assert source == checkpoint
    else:
        encoder = encoder_from_checkpoint(checkpoint, seed)
    values = {name: extract_features(encoder, splits[name], image_size=image_size, cache_root=CACHE, encoder_sha256=sha256_file(checkpoint)) for name in ("train", "validation")}
    probe, report = fit_probe(*values["train"][:2], *values["validation"][:2], seed)
    path = directory / "probe.npz"
    save_probe(path, probe)
    report.update(
        dataset=dataset, seed=seed, method=method, image_size=image_size,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        probe_path=str(path), probe_sha256=sha256_file(path), correction=provenance,
        feature_cache={name: value[2] for name, value in values.items()},
    )
    write_json(directory / "report.json", report)
    print(json.dumps({"phase": "clean_validation", "dataset": dataset, "seed": seed, "method": method, "directory": str(directory), "selected": report["selected"]}), flush=True)
    return report


def baseline_validation():
    """Evaluate all twelve preserved encoders using the corrected holdouts."""
    runs = []
    for dataset in ("eth80", "freiburg"):
        for seed in (42, 43, 44):
            for method in ("random", "lejepa"):
                source = ROOT / "outputs/experiments" / f"{dataset}-seed{seed}" / ("random-encoder.pt" if method == "random" else "checkpoint.pt")
                report = fit_condition(dataset, seed, method, 64, source, BASELINE / f"{dataset}-seed{seed}-{method}")
                runs.append(report)
    write_json(BASELINE / "validation-report.json", {"status": "passed", "stage": "pixel_clean_B0_validation", "runs": runs, "test_images_accessed": False, "environment": environment()})


def candidate_validation(dataset, candidate, seed, epoch):
    """Read a completed SSL checkpoint and fit its corrected validation head."""
    run = OUT / "training" / dataset / candidate / f"seed{seed}"
    config = json.loads((run / "config.json").read_text())
    checkpoint = run / f"checkpoint-epoch{epoch:03d}.pt"
    history = json.loads((run / "training.json").read_text())
    assert history[-1]["epoch"] >= epoch
    # A completed summary, or a later recorded epoch, ensures this checkpoint's
    # write has finished before a concurrent validation process reads it.
    assert (run / "training-summary.json").is_file() or history[-1]["epoch"] > epoch
    directory = OUT / "clean-validation" / dataset / candidate / f"seed{seed}" / f"epoch{epoch:03d}"
    report = fit_condition(dataset, seed, "lejepa", config["image_size"], checkpoint, directory)
    return report


def develop(jobs):
    """Train/reuse development candidates and lock corrected validation selection."""
    if CLEAN_LOCK.exists():
        raise RuntimeError("The corrected selection is already locked; no silent reselection.")
    baseline_validation()
    tasks = [(dataset, candidate, 42) for candidate in ("B1", "B2") for dataset in ("eth80", "freiburg")]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda args: train(*args), tasks))
    selection = {
        "protocol_lock_sha256": sha256_file(OUT / "protocol-lock.json"),
        "pixel_duplicate_audit_sha256": sha256_file(AUDIT),
        "pixel_clean_source_sha256": sha256_file(Path(__file__).with_name("pixel_clean.py")),
        "pixel_clean_policy_sha256": correction_provenance("eth80", clean_splits("eth80"))["pixel_clean_policy_sha256"],
        "coordinator_source_sha256": sha256_file(__file__),
        "development_seed": 42, "confirmation_seeds": [43, 44],
        "probe_protocol": EVALUATOR, "datasets": {}, "test_images_accessed": False,
        "selection_rule": "maximum_validation_top1_then_fewer_epochs_then_smaller_resolution_then_candidate_name",
        "environment": environment(),
    }
    for dataset in ("eth80", "freiburg"):
        path = BASELINE / f"{dataset}-seed42-lejepa" / "report.json"
        old = json.loads(path.read_text())
        candidates = [{"candidate": "B0", "epoch": 20, "image_size": 64, "validation_accuracy": old["selected"]["validation_accuracy"], "report": str(path), "report_sha256": sha256_file(path)}]
        for candidate in ("B1", "B2"):
            for epoch in (20, 50, 100):
                report = candidate_validation(dataset, candidate, 42, epoch)
                path = OUT / "clean-validation" / dataset / candidate / "seed42" / f"epoch{epoch:03d}/report.json"
                candidates.append({"candidate": candidate, "epoch": epoch, "image_size": report["image_size"], "validation_accuracy": report["selected"]["validation_accuracy"], "report": str(path), "report_sha256": sha256_file(path)})
        best = min(candidates, key=lambda item: (-item["validation_accuracy"], item["epoch"], item["image_size"], item["candidate"]))
        selection["datasets"][dataset] = {"selected": best, "candidates": candidates, "correction": correction_provenance(dataset, clean_splits(dataset))}
    write_json(CLEAN_LOCK, selection)
    print(json.dumps({"phase": "clean_selection_locked", "sha256": sha256_file(CLEAN_LOCK), "datasets": selection["datasets"]}), flush=True)


def evaluate_test(dataset, seed, candidate, epoch, method, image_size, validation_report, destination, lock_sha):
    """Evaluate fixed heads on official tests and declared pixel sensitivity masks."""
    if destination.exists():
        raise RuntimeError(f"Final evaluation directory already exists: {destination}")
    splits = clean_splits(dataset)
    report = validation_report
    checkpoint, probe_path = Path(report["checkpoint"]), Path(report["probe_path"])
    assert sha256_file(checkpoint) == report["checkpoint_sha256"]
    assert sha256_file(probe_path) == report["probe_sha256"]
    assert report["correction"] == correction_provenance(dataset, splits)
    if method == "random" or candidate == "B0":
        encoder, source = baseline_encoder(ROOT / "outputs/experiments" / f"{dataset}-seed{seed}", seed, method)
        assert checkpoint == source
    else:
        encoder = encoder_from_checkpoint(checkpoint, seed)
    probe = load_probe(probe_path)
    x, labels, cached = extract_features(encoder, splits["test"], image_size=image_size, cache_root=CACHE, encoder_sha256=sha256_file(checkpoint))
    predictions = predict_probe(probe, x)
    classes = np.arange(splits["train"].features["label"].num_classes)
    official = metrics(labels, predictions, classes)
    sensitivity = sensitivity_metrics(dataset, cached["metadata"]["ordered_image_ids"], labels, predictions, classes)
    result = {
        **official, "dataset": dataset, "seed": seed, "candidate": candidate,
        "epoch": epoch, "method": method, "image_size": image_size,
        "selection_lock_sha256": lock_sha, "checkpoint_sha256": report["checkpoint_sha256"],
        "probe_sha256": report["probe_sha256"], "correction": report["correction"],
        "official_test": official, "pixel_sensitivity": sensitivity,
        "feature_cache": cached, "head_refitted": False,
        "test_images_accessed": True,
    }
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "predictions.csv").open("w") as target:
        writer = csv.writer(target)
        writer.writerow(["image_id", "label", "prediction"])
        writer.writerows(zip(cached["metadata"]["ordered_image_ids"], labels.tolist(), predictions.tolist()))
    write_json(destination / "results.json", result)
    print(json.dumps({"phase": "clean_final_test", "dataset": dataset, "seed": seed, "candidate": candidate, "method": method, "official_accuracy": official["accuracy"], "pixel_sensitivity": sensitivity}), flush=True)
    return result


def sensitivity_metrics(dataset, ordered_ids, labels, predictions, classes):
    """Apply model-independent exact-pixel masks to the same official predictions."""
    from pixel_clean import test_sensitivity_masks

    masks = test_sensitivity_masks(dataset, ordered_ids)
    result = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (len(labels),)
        result[name] = metrics(labels[mask], predictions[mask], classes)
    return result


def confirm(jobs):
    """Run confirmation seeds under the corrected lock, then evaluate frozen heads."""
    lock = json.loads(CLEAN_LOCK.read_text())
    lock_sha = sha256_file(CLEAN_LOCK)
    assert lock["protocol_lock_sha256"] == sha256_file(OUT / "protocol-lock.json")
    assert lock["pixel_duplicate_audit_sha256"] == sha256_file(AUDIT)
    assert lock["pixel_clean_source_sha256"] == sha256_file(Path(__file__).with_name("pixel_clean.py"))
    from pixel_clean import POLICY_PATH, ensure_policy
    ensure_policy()
    assert lock["pixel_clean_policy_sha256"] == sha256_file(POLICY_PATH)
    assert lock["coordinator_source_sha256"] == sha256_file(__file__)
    assert json.dumps(lock["probe_protocol"], sort_keys=True) == json.dumps(EVALUATOR, sort_keys=True)
    for item in lock["datasets"].values():
        for candidate in item["candidates"]:
            assert sha256_file(candidate["report"]) == candidate["report_sha256"]
    tasks = [(dataset, item["selected"]["candidate"], seed, item["selected"]["epoch"]) for dataset, item in lock["datasets"].items() for seed in (43, 44) if item["selected"]["candidate"] != "B0"]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda args: train(*args), tasks))
    final = {
        "selection_lock_sha256": lock_sha, "pixel_duplicate_audit_sha256": sha256_file(AUDIT),
        "development_seed_included": 42, "independent_confirmation_seeds": [43, 44],
        "original_test_results_already_known": True, "runs": [],
    }
    # Prepare all confirmation heads from corrected validation first. No test
    # features are read while confirmation heads are being selected.
    heads = []
    for dataset in ("eth80", "freiburg"):
        chosen = lock["datasets"][dataset]["selected"]
        candidate, epoch, size = chosen["candidate"], chosen["epoch"], chosen["image_size"]
        for seed in (42, 43, 44):
            for method in ("random", "lejepa"):
                if method == "lejepa" and candidate != "B0":
                    selected = candidate_validation(dataset, candidate, seed, epoch)
                elif method == "random" and size != 64:
                    source = ROOT / "outputs/experiments" / f"{dataset}-seed{seed}/random-encoder.pt"
                    selected = fit_condition(dataset, seed, method, size, source, OUT / "clean-validation" / dataset / f"random{size}/seed{seed}")
                else:
                    selected = json.loads((BASELINE / f"{dataset}-seed{seed}-{method}/report.json").read_text())
                heads.append((dataset, seed, candidate, epoch, method, size, selected, OUT / "clean-final" / dataset / candidate / f"seed{seed}" / method))
            if candidate != "B0":
                for method in ("random", "lejepa"):
                    baseline = json.loads((BASELINE / f"{dataset}-seed{seed}-{method}/report.json").read_text())
                    heads.append((dataset, seed, "B0", 20, method, 64, baseline, OUT / "clean-final" / dataset / "B0-converged" / f"seed{seed}" / method))
    for values in heads:
        final["runs"].append(evaluate_test(*values, lock_sha))
    assert sha256_file(CLEAN_LOCK) == lock_sha
    final["status"] = "passed"
    write_json(OUT / "clean-final-report.json", final)


def main():
    """Run an explicit corrected-validation phase; test requires its immutable lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "develop", "confirm"), required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    initialize_runtime(4)
    torch.use_deterministic_algorithms(True)
    if args.phase == "baseline":
        baseline_validation()
    elif args.phase == "develop":
        develop(args.jobs)
    else:
        confirm(args.jobs)


if __name__ == "__main__":
    main()
