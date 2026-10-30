"""Run the bounded study with validation-only development and a locked final test."""

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import timm
import torch

from common import ROOT, environment, initialize_runtime, load_splits, sha256_file, write_json
from evaluate_frozen import EVALUATOR, baseline_encoder, extract_features, fit_probe, load_probe, metrics, predict_probe, save_probe

HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/experiments-v2"
CACHE = OUT / "feature-cache"
BASE_EVAL = OUT / "B0-converged-probe-validation"


def train(dataset, candidate, seed, stop_after=None):
    """Launch a fresh controlled run; never reuse a benchmark's optimized state."""
    directory = OUT / "training" / dataset / candidate / f"seed{seed}"
    if (directory / "training-summary.json").exists():
        summary = json.loads((directory / "training-summary.json").read_text())
        saved = json.loads((directory / "config.json").read_text())
        assert saved["dataset"] == dataset and saved["candidate"] == candidate and saved["seed"] == seed
        assert saved["epochs"] == 100 and saved.get("stop_after") == stop_after
        assert summary["status"] == "completed" and summary["initial_encoder_equal_b0"]
        assert summary["epochs_completed"] == (stop_after or 100)
        assert (directory / f"checkpoint-epoch{(stop_after or 100):03d}.pt").is_file()
        return directory
    if directory.exists():
        raise RuntimeError(f"Incomplete run retained for diagnosis; no silent overwrite/resume: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    log = directory.parent / f"seed{seed}-console.log"
    command = [sys.executable, str(HERE / "run_experiment.py"), "--dataset", dataset, "--candidate", candidate, "--seed", str(seed), "--output", str(directory)]
    if stop_after:
        command += ["--stop-after", str(stop_after)]
    with log.open("x") as target:
        subprocess.run(command, stdout=target, stderr=subprocess.STDOUT, check=True)
    return directory


def encoder_from_checkpoint(path, seed):
    """Load a local training checkpoint into the same frozen timm architecture."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("backbone."): value for key, value in saved["model"].items() if key.startswith("backbone.")}
    torch.manual_seed(seed)
    encoder = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True, drop_path_rate=0.)
    encoder.load_state_dict(state, strict=True)
    return encoder


def validation(dataset, candidate, seed, epoch):
    """Fit a train-only head and select its C using validation labels only."""
    directory = OUT / "validation" / dataset / candidate / f"seed{seed}" / f"epoch{epoch:03d}"
    if (directory / "report.json").exists():
        return json.loads((directory / "report.json").read_text())
    directory.mkdir(parents=True, exist_ok=False)
    run = OUT / "training" / dataset / candidate / f"seed{seed}"
    checkpoint = run / f"checkpoint-epoch{epoch:03d}.pt"
    config = json.loads((run / "config.json").read_text())
    splits = load_splits(dataset)
    encoder = encoder_from_checkpoint(checkpoint, seed)
    features = {split: extract_features(encoder, splits[split], image_size=config["image_size"], cache_root=CACHE, encoder_sha256=sha256_file(checkpoint)) for split in ("train", "validation")}
    probe, report = fit_probe(features["train"][0], features["train"][1], features["validation"][0], features["validation"][1], seed)
    save_probe(directory / "probe.npz", probe)
    report.update(dataset=dataset, candidate=candidate, seed=seed, epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), probe_path=str(directory / "probe.npz"), image_size=config["image_size"], feature_cache={s: f[2] for s, f in features.items()})
    write_json(directory / "report.json", report)
    print(json.dumps({"stage": "validation", "dataset": dataset, "candidate": candidate, "epoch": epoch, "seed": seed, "selected": report["selected"]}), flush=True)
    return report


def develop(jobs):
    """Train four development runs and lock the predeclared selection rule."""
    baseline = json.loads((BASE_EVAL / "validation-report.json").read_text())
    assert baseline["status"] == "passed" and not baseline["test_accessed"]
    tasks = [(dataset, candidate, 42) for candidate in ("B1", "B2") for dataset in ("eth80", "freiburg")]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda args: train(*args), tasks))
    selection = {"protocol_lock_sha256": sha256_file(OUT / "protocol-lock.json"), "development_seed": 42, "confirmation_seeds": [43, 44], "probe_protocol": EVALUATOR, "datasets": {}, "test_accessed": False, "environment": environment()}
    for dataset in ("eth80", "freiburg"):
        old = next(r for r in baseline["runs"] if r["dataset"] == dataset and r["seed"] == 42 and r["encoder"] == "lejepa")
        candidates = [{"candidate": "B0", "epoch": 20, "image_size": 64, "validation_accuracy": old["selected"]["validation_accuracy"], "report": str(BASE_EVAL / f"{dataset}-seed42-lejepa.json")}]
        for candidate in ("B1", "B2"):
            for epoch in (20, 50, 100):
                report = validation(dataset, candidate, 42, epoch)
                candidates.append({"candidate": candidate, "epoch": epoch, "image_size": report["image_size"], "validation_accuracy": report["selected"]["validation_accuracy"], "report": str(OUT / "validation" / dataset / candidate / "seed42" / f"epoch{epoch:03d}/report.json")})
        best = min(candidates, key=lambda c: (-c["validation_accuracy"], c["epoch"], c["image_size"], c["candidate"]))
        selection["datasets"][dataset] = {"selected": best, "candidates": candidates}
    lock = OUT / "selection-lock.json"
    if lock.exists():
        raise RuntimeError("Selection was already locked; no silent reselection is allowed.")
    write_json(lock, selection)
    print(json.dumps({"stage": "selection_locked", "datasets": selection["datasets"], "sha256": sha256_file(lock)}), flush=True)


def final_test(dataset, seed, candidate, epoch, method, image_size, probe_path, checkpoint, destination, lock_sha):
    """Evaluate a frozen selected head once, with selection provenance attached."""
    destination.mkdir(parents=True, exist_ok=False)
    if candidate == "B0" or method == "random":
        encoder, source = baseline_encoder(ROOT / "outputs/experiments" / f"{dataset}-seed{seed}", seed, method)
    else:
        source = checkpoint
        encoder = encoder_from_checkpoint(checkpoint, seed)
    probe = load_probe(probe_path)
    splits = load_splits(dataset)
    x, labels, cache = extract_features(encoder, splits["test"], image_size=image_size, cache_root=CACHE, encoder_sha256=sha256_file(source))
    predictions = predict_probe(probe, x)
    report = metrics(labels, predictions, np.arange(splits["train"].features["label"].num_classes))
    report.update(dataset=dataset, seed=seed, candidate=candidate, epoch=epoch, method=method, selection_lock_sha256=lock_sha, encoder_sha256=sha256_file(source), probe_sha256=sha256_file(probe_path), image_size=image_size, feature_cache=cache)
    with (destination / "predictions.csv").open("w") as target:
        writer = csv.writer(target)
        writer.writerow(["image_id", "label", "prediction"])
        writer.writerows(zip(cache["metadata"]["ordered_image_ids"], labels.tolist(), predictions.tolist()))
    write_json(destination / "results.json", report)
    print(json.dumps({"stage": "final_test", "dataset": dataset, "seed": seed, "candidate": candidate, "method": method, "accuracy": report["accuracy"]}), flush=True)
    return report


def confirm(jobs):
    """Run confirmation seeds with fixed selected recipe, then report final tests."""
    if (OUT / "data-audit-test-hold.json").exists():
        raise RuntimeError("Original test gate closed after exact-pixel duplicate audit; use run_clean_study.py.")
    lock_path = OUT / "selection-lock.json"
    lock = json.loads(lock_path.read_text())
    lock_sha = sha256_file(lock_path)
    assert lock["protocol_lock_sha256"] == sha256_file(OUT / "protocol-lock.json")
    assert json.dumps(lock["probe_protocol"], sort_keys=True) == json.dumps(EVALUATOR, sort_keys=True)
    tasks = [(dataset, item["selected"]["candidate"], seed, item["selected"]["epoch"]) for dataset, item in lock["datasets"].items() for seed in (43, 44) if item["selected"]["candidate"] != "B0"]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda args: train(*args), tasks))
    report = {"selection_lock_sha256": lock_sha, "runs": [], "development_seed_included": 42, "independent_confirmation_seeds": [43, 44], "original_test_results_already_known": True}
    for dataset in ("eth80", "freiburg"):
        chosen = lock["datasets"][dataset]["selected"]
        candidate, epoch, size = chosen["candidate"], chosen["epoch"], chosen["image_size"]
        for seed in (42, 43, 44):
            for method in ("random", "lejepa"):
                path = BASE_EVAL / f"{dataset}-seed{seed}-{method}-probe.npz"
                checkpoint = None
                if candidate != "B0" and method == "lejepa":
                    val = validation(dataset, candidate, seed, epoch)
                    path, checkpoint = Path(val["probe_path"]), Path(val["checkpoint"])
                elif method == "random" and size != 64:
                    random_dir = OUT / "validation" / dataset / f"random{size}" / f"seed{seed}"
                    random_dir.mkdir(parents=True, exist_ok=False)
                    splits = load_splits(dataset)
                    encoder, source = baseline_encoder(ROOT / "outputs/experiments" / f"{dataset}-seed{seed}", seed, "random")
                    f = {s: extract_features(encoder, splits[s], image_size=size, cache_root=CACHE, encoder_sha256=sha256_file(source)) for s in ("train", "validation")}
                    head, vr = fit_probe(f["train"][0], f["train"][1], f["validation"][0], f["validation"][1], seed)
                    path = random_dir / "probe.npz"
                    save_probe(path, head)
                    write_json(random_dir / "report.json", vr)
                destination = OUT / "final" / dataset / candidate / f"seed{seed}" / method
                result = final_test(dataset, seed, candidate, epoch, method, size, path, checkpoint, destination, lock_sha)
                report["runs"].append(result)
            if candidate != "B0":
                for method in ("random", "lejepa"):
                    destination = OUT / "final" / dataset / "B0-converged" / f"seed{seed}" / method
                    path = BASE_EVAL / f"{dataset}-seed{seed}-{method}-probe.npz"
                    report["runs"].append(final_test(dataset, seed, "B0", 20, method, 64, path, None, destination, lock_sha))
    assert sha256_file(lock_path) == lock_sha
    report["status"] = "passed"
    write_json(OUT / "final-report.json", report)


def main():
    """Run one explicit phase; test evaluation requires the saved selection lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("develop", "confirm"), required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    initialize_runtime(4)
    torch.use_deterministic_algorithms(True)
    if args.phase == "develop":
        develop(args.jobs)
    else:
        confirm(args.jobs)


if __name__ == "__main__":
    main()
