"""Frozen-feature extraction and validation-selected converged linear evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
EVALUATOR = {"solver": "lbfgs", "tol": 1e-6, "max_iter": 5000, "C": CS, "standardization": "train_mean_sample_std_ddof1_clamp1e-6"}


def sha256_file(path):
    """Hash a local file without changing it."""
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest_json(value):
    """Return a stable digest for experiment metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """Write a JSON artifact with finite numeric values only."""
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class EvaluationTransform:
    """Apply the preserved deterministic resize and normalization."""

    def __init__(self, size):
        self.transform = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def __call__(self, row):
        return {"image": self.transform(row["image"]), "label": row["label"]}


def extract_features(encoder, dataset, *, image_size, cache_root, encoder_sha256, batch_size=64):
    """Extract raw frozen features, keyed by encoder, ordered split and transform."""
    rows = list(dataset.remove_columns("image").with_format("raw"))
    metadata = {
        "encoder_sha256": encoder_sha256,
        "split_sha256": digest_json(rows),
        "ordered_image_ids": [r["image_id"] for r in rows],
        "ordered_labels": [int(r["label"]) for r in rows],
        "transform": {"resize": [image_size, image_size], "mean": [0.5] * 3, "std": [0.5] * 3, "interpolation": "bilinear", "antialias": True},
        "normalization": "raw_encoder_features_no_feature_standardization",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "implementation": "frozen_features_v1",
    }
    key = digest_json(metadata)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / f"{key}.npz"
    sidecar = cached.with_suffix(".json")
    if cached.exists():
        saved = json.loads(sidecar.read_text())
        assert saved["metadata"] == metadata
        assert saved["npz_sha256"] == sha256_file(cached)
        with np.load(cached, allow_pickle=False) as data:
            x, y = data["features"], data["labels"]
        assert np.array_equal(y, metadata["ordered_labels"])
        assert x.shape[0] == len(rows) and np.isfinite(x).all()
        return x, y, {"key": key, "npz": str(cached), "metadata": metadata, "cache_hit": True}
    before = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    encoder = encoder.cpu().eval().requires_grad_(False)
    loader = DataLoader(dataset.with_transform(EvaluationTransform(image_size)), batch_size=batch_size, shuffle=False, num_workers=0)
    start = time.perf_counter()
    features, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            value = encoder(batch["image"])
            assert torch.isfinite(value).all()
            features.append(value.cpu().numpy())
            labels.append(batch["label"].numpy())
    assert all(torch.equal(before[key], value.cpu()) for key, value in encoder.state_dict().items())
    assert all(p.grad is None for p in encoder.parameters())
    x, y = np.concatenate(features), np.concatenate(labels)
    assert np.array_equal(y, metadata["ordered_labels"])
    np.savez_compressed(cached, features=x, labels=y)
    write_json(sidecar, {"metadata": metadata, "npz_sha256": sha256_file(cached), "seconds": time.perf_counter() - start})
    return x, y, {"key": key, "npz": str(cached), "metadata": metadata, "cache_hit": False}


def feature_diagnostics(x):
    """Describe training feature variance and covariance effective rank."""
    centered = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(0)
    covariance = centered.T @ centered / max(1, len(x) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    positive = eigenvalues[eigenvalues > 0]
    probabilities = positive / positive.sum() if len(positive) else positive
    return {
        "dimension": int(x.shape[1]), "samples": int(len(x)),
        "std_min": float(np.std(x, axis=0, ddof=1).min()),
        "std_median": float(np.median(np.std(x, axis=0, ddof=1))),
        "std_max": float(np.std(x, axis=0, ddof=1).max()),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank": float(np.exp(-(probabilities * np.log(probabilities)).sum())) if len(positive) else 0.0,
    }


def predict_probe(probe, x):
    """Predict classes using saved training-only standardization and a frozen head."""
    normalized = (np.asarray(x, dtype=np.float64) - probe["mean"]) / probe["std"]
    scores = normalized @ probe["coef"].T + probe["intercept"]
    if len(probe["classes"]) == 2 and scores.shape[1] == 1:
        return probe["classes"][(scores[:, 0] > 0).astype(int)]
    return probe["classes"][scores.argmax(1)]


def fit_probe(train_x, train_y, validation_x, validation_y, seed=42):
    """Fit heads on training features and select C using validation top-1 only."""
    for values in (train_x, validation_x):
        assert np.isfinite(values).all()
    train_x, validation_x = np.asarray(train_x, dtype=np.float64), np.asarray(validation_x, dtype=np.float64)
    mean = train_x.mean(0)
    std = train_x.std(0, ddof=1).clip(min=1e-6)
    normalized = (train_x - mean) / std
    normalized_validation = (validation_x - mean) / std
    grid, best_score, selected = [], -1.0, None
    start = time.perf_counter()
    for c in CS:
        attempts = []
        for cap in (5000, 20000):
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=cap, tol=1e-6, random_state=seed)
            with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(normalized, train_y)
            messages = [str(w.message) for w in captured]
            convergence = [w for w in captured if issubclass(w.category, ConvergenceWarning)]
            attempts.append({"max_iter": cap, "n_iter": model.n_iter_.tolist(), "warnings": messages})
            if not convergence:
                break
        if convergence:
            raise RuntimeError(f"Logistic regression failed to converge at C={c}: {attempts}")
        train_prediction = model.predict(normalized)
        prediction = model.predict(normalized_validation)
        score = float(accuracy_score(validation_y, prediction))
        probe = {"mean": mean, "std": std, "coef": model.coef_, "intercept": model.intercept_, "classes": model.classes_}
        assert np.array_equal(prediction, predict_probe(probe, validation_x))
        trial = {
            "C": c, "validation_accuracy": score,
            "validation_balanced_accuracy": float(balanced_accuracy_score(validation_y, prediction)),
            "validation_correct": int((prediction == validation_y).sum()), "validation_count": len(validation_y),
            "training_accuracy": float(accuracy_score(train_y, train_prediction)),
            "converged": True, "attempts": attempts,
        }
        grid.append(trial)
        if score > best_score:
            best_score, selected, best_probe = score, trial, probe
    return best_probe, {"protocol": EVALUATOR, "selected": selected, "grid": grid, "training_features": feature_diagnostics(train_x), "seconds": time.perf_counter() - start, "selection": "validation_top1_first_C_on_ties", "test_accessed": False}


def save_probe(path, probe):
    """Save plain numeric head arrays without pickle serialization."""
    np.savez_compressed(path, **probe)


def load_probe(path):
    """Load a locally saved numeric linear head."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def metrics(labels, predictions, classes):
    """Compute top-1, balanced accuracy and full class-wise confusion counts."""
    matrix = confusion_matrix(labels, predictions, labels=classes)
    return {"accuracy": float(accuracy_score(labels, predictions)), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "correct": int((np.asarray(labels) == predictions).sum()), "count": len(labels), "confusion_matrix": matrix.tolist(), "per_class_recall": (np.diag(matrix) / matrix.sum(1).clip(min=1)).tolist()}


def baseline_encoder(run_dir, seed, method):
    """Reconstruct an original saved encoder without loading external weights."""
    path = run_dir / ("random-encoder.pt" if method == "random" else "checkpoint.pt")
    state = torch.load(path, map_location="cpu", weights_only=method == "random")
    if method == "lejepa":
        assert state["epoch"] == 20
        state = {key.removeprefix("backbone."): value for key, value in state["model"].items() if key.startswith("backbone.")}
    torch.manual_seed(seed)
    encoder = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True, drop_path_rate=0.0)
    encoder.load_state_dict(state, strict=True)
    return encoder, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("eth80", "freiburg"), default=["eth80", "freiburg"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    from common import environment, initialize_runtime, load_splits

    initialize_runtime(threads=4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"stage": "B0_converged_probe_validation", "status": "running", "test_accessed": False, "runs": [], "script_sha256": sha256_file(__file__), "environment": environment()}
    try:
        for dataset in args.datasets:
            splits = load_splits(dataset)
            for seed in args.seeds:
                run = PROJECT_ROOT / "outputs/experiments" / f"{dataset}-seed{seed}"
                original = json.loads((run / "splits.json").read_text())
                for split in ("train", "validation"):
                    rows = list(splits[split].remove_columns("image").with_format("raw"))
                    assert [(row["image_id"], row["label"]) for row in rows] == [(row["image_id"], row["label"]) for row in original[split]]
                for method in ("random", "lejepa"):
                    encoder, path = baseline_encoder(run, seed, method)
                    extracted = {split: extract_features(encoder, splits[split], image_size=64, cache_root=args.feature_cache, encoder_sha256=sha256_file(path)) for split in ("train", "validation")}
                    probe, result = fit_probe(extracted["train"][0], extracted["train"][1], extracted["validation"][0], extracted["validation"][1], seed)
                    name = f"{dataset}-seed{seed}-{method}"
                    probe_path = args.output / f"{name}-probe.npz"
                    save_probe(probe_path, probe)
                    result.update(dataset=dataset, seed=seed, encoder=method, encoder_path=str(path), encoder_sha256=sha256_file(path), probe_path=str(probe_path), probe_sha256=sha256_file(probe_path), feature_cache={split: value[2] for split, value in extracted.items()})
                    write_json(args.output / f"{name}.json", result)
                    report["runs"].append(result)
                    print(json.dumps({"dataset": dataset, "seed": seed, "encoder": method, "selected": result["selected"]}), flush=True)
                    write_json(args.output / "validation-report.json", report)
        report["status"] = "passed"
    except Exception as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        write_json(args.output / "validation-report.json", report)


if __name__ == "__main__":
    main()
