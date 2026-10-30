"""Cache-only ImageNet-supervised ResNet-18 reference with locked test evaluation.

This command never downloads weights. Validation and test are separate phases;
test requires the SHA-256 of an already saved validation selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from common import (
    environment,
    initialize_runtime,
    load_splits,
    sha256_file,
    split_manifest,
    write_json,
)
from evaluate_frozen import (
    EVALUATOR,
    digest_json,
    fit_probe,
    load_probe,
    metrics,
    predict_probe,
    save_probe,
)


PRETRAINING = {
    "method": "supervised",
    "dataset": "ImageNet-1K",
    "weights": "torchvision.ResNet18_Weights.IMAGENET1K_V1",
    "external_pretraining_data": True,
    "is_LeJEPA": False,
    "direct_causal_comparison_with_ViT_SSL": False,
    "source_dataset_overlap_independently_audited": False,
}


def transform_metadata():
    """Describe the official weights' actual preprocessing without downloading."""
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return {
        "weights_transform": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
        "resize_size": transform.resize_size,
        "crop_size": transform.crop_size,
        "mean": transform.mean,
        "std": transform.std,
        "interpolation": str(transform.interpolation),
        "antialias": transform.antialias,
    }


class OfficialTransform:
    """Apply the reference weights' official deterministic preprocessing."""

    def __init__(self):
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def __call__(self, row):
        return {"image": self.transform(row["image"]), "label": row["label"]}


def verified_weights_path(path, expected_sha256):
    """Require an existing local file and its full explicitly supplied checksum."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pretrained weights are not cached at {path}; automatic downloads are disabled."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(
            "Supply the expected full lowercase SHA-256 for the approved local weight file."
        )
    official_url = ResNet18_Weights.IMAGENET1K_V1.url
    expected_prefix = Path(urlparse(official_url).path).stem.rsplit("-", 1)[1]
    actual = sha256_file(path)
    if actual != expected_sha256 or not actual.startswith(expected_prefix):
        raise ValueError(
            "Pretrained weight checksum does not match the supplied full digest and official filename hash."
        )
    return path, actual


def load_encoder(path, expected_sha256):
    """Strictly load approved local weights, then remove the ImageNet classifier."""
    path, actual = verified_weights_path(path, expected_sha256)
    encoder = resnet18(weights=None)
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    encoder.fc = nn.Identity()
    return encoder.eval().requires_grad_(False), actual


def extract_reference_features(
    encoder, dataset, cache_root, weight_sha256, batch_size=32
):
    """Extract frozen features with a checksum-bound official-transform cache."""
    rows = list(dataset.remove_columns("image").with_format("raw"))
    metadata = {
        "weight_sha256": weight_sha256,
        "split_sha256": digest_json(rows),
        "ordered_image_ids": [row["image_id"] for row in rows],
        "ordered_labels": [int(row["label"]) for row in rows],
        "pretraining": PRETRAINING,
        "transform": transform_metadata(),
        "features": "resnet18_avgpool_512_before_fc_no_feature_standardization",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "implementation": "pretrained_reference_features_v1",
    }
    key = digest_json(metadata)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"{key}.npz"
    sidecar = cache.with_suffix(".json")
    if cache.exists():
        saved = json.loads(sidecar.read_text())
        assert saved["metadata"] == metadata and saved["npz_sha256"] == sha256_file(
            cache
        )
        with np.load(cache, allow_pickle=False) as source:
            features, labels = source["features"], source["labels"]
        assert features.shape == (len(rows), 512) and np.isfinite(features).all()
        assert np.array_equal(labels, metadata["ordered_labels"])
        return (
            features,
            labels,
            {"key": key, "metadata": metadata, "npz": str(cache), "cache_hit": True},
        )
    before = {
        name: value.detach().cpu().clone()
        for name, value in encoder.state_dict().items()
    }
    encoder = encoder.cpu().eval().requires_grad_(False)
    features, labels = [], []
    started = time.perf_counter()
    loader = DataLoader(
        dataset.with_transform(OfficialTransform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch in loader:
            value = encoder(batch["image"])
            assert value.shape[1] == 512 and torch.isfinite(value).all()
            features.append(value.numpy())
            labels.append(batch["label"].numpy())
    assert all(
        torch.equal(before[name], value.cpu())
        for name, value in encoder.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in encoder.parameters())
    features, labels = np.concatenate(features), np.concatenate(labels)
    assert np.array_equal(labels, metadata["ordered_labels"])
    np.savez_compressed(cache, features=features, labels=labels)
    write_json(
        sidecar,
        {
            "metadata": metadata,
            "npz_sha256": sha256_file(cache),
            "seconds": time.perf_counter() - started,
        },
    )
    return (
        features,
        labels,
        {"key": key, "metadata": metadata, "npz": str(cache), "cache_hit": False},
    )


def validate(args, encoder, weights_sha256, splits):
    """Fit train-only heads and select regularization using validation data."""
    features = {
        split: extract_reference_features(
            encoder, splits[split], args.feature_cache, weights_sha256
        )
        for split in ("train", "validation")
    }
    probe, report = fit_probe(
        *features["train"][:2], *features["validation"][:2], seed=42
    )
    probe_path = args.output / "probe.npz"
    save_probe(probe_path, probe)
    selection = {
        "phase": "validate",
        "dataset": args.dataset,
        "weights_sha256": weights_sha256,
        "official_weights_url": ResNet18_Weights.IMAGENET1K_V1.url,
        "pretraining": PRETRAINING,
        "transform": transform_metadata(),
        "probe_path": str(probe_path.resolve()),
        "probe_sha256": sha256_file(probe_path),
        "evaluation": report,
        "feature_cache": {split: item[2] for split, item in features.items()},
        "split_manifest_sha256": digest_json(split_manifest(splits)),
        "source_sha256": sha256_file(__file__),
        "evaluator_sha256": sha256_file(Path(__file__).with_name("evaluate_frozen.py")),
        "test_images_accessed": False,
        "environment": environment(),
    }
    write_json(args.output / "selection.json", selection)
    return {
        "phase": "validate",
        "selected": report["selected"],
        "test_images_accessed": False,
    }


def test_selected(args, encoder, weights_sha256, splits):
    """Evaluate the checksum-locked head without refitting or parameter search."""
    if sha256_file(args.selection) != args.selection_sha256:
        raise ValueError("Validation selection checksum differs from the locked value.")
    selection = json.loads(args.selection.read_text())
    assert selection["phase"] == "validate" and selection["dataset"] == args.dataset
    assert selection["weights_sha256"] == weights_sha256
    assert (
        selection["pretraining"] == PRETRAINING
        and selection["transform"] == transform_metadata()
    )
    assert selection["evaluation"]["protocol"] == json.loads(json.dumps(EVALUATOR))
    assert selection["source_sha256"] == sha256_file(__file__)
    assert selection["evaluator_sha256"] == sha256_file(
        Path(__file__).with_name("evaluate_frozen.py")
    )
    assert selection["split_manifest_sha256"] == digest_json(split_manifest(splits))
    probe_path = Path(selection["probe_path"])
    if sha256_file(probe_path) != selection["probe_sha256"]:
        raise ValueError(
            "Saved probe checksum differs from the validation-selected head."
        )
    probe = load_probe(probe_path)
    features, labels, cached = extract_reference_features(
        encoder, splits["test"], args.feature_cache, weights_sha256
    )
    predictions = predict_probe(probe, features)
    result = metrics(labels, predictions, probe["classes"])
    result.update(
        phase="test",
        dataset=args.dataset,
        pretraining=PRETRAINING,
        selection_sha256=args.selection_sha256,
        weights_sha256=weights_sha256,
        feature_cache=cached,
        head_refitted=False,
        test_images_accessed=True,
    )
    with (args.output / "test-predictions.csv").open("w") as target:
        writer = csv.writer(target)
        writer.writerow(("image_id", "label", "prediction"))
        for image_id, label, prediction in zip(
            cached["metadata"]["ordered_image_ids"], labels, predictions
        ):
            writer.writerow((image_id, int(label), int(prediction)))
    write_json(args.output / "test-results.json", result)
    return result


def main():
    """Require an approved local weight file and a distinct validation/test phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--dataset", choices=("eth80", "freiburg"), required=True)
    parser.add_argument("--phase", choices=("validate", "test"), required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    args = parser.parse_args()
    if args.phase == "test" and (
        args.selection is None or args.selection_sha256 is None
    ):
        parser.error(
            "test requires --selection and its externally locked --selection-sha256"
        )
    initialize_runtime(threads=4)
    torch.use_deterministic_algorithms(True)
    # Fail on a missing file before creating outputs or loading datasets.
    encoder, weights_sha256 = load_encoder(args.weights, args.weights_sha256)
    splits = load_splits(args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    result = (
        validate(args, encoder, weights_sha256, splits)
        if args.phase == "validate"
        else test_selected(args, encoder, weights_sha256, splits)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
