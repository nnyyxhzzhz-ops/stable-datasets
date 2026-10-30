"""Shared provenance, cache-only data access and network guard for experiments."""

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import requests
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT
OUTPUT_ROOT = ROOT / "outputs/experiments-v2"
REPO = ROOT / "outputs/stable-datasets"
BASELINE_ROOT = ROOT / "outputs/experiments"
SPT_REVISION = "ab836bf699a2be712dfcf980a5eb70d36391e876"
sys.path.insert(0, str(REPO))


def _no_network(*args, **kwargs):
    raise RuntimeError("New downloads are not authorized; this experiment uses existing local data only.")


_original_connect = socket.socket.connect


def _local_ipc_only(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        return _no_network(self, address)
    return _original_connect(self, address)


socket.socket.connect = _local_ipc_only
requests.sessions.Session.request = _no_network
urllib.request.urlopen = _no_network
os.environ["STABLE_DATASETS_CACHE_DIR"] = str(ROOT / "work/dataset-cache")
for _key in ("CI", "GITHUB_ACTIONS", "STABLE_DATASETS_SKIP_UNREACHABLE_HOSTS"):
    os.environ.pop(_key, None)


def sha256_file(path):
    """Hash a local file without loading its full contents into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


file_sha256 = sha256_file


def write_json(path, value):
    """Atomically write an English JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize_runtime(threads=4):
    """Limit CPU parallelism; every experiment runs offline."""
    torch.set_num_threads(threads)


def _git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def environment():
    """Record software, source versions and actual device availability."""
    return {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "timm", "numpy", "pyarrow", "scikit-learn", "stable-pretraining", "stable-datasets")},
        "dataset_commit": _git(REPO, "rev-parse", "HEAD"),
        "stable_pretraining_commit": _git(ROOT / "work/stable-pretraining", "rev-parse", "HEAD"),
        "network": "blocked", "source_files_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
    }


def split_manifest(splits):
    """Get ordered sample metadata without decoding image pixels."""
    result = {}
    for name, dataset in splits.items():
        result[name] = [
            {"image_id": row["image_id"], "label": row["label"], **({"object_id": row["object_id"]} if "object_id" in row else {})}
            for row in dataset.remove_columns("image").with_format("raw")
        ]
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not {r["image_id"] for r in result[a]} & {r["image_id"] for r in result[b]}
        if "object_id" in result[a][0]:
            assert not {r["object_id"] for r in result[a]} & {r["object_id"] for r in result[b]}
    return result


def load_splits(dataset, config=None):
    """Read revised loaders' existing caches and reproduce the original holdouts."""
    from sklearn.model_selection import train_test_split
    from stable_datasets.cache import cache_fingerprint
    from stable_datasets.images import ETH80, FreiburgGroceries

    if dataset not in ("eth80", "freiburg"):
        raise ValueError(f"Unknown dataset: {dataset}")
    if config and (config.get("freiburg_config", "split0") != "split0" or config.get("split_seed", 42) != 42 or config.get("validation_fraction", .2) != .2):
        raise ValueError("The bounded study preserves the original split0 / split seed 42 / 20% validation protocol.")
    eth = dataset == "eth80"
    builder = ETH80 if eth else FreiburgGroceries
    processed = ROOT / "work/pytest-round1-py312" / ("test_eth80_real_dataset0" if eth else "test_freiburg_released_folds0") / "processed"
    config_name = "default" if eth else "split0"
    for split in (("train", "validation", "test") if eth else ("train", "test")):
        metadata = processed / cache_fingerprint(builder.__name__, str(builder.VERSION), config_name, split) / "_metadata.json"
        if not metadata.is_file():
            raise RuntimeError(f"Approved processed cache missing; no automatic download/rebuild: {metadata}")
    official = builder(download_dir=ROOT / "work/data", processed_cache_dir=processed)
    if eth:
        result = official
    else:
        train = official["train"]
        labels = train.table.column("label").to_pylist()
        first = {}
        for index, image_id in enumerate(train.table.column("image_id").to_pylist()):
            first.setdefault(image_id, index)
        indices = [first[key] for key in sorted(first)]
        fit, val = train_test_split(indices, test_size=.2, random_state=42, stratify=[labels[i] for i in indices])
        result = {"train": train.select(sorted(fit)), "validation": train.select(sorted(val)), "test": official["test"]}
    manifest = split_manifest(result)
    original = json.loads((BASELINE_ROOT / f"{dataset}-seed42/splits.json").read_text())
    assert manifest == original, "Experiment sample membership/order changed."
    return result


load_data = load_splits
