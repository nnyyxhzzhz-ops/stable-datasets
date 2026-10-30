"""Prepare the documented project cache layout; downloads require an explicit flag."""

import argparse
import hashlib
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/stable-datasets"))


def main():
    """Build caches from local archives or explicitly enable source downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-data", action="store_true", help="Allow the builders to fetch missing official source archives.")
    parser.add_argument("--check-only", action="store_true", help="Check the experiment cache layout without rebuilding or downloading.")
    args = parser.parse_args()
    os.environ.pop("CI", None)
    os.environ.pop("GITHUB_ACTIONS", None)
    if args.check_only:
        from common import load_splits
        for name in ("eth80", "freiburg"):
            data = load_splits(name)
            print(name, {s: len(d) for s, d in data.items()})
        return
    if not args.download_data:
        original_connect = socket.socket.connect

        def offline(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError("Missing local source archive; downloads require --download-data.")
            return original_connect(self, address)

        socket.socket.connect = offline
    from stable_datasets.images import ETH80, FreiburgGroceries
    for name, builder, folder in (("eth80", ETH80, "test_eth80_real_dataset0"), ("freiburg", FreiburgGroceries, "test_freiburg_released_folds0")):
        processed = ROOT / "work/pytest-round1-py312" / folder / "processed"
        data = builder(download_dir=ROOT / "work/data", processed_cache_dir=processed)
        print(builder.__name__, {s: len(d) for s, d in data.items()})
        cache = ROOT / "work/baseline-cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        for link, target in ((cache / "downloads", ROOT / "work/data"), (cache / "processed", processed)):
            if link.exists() or link.is_symlink():
                if link.resolve() != target.resolve():
                    raise RuntimeError(f"Refusing to replace an existing cache entry: {link}")
            else:
                link.symlink_to(target, target_is_directory=True)
    aliases = {
        "eth80-cropped256.tgz": ("eth80-cropped256.*.tgz", ETH80.SOURCE.assets["train"].checksum),
        "freiburg_groceries_dataset.tar.gz": ("freiburg_groceries_dataset.tar.*.gz", FreiburgGroceries.SOURCE.assets["images"].checksum),
    }
    for alias, (pattern, expected) in aliases.items():
        target = ROOT / "work/data" / alias
        if target.exists():
            continue
        candidates = [p for p in (ROOT / "work/data").glob(pattern) if hashlib.file_digest(p.open("rb"), "sha256").hexdigest() == expected.split(":", 1)[1]]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot identify a unique checksum-verified archive for {alias}")
        target.hardlink_to(candidates[0])


if __name__ == "__main__":
    main()
