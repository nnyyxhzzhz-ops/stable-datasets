"""Record baseline integrity before follow-up work in a fresh reproduction."""

import json

from common import BASELINE_ROOT, OUTPUT_ROOT, sha256_file, write_json


def main():
    """Snapshot newly reproduced runs, never replace an existing reference."""
    destination = OUTPUT_ROOT / "baseline-integrity-before.json"
    if destination.exists():
        raise FileExistsError("An integrity reference already exists; it must not be replaced.")
    for dataset in ("eth80", "freiburg"):
        for seed in (42, 43, 44):
            folder = BASELINE_ROOT / f"{dataset}-seed{seed}"
            for name in ("checkpoint.pt", "random-encoder.pt", "config.json", "splits.json", "results.json", "training.json"):
                if not (folder / name).is_file():
                    raise FileNotFoundError(folder / name)
            config = json.loads((folder / "config.json").read_text())
            assert config["dataset"] == dataset and config["seed"] == seed
            assert config["epochs"] == 20 and config["max_batches"] == 0 and not config["skip_probe"]
    records = {str(p.relative_to(BASELINE_ROOT)): {"sha256": sha256_file(p), "size": p.stat().st_size} for p in sorted(BASELINE_ROOT.rglob("*")) if p.is_file()}
    write_json(destination, records)
    write_json(OUTPUT_ROOT / "baseline-integrity-profile.json", {"file_count": len(records), "purpose": "Fresh independent reproduction; locally generated baseline evidence."})
    print(f"Recorded {len(records)} baseline files.")


if __name__ == "__main__":
    main()
