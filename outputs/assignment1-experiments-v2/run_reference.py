"""Run supervised-reference validation, lock both heads, then separately test."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import OUTPUT_ROOT, sha256_file, write_json
from evaluate_frozen import EVALUATOR
from pixel_clean import POLICY_PATH, ensure_policy


def main():
    """Execute one explicit reference phase without any weight download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--phase", choices=("validate", "test"), required=True)
    args = parser.parse_args()
    ensure_policy()
    if sha256_file(args.weights) != args.weights_sha256:
        raise ValueError("Weight file checksum differs from the declared digest.")
    root = OUTPUT_ROOT / "pretrained-reference"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "clean-selection-lock.json"
    lock = {"policy_sha256": sha256_file(POLICY_PATH), "weights_sha256": args.weights_sha256,
            "evaluation": json.loads(json.dumps(EVALUATOR)), "original_test_scores_already_known": True,
            "reference_selections": {}}
    for dataset in ("eth80", "freiburg"):
        selection = root / dataset / "clean-validation/selection.json"
        command = [sys.executable, str(Path(__file__).with_name("run_pretrained_reference_clean.py")),
                   "--weights", str(args.weights), "--weights-sha256", args.weights_sha256,
                   "--dataset", dataset, "--feature-cache", str(root / "feature-cache")]
        if args.phase == "validate":
            if not selection.exists():
                subprocess.run(command + ["--phase", "validate", "--output", str(selection.parent)], check=True)
            saved = json.loads(selection.read_text())
            assert saved["pixel_clean_policy_sha256"] == lock["policy_sha256"]
            assert not saved["test_images_accessed"]
            assert sha256_file(saved["probe_path"]) == saved["probe_sha256"]
            lock["reference_selections"][dataset] = {"selection_sha256": sha256_file(selection), "probe_sha256": saved["probe_sha256"]}
        else:
            saved_lock = json.loads(lock_path.read_text())
            subprocess.run(command + ["--phase", "test", "--output", str(root / dataset / "clean-test"),
                                      "--selection", str(selection), "--selection-sha256", saved_lock["reference_selections"][dataset]["selection_sha256"],
                                      "--lock", str(lock_path), "--lock-sha256", sha256_file(lock_path)], check=True)
    if args.phase == "validate":
        if lock_path.exists():
            assert json.loads(lock_path.read_text()) == lock, "Refusing to replace a different reference selection."
        else:
            write_json(lock_path, lock)
        print("Locked reference heads:", sha256_file(lock_path))


if __name__ == "__main__":
    main()
