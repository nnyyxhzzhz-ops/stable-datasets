"""Render audited measurements for the corrected follow-up experiment."""

import json
import statistics

from common import OUTPUT_ROOT, ROOT


def summary(values):
    """Format a mean and sample standard deviation in percentage units."""
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main():
    """Generate the report only after all selected experiments pass their audit."""
    audit = json.loads((OUTPUT_ROOT / "clean-final-audit.json").read_text())
    assert audit["status"] == "passed"
    final = json.loads((OUTPUT_ROOT / "clean-final-report.json").read_text())
    lock = json.loads((OUTPUT_ROOT / "clean-selection-lock.json").read_text())
    lines = ["# Controlled LeJEPA follow-up results", "", f"All reported measurements come from saved configurations, checkpoints, validation traces and prediction files. The six original B0 experiments and their {audit['original_baseline_files_unchanged']} artifacts remain unchanged.", "", "## Data and corrected evaluation", "", "ETH-80 uses 1,968/656/656 instance-disjoint training/validation/test images. Freiburg retains all 3,143 SSL training images and the authors' 1,018 test paths. The validation set excludes one image whose decoded RGB pixels also occur in training, leaving 785 validation images. The loader and author membership are unchanged; this is an experiment-level correction.", "", "The source-pixel audit found no ETH-80 duplicates. Freiburg has 14 identical-RGB pairs under different paths, with no conflicting labels: five train/test pairs, three validation/test pairs, one train/validation pair, three within-train pairs and two within-test pairs. The final report gives official test accuracy alongside a 1,010-image unseen-pixel sensitivity subset and a 1,008-image unique-unseen subset. Subsets were fixed from source pixels before the new test predictions were computed.", "", "## Protocol and validation selection", "", "The original B0 uses 20 SSL epochs and 64/32-pixel global/local views. B1 declares 100 epochs with those views; B2 declares 100 epochs and 128/64-pixel views. All use the unchanged pinned LeJEPA objective, the same ViT-Tiny/16 architecture and exact original initial encoder tensors. New comparisons use frozen encoders and a converged multinomial logistic-regression head: C in [0.01,0.1,1,10,100], train-only standardization, validation-only selection, and smaller C on ties.", "", "Each dataset's candidate and checkpoint were selected using seed42 validation top-1; ties prefer earlier epochs and lower resolution. Confirmation seeds43/44 use the fixed recipe and checkpoint epoch. Seed42 is a development seed, so three-seed aggregates include development selection; the two confirmation seeds are listed individually. Original test scores were already known, and these follow-ups continue on the same retained holdouts.", "", "| Dataset | Selected candidate | Epoch | Global/local views | Seed42 validation (%) |", "| --- | --- | ---: | --- | ---: |"]
    for dataset, item in lock["datasets"].items():
        c = item["selected"]
        lines.append(f"| {dataset} | {c['candidate']} | {c['epoch']} | {c['image_size']}/{c['image_size']//2} | {100*c['validation_accuracy']:.2f} |")
    lines += ["", "## All development candidates", "", "No candidate was chosen using test scores. Early checkpoints belong to their declared 100-epoch cosine schedule; comparisons with B0 therefore include learning-rate-trajectory differences, and cannot always be interpreted solely as longer training.", "", "| Dataset | Candidate | Epoch | Validation (%) |", "| --- | --- | ---: | ---: |"]
    for dataset, item in lock["datasets"].items():
        for c in item["candidates"]:
            lines.append(f"| {dataset} | {c['candidate']} | {c['epoch']} | {100*c['validation_accuracy']:.2f} |")
    lines += ["", "## Three-seed test results", "", "Mean ± sample standard deviation, percentage units. Original B0/AdamW results used the historical validation membership and evaluator; B0/converged results use corrected validation. The selected comparison uses that same corrected evaluator and matched-resolution random encoder.", "", "| Dataset | Condition | Encoder | Official test (%) | Balanced (%) | Unseen-pixel test (%) | Unique-unseen test (%) |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for dataset in ("eth80", "freiburg"):
        for method in ("random", "lejepa"):
            old = [json.loads((ROOT / f"outputs/experiments/{dataset}-seed{seed}/results.json").read_text())[method] for seed in (42,43,44)]
            lines.append(f"| {dataset} | original B0 / AdamW | {method} | {summary([100*r['test_accuracy'] for r in old])} | {summary([100*r['test_balanced_accuracy'] for r in old])} | not evaluated | not evaluated |")
        for condition in sorted({r["candidate"] for r in final["runs"] if r["dataset"] == dataset}):
            for method in ("random", "lejepa"):
                rows = [r for r in final["runs"] if r["dataset"] == dataset and r["candidate"] == condition and r["method"] == method]
                assert len(rows) == 3
                lines.append(f"| {dataset} | {condition} / converged | {method} | {summary([100*r['accuracy'] for r in rows])} | {summary([100*r['balanced_accuracy'] for r in rows])} | {summary([100*r['pixel_sensitivity']['unseen_pixels']['accuracy'] for r in rows])} | {summary([100*r['pixel_sensitivity']['unique_unseen_pixels']['accuracy'] for r in rows])} |")
    lines += ["", "## Per-seed results", "", "| Dataset | Condition | Seed | Encoder | Clean validation (%) | Correct/official test | Official test (%) | Unseen-pixel test (%) |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for r in final["runs"]:
        if r["candidate"] == "B0" or (r["method"] == "random" and r["image_size"] == 64):
            path = OUTPUT_ROOT / f"clean-B0-validation/{r['dataset']}-seed{r['seed']}-{r['method']}/report.json"
        elif r["method"] == "random":
            path = OUTPUT_ROOT / f"clean-validation/{r['dataset']}/random{r['image_size']}/seed{r['seed']}/report.json"
        else:
            path = OUTPUT_ROOT / f"clean-validation/{r['dataset']}/{r['candidate']}/seed{r['seed']}/epoch{r['epoch']:03d}/report.json"
        v = json.loads(path.read_text())["selected"]
        lines.append(f"| {r['dataset']} | {r['candidate']} | {r['seed']} | {r['method']} | {100*v['validation_accuracy']:.2f} | {r['correct']}/{r['count']} | {100*r['accuracy']:.2f} | {100*r['pixel_sensitivity']['unseen_pixels']['accuracy']:.2f} |")
    reference = OUTPUT_ROOT / "pretrained-reference/clean-audit.json"
    if reference.is_file():
        ref = json.loads(reference.read_text())
        assert ref["status"] == "passed"
        lines += ["", "## Separate ImageNet-supervised reference", "", "The official ResNet-18 weights were frozen, using their documented 224-pixel preprocessing. A train-only logistic-regression head was selected on corrected validation. This is a deterministic external-supervised-pretraining reference, not a LeJEPA result or a same-architecture training-method comparison. Source-data overlap with ImageNet was not independently audited.", "", "| Dataset | Validation (%) | Correct/official test | Official test (%) | Unseen-pixel test (%) |", "| --- | ---: | ---: | ---: | ---: |"]
        for dataset, r in ref["datasets"].items():
            lines.append(f"| {dataset} | {100*r['validation_accuracy']:.2f} | {r['correct']}/{r['count']} | {100*r['accuracy']:.2f} | {100*r['sensitivity']['unseen_pixels']['accuracy']:.2f} |")
    lines += ["", "## Verification and limits", "", "The audit verifies unchanged original artifacts, exact B0 initialization, identical SSL training membership, declared optimizer-step counts, finite checkpoints, immutable candidate selection, saved numeric-head replay, and all CSV-derived metrics and sensitivity subsets. No new source datasets were downloaded. The optional pretrained reference used one official weight download; SSL runs remain from scratch and offline.", "", "These experiments do not reproduce the original paper's full benchmark protocol, establish statistical significance, or establish a general world-model guarantee. The B2 comparison changes global, local and evaluation resolutions jointly. One development seed, two confirmation seeds and one partition limit generalization claims. Different Freiburg physical-product instances are not annotated. Within-training exact duplicates were retained to preserve the controlled SSL input comparison; original official test duplicate counts are disclosed rather than hidden.", "", "## Artifacts", "", "- protocol-lock.json, pixel-clean-protocol.json and clean-selection-lock.json: recorded decisions and source-pixel correction.", "- training/: initial weights, complete epoch logs, configurations, software/source versions and checkpoints.", "- clean-B0-validation/ and clean-validation/: all regularization trials, convergence traces and heads.", "- clean-final/: per-image predictions, confusion matrices, per-class recall and subset metrics.", "- clean-final-report.json and clean-final-audit.json: complete machine-readable evidence.", "- pretrained-reference/: approved weight provenance and separately audited reference results.", ""]
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(OUTPUT_ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
