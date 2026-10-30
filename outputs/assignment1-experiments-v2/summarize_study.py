"""Render measured study results and preserve unsuccessful development candidates."""

import json
import statistics

from common import OUTPUT_ROOT, ROOT


def main():
    """Write an English report only after the locked final artifact audit passes."""
    audit = json.loads((OUTPUT_ROOT / "final-audit.json").read_text())
    assert audit["status"] == "passed"
    final = json.loads((OUTPUT_ROOT / "final-report.json").read_text())
    lock = json.loads((OUTPUT_ROOT / "selection-lock.json").read_text())
    validation = json.loads((OUTPUT_ROOT / "B0-converged-probe-validation/validation-report.json").read_text())
    lines = ["# Controlled LeJEPA follow-up results", "", "All values below are measured. Accuracy is reported as a percentage. The original six experiments remain unchanged.", "", "## Protocol and selection", "", "The original B0 uses 20 SSL epochs and 64/32-pixel global/local views. B1 declares a 100-epoch schedule with those views; B2 declares 100 epochs and 128/64-pixel views. The original initialization, dataset partitions and upstream LeJEPA objective are preserved. Both encoders use a train-only standardized, validation-selected converged logistic-regression evaluator for the new comparisons.", "", "Each dataset's recipe and SSL checkpoint were selected using seed42 validation accuracy, with ties resolved by earlier epoch then lower resolution. Confirmation seeds43/44 use that fixed recipe/checkpoint; only probe regularization is selected using the predeclared C grid. Seed42 is a development seed, so the three-seed aggregate includes development selection; confirmation-seed results are listed separately. Original test scores were already known, and this is continued evaluation on the same retained holdouts.", "", "| Dataset | Selected candidate | Checkpoint epoch | Global/evaluation resolution | Development validation (%) |", "| --- | --- | ---: | ---: | ---: |"]
    for dataset, item in lock["datasets"].items():
        c = item["selected"]
        lines.append(f"| {dataset} | {c['candidate']} | {c['epoch']} | {c['image_size']} | {100*c['validation_accuracy']:.2f} |")
    lines += ["", "A selected early checkpoint retains the declared 100-epoch cosine schedule. For example B1 at epoch20 has the same number of updates as B0, but a different learning-rate trajectory; such a result cannot be attributed solely to longer training.", "", "## Development candidates", "", "All candidates were evaluated only on training/validation data before the selection lock.", "", "| Dataset | Candidate | Epoch | Validation (%) |", "| --- | --- | ---: | ---: |"]
    for dataset, item in lock["datasets"].items():
        for c in item["candidates"]:
            lines.append(f"| {dataset} | {c['candidate']} | {c['epoch']} | {100*c['validation_accuracy']:.2f} |")
    lines += ["", "## Aggregate test results", "", "Mean ± sample standard deviation over seeds42/43/44. Original B0 AdamW evaluation is included separately from the new converged evaluator.", "", "| Dataset | Condition | Encoder | Test (%) | Balanced (%) |", "| --- | --- | --- | ---: | ---: |"]
    def summary(values):
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"
    for dataset in ("eth80", "freiburg"):
        for method in ("random", "lejepa"):
            originals = [json.loads((ROOT / f"outputs/experiments/{dataset}-seed{seed}/results.json").read_text())[method] for seed in (42,43,44)]
            lines.append(f"| {dataset} | original B0 / AdamW probe | {method} | {summary([100*r['test_accuracy'] for r in originals])} | {summary([100*r['test_balanced_accuracy'] for r in originals])} |")
        conditions = sorted({r["candidate"] for r in final["runs"] if r["dataset"] == dataset})
        for condition in conditions:
            for method in ("random", "lejepa"):
                rows = [r for r in final["runs"] if r["dataset"] == dataset and r["candidate"] == condition and r["method"] == method]
                assert len(rows) == 3
                lines.append(f"| {dataset} | {condition} / converged probe | {method} | {summary([100*r['accuracy'] for r in rows])} | {summary([100*r['balanced_accuracy'] for r in rows])} |")
    lines += ["", "## Per-seed results", "", "Seed42 is development; seeds43/44 are confirmation. Balanced accuracy is macro recall.", "", "| Dataset | Condition | Seed | Encoder | Validation (%) | Correct/test | Test (%) | Balanced (%) |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for r in final["runs"]:
        if r["candidate"] == "B0" or (r["method"] == "random" and r["image_size"] == 64):
            vr = next(x for x in validation["runs"] if x["dataset"] == r["dataset"] and x["seed"] == r["seed"] and x["encoder"] == r["method"])
        elif r["method"] == "random":
            vr = json.loads((OUTPUT_ROOT / f"validation/{r['dataset']}/random{r['image_size']}/seed{r['seed']}/report.json").read_text())
        else:
            vr = json.loads((OUTPUT_ROOT / f"validation/{r['dataset']}/{r['candidate']}/seed{r['seed']}/epoch{r['epoch']:03d}/report.json").read_text())
        lines.append(f"| {r['dataset']} | {r['candidate']} | {r['seed']} | {r['method']} | {100*vr['selected']['validation_accuracy']:.2f} | {r['correct']}/{r['count']} | {100*r['accuracy']:.2f} | {100*r['balanced_accuracy']:.2f} |")
    lines += ["", "## Validation and limitations", "", "The artifact audit checks original107-file integrity, exact B0 initialization, unchanged ordered split manifests, step counts, finite checkpoints, and CSV-derived final metrics. No new datasets or pretrained weights were downloaded. The encoder was frozen during feature extraction; head standardization and fitting use training features only. Feature caches and source digests are recorded per run.", "", "These small-data experiments do not reproduce the original paper's benchmark protocol or establish a general accuracy guarantee. The B2 contrast changes global, local and evaluation resolutions jointly. Development selection uses one seed and one partition; the same source objects/products remain a limitation for generalization claims. No ImageNet-pretrained transfer reference was executed as part of the offline study.", "", "## Artifacts", "", "- protocol-lock.json and selection-lock.json: predeclared protocol and validation selection.", "- B0-converged-probe-validation/: all60 regularization trials, convergence records and numeric heads.", "- training/: source/version records, full losses, exact initial weights and epoch checkpoints.", "- validation/: checkpoint-specific validation trials and selected heads.", "- final/: per-image predictions, confusion matrices and per-class recall.", "- final-report.json and final-audit.json: machine-readable results and integrity checks.", ""]
    (OUTPUT_ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(OUTPUT_ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
