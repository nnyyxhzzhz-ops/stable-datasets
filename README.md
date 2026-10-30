# ETH-80 and Freiburg Groceries validation

Reproducible dataset validation with a preserved 20-epoch LeJEPA baseline, controlled 100-epoch extensions, converged frozen linear evaluation, and a separate ImageNet-supervised ResNet-18 reference. The dataset PR contains only its ten library/test/documentation files; this supplemental project contains the experiment code and evidence.

## Recorded environment

Python 3.12.14; PyTorch 2.14.1; torchvision 0.29.1; timm 1.0.30; NumPy 2.5.4; PyArrow 26.0.0; Pillow 13.0.0; scikit-learn 1.9.1; SciPy 1.18.1. LeJEPA is imported unchanged from the local stable-pretraining checkout at `ab836bf699a2be712dfcf980a5eb70d36391e876`. All reported training used four CPU threads on an Apple M4 Mac mini. Numerical results can vary across hardware and dependencies.

The source archive is extracted into a project directory with this layout:

```text
outputs/stable-datasets/                 Dataset library checkout
outputs/assignment1-validation-source/  Original B0 source and configuration
outputs/assignment1-experiments-v2/      This project
outputs/experiments/                    Generated B0 runs
outputs/experiments-v2/                 Protocols and generated follow-ups
work/stable-pretraining/                Pinned upstream method checkout
work/data/                             Source archives
work/venv/                             Python 3.12 environment
```

Clone both libraries locally, check out the dataset contribution and the pinned method commit, then install them locally. The dataset checkout must include Freiburg builder version 1.0.1. The source snapshot includes `requirements.txt`; library checkouts and large data/model artifacts are not embedded in the supplemental source.

```bash
mkdir -p outputs work
git clone https://github.com/nnyyxhzzhz-ops/stable-datasets.git outputs/stable-datasets
git -C outputs/stable-datasets checkout 35fe052c3686146fd25fe05600ebc265418c8ef6
git clone https://github.com/galilai-group/stable-pretraining.git work/stable-pretraining
git -C work/stable-pretraining checkout ab836bf699a2be712dfcf980a5eb70d36391e876
python3.12 -m venv work/venv
work/venv/bin/python -m pip install -r outputs/assignment1-experiments-v2/requirements.txt
work/venv/bin/python -m pip install -e outputs/stable-datasets -e work/stable-pretraining
```

## Data and preserved baseline

Data preparation defaults to local archives only. Missing files fail rather than silently downloading. Explicit `--download-data` permits the normal builders to fetch their official checksum-pinned assets. The preparation command creates the project cache layout used by the measured runs and per-dataset cache links for the original CLI.

```bash
work/venv/bin/python outputs/assignment1-experiments-v2/prepare_data.py
# For a fresh project without source archives:
# work/venv/bin/python outputs/assignment1-experiments-v2/prepare_data.py --download-data
```

In a fresh project, reproduce the original six B0 runs before the follow-up. Existing baseline directories must not be overwritten. The original script, requirements and JSON configuration are preserved byte for byte from commit `df0abcef1364ab450acadb2d8207118b0ed7169e`.

```bash
for dataset in eth80 freiburg; do
  for seed in 42 43 44; do
    work/venv/bin/python outputs/assignment1-validation-source/examples/object_datasets_lejepa.py \
      --config outputs/assignment1-validation-source/examples/configs/object_datasets_lejepa.json \
      --dataset "$dataset" --seed "$seed" --device cpu \
      --cache-root "work/baseline-cache/$dataset" \
      --output "outputs/experiments/$dataset-seed$seed"
  done
done
work/venv/bin/python outputs/assignment1-experiments-v2/prepare_data.py --check-only
work/venv/bin/python outputs/assignment1-experiments-v2/snapshot_baseline.py
```

The baseline inputs contain every training-split member; no batch cap is used. Training uses batch 32 and drops an incomplete shuffled tail each epoch (16 ETH-80 or 7 Freiburg images). Validation/test loaders retain all members. SSL receives images only. The original B0 evaluator used 200 AdamW iterations per weight decay; the follow-up explicitly changes that evaluator.

## Source-pixel correction

The supplied `outputs/experiments-v2/exact-pixel-duplicate-audit.json` records checksum-verified source-pixel comparisons. No ETH-80 duplicates were found. Freiburg has 14 pairs of identical RGB arrays under different file IDs, including one pair across train/validation and eight across training/original-validation and test. There are no conflicting labels.

The current protocol excludes `SPICES/SPICES0052.png` from validation, giving 785 validation images. SSL training remains 3,143 images. Official test membership remains 1,018 paths; results additionally report 1,010 unseen-pixel and 1,008 unique-unseen-pixel test images. These masks were fixed from source pixels before new test predictions. Loaders and author split membership are unchanged.

The source archive supplies the original protocol and pixel audit inputs. Fresh executions generate their own machine-specific selection locks; recorded locks in the evidence package are historical records, not files to copy over a fresh execution. No numeric result or exclusion is selected using test accuracy.

## Current execution commands

Run the focused offline experiment tests:

```bash
OPENBLAS_NUM_THREADS=1 work/venv/bin/python -m pytest -q outputs/assignment1-experiments-v2/tests
```

Run development, confirmation, audit and report generation from the project root:

```bash
MPLCONFIGDIR=work/matplotlib OPENBLAS_NUM_THREADS=1 work/venv/bin/python \
  outputs/assignment1-experiments-v2/run_clean_study.py --phase develop --jobs 2
MPLCONFIGDIR=work/matplotlib OPENBLAS_NUM_THREADS=1 work/venv/bin/python \
  outputs/assignment1-experiments-v2/run_clean_study.py --phase confirm --jobs 2
OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/assignment1-experiments-v2/audit_clean_study.py
OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/assignment1-experiments-v2/summarize_clean_study.py
```

Development re-evaluates B0 with corrected validation and compares B1/B2 checkpoints20/50/100 using seed 42. Selection maximizes validation top1; ties prefer an earlier checkpoint, lower resolution, then candidate name. Seeds43/44 use that fixed candidate/checkpoint. Their probe regularization is selected using the same declared validation rule. All confirmation heads are prepared before final test features are extracted.

Standalone training can be timed with `run_experiment.py --mode benchmark`; timings never supply model states for formal runs. Existing completed runs are reused only after configuration/completion checks. Partial outputs are retained and cause an error. Exact training resumption is not implemented. Historical `run_study.py` confirmation is disabled after the pixel audit; `run_clean_study.py` is authoritative.

## Training and evaluation controls

| Candidate | Epoch budget | Global/local views | Evaluation size |
| --- | ---: | --- | ---: |
| B0 |20|64/32|64|
| B1 |100|64/32|64|
| B2 |100|128/64|128|

The architecture is ViT-Tiny/16 with pretrained=False. The original constructor order is reproduced and every initial encoder tensor checked against B0. Each recipe uses two global/two local views, the original crop scales/augmentations, a 192→256→128 projector,64 SIGReg slices, lambda 0.02, AdamW lr0.0005/weight decay0.05, two warmup epochs and cosine decay to1% of peak. A selected early checkpoint retains its declared 100-epoch schedule; `--stop-after` does not shorten that schedule.

Frozen features use deterministic bilinear resize, channel mean/std0.5 and training-only feature standardization (sample std, minimum1e-6). Logistic regression uses LBFGS, tolerance1e-6, no class weighting and C=[0.01,0.1,1,10,100]. A 5,000-iteration convergence warning triggers a fresh 20,000-cap numerical retry; remaining nonconvergence fails. Highest validation top1 chooses C, with the smallest C retained on ties. C is inverse regularization and is not equivalent to the original AdamW decay values.

Encoder states must remain unchanged during extraction. Raw features are cached by encoder checksum, ordered sample IDs/labels, preprocessing and versions; payload SHA-256 is checked. Linear heads use plain numeric NumPy arrays. The B2 contrast changes global/local/evaluation resolution together, so it is a resolution-setting comparison rather than a global-resolution-only ablation.

## Supervised-pretraining reference

`run_pretrained_reference_clean.py` loads only an existing local ResNet-18 weight file. Use the official torchvision ImageNet-1K V1 file, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. It uses the weights' official resize256/center-crop224/ImageNet normalization, freezes all backbone tensors, and applies the same converged linear evaluator.

Validation and testing are separate CLI phases. Testing requires the validation-selection checksum and the independent clean-selection-lock checksum. The lock binds the pixel policy, weight SHA, evaluator and both selected heads. `audit_reference.py` replays predictions and checks train-only standardization. See `REFERENCE_COMMANDS.md` for executable commands.

This reference uses external supervised pretraining. It is not a LeJEPA result, not a same-architecture causal comparison, and has no independent three-seed training variability. Possible ImageNet/source-data overlap has not been independently audited.

## Evidence and limits

The evidence package contains aggregate/per-seed tables, all final prediction CSVs, source-pixel findings, normalized numeric result summaries and source checksums. Large source images, processed caches, weights, feature caches and model checkpoints remain outside source archives. Original full logs, checkpoints and immutable locks remain in the local experiment directories.

Original B0 test scores were already known. The follow-up uses a bounded validation-selected study on those same holdouts; seed 42 is a development seed and seeds 43/44 are confirmation seeds. Training variability on one partition does not establish statistical significance or reproduce the paper's full benchmark protocol. Tests and metrics validate this integration and its stated experiments, not a general guarantee about SSL or world models.
