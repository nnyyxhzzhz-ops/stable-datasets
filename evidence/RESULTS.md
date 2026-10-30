# Controlled LeJEPA follow-up results

All reported measurements come from saved configurations, checkpoints, validation traces and prediction files. The six original B0 experiments and their 107 artifacts remain unchanged.

## Data and corrected evaluation

ETH-80 uses 1,968/656/656 instance-disjoint training/validation/test images. Freiburg retains all 3,143 SSL training images and the authors' 1,018 test paths. The validation set excludes one image whose decoded RGB pixels also occur in training, leaving 785 validation images. The loader and author membership are unchanged; this is an experiment-level correction.

The source-pixel audit found no ETH-80 duplicates. Freiburg has 14 identical-RGB pairs under different paths, with no conflicting labels: five train/test pairs, three validation/test pairs, one train/validation pair, three within-train pairs and two within-test pairs. The final report gives official test accuracy alongside a 1,010-image unseen-pixel sensitivity subset and a 1,008-image unique-unseen subset. Subsets were fixed from source pixels before the new test predictions were computed.

## Protocol and validation selection

The original B0 uses 20 SSL epochs and 64/32-pixel global/local views. B1 declares 100 epochs with those views; B2 declares 100 epochs and 128/64-pixel views. All use the unchanged pinned LeJEPA objective, the same ViT-Tiny/16 architecture and exact original initial encoder tensors. New comparisons use frozen encoders and a converged multinomial logistic-regression head: C in [0.01,0.1,1,10,100], train-only standardization, validation-only selection, and smaller C on ties.

Each dataset's candidate and checkpoint were selected using seed42 validation top-1; ties prefer earlier epochs and lower resolution. Confirmation seeds43/44 use the fixed recipe and checkpoint epoch. Seed42 is a development seed, so three-seed aggregates include development selection; the two confirmation seeds are listed individually. Original test scores were already known, and these follow-ups continue on the same retained holdouts.

| Dataset | Selected candidate | Epoch | Global/local views | Seed42 validation (%) |
| --- | --- | ---: | --- | ---: |
| eth80 | B1 | 100 | 64/32 | 77.59 |
| freiburg | B2 | 100 | 128/64 | 37.96 |

## All development candidates

No candidate was chosen using test scores. Early checkpoints belong to their declared 100-epoch cosine schedule; comparisons with B0 therefore include learning-rate-trajectory differences, and cannot always be interpreted solely as longer training.

| Dataset | Candidate | Epoch | Validation (%) |
| --- | --- | ---: | ---: |
| eth80 | B0 | 20 | 72.10 |
| eth80 | B1 | 20 | 68.90 |
| eth80 | B1 | 50 | 71.95 |
| eth80 | B1 | 100 | 77.59 |
| eth80 | B2 | 20 | 69.51 |
| eth80 | B2 | 50 | 72.56 |
| eth80 | B2 | 100 | 76.68 |
| freiburg | B0 | 20 | 29.04 |
| freiburg | B1 | 20 | 26.75 |
| freiburg | B1 | 50 | 30.96 |
| freiburg | B1 | 100 | 33.63 |
| freiburg | B2 | 20 | 33.89 |
| freiburg | B2 | 50 | 35.29 |
| freiburg | B2 | 100 | 37.96 |

## Three-seed test results

Mean ± sample standard deviation, percentage units. Original B0/AdamW results used the historical validation membership and evaluator; B0/converged results use corrected validation. The selected comparison uses that same corrected evaluator and matched-resolution random encoder.

| Dataset | Condition | Encoder | Official test (%) | Balanced (%) | Unseen-pixel test (%) | Unique-unseen test (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| eth80 | original B0 / AdamW | random | 54.88 ± 2.78 | 54.88 ± 2.78 | not evaluated | not evaluated |
| eth80 | original B0 / AdamW | lejepa | 56.81 ± 2.66 | 56.81 ± 2.66 | not evaluated | not evaluated |
| eth80 | B0 / converged | random | 57.22 ± 0.72 | 57.22 ± 0.72 | 57.22 ± 0.72 | 57.22 ± 0.72 |
| eth80 | B0 / converged | lejepa | 62.91 ± 2.68 | 62.91 ± 2.68 | 62.91 ± 2.68 | 62.91 ± 2.68 |
| eth80 | B1 / converged | random | 57.22 ± 0.72 | 57.22 ± 0.72 | 57.22 ± 0.72 | 57.22 ± 0.72 |
| eth80 | B1 / converged | lejepa | 61.69 ± 1.69 | 61.69 ± 1.69 | 61.69 ± 1.69 | 61.69 ± 1.69 |
| freiburg | original B0 / AdamW | random | 19.19 ± 0.74 | 18.20 ± 0.62 | not evaluated | not evaluated |
| freiburg | original B0 / AdamW | lejepa | 28.42 ± 0.91 | 26.77 ± 0.58 | not evaluated | not evaluated |
| freiburg | B0 / converged | random | 19.84 ± 1.21 | 17.82 ± 1.68 | 19.83 ± 1.27 | 19.87 ± 1.28 |
| freiburg | B0 / converged | lejepa | 30.03 ± 1.15 | 28.15 ± 1.14 | 29.93 ± 1.09 | 29.99 ± 1.09 |
| freiburg | B2 / converged | random | 25.80 ± 1.38 | 23.84 ± 1.52 | 25.81 ± 1.39 | 25.86 ± 1.39 |
| freiburg | B2 / converged | lejepa | 40.28 ± 0.61 | 39.83 ± 1.11 | 40.30 ± 0.65 | 40.38 ± 0.65 |

## Per-seed results

| Dataset | Condition | Seed | Encoder | Clean validation (%) | Correct/official test | Official test (%) | Unseen-pixel test (%) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| eth80 | B1 | 42 | random | 66.31 | 370/656 | 56.40 | 56.40 |
| eth80 | B1 | 42 | lejepa | 77.59 | 393/656 | 59.91 | 59.91 |
| eth80 | B0 | 42 | random | 66.31 | 370/656 | 56.40 | 56.40 |
| eth80 | B0 | 42 | lejepa | 72.10 | 396/656 | 60.37 | 60.37 |
| eth80 | B1 | 43 | random | 69.05 | 379/656 | 57.77 | 57.77 |
| eth80 | B1 | 43 | lejepa | 80.64 | 415/656 | 63.26 | 63.26 |
| eth80 | B0 | 43 | random | 69.05 | 379/656 | 57.77 | 57.77 |
| eth80 | B0 | 43 | lejepa | 73.63 | 411/656 | 62.65 | 62.65 |
| eth80 | B1 | 44 | random | 69.51 | 377/656 | 57.47 | 57.47 |
| eth80 | B1 | 44 | lejepa | 81.25 | 406/656 | 61.89 | 61.89 |
| eth80 | B0 | 44 | random | 69.51 | 377/656 | 57.47 | 57.47 |
| eth80 | B0 | 44 | lejepa | 72.56 | 431/656 | 65.70 | 65.70 |
| freiburg | B2 | 42 | random | 27.52 | 274/1018 | 26.92 | 26.93 |
| freiburg | B2 | 42 | lejepa | 37.96 | 408/1018 | 40.08 | 40.20 |
| freiburg | B0 | 42 | random | 21.40 | 216/1018 | 21.22 | 21.29 |
| freiburg | B0 | 42 | lejepa | 29.04 | 301/1018 | 29.57 | 29.41 |
| freiburg | B2 | 43 | random | 26.75 | 247/1018 | 24.26 | 24.26 |
| freiburg | B2 | 43 | lejepa | 38.73 | 417/1018 | 40.96 | 40.99 |
| freiburg | B0 | 43 | random | 18.47 | 197/1018 | 19.35 | 19.31 |
| freiburg | B0 | 43 | lejepa | 31.08 | 319/1018 | 31.34 | 31.19 |
| freiburg | B2 | 44 | random | 27.77 | 267/1018 | 26.23 | 26.24 |
| freiburg | B2 | 44 | lejepa | 36.43 | 405/1018 | 39.78 | 39.70 |
| freiburg | B0 | 44 | random | 20.89 | 193/1018 | 18.96 | 18.91 |
| freiburg | B0 | 44 | lejepa | 29.17 | 297/1018 | 29.17 | 29.21 |

## Separate ImageNet-supervised reference

The official ResNet-18 weights were frozen, using their documented 224-pixel preprocessing. A train-only logistic-regression head was selected on corrected validation. This is a deterministic external-supervised-pretraining reference, not a LeJEPA result or a same-architecture training-method comparison. Source-data overlap with ImageNet was not independently audited.

| Dataset | Validation (%) | Correct/official test | Official test (%) | Unseen-pixel test (%) |
| --- | ---: | ---: | ---: | ---: |
| eth80 | 95.58 | 606/656 | 92.38 | 92.38 |
| freiburg | 69.68 | 733/1018 | 72.00 | 71.88 |

## Verification and limits

The audit verifies unchanged original artifacts, exact B0 initialization, identical SSL training membership, declared optimizer-step counts, finite checkpoints, immutable candidate selection, saved numeric-head replay, and all CSV-derived metrics and sensitivity subsets. No new source datasets were downloaded. The optional pretrained reference used one official weight download; SSL runs remain from scratch and offline.

These experiments do not reproduce the original paper's full benchmark protocol, establish statistical significance, or establish a general world-model guarantee. The B2 comparison changes global, local and evaluation resolutions jointly. One development seed, two confirmation seeds and one partition limit generalization claims. Different Freiburg physical-product instances are not annotated. Within-training exact duplicates were retained to preserve the controlled SSL input comparison; original official test duplicate counts are disclosed rather than hidden.

## Artifacts

- protocol-lock.json, pixel-clean-protocol.json and clean-selection-lock.json: recorded decisions and source-pixel correction.
- training/: initial weights, complete epoch logs, configurations, software/source versions and checkpoints.
- clean-B0-validation/ and clean-validation/: all regularization trials, convergence traces and heads.
- clean-final/: per-image predictions, confusion matrices, per-class recall and subset metrics.
- clean-final-report.json and clean-final-audit.json: complete machine-readable evidence.
- pretrained-reference/: approved weight provenance and separately audited reference results.
