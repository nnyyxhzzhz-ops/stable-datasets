# Frozen supervised-pretraining reference

Official torchvision ResNet-18 ImageNet-1K supervised weights were frozen. The dataset loaders and SSL training membership were retained. One Freiburg validation image with identical pixels in training was excluded before head selection (785 validation images). Only a multinomial logistic-regression head was fitted on training features; C was selected on validation data before each test. This is an external-supervised-data transfer reference and is not a LeJEPA result or a same-architecture causal comparison.

| Dataset | Validation (%) | Selected C | Correct/official test | Official test (%) | Balanced (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| eth80 | 95.58 | 0.01 | 606/656 | 92.38 | 92.38 |
| freiburg | 69.68 | 0.01 | 733/1018 | 72.00 | 70.86 |

## Source-pixel sensitivity

The official Freiburg test split contains eight images with exact RGB matches in train or the original validation split. The subsets were fixed from a source-pixel audit before test predictions were computed. The unique-unseen subset additionally removes two within-test duplicate copies. Loader membership is unchanged.

| Dataset | Test subset | Correct/count | Accuracy (%) | Balanced (%) |
| --- | --- | ---: | ---: | ---: |
| eth80 | official | 606/656 | 92.38 | 92.38 |
| eth80 | unique_unseen_pixels | 606/656 | 92.38 | 92.38 |
| eth80 | unseen_pixels | 606/656 | 92.38 | 92.38 |
| freiburg | official | 733/1018 | 72.00 | 70.86 |
| freiburg | unique_unseen_pixels | 725/1008 | 71.92 | 70.75 |
| freiburg | unseen_pixels | 726/1010 | 71.88 | 70.73 |

The encoder and solver are fixed and deterministic; these are single evaluations, not three independent training seeds. All head searches, warnings, iterations, official preprocessing, checksums and per-image predictions are saved. The audit replayed the numeric head predictions and verified train-only normalization and exact test IDs/labels. Original LeJEPA experiments are unchanged. Pretraining-source overlap with these datasets has not been independently audited.

Weights were obtained from the official PyTorch distribution URL. Official filename hash prefix f37072fd was checked and the full SHA-256 recorded in weights-download.json. Test heads were locked by clean-selection-lock.json before test feature extraction.
