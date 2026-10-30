# Frozen supervised reference

The reference uses the official torchvision ResNet-18 ImageNet-1K V1 weights, not an SSL checkpoint. Obtain the file separately from the official distribution URL and verify its full checksum before evaluation. Experiment commands do not download weights.

```bash
mkdir -p work/model-weights
curl --fail --location https://download.pytorch.org/models/resnet18-f37072fd.pth \
  --output work/model-weights/resnet18-f37072fd.pth
```

Run these commands from the documented project root, with fresh reference output directories. The first phase fits heads on training features and selects regularization on corrected validation; it locks both dataset selections before the second phase reads test pixels.

```bash
OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/assignment1-experiments-v2/run_reference.py \
  --weights work/model-weights/resnet18-f37072fd.pth \
  --weights-sha256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec \
  --phase validate
OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/assignment1-experiments-v2/run_reference.py \
  --weights work/model-weights/resnet18-f37072fd.pth \
  --weights-sha256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec \
  --phase test
OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/assignment1-experiments-v2/audit_reference.py
```

The second phase refuses existing result directories. Official and exact-pixel sensitivity metrics use the same frozen head and predictions; there is no test-time refitting. These are single deterministic evaluations, not three independently pretrained models. See the measured results for the distinction between this external-supervised-data reference and the matched random/LeJEPA comparisons.
