"""Verify source-derived exclusions while preserving official test membership."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pixel_clean  # noqa: E402


@pytest.mark.parametrize("dataset,counts", [("eth80", (656, 656, 656)), ("freiburg", (1018, 1010, 1008))])
def test_fixed_mask_counts_and_pixel_only_exclusions(dataset, counts):
    """Official and clean sensitivity denominators are fixed before predictions."""
    ids = [row["image_id"] for row in pixel_clean._manifest(dataset)["test"]]
    masks = pixel_clean.test_sensitivity_masks(dataset, ids)
    assert tuple(int(mask.sum()) for mask in masks.values()) == counts
    assert all(mask.dtype == np.bool_ and len(mask) == len(ids) for mask in masks.values())
    if dataset == "freiburg":
        excluded = {image_id for image_id, keep in zip(ids, masks["unseen_pixels"]) if not keep}
        assert excluded == {
            "SPICES/SPICES0153.png",
            "SPICES/SPICES0001.png",
            "SPICES/SPICES0063.png",
            "SPICES/SPICES0182.png",
            "SPICES/SPICES0176.png",
            "COFFEE/COFFEE0193.png",
            "COFFEE/COFFEE0190.png",
            "COFFEE/COFFEE0129.png",
        }
        unique_excluded = {image_id for image_id, keep in zip(ids, masks["unique_unseen_pixels"]) if not keep}
        assert unique_excluded - excluded == {"SPICES/SPICES0169.png", "SPICES/SPICES0177.png"}
        assert masks["unique_unseen_pixels"][ids.index("SPICES/SPICES0007.png")]
        assert masks["unique_unseen_pixels"][ids.index("SPICES/SPICES0124.png")]


class MetadataPartition:
    """Keep IDs only, with the selection interface used by the clean wrapper."""

    def __init__(self, ids):
        self.ids = ids
        self.table = SimpleNamespace(column=lambda name: SimpleNamespace(to_pylist=lambda: self.ids))

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for image_id in self.ids:
            yield {"image_id": image_id}

    def remove_columns(self, name):
        """Represent metadata-only iteration over the selected view."""
        assert name == "image"
        return self

    def with_format(self, name):
        """Preserve the selected view when choosing the raw format."""
        assert name == "raw"
        return self

    def select(self, indices):
        """Return selected metadata without any image-reading capability."""
        return MetadataPartition([self.ids[index] for index in indices])


def test_freiburg_excludes_only_one_validation_image(monkeypatch):
    """Training and all 1,018 official test rows remain the same objects."""
    original = pixel_clean._manifest("freiburg")
    partitions = {name: MetadataPartition([row["image_id"] for row in rows]) for name, rows in original.items()}
    # Dataset.select keeps a larger underlying Arrow table; its iterator applies
    # the view indices. Reading .table directly would create invalid indices.
    base_ids = partitions["train"].ids + partitions["validation"].ids
    partitions["validation"].table = SimpleNamespace(column=lambda name: SimpleNamespace(to_pylist=lambda: base_ids))
    monkeypatch.setattr(pixel_clean.common, "load_splits", lambda dataset: partitions)
    cleaned = pixel_clean.load_clean_splits("freiburg")
    assert cleaned["train"] is partitions["train"] and len(cleaned["train"]) == 3143
    assert cleaned["test"] is partitions["test"] and len(cleaned["test"]) == 1018
    assert len(cleaned["validation"]) == 785
    assert set(partitions["validation"].ids) - set(cleaned["validation"].ids) == {"SPICES/SPICES0052.png"}
    assert cleaned["validation"].ids == [
        value for value in partitions["validation"].ids if value != "SPICES/SPICES0052.png"
    ]


def test_eth80_splits_are_unchanged(monkeypatch):
    """The dataset with no exact-pixel overlap retains its complete partitions."""
    original = pixel_clean._manifest("eth80")
    partitions = {name: MetadataPartition([row["image_id"] for row in rows]) for name, rows in original.items()}
    monkeypatch.setattr(pixel_clean.common, "load_splits", lambda dataset: partitions)
    assert pixel_clean.load_clean_splits("eth80") is partitions


def test_masks_reject_changed_test_order():
    """A correct count cannot hide a mask applied to the wrong sample order."""
    ids = [row["image_id"] for row in pixel_clean._manifest("freiburg")["test"]]
    with pytest.raises(ValueError, match="original ordered"):
        pixel_clean.test_sensitivity_masks("freiburg", list(reversed(ids)))
    with pytest.raises(ValueError, match="original ordered"):
        pixel_clean.test_sensitivity_masks("freiburg", ids[:-1])


def test_changed_source_audit_is_rejected(tmp_path, monkeypatch):
    """Exclusions cannot silently change after their source audit was pinned."""
    changed = tmp_path / "changed-audit.json"
    changed.write_text("{}")
    monkeypatch.setattr(pixel_clean, "AUDIT_PATH", changed)
    with pytest.raises(RuntimeError, match="Source-pixel audit changed"):
        pixel_clean.ensure_policy()


def test_existing_policy_cannot_be_silently_replaced(tmp_path, monkeypatch):
    """Reusing a policy name with different selections is an explicit error."""
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"decision_basis": "changed"}))
    monkeypatch.setattr(pixel_clean, "POLICY_PATH", path)
    with pytest.raises(RuntimeError, match="refusing silent reselection"):
        pixel_clean.ensure_policy()
