"""Offline guards for the optional supervised-pretraining reference."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_pretrained_reference import (
    OfficialTransform,
    transform_metadata,
    verified_weights_path,
)


def test_missing_weights_fail_instead_of_downloading(tmp_path):
    with pytest.raises(FileNotFoundError, match="automatic downloads are disabled"):
        verified_weights_path(tmp_path / "not-downloaded.pth", "0" * 64)
    assert not list(tmp_path.iterdir())


def test_wrong_weight_checksum_is_rejected(tmp_path):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"not approved pretrained weights")
    with pytest.raises(ValueError, match="checksum does not match"):
        verified_weights_path(path, "0" * 64)


def test_official_transform_retains_only_image_and_label():
    row = {
        "image": Image.new("RGB", (256, 256), "white"),
        "label": 2,
        "image_id": "metadata",
    }
    transformed = OfficialTransform()(row)
    assert set(transformed) == {"image", "label"}
    assert transformed["label"] == 2
    assert transformed["image"].shape == (3, 224, 224)
    assert torch.isfinite(transformed["image"]).all()
    assert transform_metadata()["resize_size"] == [256]
    assert transform_metadata()["crop_size"] == [224]
