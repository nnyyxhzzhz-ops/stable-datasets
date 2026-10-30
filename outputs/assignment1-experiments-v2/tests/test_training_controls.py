"""Checks for SSL label isolation and the declared optimization budget."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_experiment import Views, learning_rate_factor


def test_ssl_view_transform_discards_labels_and_metadata():
    image = Image.fromarray(
        np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    )
    result = Views({"image_size": 128, "local_size": 64})(
        {"image": image, "label": 7, "image_id": "held-only-by-loader"}
    )
    assert set(result) == {"global_views", "local_views"}
    assert [tuple(view.shape) for view in result["global_views"]] == [(3, 128, 128)] * 2
    assert [tuple(view.shape) for view in result["local_views"]] == [(3, 64, 64)] * 2
    assert all(view.isfinite().all() for views in result.values() for view in views)


def test_schedule_warms_then_decays_over_full_declared_budget():
    schedule = [learning_rate_factor(step, 6100, 122) for step in range(6100)]
    assert 0 < schedule[0] < schedule[121] == schedule[122] == 1
    assert all(a <= b for a, b in zip(schedule[:121], schedule[1:122]))
    assert all(a >= b for a, b in zip(schedule[122:-1], schedule[123:]))
    assert abs(schedule[-1] - 0.01) < 1e-12
    assert min(schedule) > 0
