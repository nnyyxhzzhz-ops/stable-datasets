"""Exercise convergence, head persistence and training-only normalization."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

MODULE = Path(__file__).resolve().parents[1] / "evaluate_frozen.py"
SPEC = importlib.util.spec_from_file_location("evaluate_frozen", MODULE)
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def separable():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(3), 20)
    features = np.eye(3)[labels] * 5 + rng.normal(0, 0.2, (60, 3))
    return features, labels


def test_train_only_standardization_and_numeric_roundtrip(tmp_path):
    train, labels = separable()
    validation = train + 0.1
    probe, report = evaluation.fit_probe(train, labels, validation, labels)
    assert np.allclose(probe["mean"], train.mean(0))
    assert not np.allclose(probe["mean"], np.concatenate([train, validation]).mean(0))
    assert report["selected"]["validation_accuracy"] == 1.0
    assert report["selected"]["C"] == min(evaluation.CS)
    assert all(trial["converged"] for trial in report["grid"])
    assert report["test_accessed"] is False
    output = tmp_path / "probe.npz"
    evaluation.save_probe(output, probe)
    loaded = evaluation.load_probe(output)
    assert np.array_equal(evaluation.predict_probe(loaded, validation), labels)
    assert evaluation.metrics(labels, labels, np.arange(3))["per_class_recall"] == [1.0] * 3


def test_covariance_rank_distinguishes_collapse():
    constant = np.ones((10, 4))
    random = np.random.default_rng(5).normal(size=(100, 4))
    assert evaluation.feature_diagnostics(constant)["effective_rank"] == 0
    assert evaluation.feature_diagnostics(random)["effective_rank"] > 3.5


def test_nonfinite_features_are_rejected():
    train, labels = separable()
    train[0, 0] = np.nan
    with pytest.raises(AssertionError):
        evaluation.fit_probe(train, labels, train, labels)


def test_unconverged_solver_does_not_report_a_result(monkeypatch):
    class Unconverged:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            import warnings

            self.n_iter_ = np.array([5000])
            warnings.warn("fixture did not converge", ConvergenceWarning)

    monkeypatch.setattr(evaluation, "LogisticRegression", Unconverged)
    train, labels = separable()
    with pytest.raises(RuntimeError, match="failed to converge"):
        evaluation.fit_probe(train, labels, train, labels)
