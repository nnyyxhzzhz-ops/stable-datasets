"""Guard the study's validation-only selection and fixed confirmation budgets."""

import json
import sys
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import run_study as study  # noqa: E402


@pytest.fixture
def study_paths(tmp_path, monkeypatch):
    """Isolate all orchestration records from real experiment artifacts."""
    output = tmp_path / "study"
    baseline = output / "baseline"
    baseline.mkdir(parents=True)
    monkeypatch.setattr(study, "OUT", output)
    monkeypatch.setattr(study, "BASE_EVAL", baseline)
    monkeypatch.setattr(study, "environment", lambda: {"test_fixture": True})
    study.write_json(output / "protocol-lock.json", {"candidate_set": ["B0", "B1", "B2"]})
    return output, baseline


def test_train_preserves_declared_schedule_when_stopping_early(study_paths, monkeypatch):
    """Confirmation stops at the selected epoch without changing cosine length."""
    output, _ = study_paths
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] is True

    monkeypatch.setattr(study.subprocess, "run", execute)
    expected = output / "training/eth80/B1/seed43"
    assert study.train("eth80", "B1", 43, stop_after=20) == expected
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--stop-after") + 1] == "20"
    assert "--epochs" not in command
    assert command[command.index("--seed") + 1] == "43"


def test_train_refuses_to_overwrite_partial_run(study_paths, monkeypatch):
    """A partial result cannot silently become a new or resumed experiment."""
    output, _ = study_paths
    (output / "training/eth80/B2/seed42").mkdir(parents=True)
    monkeypatch.setattr(study.subprocess, "run", lambda *args, **kwargs: pytest.fail("Training must not start"))
    with pytest.raises(RuntimeError, match="Incomplete run"):
        study.train("eth80", "B2", 42)


def test_train_refuses_a_completed_smoke_schedule(study_paths, monkeypatch):
    """A completed short schedule cannot masquerade as a formal early stop."""
    output, _ = study_paths
    directory = output / "training/eth80/B1/seed43"
    study.write_json(
        directory / "config.json",
        {
            "dataset": "eth80",
            "candidate": "B1",
            "seed": 43,
            "epochs": 20,
            "stop_after": 20,
        },
    )
    study.write_json(
        directory / "training-summary.json",
        {
            "status": "completed",
            "initial_encoder_equal_b0": True,
            "epochs_completed": 20,
        },
    )
    (directory / "checkpoint-epoch020.pt").write_bytes(b"unused fixture")
    monkeypatch.setattr(study.subprocess, "run", lambda *args, **kwargs: pytest.fail("Training must not start"))
    with pytest.raises(AssertionError):
        study.train("eth80", "B1", 43, stop_after=20)


def test_development_selects_on_validation_and_resolves_ties(study_paths, monkeypatch):
    """Selection uses only seed42 validation and favors the cheaper tied recipe."""
    output, baseline = study_paths
    study.write_json(
        baseline / "validation-report.json",
        {
            "status": "passed",
            "test_accessed": False,
            "runs": [
                {"dataset": name, "seed": 42, "encoder": "lejepa", "selected": {"validation_accuracy": 0.5}}
                for name in ("eth80", "freiburg")
            ],
        },
    )
    training_calls, validation_calls = [], []
    monkeypatch.setattr(study, "train", lambda *args: training_calls.append(args))

    def validation(dataset, candidate, seed, epoch):
        validation_calls.append((dataset, candidate, seed, epoch))
        assert not (output / "selection-lock.json").exists()
        score = 0.6
        if dataset == "eth80" and candidate == "B2" and epoch == 50:
            score = 0.9
        if dataset == "freiburg":
            score = 0.8
        return {"image_size": 64 if candidate == "B1" else 128, "selected": {"validation_accuracy": score}}

    monkeypatch.setattr(study, "validation", validation)
    monkeypatch.setattr(study, "final_test", lambda *args: pytest.fail("Development cannot test"))
    study.develop(jobs=1)
    lock = json.loads((output / "selection-lock.json").read_text())
    assert lock["test_accessed"] is False
    assert len(training_calls) == 4 and all(call[2] == 42 for call in training_calls)
    assert len(validation_calls) == 12 and all(call[2] == 42 for call in validation_calls)
    assert lock["datasets"]["eth80"]["selected"]["candidate"] == "B2"
    assert lock["datasets"]["eth80"]["selected"]["epoch"] == 50
    assert lock["datasets"]["freiburg"]["selected"]["candidate"] == "B1"
    assert lock["datasets"]["freiburg"]["selected"]["epoch"] == 20
    assert all(len(value["candidates"]) == 7 for value in lock["datasets"].values())


def test_confirmation_uses_locked_epoch_for_both_new_seeds(study_paths, monkeypatch):
    """Confirmation never performs a fresh checkpoint search on seeds43/44."""
    output, _ = study_paths
    lock = {
        "protocol_lock_sha256": study.sha256_file(output / "protocol-lock.json"),
        "probe_protocol": json.loads(json.dumps(study.EVALUATOR)),
        "datasets": {
            dataset: {"selected": {"candidate": "B1", "epoch": 20, "image_size": 64}}
            for dataset in ("eth80", "freiburg")
        },
    }
    study.write_json(output / "selection-lock.json", lock)
    calls, validation_calls, test_calls = [], [], []
    monkeypatch.setattr(study, "train", lambda *args: calls.append(args))

    def validate(dataset, candidate, seed, epoch):
        validation_calls.append((dataset, candidate, seed, epoch))
        assert epoch == 20 and candidate == "B1"
        return {"probe_path": "/unused/probe.npz", "checkpoint": "/unused/checkpoint.pt"}

    def test(*args):
        test_calls.append(args)
        assert (output / "selection-lock.json").exists()
        return {"dataset": args[0], "seed": args[1], "candidate": args[2]}

    monkeypatch.setattr(study, "validation", validate)
    monkeypatch.setattr(study, "final_test", test)
    study.confirm(jobs=1)
    assert sorted(calls) == sorted((dataset, "B1", seed, 20) for dataset in ("eth80", "freiburg") for seed in (43, 44))
    assert len(validation_calls) == 6
    assert len(test_calls) == 24
    report = json.loads((output / "final-report.json").read_text())
    assert report["status"] == "passed"
    assert report["original_test_results_already_known"] is True
    assert report["independent_confirmation_seeds"] == [43, 44]


@pytest.mark.parametrize("tamper", ["protocol", "evaluator"])
def test_confirmation_rejects_changed_locked_protocol(study_paths, monkeypatch, tamper):
    """A changed candidate/probe protocol must fail before training or testing."""
    output, _ = study_paths
    lock = {
        "protocol_lock_sha256": study.sha256_file(output / "protocol-lock.json"),
        "probe_protocol": json.loads(json.dumps(study.EVALUATOR)),
        "datasets": {},
    }
    if tamper == "protocol":
        study.write_json(output / "protocol-lock.json", {"changed_after_selection": True})
    else:
        lock["probe_protocol"]["tol"] = 0.01
    study.write_json(output / "selection-lock.json", lock)
    monkeypatch.setattr(study, "train", lambda *args: pytest.fail("Training must not start"))
    monkeypatch.setattr(study, "final_test", lambda *args: pytest.fail("Test must not start"))
    with pytest.raises(AssertionError):
        study.confirm(jobs=1)
