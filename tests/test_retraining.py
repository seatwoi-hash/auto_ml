import copy
import hashlib
import json
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

import retrain_on_degradation as monitor
from check_model_metrics import dataset_fingerprint


def make_report(model_hash, degraded=False, newer=False):
    rows = [
        {
            "expected": label,
            "predicted": "animals" if degraded else label,
            "sha256": hashlib.sha256(f"{label}{i}{newer}".encode()).hexdigest(),
        }
        for label in ("animals", "nature")
        for i in range(4)
    ]
    fingerprint = dataset_fingerprint(rows)
    return {
        "predictions": rows,
        "evaluated": len(rows),
        "failed": 0,
        "model_sha256": model_hash,
        "dataset_sha256": fingerprint,
        "class_order": ["animals", "nature"],
        "label_review_status": "reviewed",
        "independent_test": True,
        "evaluated_at_utc": "2026-11-02T12:00:00+00:00"
        if newer
        else "2026-11-01T12:00:00+00:00",
        "inference_code_sha256": {
            "service_ml.py": monitor.file_hash(
                Path(monitor.__file__).with_name("service_ml.py")
            )
        },
        "dataset_review": {
            "dataset_sha256": fingerprint,
            "reviewer": "Reviewer",
            "provenance_notes": "Reviewed new independent uploads",
            "label_review_status": "reviewed",
            "independent_test": True,
        },
    }


@pytest.fixture
def policy():
    return {
        "classes": ["animals", "nature"],
        "macro_f1_drop": 0.03,
        "class_recall_drop": 0.1,
        "min_samples_per_class": 2,
        "cooldown_hours": 24,
        "candidate_min_macro_f1_gain": 0.01,
        "candidate_max_recall_drop": 0,
    }


def test_stable_quality_does_not_trigger(policy):
    assert not monitor.detect_degradation(
        make_report("a" * 64), make_report("a" * 64, newer=True), policy
    )["degraded"]


def test_decay_is_recomputed_from_predictions(policy):
    current = make_report("a" * 64, degraded=True, newer=True)
    current["metrics"] = {"accuracy": 1, "macro avg": {"f1-score": 1}}
    result = monitor.detect_degradation(make_report("a" * 64), current, policy)
    assert result["degraded"]
    assert any("nature" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "change", ["unreviewed", "small", "model", "old", "code", "fingerprint"]
)
def test_bad_reports_do_not_trigger_training(policy, change):
    current = make_report("a" * 64, degraded=True, newer=True)
    if change == "unreviewed":
        current["label_review_status"] = "unreviewed"
    if change == "small":
        policy["min_samples_per_class"] = 5
    if change == "model":
        current["model_sha256"] = "b" * 64
    if change == "old":
        current["evaluated_at_utc"] = "2020-01-01T00:00:00+00:00"
    if change == "code":
        current["inference_code_sha256"]["service_ml.py"] = "b" * 64
    if change == "fingerprint":
        current["dataset_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        monitor.detect_degradation(make_report("a" * 64), current, policy)


@pytest.mark.parametrize("value", [float("nan"), -0.1, 2, 0])
def test_invalid_trigger_threshold_rejected(policy, value):
    policy["macro_f1_drop"] = value
    with pytest.raises(ValueError):
        monitor.validate_policy(policy)


@pytest.fixture
def scenario(tmp_path, policy, monkeypatch):
    active = tmp_path / "active.pth"
    active.write_bytes(b"production weights")
    baseline = make_report(monitor.file_hash(active))
    current = make_report(monitor.file_hash(active), degraded=True, newer=True)
    for root, offset in [(tmp_path / "train", 0), (tmp_path / "holdout", 100)]:
        for i, label in enumerate(policy["classes"]):
            folder = root / label
            folder.mkdir(parents=True)
            for j in range(4):
                Image.new("RGB", (8, 8), color=(offset + i * 20 + j, 0, 0)).save(
                    folder / f"{j}.png"
                )
    records = monitor.scan_images(tmp_path / "holdout", policy["classes"])
    for actual, row in zip(records, current["predictions"], strict=True):
        row["sha256"] = actual["sha256"]
    current["dataset_sha256"] = dataset_fingerprint(current["predictions"])
    current["dataset_review"]["dataset_sha256"] = current["dataset_sha256"]
    monitor.write_json(tmp_path / "baseline.json", baseline)
    monitor.write_json(tmp_path / "current.json", current)
    monitor.write_json(tmp_path / "policy.json", policy)
    monitor.write_json(
        tmp_path / "train_review.json",
        {
            "dataset_sha256": dataset_fingerprint(
                monitor.scan_images(tmp_path / "train", policy["classes"])
            ),
            "label_review_status": "reviewed",
            "reviewer": "Reviewer",
        },
    )
    args = Namespace(
        baseline=tmp_path / "baseline.json",
        current=tmp_path / "current.json",
        policy=tmp_path / "policy.json",
        model=active,
        execute=True,
        train_data=tmp_path / "train",
        train_metadata=tmp_path / "train_review.json",
        evaluation_data=tmp_path / "holdout",
        work_dir=tmp_path / "runs",
        trials=1,
        tuning_epochs=1,
        final_epochs=1,
    )
    calls = []

    def train(data, output, *budget):
        calls.append(output)
        output.write_bytes(b"candidate weights")

    def evaluate(model, data, metadata, output, log):
        report = copy.deepcopy(current)
        for row in report["predictions"]:
            row["predicted"] = row["expected"]
        report["model_sha256"] = monitor.file_hash(model)
        monitor.write_json(output, report)
        return report

    monkeypatch.setattr(monitor, "train_candidate", train)
    monkeypatch.setattr(monitor, "evaluate_candidate", evaluate)
    return args, calls


def test_dry_run_never_trains(scenario):
    args, calls = scenario
    args.execute = False
    assert monitor.run(args)["status"] == "retraining_recommended"
    assert not calls
    assert not args.work_dir.exists()


def test_candidate_is_separate_and_same_report_cannot_retrain(scenario):
    args, calls = scenario
    result = monitor.run(args)
    assert result["status"] == "candidate_ready"
    assert result["production_model_replaced"] is False
    assert args.model.read_bytes() == b"production weights"
    assert Path(result["candidate_path"]).read_bytes() == b"candidate weights"
    assert monitor.run(args)["status"] == "already_processed"
    assert len(calls) == 1


def test_training_overlap_with_holdout_blocks_before_training(scenario):
    args, calls = scenario
    (args.train_data / "animals" / "0.png").write_bytes(
        (args.evaluation_data / "animals" / "0.png").read_bytes()
    )
    metadata = json.loads(args.train_metadata.read_text())
    metadata["dataset_sha256"] = dataset_fingerprint(
        monitor.scan_images(args.train_data, ["animals", "nature"])
    )
    monitor.write_json(args.train_metadata, metadata)
    with pytest.raises(ValueError, match="overlap"):
        monitor.run(args)
    assert not calls


def test_holdout_changes_block_training(scenario):
    args, calls = scenario
    (args.evaluation_data / "nature" / "0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="directory differs"):
        monitor.run(args)
    assert not calls


def test_cooldown_blocks_another_run(scenario):
    args, calls = scenario
    args.work_dir.mkdir()
    monitor.write_json(
        args.work_dir / "state.json",
        {
            "attempts": [
                {
                    "input_key": "other report",
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "status": "failed",
                }
            ]
        },
    )
    assert monitor.run(args)["status"] == "cooldown"
    assert not calls


def test_failed_training_is_recorded_without_replacing_active(scenario, monkeypatch):
    args, calls = scenario

    def fail(*args):
        raise RuntimeError("training failed")

    monkeypatch.setattr(monitor, "train_candidate", fail)
    with pytest.raises(RuntimeError):
        monitor.run(args)
    state = json.loads((args.work_dir / "state.json").read_text())
    assert state["attempts"][0]["status"] == "failed"
    assert args.model.read_bytes() == b"production weights"


def test_candidate_without_improvement_is_not_recommended(scenario, monkeypatch):
    args, calls = scenario

    def unchanged(model, *unused):
        report = json.loads(args.current.read_text())
        report["model_sha256"] = monitor.file_hash(model)
        return report

    monkeypatch.setattr(monitor, "evaluate_candidate", unchanged)
    assert monitor.run(args)["status"] == "candidate_not_improved"
    assert args.model.read_bytes() == b"production weights"


def test_parallel_run_is_skipped(scenario):
    args, calls = scenario
    args.work_dir.mkdir()
    with (args.work_dir / "monitor.lock").open("a") as lock:
        monitor.fcntl.flock(lock, monitor.fcntl.LOCK_EX | monitor.fcntl.LOCK_NB)
        assert monitor.run(args)["status"] == "already_running"
    assert not calls
