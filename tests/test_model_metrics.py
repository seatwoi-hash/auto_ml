import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from check_model_metrics import check_report, dataset_fingerprint


@pytest.fixture
def inputs():
    predictions = [
        {
            "expected": c,
            "predicted": c,
            "sha256": hashlib.sha256(f"{c}{i}".encode()).hexdigest(),
        }
        for c in ("animals", "nature")
        for i in range(2)
    ]
    report = {
        "class_order": ["animals", "nature"],
        "evaluated": 4,
        "failed": 0,
        "predictions": predictions,
        "dataset_sha256": dataset_fingerprint(predictions),
        "model_sha256": "a" * 64,
        "label_review_status": "reviewed",
        "independent_test": True,
    }
    policy = {
        "classes": report["class_order"],
        "min_accuracy": 0.8,
        "min_macro_f1": 0.75,
        "min_recall_per_class": 0.6,
        "min_samples_per_class": 2,
        "max_failed_images": 0,
        "require_reviewed_labels": True,
        "require_independent_test": True,
    }
    return report, policy


def test_verified_perfect_model_passes_including_boundary(inputs):
    report, policy = inputs
    policy.update(min_accuracy=1.0, min_macro_f1=1.0, min_recall_per_class=1.0)
    assert check_report(report, policy)["passed"]


def test_gate_recomputes_scores_instead_of_trusting_report(inputs):
    report, policy = inputs
    report["metrics"] = {"accuracy": 1.0}
    report["predictions"][2]["predicted"] = "animals"
    report["predictions"][3]["predicted"] = "animals"
    result = check_report(report, policy)
    assert not result["passed"]
    assert result["accuracy"] == 0.5
    assert result["per_class"]["nature"]["recall"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("label_review_status", "unreviewed"),
        ("independent_test", False),
        ("failed", 1),
    ],
)
def test_unreliable_data_and_failed_images_block_release(inputs, field, value):
    report, policy = inputs
    report[field] = value
    assert not check_report(report, policy)["passed"]


def test_missing_class_cannot_be_hidden_by_high_accuracy(inputs):
    report, policy = inputs
    report["predictions"] = report["predictions"][:2]
    report["evaluated"] = 2
    report["dataset_sha256"] = dataset_fingerprint(report["predictions"])
    result = check_report(report, policy)
    assert result["accuracy"] == 1
    assert not result["passed"]
    assert any("insufficient samples" in error for error in result["failures"])


def test_fingerprint_binds_labels_but_not_paths_or_order(inputs):
    report, policy = inputs
    changed = copy.deepcopy(report)
    changed["predictions"].reverse()
    assert check_report(changed, policy)["passed"]
    changed["predictions"][0]["expected"] = "animals"
    with pytest.raises(ValueError, match="fingerprint"):
        check_report(changed, policy)


def test_duplicate_images_block_release(inputs):
    report, policy = inputs
    report["predictions"][1]["sha256"] = report["predictions"][0]["sha256"]
    report["dataset_sha256"] = dataset_fingerprint(report["predictions"])
    assert (
        "Exact duplicate images in evaluation"
        in check_report(report, policy)["failures"]
    )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_thresholds_cannot_silently_disable_gate(inputs, threshold):
    report, policy = inputs
    policy["min_accuracy"] = threshold
    with pytest.raises(ValueError):
        check_report(report, policy)


def test_cli_rejects_checkpoint_not_matching_report(inputs, tmp_path):
    report, policy = inputs
    report_path = tmp_path / "report.json"
    policy_path = tmp_path / "policy.json"
    model = tmp_path / "model.pth"
    output = tmp_path / "gate.json"
    report_path.write_text(json.dumps(report))
    policy_path.write_text(json.dumps(policy))
    model.write_bytes(b"different checkpoint")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "check_model_metrics.py"),
            "--report",
            str(report_path),
            "--policy",
            str(policy_path),
            "--model",
            str(model),
            "--output",
            str(output),
        ],
        capture_output=True,
    )
    assert result.returncode == 1
    assert "Checkpoint differs" in json.loads(output.read_text())["failures"][-1]


def test_cli_missing_report_fails_closed(tmp_path):
    output = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "check_model_metrics.py"),
            "--report",
            str(tmp_path / "absent.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
    )
    assert result.returncode == 1
    assert json.loads(output.read_text())["passed"] is False


@pytest.mark.parametrize("code_changed", [False, True])
def test_cli_checks_recorded_code_with_valid_model(inputs, tmp_path, code_changed):
    report, policy = inputs
    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "model.pth"
    model.write_bytes(b"evaluated checkpoint")
    report["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    report["inference_code_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("service_ml.py", "evaluate_model.py")
    }
    if code_changed:
        report["inference_code_sha256"]["service_ml.py"] = "b" * 64
    report_path = tmp_path / "report.json"
    policy_path = tmp_path / "policy.json"
    output = tmp_path / "gate.json"
    report_path.write_text(json.dumps(report))
    policy_path.write_text(json.dumps(policy))
    result = subprocess.run(
        [
            sys.executable,
            str(root / "check_model_metrics.py"),
            "--report",
            str(report_path),
            "--policy",
            str(policy_path),
            "--model",
            str(model),
            "--output",
            str(output),
        ],
        capture_output=True,
    )
    assert result.returncode == int(code_changed)
    assert json.loads(output.read_text())["passed"] is not code_changed
