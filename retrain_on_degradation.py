"""Detect quality decay from reviewed reports and train a separate candidate."""

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from check_model_metrics import check_report, dataset_fingerprint


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_policy(policy):
    for key in (
        "macro_f1_drop",
        "class_recall_drop",
        "candidate_min_macro_f1_gain",
        "candidate_max_recall_drop",
    ):
        value = policy[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid policy threshold: {key}")
    if policy["macro_f1_drop"] <= 0 or policy["class_recall_drop"] <= 0:
        raise ValueError("Degradation thresholds must be positive")
    cooldown = policy["cooldown_hours"]
    if (
        type(cooldown) not in (int, float)
        or not math.isfinite(cooldown)
        or cooldown <= 0
    ):
        raise ValueError("Cooldown must be positive")
    if (
        type(policy["min_samples_per_class"]) is not int
        or policy["min_samples_per_class"] < 1
    ):
        raise ValueError("Minimum support must be positive")


def reviewed_metrics(report, policy):
    result = check_report(
        report,
        {
            "classes": policy["classes"],
            "min_accuracy": 0,
            "min_macro_f1": 0,
            "min_recall_per_class": 0,
            "min_samples_per_class": policy["min_samples_per_class"],
            "max_failed_images": 0,
            "require_reviewed_labels": True,
            "require_independent_test": True,
        },
    )
    review = report.get("dataset_review", {})
    if (
        review.get("dataset_sha256") != report["dataset_sha256"]
        or review.get("label_review_status") != "reviewed"
        or review.get("independent_test") is not True
        or not review.get("reviewer")
        or not review.get("provenance_notes")
    ):
        raise ValueError(
            "Dataset review must name reviewer, fingerprint and provenance"
        )
    if not result["passed"]:
        raise ValueError("Unusable monitoring report: " + "; ".join(result["failures"]))
    return result


def report_time(report):
    value = datetime.fromisoformat(report["evaluated_at_utc"])
    if value.tzinfo is None:
        raise ValueError("Evaluation time must include timezone")
    return value


def detect_degradation(baseline, current, policy):
    validate_policy(policy)
    old = reviewed_metrics(baseline, policy)
    new = reviewed_metrics(current, policy)
    if old["model_sha256"] != new["model_sha256"]:
        raise ValueError("Monitoring periods must refer to the same model version")
    if report_time(current) <= report_time(baseline):
        raise ValueError("Current evaluation must be newer than baseline")
    before = baseline.get("inference_code_sha256", {}).get("service_ml.py")
    after = current.get("inference_code_sha256", {}).get("service_ml.py")
    if not before or before != after:
        raise ValueError("Monitoring periods must use the same inference code")
    reasons = []
    drop = old["macro_f1"] - new["macro_f1"]
    if drop > policy["macro_f1_drop"] + 1e-12:
        reasons.append(f"macro-F1 fell by {drop:.4f}")
    for label in policy["classes"]:
        recall_drop = (
            old["per_class"][label]["recall"] - new["per_class"][label]["recall"]
        )
        if recall_drop > policy["class_recall_drop"] + 1e-12:
            reasons.append(f"{label}: recall fell by {recall_drop:.4f}")
    return {
        "degraded": bool(reasons),
        "reasons": reasons,
        "baseline": old,
        "current": new,
    }


def scan_images(root, classes):
    # The same extension list as torchvision ImageFolder, without importing torch.
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    }
    root = Path(root)
    directories = {p.name for p in root.iterdir() if p.is_dir()}
    if directories != set(classes):
        raise ValueError("Dataset class directories differ from monitoring classes")
    records = []
    for label in classes:
        images = sorted(
            p
            for p in (root / label).rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        if not images:
            raise ValueError(f"No images in {label}")
        records.extend({"expected": label, "sha256": file_hash(p)} for p in images)
    return records


def training_preflight(
    train_dir, evaluation_dir, train_metadata, current, baseline, policy
):
    training = scan_images(train_dir, policy["classes"])
    evaluation = scan_images(evaluation_dir, policy["classes"])
    if dataset_fingerprint(evaluation) != current["dataset_sha256"]:
        raise ValueError("Evaluation directory differs from reviewed current report")
    metadata = json.loads(Path(train_metadata).read_text())
    if (
        metadata.get("dataset_sha256") != dataset_fingerprint(training)
        or metadata.get("label_review_status") != "reviewed"
        or not metadata.get("reviewer")
    ):
        raise ValueError("Training data must have matching reviewed metadata")
    training_hashes = {r["sha256"] for r in training}
    holdout_hashes = {r["sha256"] for r in evaluation + baseline["predictions"]}
    if training_hashes & holdout_hashes:
        raise ValueError("Training images overlap monitoring/holdout images")
    return metadata


def train_candidate(data, output, trials, tuning_epochs, final_epochs):
    from auto_ml import TrainingConfig
    from auto_ml import main as train

    train(
        TrainingConfig(
            data_dir=Path(data),
            output_path=Path(output),
            trials=trials,
            tuning_epochs=tuning_epochs,
            final_epochs=final_epochs,
        )
    )


def evaluate_candidate(model, data, metadata, output, log):
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_model.py")),
        "--model",
        str(model),
        "--data",
        str(data),
        "--per-class",
        "0",
        "--metadata",
        str(metadata),
        "--output",
        str(output),
    ]
    with Path(log).open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return json.loads(Path(output).read_text())


def run(args):
    baseline = json.loads(args.baseline.read_text())
    current = json.loads(args.current.read_text())
    policy = json.loads(args.policy.read_text())
    decision = detect_degradation(baseline, current, policy)
    if not decision["degraded"]:
        return {"status": "stable", **decision}
    if not args.execute:
        return {"status": "retraining_recommended", **decision}
    if not all((args.train_data, args.train_metadata, args.evaluation_data)):
        raise ValueError(
            "Execution requires --train-data, --train-metadata and --evaluation-data"
        )
    if file_hash(args.model) != current["model_sha256"]:
        raise ValueError("Current checkpoint does not match monitoring report")
    if current["inference_code_sha256"]["service_ml.py"] != file_hash(
        Path(__file__).with_name("service_ml.py")
    ):
        raise ValueError("Local inference code differs from monitored version")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        state_path = args.work_dir / "state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"attempts": []}
        )
        key = hashlib.sha256(
            (current["model_sha256"] + current["dataset_sha256"]).encode()
        ).hexdigest()
        if any(attempt["input_key"] == key for attempt in state["attempts"]):
            return {"status": "already_processed", **decision}
        now = datetime.now(UTC)
        if state["attempts"]:
            last = datetime.fromisoformat(state["attempts"][-1]["started_at_utc"])
            if (now - last).total_seconds() < policy["cooldown_hours"] * 3600:
                return {"status": "cooldown", **decision}
        metadata = training_preflight(
            args.train_data,
            args.evaluation_data,
            args.train_metadata,
            current,
            baseline,
            policy,
        )
        run_dir = args.work_dir / (
            now.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        run_dir.mkdir()
        checkpoint = run_dir / "candidate.pth"
        attempt = {
            "input_key": key,
            "started_at_utc": now.isoformat(),
            "status": "training",
            "run_dir": str(run_dir),
        }
        state["attempts"].append(attempt)
        write_json(state_path, state)
        write_json(
            run_dir / "training_context.json",
            {
                "decision": decision,
                "training_data_review": metadata,
                "trials": args.trials,
                "tuning_epochs": args.tuning_epochs,
                "final_epochs": args.final_epochs,
                "active_checkpoint": str(args.model),
            },
        )
        try:
            train_candidate(
                args.train_data,
                checkpoint,
                args.trials,
                args.tuning_epochs,
                args.final_epochs,
            )
            review_path = run_dir / "evaluation_metadata.json"
            write_json(review_path, current["dataset_review"])
            candidate_report = evaluate_candidate(
                checkpoint,
                args.evaluation_data,
                review_path,
                run_dir / "candidate_report.json",
                run_dir / "evaluation.log",
            )
            candidate = reviewed_metrics(candidate_report, policy)
            if candidate["model_sha256"] != file_hash(checkpoint):
                raise ValueError("Candidate report does not match trained checkpoint")
            if candidate["dataset_sha256"] != current["dataset_sha256"]:
                raise ValueError("Candidate evaluated on different holdout")
            gain = candidate["macro_f1"] - decision["current"]["macro_f1"]
            recalls_ok = all(
                candidate["per_class"][c]["recall"]
                + policy["candidate_max_recall_drop"]
                + 1e-12
                >= decision["current"]["per_class"][c]["recall"]
                for c in policy["classes"]
            )
            recommended = (
                gain + 1e-12 >= policy["candidate_min_macro_f1_gain"] and recalls_ok
            )
            result = {
                "status": "candidate_ready"
                if recommended
                else "candidate_not_improved",
                "candidate_path": str(checkpoint),
                "macro_f1_gain": gain,
                "manual_approval_required": True,
                "production_model_replaced": False,
                "candidate": candidate,
                **decision,
            }
            attempt["status"] = result["status"]
            write_json(run_dir / "result.json", result)
            return result
        except Exception as error:
            attempt["status"] = "failed"
            write_json(
                run_dir / "result.json", {"status": "failed", "error": str(error)}
            )
            raise
        finally:
            attempt["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_json(state_path, state)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("retraining_policy.json"))
    parser.add_argument("--model", type=Path, default=Path("model_checkpoint.pth"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--train-metadata", type=Path)
    parser.add_argument("--evaluation-data", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("retraining_runs"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/retraining_decision.json")
    )
    parser.add_argument("--trials", type=positive, default=3)
    parser.add_argument("--tuning-epochs", type=positive, default=2)
    parser.add_argument("--final-epochs", type=positive, default=5)
    args = parser.parse_args()
    try:
        result = run(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        result = {"status": "blocked_or_failed", "error": str(error)}
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2) from error
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
