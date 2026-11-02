"""Check a saved evaluation against a versioned quality policy (no ML runtime)."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def dataset_fingerprint(predictions):
    """Bind directory labels and evaluated bytes, independent of path/order."""
    rows = sorted((r["expected"], r["sha256"]) for r in predictions)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def check_report(report, policy):
    classes = policy["classes"]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Policy must specify distinct classes")
    for key in ("min_accuracy", "min_macro_f1", "min_recall_per_class"):
        value = policy[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid threshold: {key}")
        if not 0 <= value <= 1:
            raise ValueError(f"Threshold outside [0, 1]: {key}")
    for key in ("min_samples_per_class", "max_failed_images"):
        if type(policy[key]) is not int or policy[key] < 0:
            raise ValueError(f"Invalid count: {key}")
    for key in ("require_reviewed_labels", "require_independent_test"):
        if type(policy[key]) is not bool:
            raise ValueError(f"Invalid boolean: {key}")
    if set(report["class_order"]) != set(classes):
        raise ValueError("Report class set differs from policy")
    if not re.fullmatch(r"[0-9a-f]{64}", report["model_sha256"]):
        raise ValueError("Missing or invalid model SHA-256")
    predictions = report["predictions"]
    if not predictions or len(predictions) != report["evaluated"]:
        raise ValueError("Empty or inconsistent prediction count")
    if type(report["failed"]) is not int or report["failed"] < 0:
        raise ValueError("Invalid failed image count")
    support, predicted, correct = Counter(), Counter(), Counter()
    for row in predictions:
        actual, prediction = row["expected"], row["predicted"]
        if actual not in classes or prediction not in classes:
            raise ValueError("Unknown class in predictions")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid image SHA-256")
        support[actual] += 1
        predicted[prediction] += 1
        correct[actual] += actual == prediction
    fingerprint = dataset_fingerprint(predictions)
    if report.get("dataset_sha256") != fingerprint:
        raise ValueError("Dataset fingerprint missing or inconsistent")
    per_class = {}
    for label in classes:
        precision = correct[label] / predicted[label] if predicted[label] else 0
        recall = correct[label] / support[label] if support[label] else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[label],
        }
    accuracy = sum(correct.values()) / len(predictions)
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(classes)
    failures = []
    if accuracy < policy["min_accuracy"]:
        failures.append(f"accuracy {accuracy:.4f} < {policy['min_accuracy']}")
    if macro_f1 < policy["min_macro_f1"]:
        failures.append(f"macro_f1 {macro_f1:.4f} < {policy['min_macro_f1']}")
    for label, metrics in per_class.items():
        if metrics["recall"] < policy["min_recall_per_class"]:
            failures.append(
                f"{label}: recall {metrics['recall']:.4f} < "
                f"{policy['min_recall_per_class']}"
            )
        if metrics["support"] < policy["min_samples_per_class"]:
            failures.append(f"{label}: insufficient samples ({metrics['support']})")
    if report["failed"] > policy["max_failed_images"]:
        failures.append(f"Failed images: {report['failed']}")
    if (
        policy["require_reviewed_labels"]
        and report.get("label_review_status") != "reviewed"
    ):
        failures.append("Test labels are not reviewed")
    if (
        policy["require_independent_test"]
        and report.get("independent_test") is not True
    ):
        failures.append("Test independence is not verified")
    if len({row["sha256"] for row in predictions}) != len(predictions):
        failures.append("Exact duplicate images in evaluation")
    return {
        "policy": policy,
        "passed": not failures,
        "failures": failures,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "model_sha256": report["model_sha256"],
        "dataset_sha256": fingerprint,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, help="Also verify the released checkpoint hash"
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("model_metrics_policy.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/model_metrics_gate.json")
    )
    args = parser.parse_args()
    try:
        result = check_report(
            json.loads(args.report.read_text()), json.loads(args.policy.read_text())
        )
        saved_report = json.loads(args.report.read_text())
        for name in ("service_ml.py", "evaluate_model.py"):
            current_hash = hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            if saved_report.get("inference_code_sha256", {}).get(name) != current_hash:
                result["passed"] = False
                result["failures"].append(
                    f"Evaluation code changed or unrecorded: {name}"
                )
        if args.model:
            actual_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
            if actual_hash != result["model_sha256"]:
                result["passed"] = False
                result["failures"].append("Checkpoint differs from evaluated model")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"passed": False, "failures": [f"Invalid evaluation: {error}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
