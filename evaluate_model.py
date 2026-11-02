"""Reproducible directory-label diagnostics; independence is not established."""

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from check_model_metrics import check_report, dataset_fingerprint
from service_ml import PhotoClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("my_dataset/train"))
    parser.add_argument("--model", type=Path, default=Path("model_checkpoint.pth"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_report.json"))
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--metadata", type=Path, help="Dataset review attestation JSON")
    parser.add_argument("--policy", type=Path, help="Fail if quality policy is not met")
    parser.add_argument("--history", type=Path, help="Append summary to JSONL history")
    args = parser.parse_args()
    if args.per_class < 0:
        parser.error("--per-class must be nonnegative (0 evaluates all files)")
    torch.set_num_threads(4)
    classifier = PhotoClassifier(args.model)
    rng = random.Random(42)
    records, errors, inventory = [], [], {}
    hashes = Counter()
    for label in classifier.class_names:
        files = sorted(p for p in (args.data / label).rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"No files for class {label}")
        inventory[label] = len(files)
        selected = (
            rng.sample(files, min(args.per_class, len(files)))
            if args.per_class
            else files
        )
        for path in selected:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            hashes[digest] += 1
            result = classifier._predict(data, path.name)
            if not result["success"]:
                errors.append({"path": str(path), "error": result["error"]})
                continue
            records.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "expected": label,
                    "predicted": result["class_name"],
                    "confidence": result["confidence"],
                }
            )
        print(f"Completed {label}", flush=True)
    if not records:
        raise ValueError("No valid images evaluated")
    actual = [r["expected"] for r in records]
    predicted = [r["predicted"] for r in records]
    report = {
        "evaluation_type": "diagnostic_on_labeled_directory",
        "data_path": str(args.data),
        "label_source": "directory_names_not_manually_verified",
        "independent_test": False,
        "label_review_status": "unreviewed",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": dataset_fingerprint(records),
        "inference_code_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("service_ml.py", "evaluate_model.py")
        },
        "limitation": "Unknown overlap with legacy model training; not generalization evidence",
        "seed": 42,
        "requested_per_class": args.per_class,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "threads": 4,
        "class_order": classifier.class_names,
        "inventory": inventory,
        "evaluated": len(records),
        "failed": len(errors),
        "exact_duplicate_excess_in_sample": sum(n - 1 for n in hashes.values()),
        "metrics": classification_report(
            actual,
            predicted,
            labels=classifier.class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            actual, predicted, labels=classifier.class_names
        ).tolist(),
        "errors": errors,
        "predictions": records,
    }
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
        if metadata.get("dataset_sha256") != report["dataset_sha256"]:
            parser.error(
                "Dataset review metadata fingerprint differs from evaluated files"
            )
        if metadata.get("label_review_status") == "reviewed" and not metadata.get(
            "reviewer"
        ):
            parser.error("Reviewed labels require a named reviewer")
        if metadata.get("independent_test") is True and not metadata.get(
            "provenance_notes"
        ):
            parser.error("Independent test requires documented provenance")
        report["dataset_review"] = metadata
        report["label_review_status"] = metadata.get(
            "label_review_status", "unreviewed"
        )
        report["independent_test"] = metadata.get("independent_test") is True
        if report["label_review_status"] == "reviewed":
            report["label_source"] = "reviewed_directory_labels"
        report["limitation"] = (
            "Dataset review and independence are maintainer attestations; "
            "the tool verifies fingerprints, not provenance."
        )
    if args.policy:
        report["quality_gate"] = check_report(
            report, json.loads(args.policy.read_text())
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            k: report[k]
            for k in (
                "evaluated_at_utc",
                "model_sha256",
                "dataset_sha256",
                "evaluated",
                "failed",
                "metrics",
                "label_review_status",
                "independent_test",
            )
        }
        summary["quality_gate_passed"] = report.get("quality_gate", {}).get("passed")
        summary["quality_policy"] = report.get("quality_gate", {}).get("policy")
        with args.history.open("a") as history:
            history.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"predictions", "errors"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.policy and not report["quality_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
