"""Export explicit user feedback into a fresh candidate dataset snapshot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback import FeedbackStore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("feedback_data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        required=True,
        help="Existing train/test/baseline directories; repeat for each",
    )
    args = parser.parse_args()
    result = FeedbackStore(args.store).export(args.output, args.exclude)
    print(
        f"Exported {result['count']} candidates; review and independence remain unverified"
    )


if __name__ == "__main__":
    main()
