"""Opt-in feedback storage. User labels are claims, not independent validation."""

import hashlib
import io
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

CLASSES = ("animals", "city", "documents", "nature", "people")


class FeedbackStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY, image BLOB NOT NULL, sha256 TEXT NOT NULL,
                predicted TEXT NOT NULL, confidence REAL NOT NULL, model_sha256 TEXT,
                created_at TEXT NOT NULL, label TEXT, reviewed_at TEXT)""")

    def connect(self):
        return sqlite3.connect(self.root / "feedback.sqlite3", timeout=30)

    def record(self, image: bytes, predicted: str, confidence: float, model_sha256):
        sample_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    sample_id,
                    image,
                    hashlib.sha256(image).hexdigest(),
                    predicted,
                    confidence,
                    model_sha256,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return sample_id

    def confirm(self, sample_id: str, label: str):
        if label not in CLASSES:
            raise ValueError("Неизвестная категория")
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE samples SET label=?, reviewed_at=? WHERE id=?",
                (label, datetime.now(UTC).isoformat(), sample_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(sample_id)

    def export(self, output: Path, exclude: list[Path]):
        """Create a new candidate snapshot; never certify test independence."""
        excluded = set()
        for directory in exclude:
            if not directory.is_dir():
                raise ValueError(f"Каталог исключения не найден: {directory}")
            for path in directory.rglob("*"):
                if path.is_file():
                    excluded.add(hashlib.sha256(path.read_bytes()).hexdigest())
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, image, sha256, label FROM samples "
                "WHERE label IS NOT NULL ORDER BY id"
            ).fetchall()
        labels = {}
        for _, _, digest, label in rows:
            labels.setdefault(digest, set()).add(label)
        output.mkdir(parents=True, exist_ok=False)
        manifest = []
        seen = set()
        for sample_id, image, digest, label in rows:
            if digest in excluded or digest in seen or len(labels[digest]) != 1:
                continue
            # Preserve supported originals; convert other formats for ImageFolder.
            exported = image
            with Image.open(io.BytesIO(image)) as decoded:
                suffix = {
                    "PNG": ".png",
                    "JPEG": ".jpg",
                    "WEBP": ".webp",
                    "BMP": ".bmp",
                    "TIFF": ".tiff",
                    "PPM": ".ppm",
                }.get(decoded.format)
                if suffix is None:
                    buffer = io.BytesIO()
                    decoded.convert("RGB").save(buffer, format="PNG")
                    exported = buffer.getvalue()
                    suffix = ".png"
                else:
                    decoded.verify()
            exported_hash = hashlib.sha256(exported).hexdigest()
            if exported_hash in excluded:
                continue
            path = output / label / f"{digest}{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(exported)
            manifest.append(
                {
                    "sample_id": sample_id,
                    "path": str(path.relative_to(output)),
                    "source_sha256": digest,
                    "sha256": exported_hash,
                    "label": label,
                }
            )
            seen.add(digest)
        metadata = {
            "label_source": "explicit_user_feedback",
            "label_review_status": "unreviewed",
            "independence_status": "unverified",
            "created_at": datetime.now(UTC).isoformat(),
            "samples": manifest,
            "count": len(manifest),
        }
        (output / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        return metadata
