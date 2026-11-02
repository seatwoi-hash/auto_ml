"""Collect licensed Wikimedia candidates, NOT a validated independent test set."""

import hashlib
import io
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "my_dataset" / "test_candidates"
QUERIES = {
    "animals": [
        "cat photograph",
        "dog photograph",
        "wild bird photograph",
        "elephant photograph",
    ],
    "city": [
        "city skyline photograph",
        "urban street photograph",
        "city buildings photograph",
        "town square photograph",
    ],
    "documents": [
        "scanned letter",
        "scanned manuscript page",
        "printed document scan",
        "typewritten letter",
    ],
    "nature": [
        "mountain landscape photograph",
        "forest landscape photograph",
        "waterfall landscape photograph",
        "lake landscape photograph",
    ],
    "people": [
        "portrait photograph person",
        "people walking photograph",
        "group portrait photograph",
        "athlete portrait photograph",
    ],
}


class RateLimited(RuntimeError):
    """Stop collection rather than repeatedly hitting a throttled source."""


def fetch(url):
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "0",
            "--max-time",
            "45",
            "--user-agent",
            "PhotoDistributionEvaluation/1.0 (research dataset collection)",
            url,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        error = result.stderr.decode(errors="replace")
        if "429" in error:
            raise RateLimited("Wikimedia returned HTTP 429; retry later")
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return result.stdout


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "manifest.json"
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    seen = {r["page_id"] for r in records}
    hashes = {r["download_sha256"] for r in records}
    for label, queries in QUERIES.items():
        folder = DEST / label
        folder.mkdir(exist_ok=True)
        for query in queries:
            count = sum(r["candidate_label"] == label for r in records)
            if count >= 100:
                break
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": "filetype:bitmap " + query,
                "gsrnamespace": 6,
                "gsrlimit": 40,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 640,
            }
            response = json.loads(
                fetch("https://commons.wikimedia.org/w/api.php?" + urlencode(params))
            )
            pages = sorted(
                response.get("query", {}).get("pages", {}).values(),
                key=lambda p: p.get("index", 0),
            )
            accepted = 0
            for page in pages:
                if (
                    page["pageid"] in seen
                    or accepted >= 25
                    or sum(r["candidate_label"] == label for r in records) >= 100
                ):
                    continue
                infos = page.get("imageinfo", [])
                if not infos:
                    continue
                info = infos[0]
                meta = info.get("extmetadata", {})

                def value(key, metadata=meta):
                    return metadata.get(key, {}).get("value", "")

                license_name = value("LicenseShortName")
                if not (
                    license_name.startswith("CC BY")
                    or license_name in {"CC0", "Public domain"}
                ):
                    continue
                url = info.get("thumburl", info["url"])
                try:
                    data = fetch(url)
                    with Image.open(io.BytesIO(data)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(data)) as image:
                        if min(image.size) < 100:
                            continue
                        extension = {
                            "JPEG": ".jpg",
                            "PNG": ".png",
                            "WEBP": ".webp",
                        }.get(image.format)
                    if extension is None:
                        continue
                except (subprocess.CalledProcessError, OSError, ValueError) as error:
                    print(
                        f"Skipped {page['pageid']}: {type(error).__name__}", flush=True
                    )
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in hashes:
                    continue
                path = folder / (str(page["pageid"]) + extension)
                path.write_bytes(data)
                records.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "candidate_label": label,
                        "review_status": "unreviewed",
                        "independence": "unverified",
                        "page_id": page["pageid"],
                        "title": page["title"],
                        "source_page": info["descriptionurl"],
                        "download_url": url,
                        "original_url": info["url"],
                        "license": license_name,
                        "license_url": value("LicenseUrl"),
                        "artist_html": value("Artist"),
                        "credit_html": value("Credit"),
                        "description_html": value("ImageDescription"),
                        "restrictions": value("Restrictions"),
                        "query": query,
                        "download_sha256": digest,
                    }
                )
                seen.add(page["pageid"])
                hashes.add(digest)
                accepted += 1
                manifest_path.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2) + "\n"
                )
                time.sleep(3)
            print(
                f"{label}: {sum(r['candidate_label'] == label for r in records)} candidates",
                flush=True,
            )
    print(f"Saved {len(records)} unreviewed candidates to {DEST}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RateLimited as error:
        raise SystemExit(str(error)) from error
