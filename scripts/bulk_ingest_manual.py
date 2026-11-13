
import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "manual_summaries" / "manual_summaries.json"

BASE_URL = "http://127.0.0.1:8000"
DOCUMENT_ID = "ee431476-669e-4980-9a8c-b52d82571e43"


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    if data.get("uuid") != DOCUMENT_ID:
        print("ERROR: Document UUID does not match.")
        sys.exit(1)

    sections = data.get("sections", [])

    if len(sections) != 37:
        print(f"ERROR: Expected 37 sections, found {len(sections)}.")
        sys.exit(1)

    for section in sections:
        if not str(section.get("summary", "")).strip():
            print(f"ERROR: Missing summary for section {section.get('section_id')}")
            sys.exit(1)

    print(f"Validated {len(sections)} sections.")
    print("Starting ingestion...\n")

    summary_success = 0
    content_success = 0
    failures = []

    for index, section in enumerate(sections, start=1):
        section_id = str(section["section_id"])
        header = section.get("header", f"Section {section_id}")

        print(f"[{index}/{len(sections)}] {header} (ID: {section_id})")

        try:
            # Store the manually prepared summary.
            summary_response = requests.post(
                f"{BASE_URL}/api/v1/documents/{DOCUMENT_ID}/sections/{section_id}/summary",
                json={"summary": section["summary"]},
                timeout=300,
            )
            summary_response.raise_for_status()
            summary_success += 1
            print("  Summary: stored")

            # Store original extracted content as searchable chunks.
            content_response = requests.post(
                f"{BASE_URL}/api/v1/documents/{DOCUMENT_ID}/sections/{section_id}/content",
                timeout=600,
            )
            content_response.raise_for_status()
            content_result = content_response.json()
            content_success += 1

            print(
                f"  Content: stored "
                f"({content_result.get('chunks_created', 0)} chunks)"
            )

        except requests.RequestException as exc:
            failures.append({
                "section_id": section_id,
                "header": header,
                "error": str(exc),
            })
            print(f"  ERROR: {exc}")

            if getattr(exc, "response", None) is not None:
                print("  Response:", exc.response.text[:1000])

        print()

    print("=" * 55)
    print("INGESTION SUMMARY")
    print("=" * 55)
    print(f"Total sections:       {len(sections)}")
    print(f"Summaries stored:     {summary_success}")
    print(f"Content stored:       {content_success}")
    print(f"Failed sections:      {len(failures)}")

    if failures:
        print("\nFailed sections:")
        for failure in failures:
            print(failure)
        sys.exit(1)

    print("\nAll sections ingested successfully.")


if __name__ == "__main__":
    main()
