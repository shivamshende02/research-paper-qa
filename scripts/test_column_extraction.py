
import json
from pathlib import Path

from src.pdf_extractor import extract_research_paper


PDF_PATH = Path(
    "data/feec53e3-4d20-4ce3-bb77-b58c7484efe9.pdf"
)

OUTPUT_PATH = Path("outputs/column_extraction.json")


def main() -> None:
    """Run PDF extraction and save the structured output."""
    result = extract_research_paper(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Document UUID: {result['uuid']}")
    print(f"Sections extracted: {len(result['sections'])}")
    print(f"Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
