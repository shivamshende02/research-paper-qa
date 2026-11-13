
import json
from pathlib import Path

import pymupdf


# PDF configuration
PDF_PATH = Path(
    "data/e956b925-cdfe-47bb-8de3-2e2db06cfcbc.pdf"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "pymupdf_inspection.txt"


def extract_pdf(pdf_path: Path) -> None:
    """
    Extract text blocks and layout metadata from every PDF page.

    Records page numbers, bounding boxes, font information,
    and extracted text for manual inspection.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    try:
        with OUTPUT_FILE.open("w", encoding="utf-8") as output:
            output.write(f"PDF: {pdf_path.name}\n")
            output.write(f"Total pages: {len(document)}\n")
            output.write("=" * 90 + "\n")

            for page_index, page in enumerate(document):
                page_number = page_index + 1

                # Extract structured text and sort blocks top-to-bottom.
                page_data = page.get_text("dict", sort=True)

                output.write(f"\n\nPAGE {page_number}\n")
                output.write("=" * 90 + "\n")

                for block_index, block in enumerate(
                    page_data.get("blocks", [])
                ):
                    # Ignore image blocks for this initial text test.
                    if block.get("type") != 0:
                        continue

                    bbox = block.get("bbox", [])

                    output.write(
                        f"\nBLOCK {block_index} | "
                        f"BBOX: {bbox}\n"
                    )

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()

                            if not text:
                                continue

                            font_info = {
                                "font": span.get("font"),
                                "size": round(span.get("size", 0), 2),
                                "flags": span.get("flags"),
                                "bold_or_italic": span.get("font"),
                            }

                            output.write(
                                f"FONT: {json.dumps(font_info)}\n"
                            )
                            output.write(f"TEXT: {text}\n")

                output.write("\n")

        print(f"Extraction completed: {OUTPUT_FILE}")
        print(f"Pages processed: {len(document)}")

    finally:
        document.close()


if __name__ == "__main__":
    extract_pdf(PDF_PATH)
