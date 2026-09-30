from pathlib import Path
import json

import pdfplumber


PDF_DIR = Path("data/pdfs")
OUTPUT_DIR = Path("data/raw")


def extract_pdf(pdf_path: Path) -> dict:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            pages.append(
                {
                    "page": page_number,
                    "text": text or "",
                }
            )

    return {
        "source": pdf_path.name,
        "pages": pages,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")

        document = extract_pdf(pdf_path)

        output_file = (
            OUTPUT_DIR / f"{pdf_path.stem}.json"
        )

        output_file.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()