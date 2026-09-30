from pathlib import Path
import json
import re


INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/clean")


def clean_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Fix words split across lines.
    # education-
    # al
    # becomes:
    # educational
    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text,
    )

    # Normalize spaces and tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Replace single line breaks with spaces
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Remove whitespace around paragraph boundaries
    text = re.sub(
        r" *\n\n *",
        "\n\n",
        text,
    )

    return text.strip()


def clean_document(document: dict) -> dict:
    cleaned_pages = []

    for page in document["pages"]:
        cleaned_pages.append(
            {
                "page": page["page"],
                "text": clean_text(page["text"]),
            }
        )

    return {
        "source": document["source"],
        "pages": cleaned_pages,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_files = list(INPUT_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {INPUT_DIR}"
        )

    for input_file in json_files:
        print(f"Cleaning: {input_file.name}")

        document = json.loads(
            input_file.read_text(
                encoding="utf-8"
            )
        )

        cleaned_document = clean_document(
            document
        )

        output_file = (
            OUTPUT_DIR / input_file.name
        )

        output_file.write_text(
            json.dumps(
                cleaned_document,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()