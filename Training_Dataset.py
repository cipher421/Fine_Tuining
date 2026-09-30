from pathlib import Path
import json
import re


INPUT_DIR = Path("data/clean")
OUTPUT_FILE = Path("data/processed/training.jsonl")


MIN_WORDS = 20


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if len(paragraph.split()) >= MIN_WORDS
    ]


def create_training_example(
    paragraph: str,
    source: str,
    page: int,
) -> dict:

    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the following "
                    "educational content clearly."
                    "\n\n"
                    + paragraph
                ),
            },
            {
                "role": "assistant",
                "content": paragraph,
            },
        ],
        "metadata": {
            "source": source,
            "page": page,
        },
    }


def main():
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_files = list(
        INPUT_DIR.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {INPUT_DIR}"
        )

    total_examples = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for input_file in json_files:

            document = json.loads(
                input_file.read_text(
                    encoding="utf-8"
                )
            )

            for page in document["pages"]:

                paragraphs = split_into_paragraphs(
                    page["text"]
                )

                for paragraph in paragraphs:

                    example = create_training_example(
                        paragraph=paragraph,
                        source=document["source"],
                        page=page["page"],
                    )

                    output.write(
                        json.dumps(
                            example,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_examples += 1

    print(
        f"Created {total_examples:,} training examples"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()