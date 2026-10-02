import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

INPUT_FILE = BASE_DIR / "data" / "processed" / "training.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "tokenized"

MAX_LENGTH = 512


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Training dataset not found: {INPUT_FILE}")
    if INPUT_FILE.stat().st_size == 0:
        raise ValueError(
            f"Training dataset is empty: {INPUT_FILE}. "
            "Run Training_Dataset.py and check that source paragraphs meet the minimum length."
        )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=str(INPUT_FILE),
        split="train",
    )

    if len(dataset) == 0:
        raise ValueError(f"Training dataset contains no examples: {INPUT_FILE}")

    print(f"Training examples: {len(dataset):,}")

    def tokenize_example(example):
        messages = example["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    print("Tokenizing...")

    tokenized_dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    output_path = Path(OUTPUT_DIR)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenized_dataset.save_to_disk(
        str(output_path)
    )

    print()
    print("Tokenization complete.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()