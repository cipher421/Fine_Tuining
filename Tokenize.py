from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"

INPUT_FILE = "data/processed/training.jsonl"
OUTPUT_DIR = "data/tokenized"

MAX_LENGTH = 2048


def main():
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
        data_files=INPUT_FILE,
        split="train",
    )

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