#!/usr/bin/env python3
"""Smoke tests for the Fine_Tuining project.

This script validates that the repo is structurally sound and that the expected
training inputs are present before attempting a full fine-tuning run.

Usage:
    python test.py
    python test.py --skip-data
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fail(message: str) -> None:
    raise AssertionError(message)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"Missing required {label}: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        fail(f"Missing required {label}: {path}")


def assert_import(name: str) -> None:
    try:
        importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        fail(f"Unable to import '{name}': {exc}")


def validate_environment() -> None:
    if sys.version_info[:2] < (3, 10):
        fail("Python 3.10+ is required.")

    required_modules = [
        "torch",
        "datasets",
        "transformers",
        "peft",
        "trl",
        "accelerate",
    ]

    for module_name in required_modules:
        assert_import(module_name)

    require_file(ROOT / "training_common.py", "training config")
    require_file(ROOT / "Tokenize.py", "tokenizer script")
    require_file(ROOT / "fine_tuneing.py", "CPU training script")
    require_file(ROOT / "fine_tuneing_qlora.py", "GPU training script")
    require_directory(ROOT / "data", "dataset directory")


def validate_model_config() -> None:
    sys.path.insert(0, str(ROOT))
    import training_common

    if not training_common.MODEL_NAME.strip():
        fail("MODEL_NAME is empty in training_common.py")

    model_name = training_common.MODEL_NAME.lower()
    if "llama" not in model_name and "qwen" not in model_name and "gemma" not in model_name:
        fail(f"MODEL_NAME '{training_common.MODEL_NAME}' does not look like a supported Hugging Face model.")

    for path_value in [
        training_common.DATASET_DIR,
        training_common.CPU_ADAPTER_DIR,
        training_common.QLORA_ADAPTER_DIR,
        training_common.MERGED_DIR,
    ]:
        if not path_value.name:
            fail(f"Invalid path value in training_common.py: {path_value!r}")


def validate_dataset(args: argparse.Namespace) -> None:
    processed_file = ROOT / "data" / "processed" / "training.jsonl"
    tokenized_dir = ROOT / "data" / "tokenized"

    if not processed_file.exists() and not args.skip_data:
        fail(f"Training dataset is missing: {processed_file}")

    if processed_file.exists():
        with processed_file.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
            if not first_line:
                fail(f"Training dataset is empty: {processed_file}")
            try:
                payload = json.loads(first_line)
            except json.JSONDecodeError as exc:
                fail(f"Training dataset line is not valid JSON: {exc}")

            if "messages" not in payload:
                fail("Training dataset record is missing the 'messages' field.")

            if not isinstance(payload["messages"], list) or len(payload["messages"]) < 2:
                fail("Training dataset record must include at least two chat messages.")

    if tokenized_dir.exists() and not tokenized_dir.is_dir():
        fail(f"Tokenized dataset path exists but is not a directory: {tokenized_dir}")

    if not processed_file.exists() and args.skip_data:
        print("[SKIP] Data checks skipped because training.jsonl is not present yet.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Fine_Tuining codebase.")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip validation of the generated dataset files.",
    )
    args = parser.parse_args()

    try:
        validate_environment()
        validate_model_config()
        validate_dataset(args)
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS: project environment and configuration look valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())