import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments


BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

DATASET_DIR = BASE_DIR / "data" / "tokenized"
CPU_ADAPTER_DIR = BASE_DIR / "models" / "llama-3.2-3b-instruct-lora"
QLORA_ADAPTER_DIR = BASE_DIR / "models" / "llama-3.2-3b-instruct-qlora"
MERGED_DIR = BASE_DIR / "models" / "llama-3.2-3b-instruct-merged"


def load_training_data():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Tokenized dataset not found: {DATASET_DIR}")

    dataset = load_from_disk(str(DATASET_DIR))
    if len(dataset) == 0:
        raise ValueError(f"Tokenized dataset is empty: {DATASET_DIR}")
    missing_columns = {"input_ids", "attention_mask"} - set(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tokenized dataset is missing required columns: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, dataset


def create_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def create_training_arguments(output_dir: Path, *, use_gpu: bool):
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        fp16=use_gpu,
        bf16=False,
        gradient_checkpointing=use_gpu,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gpu else None,
        optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=use_gpu,
    )


def save_adapter(trainer, tokenizer, output_dir: Path):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))