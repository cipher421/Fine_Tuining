from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen3-0.6B"

DATASET_DIR = "data/tokenized"

ADAPTER_DIR = "models/qwen3-0.6b-lora"
MERGED_DIR = "models/qwen3-0.6b-merged"

MAX_LENGTH = 1024


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False

    print("Loading dataset...")

    dataset = load_from_disk(DATASET_DIR)

    print(f"Training examples: {len(dataset):,}")

    lora_config = LoraConfig(
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

    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,

        learning_rate=2e-4,

        logging_steps=10,

        save_steps=500,
        save_total_limit=2,

        fp16=True,

        gradient_checkpointing=True,

        optim="adamw_torch",

        report_to="none",

        remove_unused_columns=False,
    )

    print("\nStarting fine tuning...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    trainer.train()

    print("\nSaving LoRA adapter...")

    Path(ADAPTER_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print(f"LoRA adapter saved to: {ADAPTER_DIR}")

    print("\nMerging LoRA adapter with base model...")

    merged_model = trainer.model.merge_and_unload()

    Path(MERGED_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(MERGED_DIR)

    print(f"Merged model saved to: {MERGED_DIR}")

    print("\nFine tuning complete.")


if __name__ == "__main__":
    main()