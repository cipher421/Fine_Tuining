import gc
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

from training_common import (
    CPU_ADAPTER_DIR,
    MERGED_DIR,
    MODEL_NAME,
    create_lora_config,
    create_training_arguments,
    load_training_data,
    save_adapter,
)


def main():
    print("Training on CPU.")

    tokenizer, dataset = load_training_data()

    print(f"Training examples: {len(dataset):,}")
    print("Loading model in float32 on CPU...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    model.config.use_cache = False
    model = get_peft_model(model, create_lora_config())
    model.print_trainable_parameters()

    training_args = create_training_arguments(CPU_ADAPTER_DIR, use_gpu=False)
    trainer = None
    merged_model = None

    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        print("\nStarting fine tuning...")
        trainer.train()

        print("\nSaving LoRA adapter...")
        save_adapter(trainer, tokenizer, CPU_ADAPTER_DIR)
        print(f"LoRA adapter saved to: {CPU_ADAPTER_DIR}")

        print("\nMerging LoRA adapter with base model...")
        merged_model = trainer.model.merge_and_unload()
        MERGED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        merged_model.save_pretrained(
            str(MERGED_DIR),
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(MERGED_DIR))
        print(f"Merged model saved to: {MERGED_DIR}")
        print("\nFine tuning complete.")
    finally:
        del merged_model
        del trainer
        del model
        gc.collect()


if __name__ == "__main__":
    main()