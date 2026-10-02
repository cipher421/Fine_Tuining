import gc
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from trl import SFTTrainer

from training_common import (
    MODEL_NAME,
    QLORA_ADAPTER_DIR,
    create_training_arguments,
    load_training_data,
    save_adapter,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA requires a CUDA GPU. Use fine_tuneing.py for CPU LoRA training."
        )

    try:
        import bitsandbytes
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "QLoRA requires a working bitsandbytes installation. "
            "Install the GPU dependencies with: "
            "python -m pip install -r requirements-gpu.txt"
        ) from exc

    try:
        from unsloth import FastLanguageModel
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "GPU QLoRA requires Unsloth. Install the GPU dependencies with: "
            "python -m pip install -r requirements-gpu.txt"
        ) from exc

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bitsandbytes: {bitsandbytes.__version__}")

    tokenizer, dataset = load_training_data()
    print(f"Training examples: {len(dataset):,}")

    print("Loading model with Unsloth in 4-bit QLoRA mode...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.config.use_cache = False

    training_args = create_training_arguments(QLORA_ADAPTER_DIR, use_gpu=True)
    trainer = None

    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.model.print_trainable_parameters()

        print("\nStarting GPU QLoRA fine tuning...")
        trainer.train()

        print("\nSaving QLoRA adapter...")
        save_adapter(trainer, tokenizer, QLORA_ADAPTER_DIR)
        print(f"QLoRA adapter saved to: {QLORA_ADAPTER_DIR}")
        print("\nQLoRA fine tuning complete.")
    finally:
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()