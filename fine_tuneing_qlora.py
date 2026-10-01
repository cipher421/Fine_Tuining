import gc
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer

from training_common import (
    MODEL_NAME,
    QLORA_ADAPTER_DIR,
    create_lora_config,
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

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bitsandbytes: {bitsandbytes.__version__}")

    tokenizer, dataset = load_training_data()
    print(f"Training examples: {len(dataset):,}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading model in 4-bit QLoRA mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
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
            peft_config=create_lora_config(),
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