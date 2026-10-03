# Fine_Tuining

Fine-tuning pipeline for a local educational dataset using Qwen3 0.6B with LoRA and QLoRA adapters.

## Overview

This project prepares educational content, converts it into a chat-style instruction dataset, tokenizes it, and then fine-tunes Qwen3 0.6B with PEFT-based adapters. The workflow is split into stages so each part can be rerun independently and validated before the full training run.

## Current model

The project is configured to use:

```text
unsloth/Qwen3-0.6B
```

This is the Hugging Face model ID for the Qwen3 0.6B model. The shared setting lives in [training_common.py](training_common.py), and both tokenization and training use it.

## Project structure

- `Clean_Text.py` — cleans raw extracted text from JSON documents.
- `Extract_Pdf.py` — extracts text from PDF files into JSON pages.
- `Training_Dataset.py` — converts cleaned JSON pages into a chat-style training dataset in JSONL format.
- `Tokenize.py` — loads the JSONL training set and tokenizes it for model training.
- `fine_tuneing.py` — CPU-only LoRA fallback and optional merge of the adapter back into the base model.
- `fine_tuneing_qlora.py` — recommended CUDA-only 4-bit QLoRA training path using Unsloth.
- `training_common.py` — shared model, dataset, adapter, and training configuration.
- `test.py` — smoke test for checking the environment, dependencies, config, and dataset shape.
- `data/` — raw, cleaned, processed, and tokenized datasets.
- `models/` — generated adapter and merged model outputs.

## Typical workflow

1. Place source PDFs in `data/pdfs/`.
2. Run `Extract_Pdf.py` to extract raw text.
3. Run `Clean_Text.py` to normalize the extracted content.
4. Run `Training_Dataset.py` to build the chat-style instruction dataset.
5. Run `Tokenize.py` to prepare the dataset for model training.
6. Run `test.py` to validate the environment and config before training.
7. Use `fine_tuneing_qlora.py` for Unsloth QLoRA on a supported CUDA GPU, or `fine_tuneing.py` for CPU LoRA.

## Setup

Requirements: Python 3.10 through 3.13 and pip. Python 3.14 is not currently supported by the dependency range used here. A CUDA-enabled NVIDIA GPU is required for Unsloth QLoRA; the CPU LoRA fallback does not require CUDA. The first run downloads the model from Hugging Face, so internet access is needed.

For Unsloth QLoRA, install a CUDA-enabled PyTorch build that matches your system first. From the project directory, create and activate a virtual environment, then install the project dependencies:

```bash
cd /path/to/Fine_Tuining
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validate the project

Before starting training, run the smoke test:

```bash
python test.py
```

You can skip the dataset validation if the training JSONL has not been generated yet:

```bash
python test.py --skip-data
```

## Prepare data and train

Create the PDF input directory and place your source PDFs there:

```bash
mkdir -p data/pdfs
```

Run the pipeline from the project root:

```bash
python Extract_Pdf.py
python Clean_Text.py
python Training_Dataset.py
python Tokenize.py
python test.py
```

Choose one training path:

```bash
# Recommended: CUDA GPU QLoRA with Unsloth (requires the GPU setup above)
python fine_tuneing_qlora.py

# Optional: CPU LoRA fallback
python fine_tuneing.py
```

After changing the model, replace any existing generated `data/tokenized/` dataset by rerunning `Tokenize.py` with the Qwen3 tokenizer. If you open a new terminal later, reactivate the environment with `source .venv/bin/activate` before running commands. On Windows, use `.venv\Scripts\activate` instead.

## Hardware notes

The recommended GPU path uses Unsloth with 4-bit QLoRA and requires a CUDA-enabled PyTorch installation plus working CUDA support in `bitsandbytes`. The optional CPU path loads the base model in float32 and trains an unquantized LoRA adapter; it is slower and needs enough memory for the model and activations. Both use a 512-token sequence cap.

## Important notes

- The Ollama tag `qwen3:0.6b` corresponds to the Hugging Face model ID `unsloth/Qwen3-0.6B` used here.
- Do not pass a `Path` object to `datasets.load_dataset(..., data_files=...)`; use a string path.
- `requirements.txt` requires TRL 0.15.2 or newer for the `processing_class` trainer API.
- `bitsandbytes` and Unsloth are used by the GPU QLoRA path; both are included in `requirements.txt`.
- Training scripts disable tokenizer parallelism and clean up model references after training.
- The smoke test is meant to catch common setup problems before a time-consuming fine-tuning run begins.

## Outputs

- `data/processed/training.jsonl` — final instruction dataset
- `data/tokenized/` — tokenized training dataset
- `models/qwen3-0.6b-lora/` — the CPU LoRA adapter
- `models/qwen3-0.6b-qlora/` — the Unsloth GPU QLoRA adapter
- `models/qwen3-0.6b-merged/` — merged model output

## Troubleshooting

- For CPU out-of-memory errors, lower the batch size or sequence length.
- `fine_tuneing_qlora.py` exits immediately without CUDA; install a CUDA-enabled PyTorch build before running it.
- If `test.py` fails with missing dependencies, run `python -m pip install -r requirements.txt` inside the active environment.
- If tokenizer warnings appear at shutdown, the project suppresses parallelism and performs cleanup automatically.
