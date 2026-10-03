# Fine_Tuining

A Linux-first, CPU-only pipeline for preparing educational text from PDFs and
fine-tuning a causal language model with LoRA.

> **Model compatibility notice:** the configured label `qwen3.5:0.8b` currently
> resolves to `Qwen/Qwen3-0.6B`. The code does **not** load Unsloth's Qwen3.5
> 0.8B weights under that label. This fallback was selected because the
> installed Transformers stack in this environment did not recognize the
> Qwen3.5 architecture. The backend model is therefore Qwen3 0.6B, not Qwen3.5
> 0.8B. Do not interpret the label or output-folder names as proof that the
> Qwen3.5 checkpoint was trained.

## What the project does

The pipeline has four data-preparation stages followed by CPU fine-tuning:

1. Extract text from PDFs into page-level JSON.
2. Clean extracted text and normalize paragraphs.
3. Turn paragraphs into user/assistant chat examples in JSONL.
4. Apply the selected model tokenizer's chat template and save token IDs.
5. Fine-tune the base model with a PEFT LoRA adapter, save the adapter, and
   merge it with the base model.

All commands below are intended to run from the repository root.

## Project layout

| Path | Purpose |
| --- | --- |
| `Extract_Pdf.py` | Reads `data/pdfs/*.pdf`; writes page-level JSON under `data/raw/`. |
| `Clean_Text.py` | Normalizes the extracted JSON; writes cleaned documents under `data/clean/`. |
| `Training_Dataset.py` | Builds `data/processed/training.jsonl` from cleaned page text. |
| `Tokenize.py` | Applies the configured model tokenizer and writes `data/tokenized/`. |
| `fine_tuneing.py` | Validates the tokenized data, trains the CPU LoRA adapter, and saves adapter and merged outputs under `models/`. |
| `test.py` | Checks the Python/dependency setup and validates the generated dataset and token IDs. |
| `requirements.txt` | Python dependencies and compatible PyTorch/torchvision pins. |
| `data/pdfs/` | PDF inputs. These are source data, not disposable build output. |
| `data/raw/`, `data/clean/`, `data/processed/` | Regenerable intermediate datasets. |
| `data/tokenized/` | Regenerable Hugging Face dataset cache; excluded from Git. |
| `models/` | Generated training outputs; excluded from Git. |

## Requirements

- Linux (the project is configured and documented for CPU-only Linux use).
- Python 3.12 is the target environment; Python 3.12.15 was used for the
  development and training-step checks.
- Enough system memory for the float32 base model, LoRA optimizer state, and
  training activations. CPU training can be terminated by Linux when memory is
  exhausted; gradient checkpointing and the short sequence limit reduce but do
  not eliminate that risk.
- Internet access on first model/tokenizer load, unless the Hugging Face files
  are already cached locally.

No CUDA, GPU, or running Ollama service is used by the training script. The
`qwen3.5:0.8b` string is only a project alias; it is not an Ollama invocation.

## Installation

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For later terminals, activate the environment again:

```bash
source .venv/bin/activate
```

The dependency file pins the PyTorch and torchvision versions together to avoid
the import-time operator mismatch encountered with incompatible installations.
If you already have a different environment, install into the project virtual
environment rather than mixing system packages with the project dependencies.

## Prepare data

Place PDF inputs in `data/pdfs/`. Then run the pipeline in order:

```bash
python Extract_Pdf.py
python Clean_Text.py
python Training_Dataset.py
python Tokenize.py
```

The stages report an error if their required input directory has no files.
`Training_Dataset.py` skips paragraphs shorter than 20 words and reports an
error if it produces no examples.

Each JSONL record has this shape:

```json
{
  "messages": [
    {"role": "user", "content": "Explain the following educational content clearly.\n\n..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {"source": "book.pdf", "page": 1}
}
```

The tokenizer uses the configured model's chat template and truncates each
example to 128 tokens. `Tokenize.py` stages the replacement dataset before
swapping it into `data/tokenized/`, so a failed rebuild leaves the previous
cache available.

## Validate the data and environment

After preparing data, run:

```bash
python test.py
```

The full check verifies the supported Python version, required imports, model
alias resolution, JSONL structure, tokenized dataset shape, token ID range, and
the training sequence limit. It loads the tokenizer, so the first run may need
network access.

To validate the environment and model alias before preparing data:

```bash
python test.py --skip-data
```

The normal check also prevents the earlier embedding-index failure: it checks
that every cached token ID fits the tokenizer and rejects stale or incompatible
tokenized data.

## Train

Start training from the repository root:

```bash
python fine_tuneing.py
```

The training configuration is deliberately CPU-only:

- Loads the base model in float32 and explicitly places it on the CPU.
- Uses LoRA with rank 16, alpha 32, dropout 0.05, and attention/MLP projection
  modules as targets.
- Trains for one epoch with batch size 1 and gradient accumulation of 8.
- Uses a 128-token maximum sequence length and non-reentrant gradient
  checkpointing to reduce activation memory.
- Disables fp16/bf16 and pinned-memory data loading.
- Aligns the model and generation configuration's BOS/EOS/PAD IDs to the
  tokenizer before training.

CPU fine-tuning is slow. If the operating system prints `Killed`, it terminated
the process, usually because memory was exhausted. The code already applies
gradient checkpointing and a short sequence limit; close other memory-heavy
applications before trying again. Do not increase the sequence length or batch
size on a memory-constrained machine.

## Outputs and cleanup

Successful training writes:

- `models/qwen3.5-0.8b-lora/` — PEFT LoRA adapter plus tokenizer files.
- `models/qwen3.5-0.8b-merged/` — merged model plus tokenizer files.

The directory names retain the project's configured alias; they do not change
which base checkpoint was loaded. The actual checkpoint is printed near
startup.

Generated raw/clean/processed/tokenized data and model output directories are
ignored by Git (except for the source PDFs currently included in the
repository). The tokenized dataset can be rebuilt by running
`python Tokenize.py` after `data/processed/training.jsonl` exists. The virtual
environment can be recreated from `requirements.txt`; neither the environment
nor generated model outputs should be manually deleted while a run is active.

## Troubleshooting

### `IndexError: index out of range in self`

The tokenized cache was created with token IDs that do not fit the active
model/tokenizer. Rebuild it and validate:

```bash
python Tokenize.py
python test.py
```

### The training process ends with `Killed`

This is an operating-system process termination rather than a Python exception;
on a small-memory machine it is commonly an out-of-memory kill. Keep the
configured batch size and 128-token sequence limit, close other large
applications, and ensure there is adequate available memory and swap. Swap may
avoid an immediate kill but can make CPU training substantially slower.

### `qwen3_5` is an unrecognized model type

The current code maps the alias to `Qwen/Qwen3-0.6B` for compatibility; it does
not load the Unsloth Qwen3.5 0.8B architecture. To train the actual Qwen3.5
checkpoint, first use a Transformers release that supports its architecture,
then update the model mapping in both `Tokenize.py` and `fine_tuneing.py` and
regenerate the tokenized dataset. Do not change only the alias text.

### Missing package/import error

Activate `.venv` and reinstall the declared dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The PyTorch `KernelPreference` / `ScaleCalculationMode` Enum deprecation
warnings are emitted by the installed PyTorch stack and are not, by themselves,
training failures.
