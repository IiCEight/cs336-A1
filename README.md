# CS336 Lab 1: Language Model Basics

<p align="center">
    <a href="./README.zh-CN.md">
        <img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E4%B8%AD%E6%96%87-red?style=for-the-badge" alt="中文" />
    </a>
    &nbsp;&nbsp;
    <a href="#">
        <img src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge" alt="English" />
    </a>
</p>

This repository contains my completed implementation for CS336 Lab 1.
The project builds a compact GPT-style language modeling pipeline from scratch, including tokenizer training, Transformer components, optimization utilities, training and evaluation, checkpointing, and text generation.

## Highlights

- Byte Pair Encoding (BPE) tokenizer training and serialization
- Tokenization and memory-mapped binary dataset preparation
- Core Transformer components implemented from scratch:
	- Embeddings
	- Multi-head self-attention
	- RoPE positional encoding
	- RMSNorm
	- SwiGLU feed-forward block
- Custom AdamW optimizer implementation
- Learning-rate scheduling and gradient clipping
- Training loop with validation, checkpointing, and optional Weights & Biases logging
- Autoregressive inference script for text generation

## Project Structure

- `cs336_basics/tokenizer/`: tokenizer implementation and BPE training
- `cs336_basics/module/`: neural modules (attention, normalization, Transformer block, LM)
- `cs336_basics/optimizer/`: custom optimizer
- `cs336_basics/utils/`: training and math utilities
- `cs336_basics/load/`: dataset loading and batching helpers
- `cs336_basics/checkpoint/`: checkpoint save and load
- `cs336_basics/main.py`: training entry
- `cs336_basics/prepare_tokens.py`: tokenizer training + token file generation
- `cs336_basics/generate.py`: text generation from checkpoint
- `tests/`: assignment tests and fixtures

## Environment Setup

This project uses `uv` for dependency and environment management.

Install `uv`:

```bash
# Linux/macOS (recommended)
# see: https://docs.astral.sh/uv/getting-started/installation/

# Alternative
pip install uv
```

Install dependencies:

```bash
uv sync
```

Run any command inside the project environment:

```bash
uv run <command>
```

## Data Preparation

Download TinyStories train/valid files into `data/`:

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..
```

## Quick Start

### 1) Run Unit Tests

```bash
uv run pytest
```

### 2) Train Tokenizer and Prepare Token Binaries

```bash
uv run python -m cs336_basics.prepare_tokens
```

Default outputs:

```text
data/tinystories_vocab.json
data/tinystories_merges.txt
data/train_token_origin.bin
data/valid_token_origin.bin
```

If you want to use default training arguments directly, make sure token files match:

```text
data/train_token.bin
data/valid_token.bin
```

### 3) Train the Language Model

```bash
uv run python -m cs336_basics.main \
	--device cpu \
	--iters 20000 \
	--batch-size 64
```

Useful options:

```text
--checkpoint-dir
--iters-per-checkpoint
--iters-per-evaluation
--wandb-project
--train-tokens-filepath
--valid-tokens-filepath
```

### 4) Generate Text from a Checkpoint

```bash
uv run python -m cs336_basics.generate "Once upon a time" \
	--checkpoint-path ./checkpoint/best_checkpoint_iter_20000.pt \
	--max-new-tokens 200 \
	--temperature 0.8
```

## Reproducibility

- Python version: 3.11+
- Dependencies are declared in `pyproject.toml`
- Checkpoints are saved under `checkpoint/`
- Training includes periodic validation and best-checkpoint tracking

## Implementation Coverage

This submission includes all major Lab 1 components:

- Tokenizer training and encoding pipeline
- Transformer modules and end-to-end LM
- Loss, optimizer, LR scheduler, and clipping
- Full training and evaluation loop
- Checkpoint save/load and generation

## Acknowledgments

- Stanford CS336 course staff for assignment design and testing infrastructure
- TinyStories dataset authors and hosting providers

