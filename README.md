# Tiny LLM From Scratch

This project is an educational implementation of a GPT-style, decoder-only language model built completely from scratch in PyTorch. The goal is to deeply understand every component of a generative transformer by implementing it manually, without relying on pre-built libraries like Hugging Face `transformers` or directly copying existing minimalist implementations like NanoGPT.

## Features

This repository contains the building blocks of a language model, including a custom tokenizer, dataset loader, and the transformer architecture itself. 

### 1. Custom BPE Tokenizer (`bpe_tokenizer.py`)
A custom Byte-Pair Encoding (BPE) tokenizer implemented from the ground up:
- Operates natively on bytes, ensuring any arbitrary string can be tokenized.
- Trains by iteratively finding and merging the most frequent adjacent pairs.
- Supports encoding text into token IDs, decoding IDs back to text, and handling special `<EOS>` tokens.
- Serialization methods (`save` and `load`) for exporting the tokenizer's vocabulary and merges to JSON.

### 2. Dataset Processing (`dataset.py`)
A simple PyTorch `Dataset` utility:
- Consumes a contiguous stream of token IDs and chunks them into overlapping `(context, target)` pairs for autoregressive next-token prediction.
- Seamlessly integrates with PyTorch's `DataLoader` for batching and shuffling during training.

### 3. Transformer Model (`model.py`)
A PyTorch implementation of a decoder-only transformer network:
- **Embeddings:** Learned token embeddings and absolute position embeddings.
- **Transformer Blocks:** Stacked layers utilizing the Pre-LayerNorm architecture for better training stability.
- **Causal Self-Attention:** A from-scratch implementation of masked self-attention ensuring that the model cannot "look ahead" at future tokens. *(Currently implements single-head attention)*
- **Feed-Forward Network:** A standard two-layer MLP with a GELU activation function.
- **Custom LayerNorm:** A manual implementation of Layer Normalization.
- **Weight Tying:** The weights of the final output projection layer are tied to the input token embeddings, significantly saving parameters.

### 4. Experimental Notebooks (`notebooks/`)
The `notebooks/` directory contains scratchpads where the model and training loops were prototyped on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
- `tiny_gpt_v1.py`: Initial experiments outlining the minimal components, including trivial character-level tokenization vs BPE, sanity checks, basic training loops, and an initial generative inference script.
- `tiny_gpt_v2.py`: A more refined iteration featuring:
  - Integration of the final `BPETokenizer`.
  - An encapsulated `Trainer` class supporting GPU (`cuda`), evaluation intervals, loss tracking, and throughput (tokens/sec) calculations.
  - Generative text sampling utilities with parameters for temperature, top-k filtering, and greedy decoding.
  - Visualization of the training/validation loss curves.

## Goal
This repository serves as a learning sandbox. The intentional choice to avoid multi-head attention abstractions (by using a flat single-head projection initially) and built-in normalization functions ensures a comprehensive understanding of the mathematical and architectural underpinnings of modern Large Language Models.
