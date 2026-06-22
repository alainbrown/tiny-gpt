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

### 3. Transformer Model (`src/model.py` & `src/hf_model.py`)
A PyTorch implementation of a decoder-only transformer network:
- **Core Math (`model.py`):** Learned token/position embeddings, Pre-LayerNorm, Multi-Head Attention, and a standard two-layer MLP with a GELU activation function.
- **Hugging Face Wrapper (`hf_model.py`):** The pure mathematical model is cleanly wrapped in a `PreTrainedModel` and `PretrainedConfig` interface, granting access to the industry-standard `.save_pretrained()` and `.from_pretrained()` workflow.

### 4. Training Pipeline (`src/pipeline.py` & `src/trainer.py`)
A complete, isolated training script capable of training the model on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
- Features a custom `Trainer` class that exposes the raw PyTorch backpropagation loop while abstracting away boilerplate device routing and configuration via `TrainerConfig`.
- Exports checkpoints perfectly compatible with Hugging Face (`safetensors` + `config.json`).

### 5. Gradio UI for HF Spaces (`app.py`)
A fully functional `Gradio` ChatInterface hooked up to the token-by-token generation logic in `src/generate.py`. It is structured to be deployed instantly as a Hugging Face Space with ZeroGPU support (`@spaces.GPU`).

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
You can start a training run using the pipeline script. This will download the TinyStories dataset, train a new BPE tokenizer, train the model, and save a Hugging Face checkpoint.
```bash
export PYTHONPATH=.
python src/pipeline.py \
    --checkpoint_dir checkpoints/tiny_gpt \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --n_stories 1000 \
    --eval_interval 500
```

### Launch the UI
After training, launch the interactive Gradio demo:
```bash
export PYTHONPATH=.
python app.py
```
