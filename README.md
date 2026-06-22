# Tiny GPT From Scratch

Tiny GPT is an educational decoder-only language model implemented from
scratch in PyTorch and trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset.

## Repository layout

```text
tiny-gpt/
├── src/tiny_gpt/              # Model, tokenizer, trainer, and HF integration
├── apps/gradio/               # Files deployed to the Hugging Face Space
├── scripts/export_model.py    # Builds the Hugging Face model artifact
├── checkpoints/
│   ├── tiny_gpt/              # Local resumable training checkpoint (ignored)
│   └── tiny_gpt_hub/          # Generated model release artifact (ignored)
├── train.py
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

This GitHub repository is the development source for two Hugging Face targets:

```text
checkpoints/tiny_gpt_hub/  →  Model: alainbrown/tiny-gpt
apps/gradio/               →  Space: alainbrown/tiny-gpt-demo
```

The model repository stores weights, tokenizer files, custom Transformers
code, and the model card. The Space contains the complete chat-generation
application and loads the model from the model repository.

## Model

The implementation includes:

- Byte-level BPE tokenization
- Learned token and positional embeddings
- Causal multi-head self-attention
- Pre-LayerNorm Transformer blocks
- GELU feed-forward layers
- Tied input and output embeddings
- Hugging Face-compatible checkpoint serialization

## Install

```bash
pip install -e .
```

Runtime dependencies are defined in `pyproject.toml`. The Space has its own
small `apps/gradio/requirements.txt` because Hugging Face Spaces installs
dependencies from that file after the directory is deployed.

## Train

Run the configured Docker training service:

```bash
docker compose --profile training run --rm training
```

The resumable checkpoint is written to `checkpoints/tiny_gpt` and includes
model weights, tokenizer data, optimizer state, and training progress.

## Export the model

Create a clean Hugging Face model artifact from the local checkpoint:

```bash
docker compose --profile tools run --rm hub-export
```

The command writes `checkpoints/tiny_gpt_hub` and validates that the exported
model can be loaded through `AutoModelForCausalLM`. Optimizer and progress
state are not included in the release artifact.

## Run the Gradio app locally

The app defaults to the public model ID `alainbrown/tiny-gpt`:

```bash
docker compose --profile gradio up gradio
```

Override the model or revision when needed:

```bash
MODEL_ID=alainbrown/tiny-gpt MODEL_REVISION=main \
  docker compose --profile gradio up gradio
```

The local app is available at <http://localhost:7860>.

## Deployments

Upload the contents of `checkpoints/tiny_gpt_hub` to the Hugging Face **Model**
repository. Upload the contents of `apps/gradio` to the Hugging Face **Space**.
The two deployment targets are generated or maintained from this single
GitHub source repository.

## License

MIT
