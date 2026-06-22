# Tiny GPT From Scratch

Tiny GPT is an educational decoder-only language model implemented from
scratch in PyTorch and trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset.

- Source: <https://github.com/alainbrown/tiny-gpt>
- Model: <https://huggingface.co/alainbrown/tiny-gpt>
- ZeroGPU demo: <https://huggingface.co/spaces/alainbrown/tiny-gpt-demo>

## Model architecture

The published model has approximately 10 million parameters:

| Setting | Value |
| --- | ---: |
| Context length | 512 tokens |
| Vocabulary size | 10,000 |
| Hidden size | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |

The implementation includes:

- Byte-level BPE tokenization
- Learned token and positional embeddings
- Causal multi-head self-attention
- Pre-LayerNorm Transformer blocks
- GELU feed-forward layers
- Tied input and output embeddings
- Safetensors checkpoints and Hugging Face AutoClass support

## Repository layout

This is the single development repository for both Hugging Face deployments:

```text
tiny-gpt/
├── src/tiny_gpt/              # Model, tokenizer, trainer, and HF integration
├── apps/gradio/               # Source deployed to the Hugging Face Space
├── scripts/train.py           # Runs the training pipeline
├── scripts/export_model.py    # Builds the Hugging Face model artifact
├── checkpoints/
│   ├── tiny_gpt/              # Local resumable training state (ignored)
│   └── tiny_gpt_hub/          # Generated model release artifact (ignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

Deployment mapping:

```text
checkpoints/tiny_gpt_hub/  →  HF Model: alainbrown/tiny-gpt
apps/gradio/               →  HF Space: alainbrown/tiny-gpt-demo
```

The model repository contains weights, tokenizer files, custom Transformers
code, and the model card. The Space contains the Gradio chat application and
loads the published model from the model repository.

## Install locally

```bash
pip install -e .
```

Project dependencies are defined in `pyproject.toml`. The Space has a separate
`apps/gradio/requirements.txt` because its ZeroGPU runtime uses a different
supported PyTorch version from the local training image.

## Train

Run the configured CUDA training container:

```bash
docker compose --profile training run --rm training
```

The training configuration is defined by the `training` service in
`docker-compose.yml`. The resumable checkpoint is written to
`checkpoints/tiny_gpt` and contains:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `optimizer.pt`
- `TinyStories.progress`

## Export the Hugging Face model

Create an inference-only Hub artifact from the local training checkpoint:

```bash
docker compose --profile tools run --rm hub-export
```

The exporter writes `checkpoints/tiny_gpt_hub` and validates that the result
loads through `AutoModelForCausalLM` with `trust_remote_code=True`. Optimizer
and training-progress state are intentionally excluded.

The generated directory is the content published to
[alainbrown/tiny-gpt](https://huggingface.co/alainbrown/tiny-gpt).

## Use the published model

The model includes custom Transformers code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "alainbrown/tiny-gpt"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)

inputs = tokenizer("Once upon a time", return_tensors="pt")
logits = model(**inputs).logits
```

Review remote model code before enabling `trust_remote_code` for repositories
you do not control.

## Gradio Space

The public demo runs on Hugging Face ZeroGPU:

<https://huggingface.co/spaces/alainbrown/tiny-gpt-demo>

Its supported runtime is pinned in `apps/gradio`:

- Python 3.12.12
- PyTorch 2.8.0
- Gradio 6.19.0
- Transformers 5.12.1
- Spaces 0.50.4

The model is placed on CUDA during startup, as required by ZeroGPU, and the
streaming generation callback requests GPU access with `@spaces.GPU`.

To run the same app locally with Docker and a CUDA-capable host:

```bash
docker compose --profile gradio up gradio
```

The app defaults to `alainbrown/tiny-gpt`. Override the model or revision with:

```bash
MODEL_ID=alainbrown/tiny-gpt MODEL_REVISION=main \
  docker compose --profile gradio up gradio
```

The local app is available at <http://localhost:7860>.

## Limitations

This is a small educational model trained on synthetic children's stories.
It is not instruction-tuned or intended for production, factual question
answering, or safety-critical applications. Generated text may be repetitive,
incoherent, incorrect, or inappropriate.

## License

MIT
