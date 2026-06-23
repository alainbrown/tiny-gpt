# Tiny GPT From Scratch

Tiny GPT is an educational decoder-only language model built from first
principles in PyTorch. The current model is a small story-completion model
trained on [skeskinen/TinyStories-hf](https://huggingface.co/datasets/skeskinen/TinyStories-hf).

The production model uses PyTorch scaled dot-product attention, fused QKV
projection, and native LayerNorm. A separate explicit reference model remains
available for learning, numerical comparison, and correctness tests.

- Source: <https://github.com/alainbrown/tiny-gpt>
- Model: <https://huggingface.co/alainbrown/tiny-gpt>
- Storyteller demo: <https://huggingface.co/spaces/alainbrown/tiny-gpt-demo>

## Current status

- A complete byte-level BPE tokenizer and decoder-only Transformer
- A published 24.28M-parameter TinyStories storyteller checkpoint
- Resumable training and Hugging Face export
- Reproducible validation and generation evaluation
- Forward/backward throughput and memory benchmarking
- Tests covering tensor shapes, context limits, head reshaping, and causality
- SDPA attention with fused QKV projection

The model execution path is optimized, while `torch.compile`, fused AdamW,
KV-cached generation, and broader non-story pretraining remain future
improvements.

## Architecture

The published model has 24,282,624 tied parameters:

| Setting | Value |
| --- | --- |
| Context length | 1,024 tokens |
| Vocabulary size | 16,000 |
| Hidden size | 384 |
| Transformer layers | 10 |
| Attention heads | 6 |
| Training dataset | `skeskinen/TinyStories-hf` |
| Training length | 3 epochs |

Latest full evaluation on 5,000 TinyStories validation examples:

| Metric | Value |
| --- | ---: |
| Validation loss | 1.4934 |
| Validation perplexity | 4.4521 |
| Evaluation tokens | 973,824 |
| Generated samples | 72 |

The implementation includes:

- Byte-level BPE tokenization
- Learned token and positional embeddings
- Causal multi-head self-attention
- Pre-LayerNorm Transformer blocks
- GELU feed-forward layers
- Tied token-embedding and output-projection weights
- Safetensors checkpoints and Hugging Face AutoClass support

The production architecture lives in `src/tiny_gpt/model.py`. The explicit
mathematical version lives in `src/tiny_gpt/ref_model.py` and is not used by
training, evaluation, export, or inference.

## Project direction

The reference model remains useful for:

- Mathematical comparison
- Numerical-equivalence tests
- Throughput and memory benchmarks
- Debugging causal behavior

Architectural changes such as RoPE, RMSNorm, SwiGLU, or grouped-query attention
should be evaluated separately from execution optimizations such as SDPA or
compilation.

## Repository

This is the single development repository for both Hugging Face deployments:

```text
tiny-gpt/
├── src/tiny_gpt/
│   ├── model.py               # Production SDPA and fused-QKV architecture
│   ├── ref_model.py           # Explicit educational architecture
│   ├── ref_tokenizer.py       # Explicit educational byte-level BPE
│   ├── modeling_tiny_gpt.py   # Hugging Face causal-LM wrapper
│   ├── trainer.py             # Custom training loop
│   ├── dataset.py             # Packed next-token dataset
├── apps/gradio/               # Source deployed to the Hugging Face Space
├── scripts/train_tokenizer.py # Trains the Rust-backed byte-level BPE
├── scripts/train.py           # Runs the training pipeline
├── scripts/evaluate.py        # Full storyteller eval and sample comparison
├── scripts/benchmark.py       # Throughput and memory benchmark
├── scripts/export_model.py    # Builds the Hugging Face model artifact
├── tests/                     # Reference behavior and causality tests
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
code, and the model card. The Space contains the Gradio storyteller generation
form and loads the published model from the model repository.

## Setup

The supported training environment is the CUDA-enabled Docker image:

```bash
docker compose build training
```

For local development, install the package in an environment with the
dependencies from `pyproject.toml`:

```bash
pip install -e .
```

The Space uses separate dependencies in `apps/gradio/requirements.txt` because
the public demo environment differs from the local CUDA training image.

## Train

Train the byte-level BPE tokenizer first:

```bash
docker compose --profile tools run --rm tokenizer-training
```

The tokenizer service uses the complete configured dataset split by default.
Its command in `docker-compose.yml` configures the dataset, split, text column,
vocabulary size, and output path. For a limited experiment, run the script
directly with `--num-examples`:

```bash
python scripts/train_tokenizer.py \
  --dataset skeskinen/TinyStories-hf \
  --split train \
  --text-column text \
  --vocab-size 16000 \
  --num-examples 100000 \
  --output checkpoints/tiny_gpt/tokenizer.json
```

Then run the configured CUDA training container:

```bash
docker compose --profile training run --rm training
```

The training configuration is defined by the `training` service in
`docker-compose.yml`. The checkpoint root is `checkpoints/tiny_gpt` and uses a
resumable layout:

- `latest/` contains the most recent model, tokenizer, trainer state, progress,
  and run metadata.
- `best/` contains the best checkpoint by validation loss.
- `steps/step_XXXXXXXX/` contains retained historical checkpoints. The
  retention count is controlled by `--keep_step_checkpoints`.
- `training_progress.json`, `run_metadata.json`, and `final_summary.json` are
  written at the checkpoint root for quick inspection.

The default model configuration is specified in `docker-compose.yml`. Command
line flags in `scripts/train.py` can be used for smaller experiments. Training
requires an existing `tokenizer.json` and will not train one implicitly.
`--epochs` controls the normal training length. `--max_steps` is optional and
only acts as a safety cap when explicitly provided.

Training logs practical observability to Aim:

- train and validation loss
- train and validation perplexity
- train/validation loss gap
- gradient norm mean/max/last
- gradient clipping rate
- learning rate
- tokens/sec
- tokens processed
- stories seen
- epoch progress
- elapsed time and ETA
- checkpoint save events
- generated sample count

Generated samples are also appended to `runs/training_samples.jsonl` by default.

Start the local Aim metrics UI:

```bash
docker compose --profile metrics up aim
```

The UI is served on http://localhost:43800. Docker Compose stores Aim data in
the named volume `aim-data`, mounted at `/opt/aim` in the training container.

## Evaluate

Evaluation loads a checkpoint, packs the TinyStories validation split exactly
as next-token training data, reports loss and perplexity, and generates a
multi-prompt, multi-seed, multi-setting storyteller sample report. If the
checkpoint root contains `latest/`, evaluation resolves that automatically:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/tiny_gpt \
  --output runs/eval.json
```

Reports are printed as JSON and written to disk. A Markdown report is written
next to the JSON output. Use repeated `--prompt` flags to replace the default
prompts:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/tiny_gpt \
  --prompt "Once upon a time" \
  --prompt "The old robot found a garden"
```

## Benchmark

Measure forward/backward throughput and peak allocated CUDA memory:

```bash
python scripts/benchmark.py \
  --context-size 1024 \
  --vocab-size 16000 \
  --d-model 384 \
  --n-layers 10 \
  --n-heads 6 \
  --batch-sizes 8,16,32 \
  --effective-batch-size 128 \
  --warmup-steps 20 \
  --steps 100 \
  --repetitions 3 \
  --output runs/storyteller-batch-benchmark.json
```

Each configuration runs in a fresh process. The benchmark measures complete
BF16 optimizer steps, including forward, backward, gradient accumulation,
gradient clipping, and AdamW. It reports:

- Parameter count
- Median, mean, standard deviation, minimum, and maximum tokens per second
- Milliseconds per optimizer step
- Final synthetic-batch loss
- Peak allocated CUDA memory

Benchmark results depend on hardware, precision, batch size, context length,
and warmup. Compare implementations using identical arguments.

## Test

Run the model and trainer behavior tests:

```bash
python -m unittest discover -s tests
```

The tests verify causal behavior and numerical equivalence between converted
reference weights and the optimized model.

## Export

Create an inference-only Hub artifact from the local training checkpoint:

```bash
docker compose --profile tools run --rm hub-export
```

The exporter writes `checkpoints/tiny_gpt_hub` and validates that the result
loads through `AutoModelForCausalLM` with `trust_remote_code=True`. If
`checkpoints/tiny_gpt/latest` exists, the exporter uses it automatically.
Optimizer and training-progress state are intentionally excluded.

The generated directory is the content published to
[alainbrown/tiny-gpt](https://huggingface.co/alainbrown/tiny-gpt).

## Use the Model

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

## Demo

The public demo runs on Hugging Face Spaces:

<https://huggingface.co/spaces/alainbrown/tiny-gpt-demo>

Its supported runtime is pinned in `apps/gradio`:

- Python 3.12.12
- PyTorch 2.8.0
- Gradio 6.19.0
- Transformers 5.12.1

The currently published Space is running on `cpu-basic`. The app automatically
uses CUDA when it is available, otherwise it runs on CPU.

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
It is best understood as a TinyStories-style text completer, not a general
chatbot. It is not instruction-tuned or intended for production, factual
question answering, or safety-critical applications. Its 512-token context and
small parameter count limit plot continuity and prompt adherence. Generated
text may be repetitive, incoherent, incorrect, or inappropriate.

## License

MIT
