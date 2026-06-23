import math
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get("MODEL_ID", "alainbrown/tiny-gpt")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRESET_PROMPTS = [
    "Once upon a time, a small rabbit found a glowing key in the grass.",
    "Mia opened the mysterious blue door and saw",
    "The little fox looked up at the moon and",
    "Tom found a tiny dragon under his bed.",
    "Ella broke her brother's toy and wanted to make things right.",
    "Write a short bedtime story about a robot who learns to share.",
]

DECODING_PRESETS = {
    "Balanced": {"temperature": 0.8, "top_k": 40, "max_new_tokens": 180},
    "Safer": {"temperature": 0.6, "top_k": 20, "max_new_tokens": 160},
    "More varied": {"temperature": 1.0, "top_k": 80, "max_new_tokens": 200},
    "No top-k": {"temperature": 0.8, "top_k": 0, "max_new_tokens": 180},
}

CSS = """
:root {
  --paper: #f7efe1;
  --ink: #24170f;
  --clay: #b65f3d;
  --moss: #536b45;
  --night: #1f2b3a;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(182, 95, 61, 0.22), transparent 32rem),
    linear-gradient(135deg, #f9f2e5 0%, #ead9be 46%, #d8c3a2 100%);
  color: var(--ink);
}

#hero {
  padding: 1.4rem 1.6rem;
  border: 1px solid rgba(36, 23, 15, 0.16);
  border-radius: 24px;
  background: rgba(255, 250, 239, 0.72);
  box-shadow: 0 18px 50px rgba(60, 39, 21, 0.16);
}

#hero h1 {
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2.2rem, 7vw, 5.2rem);
  line-height: 0.92;
  margin-bottom: 0.4rem;
  color: var(--night);
}

#hero p {
  max-width: 52rem;
  font-size: 1.04rem;
}

#story_output textarea {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.08rem;
  line-height: 1.65;
}

.compact-note {
  color: rgba(36, 23, 15, 0.72);
  font-size: 0.92rem;
}
"""


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model.to(DEVICE)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    return model, tokenizer


def validate_generation_settings(temperature, top_k, max_new_tokens):
    try:
        temperature = float(temperature) if temperature is not None else 0.8
        top_k = int(top_k) if top_k is not None else 40
        max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else 180
    except (TypeError, ValueError, OverflowError) as error:
        raise gr.Error("Generation settings must be numeric values.") from error

    if not math.isfinite(temperature) or not 0.1 <= temperature <= 2.0:
        raise gr.Error("Temperature must be between 0.1 and 2.0.")
    if not 0 <= top_k <= model.config.vocab_size:
        raise gr.Error(
            f"Top-k must be between 0 and {model.config.vocab_size}."
        )
    if not 16 <= max_new_tokens <= 320:
        raise gr.Error("Maximum new tokens must be between 16 and 320.")

    return temperature, top_k, max_new_tokens


model, tokenizer = load_model()
print(f"Loaded {MODEL_ID}@{MODEL_REVISION} on {DEVICE}.")


def apply_preset(preset_name):
    preset = DECODING_PRESETS[preset_name]
    return (
        preset["temperature"],
        preset["top_k"],
        preset["max_new_tokens"],
    )


def generate_story(prompt, temperature, top_k, max_new_tokens):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a story opening first.")

    temperature, top_k, max_new_tokens = validate_generation_settings(
        temperature,
        top_k,
        max_new_tokens,
    )

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    context_size = model.config.context_size
    input_ids = input_ids[:, -context_size:]
    generated_ids = []

    yield prompt

    with torch.no_grad():
        for _ in range(max_new_tokens):
            model_inputs = input_ids[:, -context_size:]
            logits = model(input_ids=model_inputs).logits[:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                filtered_logits = torch.full_like(logits, float("-inf"))
                filtered_logits.scatter_(1, indices, values)
                logits = filtered_logits

            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)

            token_id = next_id.item()
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            continuation = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            yield prompt + continuation


APP_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="green",
    neutral_hue="stone",
)


with gr.Blocks(title="Tiny GPT Storyteller") as demo:
    gr.Markdown(
        """
        <section id="hero">
          <h1>Tiny GPT Storyteller</h1>
          <p>
            A small decoder-only Transformer trained from scratch on TinyStories.
            Give it a story opening, choose a decoding style, and let it continue.
            This is a story generator, not a chat assistant.
          </p>
        </section>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            prompt = gr.Textbox(
                label="Story opening",
                value=PRESET_PROMPTS[0],
                lines=6,
                max_lines=10,
                placeholder="Start a children's story...",
            )
            generate_button = gr.Button(
                "Generate Story",
                variant="primary",
                size="lg",
            )
        with gr.Column(scale=3):
            preset = gr.Dropdown(
                choices=list(DECODING_PRESETS),
                value="Balanced",
                label="Decoding preset",
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=DECODING_PRESETS["Balanced"]["temperature"],
                step=0.1,
                label="Temperature",
            )
            top_k = gr.Slider(
                minimum=0,
                maximum=120,
                value=DECODING_PRESETS["Balanced"]["top_k"],
                step=1,
                label="Top-K (0 disables top-k)",
            )
            max_new_tokens = gr.Slider(
                minimum=16,
                maximum=320,
                value=DECODING_PRESETS["Balanced"]["max_new_tokens"],
                step=16,
                label="Maximum new tokens",
            )
            gr.Markdown(
                """
                <p class="compact-note">
                  Lower temperature/top-k is safer. Higher values are more varied
                  but can break story logic.
                </p>
                """
            )

    output = gr.Textbox(
        label="Generated story",
        lines=16,
        buttons=["copy"],
        elem_id="story_output",
    )

    gr.Examples(
        examples=[[item] for item in PRESET_PROMPTS],
        inputs=[prompt],
        label="Story openings",
    )

    preset.change(
        fn=apply_preset,
        inputs=[preset],
        outputs=[temperature, top_k, max_new_tokens],
    )
    generate_button.click(
        fn=generate_story,
        inputs=[prompt, temperature, top_k, max_new_tokens],
        outputs=[output],
    )
    prompt.submit(
        fn=generate_story,
        inputs=[prompt, temperature, top_k, max_new_tokens],
        outputs=[output],
    )


if __name__ == "__main__":
    demo.launch(theme=APP_THEME, css=CSS)
