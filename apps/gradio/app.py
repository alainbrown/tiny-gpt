import os

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get("MODEL_ID", "alainbrown/tiny-gpt")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    model.eval()
    return model, tokenizer


def build_prompt(message, history):
    parts = []

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            label = "Assistant" if role == "assistant" else "User"
            parts.append(f"{label}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            if user_message:
                parts.append(f"User: {user_message}")
            if assistant_message:
                parts.append(f"Assistant: {assistant_message}")

    parts.append(f"User: {message}")
    parts.append("Assistant:")
    return "\n".join(parts)


try:
    model, tokenizer = load_model()
    load_error = None
    print(f"Loaded {MODEL_ID}@{MODEL_REVISION} on {DEVICE}.")
except Exception as error:
    model = None
    tokenizer = None
    load_error = str(error)
    print(f"Could not load {MODEL_ID}@{MODEL_REVISION}: {error}")


@spaces.GPU
def stream_chat(message, history, temperature, top_k, max_new_tokens):
    if model is None or tokenizer is None:
        yield f"Model {MODEL_ID} is currently unavailable."
        return

    message = message.strip()
    if not message:
        yield "Please enter a message."
        return

    prompt = build_prompt(message, history)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    context_size = model.config.context_size
    input_ids = input_ids[:, -context_size:]
    output_text = ""

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            model_inputs = input_ids[:, -context_size:]
            logits = model(input_ids=model_inputs).logits[:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                values, indices = torch.topk(logits, int(top_k))
                filtered_logits = torch.full_like(logits, float("-inf"))
                filtered_logits.scatter_(1, indices, values)
                logits = filtered_logits

            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)

            token_id = next_id.item()
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

            chunk = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
            )
            if chunk:
                output_text += chunk
                yield output_text

    if not output_text:
        yield "The model ended its response immediately. Please try another prompt."


demo = gr.ChatInterface(
    fn=stream_chat,
    title="Tiny GPT",
    description="An educational GPT model trained from scratch on TinyStories.",
    additional_inputs=[
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1,
            label="Temperature",
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=20,
            step=1,
            label="Top-K (0 to disable)",
        ),
        gr.Slider(
            minimum=16,
            maximum=300,
            value=128,
            step=16,
            label="Maximum new tokens",
        ),
    ],
    examples=[
        ["Tell me a short story about a brave little fox."],
        ["Once upon a time, a robot found a tiny garden."],
        ["Write a bedtime story about the moon and the sea."],
    ],
)


if __name__ == "__main__":
    demo.launch()
