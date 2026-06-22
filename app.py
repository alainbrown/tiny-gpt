import gradio as gr
import torch
import os
import spaces

from tiny_gpt.generate import TextGenerator
from tiny_gpt.hf_model import TinyGPTForCausalLM
from transformers import PreTrainedTokenizerFast

checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints/tiny_gpt")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "checkpoints/tiny_gpt/tokenizer.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, eos_token="<EOS>")
    model = TinyGPTForCausalLM.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    context_size = model.config.context_size
    text_generator = TextGenerator(model, tokenizer, context_size)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded at startup. {e}")
    tokenizer = None
    model = None
    context_size = 32
    text_generator = None

@spaces.GPU
def stream_chat(message, history, temperature, top_k):
    if text_generator is None:
        yield "Model not found. Please ensure the checkpoint exists."
        return

    output_text = ""
    for chunk in text_generator.generate_stream(
        prompt_text=message,
        max_new_tokens=200,
        mode="sample",
        temperature=temperature,
        top_k=int(top_k) if top_k > 0 else None,
    ):
        output_text += chunk
        yield output_text

demo = gr.ChatInterface(
    fn=stream_chat,
    title="Tiny GPT Inference Demo",
    description="An educational GPT implementation from scratch, trained on TinyStories.",
    additional_inputs=[
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Top-K (0 to disable)"),
    ],
)

if __name__ == "__main__":
    demo.launch()
