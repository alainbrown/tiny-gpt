import gradio as gr
import torch
import os
import spaces

from src.generate import generate_stream
from src.hf_model import TinyGPTForCausalLM
from src.bpe_tokenizer import BPETokenizer

checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints/tiny_gpt")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "tiny_bpe_v2.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = BPETokenizer.load(tokenizer_path)
    model = TinyGPTForCausalLM.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    context_size = model.config.context_size
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded at startup. {e}")
    tokenizer = None
    model = None
    context_size = 32

@spaces.GPU
def stream_chat(message, history, temperature, top_k):
    if model is None:
        yield "Model not found. Please ensure the checkpoint exists."
        return

    prompt_ids = tokenizer.encode(message)
    
    generator = generate_stream(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=200,
        context_size=context_size,
        tokenizer=tokenizer,
        mode="sample",
        temperature=temperature,
        top_k=int(top_k) if top_k > 0 else None,
    )
    
    output_ids = []
    for token_id in generator:
        output_ids.append(token_id)
        yield tokenizer.decode(output_ids, skip_special=True)

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
