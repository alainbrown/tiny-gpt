import torch
import os
import argparse
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import Dataset
from bpe_tokenizer import BPETokenizer
from transformers import PreTrainedTokenizerFast
from hf_model import TinyGPTForCausalLM, TinyGPTConfig
from trainer import Trainer, TrainerConfig

def _make_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

BYTES_TO_UNICODE = _make_bytes_to_unicode()

def export_hf_tokenizer_json(tokenizer, path):
    byte_encoder = BYTES_TO_UNICODE
    vocab_export = {}
    for idx, token_bytes in tokenizer.vocab.items():
        if idx == tokenizer.eos_id:
            continue
        token_str = "".join([byte_encoder[b] for b in token_bytes])
        vocab_export[token_str] = idx

    merges_export = []
    sorted_merges = sorted(tokenizer.merges.items(), key=lambda x: x[1])
    for (left, right), new_id in sorted_merges:
        left_str = "".join([byte_encoder[b] for b in tokenizer.vocab[left]])
        right_str = "".join([byte_encoder[b] for b in tokenizer.vocab[right]])
        merges_export.append(f"{left_str} {right_str}")

    data = {
        "version": "1.0",
        "added_tokens": [
            {"id": tokenizer.eos_id, "content": "<EOS>", "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
        ],
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True
        },
        "model": {
            "type": "BPE",
            "vocab": vocab_export,
            "merges": merges_export,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": ""
        }
    }

    tokenizer_dir = os.path.dirname(path)
    if tokenizer_dir:
        os.makedirs(tokenizer_dir, exist_ok=True)
        
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_checkpoint(model, optimizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

def load_checkpoint(optimizer, path):
    opt_state = torch.load(os.path.join(path, "optimizer.pt"))
    optimizer.load_state_dict(opt_state)

def create_dataloaders(ds, tokenizer, context, batch_size, n_stories, start_story=0):
    encoding = []
    end_story = start_story + n_stories
    for story in ds["train"][start_story:end_story]["text"]:
        ids = tokenizer.encode(story)
        ids.append(tokenizer.eos_token_id)
        encoding.extend(ids)

    train_ds = Dataset(encoding, context=context)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    encoding = []
    for story in ds["validation"][:n_stories]["text"]:
        ids = tokenizer.encode(story)
        ids.append(tokenizer.eos_token_id)
        encoding.extend(ids)

    val_ds = Dataset(encoding, context=context)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

def main(args):
    # 1. Load Dataset
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    # 2. Tokenizer
    if not os.path.exists(args.tokenizer_path):
        print("Training new tokenizer...")
        custom_tokenizer = BPETokenizer()
        tokenizer_texts = ds["train"][:args.tokenizer_train_stories]["text"]
        custom_tokenizer.train(tokenizer_texts, vocab_size=args.tokenizer_vocab_size)
        export_hf_tokenizer_json(custom_tokenizer, args.tokenizer_path)
        print(f"Tokenizer saved to {args.tokenizer_path}")

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path, eos_token="<EOS>")

    progress_file = os.path.join(args.checkpoint_dir, "TinyStories.progress")
    start_story = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            start_story = int(f.read().strip())
        print(f"Resuming from story {start_story}...")

    # 3. Model Initialization
    # If a checkpoint exists, always resume from it automatically
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        print(f"Resuming model from {args.checkpoint_dir}...")
        model = TinyGPTForCausalLM.from_pretrained(args.checkpoint_dir)
    else:
        print("Initializing new Model...")
        vocab_size = len(tokenizer)
        
        config = TinyGPTConfig(
            context_size=args.context_size,
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout
        )
        model = TinyGPTForCausalLM(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model moved to {device}")

    # 5. Training Setup
    train_config = TrainerConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_dir=args.checkpoint_dir
    )

    trainer = Trainer(
        model=model,
        train_loader=None, # Will be set in loop
        val_loader=None,
        config=train_config,
    )

    opt_path = os.path.join(args.checkpoint_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        print("Resuming optimizer state...")
        load_checkpoint(trainer.optimizer, args.checkpoint_dir)

    # 5. Training Loop in Chunks
    print("Starting continuous training...")
    chunk_size = 25000  # Number of stories to process before saving progress
    
    while start_story < len(ds["train"]):
        print(f"\n--- Processing stories {start_story} to {start_story + chunk_size} ---")
        train_loader, val_loader = create_dataloaders(
            ds=ds, 
            tokenizer=tokenizer, 
            context=args.context_size, 
            batch_size=args.batch_size, 
            n_stories=chunk_size,
            start_story=start_story
        )
        
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader

        trainer.train()

        # Save Checkpoint & Progress
        save_checkpoint(model, trainer.optimizer, args.checkpoint_dir)
        start_story += chunk_size
        
        with open(progress_file, "w") as f:
            f.write(str(start_story))
            
        print(f"Checkpoint saved. Progress: {start_story} stories processed.")

    print("Finished training on all stories!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyGPT")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tiny_gpt/tokenizer.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/tiny_gpt")
    
    parser.add_argument("--tokenizer_vocab_size", type=int, default=10000)
    parser.add_argument("--tokenizer_train_stories", type=int, default=100000)
    
    parser.add_argument("--context_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=100)
    
    args = parser.parse_args()
    main(args)
