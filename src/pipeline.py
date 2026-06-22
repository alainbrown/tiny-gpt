import torch
import os
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import Dataset
from bpe_tokenizer import BPETokenizer
from hf_model import TinyGPTForCausalLM, TinyGPTConfig
from trainer import Trainer, TrainerConfig

def create_dataloaders(ds, tokenizer, context, batch_size, n_stories):
    encoding = []
    for story in ds["train"][:n_stories]["text"]:
        encoding.extend(tokenizer.encode(story, add_eos=True))

    train_ds = Dataset(encoding, context=context)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    encoding = []
    for story in ds["validation"][:n_stories]["text"]:
        encoding.extend(tokenizer.encode(story, add_eos=True))

    val_ds = Dataset(encoding, context=context)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

def main(args):
    # 1. Load Dataset
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    # 2. Tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(args.tokenizer_path):
        print(f"Loading tokenizer from {args.tokenizer_path}...")
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer_texts = ds["train"][:1000]["text"]
        tokenizer.train(tokenizer_texts, vocab_size=1024)
        tokenizer.save(args.tokenizer_path)
        print(f"Tokenizer saved to {args.tokenizer_path}")

    # 3. Dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        ds=ds, 
        tokenizer=tokenizer, 
        context=args.context_size, 
        batch_size=args.batch_size, 
        n_stories=args.n_stories
    )

    # 4. Model Initialization
    print("Initializing Model...")
    vocab_size = len(tokenizer.vocab)
    
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
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
    )

    # 6. Training Loop
    print("Starting training...")
    val_steps, train_losses, val_losses, throughput = trainer.train()

    # 7. Save Checkpoint
    print(f"Training complete. Saving checkpoint to {args.checkpoint_dir}...")
    trainer.save_checkpoint(args.checkpoint_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyGPT")
    parser.add_argument("--tokenizer_path", type=str, default="tiny_bpe_v2.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/tiny_gpt")
    
    parser.add_argument("--context_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_stories", type=int, default=1000, help="Number of stories to train on")
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=100)
    
    args = parser.parse_args()
    main(args)
