import torch

class TextGenerator:
    def __init__(self, model, tokenizer, context_size):
        self.model = model
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.device = next(model.parameters()).device
        
        self.model.eval()

    def generate_stream(self, prompt_text, max_new_tokens, mode="sample", temperature=1.0, top_k=None):
        """Yields text chunks incrementally."""
        prompt_ids = self.tokenizer.encode(prompt_text)
        
        x = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        eos_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_cond = x[:, -self.context_size:]

                logits = self.model(x_cond)
                logits = logits[:, -1, :]

                if mode == "greedy":
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                elif mode == "sample":
                    logits = logits / temperature

                    if top_k is not None:
                        values, indices = torch.topk(logits, top_k)
                        filtered_logits = torch.full_like(logits, float("-inf"))
                        filtered_logits.scatter_(dim=-1, index=indices, src=values)
                        logits = filtered_logits

                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                else:
                    raise ValueError(f"Unknown generation mode: {mode}")

                x = torch.cat([x, next_id], dim=1)
                token_id = next_id.item()
                
                if eos_id is not None and token_id == eos_id:
                    break
                    
                yield self.tokenizer.decode([token_id], skip_special_tokens=True)

    def generate(self, prompt_text, max_new_tokens, mode="sample", temperature=1.0, top_k=None):
        """Returns the complete generated string."""
        chunks = []
        for chunk in self.generate_stream(prompt_text, max_new_tokens, mode, temperature, top_k):
            chunks.append(chunk)
            
        return "".join(chunks)
