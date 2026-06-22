import torch

def generate_stream(
    model,
    prompt_ids,
    max_new_tokens,
    context_size,
    tokenizer,
    mode="sample",
    temperature=1.0,
    top_k=None,
):
    model.eval()
    device = next(model.parameters()).device

    x = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    eos_id = tokenizer.eos_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop the sequence to the maximum context size
            x_cond = x[:, -context_size:]

            logits = model(x_cond)
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

            # Append the predicted token to the running sequence
            x = torch.cat([x, next_id], dim=1)

            token_id = next_id.item()
            
            # Stop if we hit the end-of-sequence token
            if eos_id is not None and token_id == eos_id:
                break
                
            # Yield the generated token ID
            yield token_id
