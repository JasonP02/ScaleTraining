import torch
from torch.amp import autocast
from transformers import AutoTokenizer
from scaletraining.config import Config

@torch.no_grad()
def main():
    cfg = Config()

    prompt = cfg.sample_gen_prompt
    max_new_tokens = cfg.generation_max_tokens
    temperature = cfg.generation_temperature
    top_k = cfg.generation_top_k

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(cfg.device)
    for _ in range(max_new_tokens):
        with autocast(dtype=torch.bfloat16, device_type='cuda'):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / max(1e-6, temperature)

            if top_k is not None and top_k > 0 and top_k < next_token_logits.size(-1):
                topk_vals, _ = torch.topk(next_token_logits, top_k)
                min_topk = topk_vals[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(next_token_logits < min_topk, torch.full_like(next_token_logits, float('-inf')), next_token_logits)

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop on EOS if defined
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\n=== Generated Story Sample ===\n" + text + "\n==============================\n")
    return text