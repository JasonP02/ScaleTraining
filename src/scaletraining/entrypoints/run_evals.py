"""
Entrypoint for evaluating model performance across supported benchmarks
Currently supported benchmarks:

To run an eval on a single benchmark, use the --{benchmark} flag, otherwise all supported benchmarks will run
"""
from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from scaletraining.config import load_project_config
from scaletraining.util import resolve_device
from scaletraining.util.eval_utils import evaluate_perplexity
from scaletraining.data_processing import build_loaders, get_loader_kwargs
from scaletraining.util.eval_utils import load_pretrained_model_and_tokenizer

# Remove EvalTokenizer class and use a simple function for clarity
def tokenize_fn(tok, example, col):
    return tok(example[col], truncation=True, padding="max_length")

def eval_on_gsm8k(cfg, model, tok):
    """Requires instruction fine-tuning, will have to put on hold or come back to this when I do SFT"""
    dataset = load_dataset('openai/gsm8k', 'main')
    tokenized_dataset = dataset.map(lambda ex: tokenize_fn(tok, ex, "question"), batched=True)

    loader_kwargs = get_loader_kwargs(cfg.training)
    print(loader_kwargs)

    batch_size = int(getattr(cfg.training, "eval_batch_size", 1))

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # Call accuracy estimation for manual/automatic review
    correct = 0
    total = 0
    model.eval()
    for batch in dataloader:
        questions = batch["question"]
        gold_answers = batch["answer"]
        # NOTE: Replace this with your actual model generation logic
        # Example: preds = model.generate(...)
        preds = [""] * len(questions)  # Placeholder: model not instruction-tuned
        for q, pred, gold in zip(questions, preds, gold_answers):
            print(f"Q: {q}\nPred: {pred}\nGold: {gold}\n---")
            if str(pred).strip() == str(gold).strip():
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    print(f"GSM8K accuracy (string match): {accuracy:.4f} ({correct}/{total})")
    return accuracy

def make_arc_prompts(dataset):
    samples = []
    for problem in dataset:
        choice_lines = [
            f"{label}: {text}"
            for label, text in zip(problem["choices"]["label"], problem["choices"]["text"])
        ]
        prompt = (
            "Answer the following question:\n"
            f"{problem['question']}\n"
            "Choices:\n"
            + "\n".join(choice_lines)
            + "Answer: "
        )
        samples.append(
            {
                "prompt": prompt,
                "label": problem["answerKey"],
                "options": list(problem["choices"]["label"]),
            }
        )
    return samples


def eval_on_arc(cfg, model, tok):
    dataset = load_dataset("ai2_arc", "ARC-Easy", split='test')
    arc_prompts = make_arc_prompts(dataset)

    correct = 0
    total = 0
    model.eval()
    device = next(model.parameters()).device

    for sample in arc_prompts:
        prompt = sample['prompt']
        label = sample['label']
        options = sample['options']

        prompt_tokens = tok(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = prompt_tokens.input_ids.shape[1]
        log_probs_per_option = []

        for option in options:
            full_text = prompt + option
            inputs = tok(full_text, return_tensors='pt', add_special_tokens=False)
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                logits = model(input_ids)

            # Align logits with targets: logits[:, :-1, :] predicts input_ids[:, 1:]
            shifted_logits = logits[:, :-1, :]
            shifted_targets = input_ids[:, 1:]

            option_logits = shifted_logits[:, prompt_len-1:, :]
            option_ids = shifted_targets[:, prompt_len-1:]

            log_probs = F.log_softmax(option_logits, dim=-1)
            # Lookup to determine our scores
            option_log_probs = torch.gather(log_probs, -1, option_ids.unsqueeze(-1)).squeeze(-1)

            # Sum probs to get our value
            total_option_score = option_log_probs.sum(dim=-1).item()
            log_probs_per_option.append(total_option_score)

        predicted_idx = torch.tensor(log_probs_per_option).argmax().item()
        predicted_label = options[predicted_idx]

        if predicted_label == label:
            correct += 1
        total += 1

        # Optional: print progress
        if total % 50 == 0:
            print(f"Processed: {total}/{len(arc_prompts)} | Accuracy: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"Final ARC-Easy Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy




@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = load_project_config(cfg)
    resolve_device(cfg)
    model, tok = load_pretrained_model_and_tokenizer(cfg)
    eval_on_arc(cfg, model, tok)
    # eval_on_gsm8k(cfg, model, tok)


if __name__ == "__main__":
    main()
