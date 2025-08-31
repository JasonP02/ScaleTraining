from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig
from scaletraining.util.utils import tokenized_dir, write_metadata, _cfg_subset


def get_tokenizer(tok_name: str):
    """Load a HuggingFace tokenizer and ensure EOS/PAD exist (pad==eos).

    Returns:
        (tokenizer, eos_id)
    """
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tok.eos_token_id is None:
        print(f"Warning, eos token does not exist, using '' as eos token")
        tok.add_special_tokens({"eos_token": ""})
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, tok.eos_token_id


def tokenize_dataset(cfg) -> None:
    """Tokenize text -> input_ids (+ optional attention_mask) and save to disk.

    Appends a single EOS to each sequence to enable concatenation+packing cleanly.

    Args:
        cfg: Hydra DictConfig-like object with keys:
             - tokenizer_name: str
             - max_seq_len: int
             - tokenized_path: str
             - use_attention_mask: bool
             - num_proc: int
             - hf_dataset_names: str | dict
    """
    tok, eos_id = get_tokenizer(cfg.tokenizer_name)
    save_path = tokenized_dir(cfg)
    use_mask = cfg.use_attention_mask
    max_len = int(cfg.max_seq_len)

    try:
        ds = load_dataset(cfg.hf_dataset_names)
    except Exception as e:
        raise RuntimeError(f"Could not load dataset: {e}")

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        out = tok(
            examples["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 1,
            padding=False,
            return_attention_mask=use_mask,
        )
        input_ids = out["input_ids"]
        input_ids = [ids + [eos_id] for ids in input_ids]
        if use_mask:
            attn = out.get("attention_mask", [[1] * len(ids) for ids in input_ids])
            attn = [m + [1] for m in attn]
            return {"input_ids": input_ids, "attention_mask": attn}
        else:
            return {"input_ids": input_ids}

    train_split = "train" if "train" in ds else list(ds.keys())[0]
    val_split = "validation" if "validation" in ds else ("test" if "test" in ds else None)

    tokenized_train = ds[train_split].map(
        tokenize_function,
        remove_columns=ds[train_split].column_names,
        batched=True,
        num_proc=cfg.num_proc,
        load_from_cache_file=True,
        desc="Tokenizing train",
    )
    tokenized_train.save_to_disk(f"{save_path}/train")

    if val_split:
        tokenized_val = ds[val_split].map(
            tokenize_function,
            remove_columns=ds[val_split].column_names,
            batched=True,
            num_proc=cfg.num_proc,
            load_from_cache_file=True,
            desc="Tokenizing val",
        )
        tokenized_val.save_to_disk(f"{save_path}/val")

    write_metadata(save_path, {
        "config": _cfg_subset(cfg),
        "tokenizer_name": cfg.tokenizer_name,
        "eos_token_id": eos_id,
        "tokenizer_vocab_size": tok.vocab_size,
    })


@hydra.main(version_base=None, config_path='../../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    """Hydra console script entrypoint for tokenization."""
    tokenize_dataset(cfg)
