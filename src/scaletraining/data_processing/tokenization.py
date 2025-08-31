from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple, Dict, Any, List, Optional
import hydra
from omegaconf import DictConfig
from scaletraining.util.utils import tokenized_dir, write_metadata, _cfg_subset

class Tokenization:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer, self.eos_id = self._get_tokenizer_and_eos(cfg.tokenizer_name)
        self.max_length = cfg.max_seq_len
        self.save_path = tokenized_dir(cfg)
        self.use_attention_mask = cfg.use_attention_mask

        try:
            # cfg.hf_dataset_names: e.g., "openwebtext" or {"text": ["file1.txt", ...]}
            self.dataset = load_dataset(cfg.hf_dataset_names)
        except Exception as e:
            raise RuntimeError(f"Could not load dataset: {e}")

    def _get_tokenizer_and_eos(self, tok_name: str):
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        # Ensure we have pad & eos
        if tok.eos_token_id is None:
            # Worst case, set eos to sep or newline; but prefer real eos.
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            # For causal LMs, pad==eos is common/convenient
            tok.pad_token = tok.eos_token
        return tok, tok.eos_token_id

    def tokenize_dataset(self):
        """
        Tokenize text -> input_ids (+ optional attention_mask).
        We append a single EOS to each sequence to enable later concatenation+packing cleanly.
        """
        def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
            out = self.tokenizer(
                examples["text"],
                add_special_tokens=False,
                truncation=True,              # keep this conservative; we’ll repack later
                max_length=self.max_length-1, # leave room for EOS we append
                padding=False,
                return_attention_mask=self.use_attention_mask,
            )
            input_ids = out["input_ids"]
            # Append EOS id to each sequence
            input_ids = [ids + [self.eos_id] for ids in input_ids]

            if self.use_attention_mask:
                # HF returns masks only if sequences were truncated; we appended EOS so extend mask
                attn = out.get("attention_mask", [[1]*len(ids) for ids in input_ids])
                attn = [m + [1] for m in attn]
                return {"input_ids": input_ids, "attention_mask": attn}
            else:
                return {"input_ids": input_ids}

        ds = self.dataset
        # Be tolerant about splits
        train_split = "train" if "train" in ds else list(ds.keys())[0]
        val_split = "validation" if "validation" in ds else ("test" if "test" in ds else None)

        tokenized_train = ds[train_split].map(
            tokenize_function,
            remove_columns=ds[train_split].column_names,
            batched=True,
            num_proc=self.cfg.num_proc,
            load_from_cache_file=True,
            desc="Tokenizing train"
        )

        if val_split:
            tokenized_val = ds[val_split].map(
                tokenize_function,
                remove_columns=ds[val_split].column_names,
                batched=True,
                num_proc=self.cfg.num_proc,
                load_from_cache_file=True,
                desc="Tokenizing val"
            )
        else:
            tokenized_val = None

        tokenized_train.save_to_disk(f"{self.save_path}/train")
        if tokenized_val is not None:
            tokenized_val.save_to_disk(f"{self.save_path}/val")

        # Save metadata for compatibility checks
        write_metadata(self.save_path, {
            "config": _cfg_subset(self.cfg),
            "tokenizer_name": self.cfg.tokenizer_name,
            "eos_token_id": self.eos_id,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
        })


@hydra.main(version_base=None, config_path='../../../conf', config_name='config')
def _hydra_tokenize(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint for tokenization.

    Args:
        cfg: Hydra DictConfig with dataset/tokenizer fields.
    """
    runner = Tokenization(cfg)
    runner.tokenize_dataset()


def main() -> None:
    """Console script entrypoint for tokenization using Hydra config.

    This wraps the Hydra-decorated function to allow `scaletraining-tokenize` to run.
    """
    _hydra_tokenize()
