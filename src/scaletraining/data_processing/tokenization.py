from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig
from scaletraining.util.utils import tokenized_dir, write_metadata, _cfg_subset, flatten_cfg
from pathlib import Path
import subprocess
import sys


def get_tokenizer_name_from_dataset(dataset_specs, vocab_size: int | None = None):
    """Generate tokenizer name based on dataset specifications.
    
    Args:
        dataset_specs: Single dataset name or list of dataset names
        
    Returns:
        Path to the corresponding tokenizer file
    """
    # Handle single dataset or list
    specs = dataset_specs if isinstance(dataset_specs, list) else [dataset_specs]
    
    # Generate safe name from dataset specs
    safe_name = "_".join([spec.replace("/", "_").replace("-", "_") for spec in specs])
    base_dir = Path.cwd() / "tokenizers"
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_v{int(vocab_size)}" if vocab_size is not None else ""
    tokenizer_path = base_dir / f"tokenizer_{safe_name}{suffix}.json"
    return str(tokenizer_path)


def ensure_dataset_tokenizer_exists(cfg):
    """Ensure a tokenizer exists for the specified dataset(s).
    
    If no dataset-specific tokenizer exists, train one automatically.
    """
    dataset_tokenizer = get_tokenizer_name_from_dataset(cfg.hf_dataset_names, getattr(cfg, "tokenizer_vocab_size", None))
    
    if not Path(dataset_tokenizer).exists():
        print(f"Dataset-specific tokenizer not found: {dataset_tokenizer}")
        print("Training new tokenizer for dataset(s):", cfg.hf_dataset_names)
        
        # Run the tokenizer training
        try:
            result = subprocess.run([
                sys.executable, "-m", "scaletraining.data_processing.train_tokenizer"
            ], cwd=str(Path.cwd()), capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Tokenizer training completed successfully")
                print(result.stdout)
            else:
                print("Tokenizer training failed:")
                print(result.stderr)
                raise RuntimeError("Failed to train tokenizer")
                
        except Exception as e:
            print(f"Error training tokenizer: {e}")
            raise
    else:
        print(f"Using existing dataset-specific tokenizer: {dataset_tokenizer}")


def get_tokenizer(tok_name: str):
    """Load a HuggingFace tokenizer and ensure EOS/PAD exist (pad==eos).

    Returns:
        (tokenizer, eos_id)
    """
    from pathlib import Path
    from tokenizers import Tokenizer
    
    # Check if it's a local .json tokenizer file
    if Path(tok_name).exists() and tok_name.endswith('.json'):
        print(f"Loading local tokenizer file: {tok_name}")
        # Load the tokenizer using tokenizers library
        tokenizer_obj = Tokenizer.from_file(tok_name)
        
        # Wrap it in a minimal AutoTokenizer-compatible object
        class LocalTokenizer:
            def __init__(self, tokenizer_obj):
                self._tokenizer = tokenizer_obj
                self.vocab_size = tokenizer_obj.get_vocab_size()
                # Set EOS token (assume it's one of the special tokens)
                vocab = tokenizer_obj.get_vocab()
                self.eos_token_id = vocab.get("[SEP]", vocab.get("</s>", vocab.get("<|endoftext|>", 2)))
                self.pad_token_id = vocab.get("[PAD]", self.eos_token_id)
                
            def __call__(self, text, **kwargs):
                # Handle batch tokenization
                if isinstance(text, list):
                    results = [self._tokenizer.encode(t) for t in text]
                    input_ids = [r.ids for r in results]
                else:
                    result = self._tokenizer.encode(text)
                    input_ids = result.ids
                
                output = {"input_ids": input_ids}
                if kwargs.get("return_attention_mask", False):
                    if isinstance(text, list):
                        output["attention_mask"] = [[1] * len(ids) for ids in input_ids]
                    else:
                        output["attention_mask"] = [1] * len(input_ids)
                
                return output
        
        tok = LocalTokenizer(tokenizer_obj)
        
    else:
        # Use standard AutoTokenizer for HF models
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
    # Ensure dataset-specific tokenizer exists
    ensure_dataset_tokenizer_exists(cfg)
    
    # Use dataset-specific tokenizer
    tokenizer_name = get_tokenizer_name_from_dataset(cfg.hf_dataset_names, getattr(cfg, "tokenizer_vocab_size", None))
    print(f"Using dataset-specific tokenizer: {tokenizer_name}")
    
    tok, eos_id = get_tokenizer(tokenizer_name)
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
        "tokenizer_name": tokenizer_name,
        "eos_token_id": eos_id,
        "tokenizer_vocab_size": tok.vocab_size,
    })


@hydra.main(version_base=None, config_path='../../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    """Hydra console script entrypoint for tokenization."""
    cfg = flatten_cfg(cfg)
    tokenize_dataset(cfg)
