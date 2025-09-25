from transformers import AutoTokenizer
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig
from scaletraining.data_processing.dataset_utils import dataset_safe_name, load_hf_dataset
from scaletraining.data_processing.tokenizer import Tokenizer
from scaletraining.util.artifacts import write_metadata
from scaletraining.util.config import _cfg_subset, flatten_cfg
from scaletraining.util.path_utils import get_tokenized_directory
from pathlib import Path


def get_tokenizer_name_from_dataset(
    dataset_specs,
    vocab_size: int | None = None,
    dataset_configs=None,
):
    """Generate tokenizer name based on dataset specifications.
    
    Args:
        dataset_specs: Single dataset name or list of dataset names
        
    Returns:
        Path to the corresponding tokenizer file
    """
    # Handle single dataset or list
    specs = dataset_specs if isinstance(dataset_specs, list) else [dataset_specs]
    configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs] if dataset_configs else [None] * len(specs)
    if len(configs) == 1 and len(specs) > 1:
        configs = configs * len(specs)
    safe_name = dataset_safe_name(specs, configs)
    base_dir = Path.cwd() / "tokenizers"
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_v{int(vocab_size)}" if vocab_size is not None else ""
    tokenizer_path = base_dir / f"tokenizer_{safe_name}{suffix}.json"
    return str(tokenizer_path)


def tokenize_dataset(cfg, tok: Tokenizer) -> None:
    """Tokenize text -> input_ids and save to disk.

    Appends a single EOS to each sequence to enable concatenation+packing cleanly.

    Args:
        cfg: Hydra DictConfig-like object with keys:
             - tokenizer_name: str
             - max_seq_len: int
             - tokenized_path: str
             - num_proc: int
             - hf_dataset_names: str | dict
    """
    save_path = tok.get_tokenized_directory(cfg)
    max_len = int(cfg.max_seq_len)

    try:
        ds = load_hf_dataset(
            cfg.hf_dataset_names,
            getattr(cfg, "hf_dataset_config_name", None),
        )
    except Exception as e:
        raise RuntimeError(f"Could not load dataset: {e}")

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        out = tok(
            examples["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 1,
            padding=False,
        )
        input_ids = out["input_ids"]
        input_ids = [ids + [eos_id] for ids in input_ids]
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
    tokenize_dataset(cfg)
