import os
from typing import Any

from datasets import load_from_disk
from torch.utils.data import DataLoader

from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.data_processing.tokenization import tokenize_dataset
from scaletraining.util.artifacts import read_metadata
from scaletraining.util.config import _cfg_subset
from scaletraining.util.path_utils import get_packed_directory, get_tokenized_directory
from dataclasses import dataclass

def check_tokenizer_metadata_match(cfg, dataset_root, tok_dir, pk_dir):
    meta = read_metadata(dataset_root) or read_metadata(tok_dir) or read_metadata(pk_dir)
    if meta:
        if cfg.strict_dataset_compat:
            current = _cfg_subset(cfg)
            saved = meta.get("config", {})
            if any(saved.get(k) != current.get(k) for k in current.keys()):
                raise RuntimeError(f"Dataset/tokenizer mismatch. Saved={saved} vs Current={current}")
        saved_vocab = meta.get("tokenizer_vocab_size")
        if saved_vocab is not None:
            try:
                cfg.vocab_size = int(saved_vocab)
            except Exception:
                pass
        saved_tok = meta.get("tokenizer_name")
        if saved_tok:
            try:
                cfg.tokenizer_name = saved_tok
            except Exception:
                pass

def is_tokenized(tokenized_path):
    return os.path.isdir(tokenized_path)

def is_packed(packed_path):
    return os.path.isdir(packed_path)


def get_loader_kwargs(cfg):
    num_workers = int(getattr(cfg, "loader_num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(cfg, "loader_pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(cfg, "loader_persistent_workers", False))
        prefetch = getattr(cfg, "loader_prefetch_factor", None)
        if prefetch:
            loader_kwargs["prefetch_factor"] = int(prefetch)
    return loader_kwargs
            
def build_loaders(cfg, for_training: bool = True):
    """Build PyTorch DataLoaders from dataset artifacts.

    When `for_training` is True (default) we operate on packed, fixed-length
    shards and create shuffled loaders suitable for training. When False we
    work directly from the tokenized split directories so evaluation code can
    reuse variable-length text without repacking.
    """
    tok_dir = get_tokenized_directory(cfg, for_training)
    tokenized_train_dir = os.path.join(tok_dir, "train")
    # We dont want to tokenize for evals.
    if not is_tokenized(tokenized_train_dir) and for_training:
        tokenize_dataset(cfg)
        
    pk_dir = get_packed_directory(cfg, for_training)  # expected packed dataset location for this config
    if for_training:
        dataset_root = pk_dir
        packed_data_dir = os.path.join(pk_dir, "train")
        if not is_packed(packed_data_dir) or cfg.do_packing:
            pack_and_save(
                tokenized_path=tok_dir,
                packed_path=pk_dir,
                block_size=cfg.max_seq_len,
                num_proc=cfg.pack_num_proc,
                map_batch_size=cfg.pack_map_batch_size,
                writer_batch_size=cfg.pack_writer_batch_size,
                metadata={"config": _cfg_subset(cfg)}
            )
    else:
        dataset_root = tok_dir
    # Compatibility/metadata sync with persisted artifacts.
    check_tokenizer_metadata_match(cfg, dataset_root, tok_dir, pk_dir)

    train = load_from_disk(f"{dataset_root}/train").with_format("torch", columns=["input_ids"])
    loader_kwargs = get_loader_kwargs(cfg)

    eval_bsz = getattr(cfg, "eval_batch_size", cfg.batch_size)
    bsz = int(cfg.batch_size if for_training else eval_bsz)

    train_loader = DataLoader(
        train,
        batch_size=bsz,
        shuffle=bool(for_training),
        drop_last=bool(for_training),
        **loader_kwargs,
    )

    val_loader = None
    try:
        val = load_from_disk(f"{dataset_root}/val").with_format("torch", columns=["input_ids"])
        val_loader = DataLoader(
            val,
            batch_size=bsz,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    except Exception:
        pass
    return train_loader, val_loader
