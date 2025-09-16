from torch.utils.data import DataLoader
from datasets import load_from_disk
from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.data_processing.tokenization import tokenize_dataset
from scaletraining.util.artifacts import read_metadata
from scaletraining.util.config import _cfg_subset
from scaletraining.util.paths import packed_dir, tokenized_dir
import os

def build_loaders(cfg):
    """
    Build PyTorch DataLoaders from packed datasets using configuration.
    If cfg.do_packing is True, will pack tokenized datasets first.
    """
    # Ensure tokenized datasets exist; if not, run tokenization
    tok_dir = tokenized_dir(cfg)
    tokenized_train_dir = os.path.join(tok_dir, "train")
    if not os.path.isdir(tokenized_train_dir):
        tokenize_dataset(cfg)

    # Ensure packed datasets exist; if not (or if do_packing True), run packing
    pk_dir = packed_dir(cfg)
    packed_train_dir = os.path.join(pk_dir, "train")
    if cfg.do_packing or not os.path.isdir(packed_train_dir):
        pack_and_save(
            tokenized_path=tok_dir,
            packed_path=pk_dir,
            block_size=cfg.max_seq_len,
            num_proc=cfg.pack_num_proc,
            map_batch_size=cfg.pack_map_batch_size,
            writer_batch_size=cfg.pack_writer_batch_size,
            metadata={"config": _cfg_subset(cfg)}
        )

    # Compatibility check
    meta = read_metadata(pk_dir) or read_metadata(tok_dir)
    if meta:
        # Enforce dataset fundamentals match current intent
        if cfg.strict_dataset_compat:
            current = _cfg_subset(cfg)
            saved = meta.get("config", {})
            if any(saved.get(k) != current.get(k) for k in current.keys()):
                raise RuntimeError(f"Dataset/tokenizer mismatch. Saved={saved} vs Current={current}")
        # Auto-set vocab size from metadata for model creation later
        saved_vocab = meta.get("tokenizer_vocab_size")
        if saved_vocab is not None:
            try:
                cfg.vocab_size = int(saved_vocab)
            except Exception:
                pass
        # Propagate canonical tokenizer path into cfg for downstream usage (generation, logging)
        saved_tok = meta.get("tokenizer_name")
        if saved_tok:
            try:
                cfg.tokenizer_name = saved_tok
            except Exception:
                pass

    train = load_from_disk(f"{pk_dir}/train").with_format("torch", columns=["input_ids"])
    # DataLoader performance knobs: parallel workers + pinned memory for faster H2D copies.
    # persistent_workers avoids process respawn overhead per epoch; prefetch_factor overlaps host/GPU work.
    train_loader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=getattr(cfg, 'loader_num_workers', 0),
        pin_memory=bool(getattr(cfg, 'loader_pin_memory', False)),
        persistent_workers=bool(getattr(cfg, 'loader_persistent_workers', False)) if getattr(cfg, 'loader_num_workers', 0) > 0 else False,
        prefetch_factor=int(getattr(cfg, 'loader_prefetch_factor', 2)) if getattr(cfg, 'loader_num_workers', 0) > 0 else None,
    )

    val_loader = None
    try:
        val = load_from_disk(f"{pk_dir}/val").with_format("torch", columns=["input_ids"])
        val_loader = DataLoader(
            val,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=getattr(cfg, 'loader_num_workers', 0),
            pin_memory=bool(getattr(cfg, 'loader_pin_memory', False)),
            persistent_workers=bool(getattr(cfg, 'loader_persistent_workers', False)) if getattr(cfg, 'loader_num_workers', 0) > 0 else False,
            prefetch_factor=int(getattr(cfg, 'loader_prefetch_factor', 2)) if getattr(cfg, 'loader_num_workers', 0) > 0 else None,
        )
    except Exception:
        pass
    return train_loader, val_loader
