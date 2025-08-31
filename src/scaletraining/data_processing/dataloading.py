from torch.utils.data import DataLoader
from datasets import load_from_disk
from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.data_processing.tokenization import Tokenization
from scaletraining.util.utils import tokenized_dir, packed_dir, read_metadata, _cfg_subset
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
        tokenizer_runner = Tokenization(cfg)
        tokenizer_runner.tokenize_dataset()

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

    train = load_from_disk(f"{pk_dir}/train").with_format("torch", columns=["input_ids", "labels"])
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    val_loader = None
    try:
        val = load_from_disk(f"{pk_dir}/val").with_format("torch", columns=["input_ids", "labels"])
        val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    except Exception:
        pass
    return train_loader, val_loader
