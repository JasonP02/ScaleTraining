from torch.utils.data import DataLoader
from datasets import load_from_disk
from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.data_processing.tokenization import Tokenization
import os

def build_loaders(cfg):
    """
    Build PyTorch DataLoaders from packed datasets using configuration.
    If cfg.do_packing is True, will pack tokenized datasets first.
    """
    # Ensure tokenized datasets exist; if not, run tokenization
    tokenized_train_dir = os.path.join(cfg.tokenized_path, "train")
    if not os.path.isdir(tokenized_train_dir):
        tokenizer_runner = Tokenization(cfg)
        tokenizer_runner.tokenize_dataset()

    # Ensure packed datasets exist; if not (or if do_packing True), run packing
    packed_train_dir = os.path.join(cfg.batched_tokenized_path, "train")
    if cfg.do_packing or not os.path.isdir(packed_train_dir):
        pack_and_save(
            tokenized_path=cfg.tokenized_path,
            packed_path=cfg.batched_tokenized_path,
            block_size=cfg.max_seq_len,
            num_proc=cfg.pack_num_proc,
            map_batch_size=cfg.batch_size,
            writer_batch_size=cfg.pack_writer_batch_size,
        )

    train = load_from_disk(f"{cfg.batched_tokenized_path}/train").with_format("torch", columns=["input_ids", "labels"])
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    val_loader = None
    try:
        val = load_from_disk(f"{cfg.batched_tokenized_path}/val").with_format("torch", columns=["input_ids", "labels"])
        val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    except Exception:
        pass
    return train_loader, val_loader
