from torch.utils.data import DataLoader
from datasets import load_from_disk
from scaletraining.data_processing.batch_packer import pack_and_save

def build_loaders(cfg):
    """
    Build PyTorch DataLoaders from packed datasets using configuration.
    If cfg.do_packing is True, will pack tokenized datasets first.
    """
    # Optionally pack datasets based on config
    if cfg.do_packing:
        pack_and_save(
            tokenized_path=cfg.tokenized_path,
            packed_path=cfg.batched_tokenized_path,
            block_size=cfg.max_seq_len,
            num_proc=cfg.pack_num_proc,
            map_batch_size=cfg.pack_map_batch_size,
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
