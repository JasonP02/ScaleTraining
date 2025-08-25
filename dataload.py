from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

# Loading tinystories
def load_tiny_stories(cfg):
    tsplit = load_dataset("roneneldan/TinyStories", split="train")
    vsplit = load_dataset("roneneldan/TinyStories", split="validation")
    # Remove torch format for text column as it's unsupported
    tsplit = tsplit.with_format(None)
    vsplit = vsplit.with_format(None)
    train_loader = DataLoader(tsplit, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(vsplit, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader
