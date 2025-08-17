from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

# Loading tinystories
def load_tiny_stories(cfg):
    tsplit = load_dataset("roneneldan/TinyStories", split="train")
    vsplit = load_dataset("roneneldan/TinyStories", split="validation")
    tsplit.set_format(type='torch', columns=['text']) # type: ignore
    vsplit.set_format(type='torch', columns=['text']) # type: ignore
    train_loader = DataLoader(tsplit, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(vsplit, batch_size=cfg.batch_size, shuffle=True)
    return train_loader, val_loader
