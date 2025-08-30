from torch.utils.data import DataLoader
from datasets import load_from_disk

def build_loaders(packed_path: str, batch_size: int):
    train = load_from_disk(f"{packed_path}/train").with_format("torch", columns=["input_ids", "labels"])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = None
    try:
        val = load_from_disk(f"{packed_path}/val").with_format("torch", columns=["input_ids", "labels"])
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)
    except Exception:
        pass
    return train_loader, val_loader
