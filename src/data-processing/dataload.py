from torch.utils.data import DataLoader
from config import Config

def collate_function(batch):
    for item in batch:
        print(item[1])
        print(item[0])

def load_tokenized_dataset(cfg):
    """
    Load the tokenized dataset from the disk
    Args:
        cfg: Config object
    Returns:
        train_loader: DataLoader object for the train dataset
        val_loader: DataLoader object for the validation dataset
    """
    from datasets import load_from_disk
    
    # Load the saved datasets
    train_dataset = load_from_disk(f'{cfg.data_path}/train')
    val_dataset = load_from_disk(f'{cfg.data_path}/val')
    
    # Create collate function with the correct pad_token
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=self.collate_function)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=self.collate_function)
    return train_loader, val_loader


if __name__ == '__main__':
    cfg = Config()
    train_loader, val_loader = load_tokenized_dataset(cfg)
    print("Train batch example:")
    print(next(iter(train_loader)))
    print("\nValidation batch example:")
    print(next(iter(val_loader)))


