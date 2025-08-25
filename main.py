from trainer import TransformerNetwork
from config import Config
from dataload import load_tokenized_dataset

if __name__ == "main":
    cfg = Config()
    train_loader, val_loader = load_tokenized_dataset(cfg)
    model = TransformerNetwork(cfg, train_loader=train_loader, val_loader=val_loader)