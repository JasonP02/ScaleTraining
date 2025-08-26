from trainer import LLMTrainer
from config import Config
from dataload import load_tokenized_dataset
from model import TransformerNetwork

if __name__ == "__main__":
    cfg = Config()
    train_loader, val_loader = load_tokenized_dataset(cfg)
    model = TransformerNetwork(cfg)
    trainer = LLMTrainer(cfg, model, train_loader, val_loader)
    trainer.training_run()
    