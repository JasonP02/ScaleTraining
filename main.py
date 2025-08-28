from trainer import LLMTrainer
from config import Config
from dataload import load_tokenized_dataset
from model import TransformerNetwork
from utils import configure_rocm_and_sdp, clear_cuda_cache
import wandb


if __name__ == "__main__":
    cfg = Config()
    configure_rocm_and_sdp(cfg)
    clear_cuda_cache()


    train_loader, val_loader = load_tokenized_dataset(cfg)
    print(f"Data loaded")


    model = TransformerNetwork(cfg)
    trainer = LLMTrainer(cfg, model, train_loader, val_loader)

    trainer.training_run()
    wandb.finish()
    