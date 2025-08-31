from scaletraining.model import LLMTrainer
from scaletraining.config import Config
from scaletraining.data_processing import build_loaders
from scaletraining.util import configure_rocm_and_sdp, clear_cuda_cache
import wandb


if __name__ == "__main__":
    cfg = Config()
    configure_rocm_and_sdp(cfg)
    clear_cuda_cache()

    train_loader, val_loader = build_loaders(cfg)

    print(f"Data loaded")

    trainer = LLMTrainer(cfg, train_loader, val_loader)

    trainer.training_run()
    wandb.finish()
    
