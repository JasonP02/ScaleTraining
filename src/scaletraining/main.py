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
    # Simple qualitative eval: generate a short story
    try:
        trainer.generate_sample_story(
            prompt="Once upon a time in a small village,",
            max_new_tokens=100,
            temperature=1.0,
            top_k=50,
        )
    except Exception as e:
        print(f"Generation failed: {e}")
    wandb.finish()
    
