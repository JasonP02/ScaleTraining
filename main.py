from trainer import LLMTrainer
from config import Config
from dataload import load_tokenized_dataset
from model import TransformerNetwork
import torch
import gc
import os

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def configure_rocm_and_sdp(cfg):
    os.enviorn.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    
    try:
        torch.backends.cuda.enable_flash_sdp(cfg.use_flash_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(cfg.use_mem_efficent_sdp)
        torch.backends.cuda.enable_math_sdp(cfg.use_math_sdp)
    except Exception as e:
        print(f"SDP backend config skipped: {e}")
    

if __name__ == "__main__":
    cfg = Config()
    configure_rocm_and_sdp(cfg)

    clear_cuda_cache()
    train_loader, val_loader = load_tokenized_dataset(cfg)
    print(f"Data loaded")
    model = TransformerNetwork(cfg)
    trainer = LLMTrainer(cfg, model, train_loader, val_loader)

    trainer.training_run()
    