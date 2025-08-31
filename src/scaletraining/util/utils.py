import torch
import gc
import os

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def configure_rocm_and_sdp(cfg):
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.backends.cuda.enable_flash_sdp(cfg.use_flash_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(cfg.use_mem_efficient_sdp)
        torch.backends.cuda.enable_math_sdp(cfg.use_math_sdp)
    except Exception as e:
        print(f"SDP backend config skipped: {e}")