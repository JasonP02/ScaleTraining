from dataclasses import dataclass
import torch

@dataclass
class Config():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 64
    accum_steps: int = 8
    vocab_size: int = 50257
    n_layer: int = 1
    max_seq_len: int = 1000
    grad_clip_norm: float = 1.0

    use_checkpoint: bool = True

    # Kernel stuff
    use_flash_sdp: bool = False
    use_mem_efficient_sdp: bool = True
    use_math_sdp: bool = False

    debug_memory: bool = True # For debug...

    n_head: int = 4
    n_embed: int = 256
    bias: bool = True
    attn_dropout: float = 0.2
    resid_dropout: float = 0.2
    UE_bias: bool = False

    n_hidden: int = 256*4
    momentum_coef: float = 0.99
    lr: float = 1e-4
    max_train_tokens: int = 100000
    max_val_tokens: int = 100000
    
    # Optimizer parameters
    beta: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    ns_iters: int = 5
    eps: float = 1e-8
    
    # Data path
    tokenized_path: str = 'datasets'
    wandb_project_name: str = 'tiny-stories-base'
    hf_dataset_names: str = 'roneneldan/TinyStories'

    tokenizer: str = 'EleutherAI/gpt-neo-125M'
    tokenizer_type: str = 'hf'
