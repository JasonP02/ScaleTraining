from dataclasses import dataclass
import torch
import os

@dataclass
class Config():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 16
    accum_steps: int = 4
    vocab_size: int = 50257
    n_layer: int = 2
    max_seq_len: int = 1000
    grad_clip_norm: float = 1.0

    use_checkpoint: bool = True

    # Kernel stuff
    use_flash_sdp: bool = False
    use_mem_efficient_sdp: bool = True
    use_math_sdp: bool = False

    debug_memory: bool = True # For debug...

    n_head: int = 8
    n_embed: int = 256
    bias: bool = True
    attn_dropout: float = 0.2
    resid_dropout: float = 0.2
    UE_bias: bool = False

    n_hidden: int = 256*4
    momentum_coef: float = 0.99
    lr: float = 1e-4
    max_train_tokens: int = 100000000
    max_val_tokens: int = 1000000000
    
    # Optimizer parameters
    beta: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    ns_iters: int = 5
    eps: float = 1e-8
    
    # Data path (relative to project root unless absolute)
    tokenized_path: str = 'datasets/tokenized_base'
    batched_tokenized_path: str = 'datasets/tokenized_batched'
    use_attention_mask: bool = False
    wandb_project_name: str = 'tiny-stories-base'
    hf_dataset_names: str = 'roneneldan/TinyStories'
    do_packing: bool = False
    num_proc: int = 4
    pack_num_proc: int = 8
    pack_map_batch_size: int = 400
    pack_writer_batch_size = 4000

    # Tokenization workers (used in tokenization.py)
    num_proc: int = 8

    # Optimizer selection for functional trainer
    primary_optimizer: str = 'adamuon'  # one of: 'adamuon', 'muon', 'adamw'

    tokenizer_name: str = 'EleutherAI/gpt-neo-125M'
    tokenizer_type: str = 'hf'

    # Training memory controls
    logits_chunk_size: int = 256  # set 0 to disable chunked loss

    def __post_init__(self):
        # Compute project root from this file location
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if not os.path.isabs(self.tokenized_path):
            self.tokenized_path = os.path.join(project_root, self.tokenized_path)
        if not os.path.isabs(self.batched_tokenized_path):
            self.batched_tokenized_path = os.path.join(project_root, self.batched_tokenized_path)
