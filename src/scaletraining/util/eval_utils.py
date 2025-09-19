import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scaletraining.model.model import TransformerNetwork
from scaletraining.util import find_latest_model_path

def load_pretrained_model_and_tokenizer(flat):
    model_path = getattr(flat, "model_path", None)
    if not model_path or str(model_path).lower() == "latest":
        # Auto-discover latest model under outputs
        output_root = getattr(flat, "output_dir", "outputs")
        auto_path = find_latest_model_path(output_root)
        if not auto_path:
            raise RuntimeError("No model_path provided and no latest model found under outputs/. Pass model_path=... or create outputs/<run>/model.pt.")
        print(f"[generate] Using latest model: {auto_path}")
        model_path = auto_path

    # Build model from config and load weights
    model = TransformerNetwork(flat).to(flat.device)
    ckpt = torch.load(model_path, map_location=flat.device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Normalize keys from compiled/DataParallel checkpoints if present
    def _strip_prefix(sd, prefix: str):
        if any(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd
    state_dict = _strip_prefix(state_dict, "_orig_mod.")
    state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict)
    model.eval()

    # Load tokenizer, supporting local JSON (dataset-specific) via PreTrainedTokenizerFast
    tok_path = flat.tokenizer_name
    from pathlib import Path as _P
    if isinstance(tok_path, str) and _P(tok_path).exists() and tok_path.endswith('.json'):
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    else:
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token