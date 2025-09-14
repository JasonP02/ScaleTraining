import torch
import gc
import os
import json
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def configure_rocm_and_sdp(cfg):
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    # Optional SDP toggles; only apply when present in config
    if hasattr(cfg, 'use_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(bool(cfg.use_flash_sdp))
    if hasattr(cfg, 'use_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(bool(cfg.use_mem_efficient_sdp))
    if hasattr(cfg, 'use_math_sdp'):
        torch.backends.cuda.enable_math_sdp(bool(cfg.use_math_sdp))

def resolve_device(cfg) -> None:
    """Resolve cfg.device when set to 'auto'."""
    if getattr(cfg, 'device', None) == 'auto':
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---- Dataset/versioning helpers ----

_FINGERPRINT_FIELDS = (
    "hf_dataset_names",
    "tokenizer_name",
    "max_seq_len",
    "use_attention_mask",
)

def _cfg_subset(cfg) -> Dict[str, Any]:
    out = {}
    for k in _FINGERPRINT_FIELDS:
        out[k] = getattr(cfg, k)
    return out

def config_fingerprint(cfg) -> str:
    payload = json.dumps(_cfg_subset(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _sanitize(s: str) -> str:
    return str(s).replace("/", "-").replace(" ", "_")

def tokenized_dir(cfg) -> str:
    fp = config_fingerprint(cfg)[:8]
    base = cfg.tokenized_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, 'dataset_tag', '') else ""
    name = f"{tag}ds={_sanitize(cfg.hf_dataset_names)}__tok={_sanitize(cfg.tokenizer_name)}__L={cfg.max_seq_len}__mask={int(cfg.use_attention_mask)}__v={fp}"
    return os.path.join(base, name)

def packed_dir(cfg) -> str:
    fp = config_fingerprint(cfg)[:8]
    base = cfg.batched_tokenized_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, 'dataset_tag', '') else ""
    name = f"{tag}ds={_sanitize(cfg.hf_dataset_names)}__tok={_sanitize(cfg.tokenizer_name)}__L={cfg.max_seq_len}__mask={int(cfg.use_attention_mask)}__v={fp}"
    return os.path.join(base, name)

def write_metadata(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"Warning: could not write metadata to {path}: {e}")

def read_metadata(path: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not read metadata, returning empty dictionary: {e}")
        return {}


def save_run_manifest(cfg, out_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "time": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "dataset": _cfg_subset(cfg),
        "optimizer": {
            "primary": cfg.primary_optimizer,
            "lr": cfg.lr,
            "beta": cfg.beta,
            "beta2": cfg.beta2,
            "weight_decay": cfg.weight_decay,
            "ns_iters": cfg.ns_iters,
            "eps": cfg.eps,
        },
        "training": {
            "batch_size": cfg.batch_size,
            "accum_steps": cfg.accum_steps,
            "effective_batch_size": cfg.batch_size * cfg.accum_steps,
            "grad_clip_norm": cfg.grad_clip_norm,
            "logits_chunk_size": cfg.logits_chunk_size,
            "device": cfg.device,
        },
        "model": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embed": cfg.n_embed,
            "n_hidden": cfg.n_hidden,
            "vocab_size": cfg.vocab_size,
            "UE_bias": cfg.UE_bias,
            "use_checkpoint": cfg.use_checkpoint,
        },
        "tokenizer": {
            "tokenizer_name": cfg.tokenizer_name,
            "tokenizer_type": cfg.tokenizer_type,
        },
        "dataset_tag": cfg.dataset_tag,
        "fingerprint": config_fingerprint(cfg),
    }
    
    # Add implementation details
    manifest['implementation'] = {
        'optimizer': 'baseline_adam' if cfg.use_baseline_adam else cfg.primary_optimizer,
        'rope': {
            'implementation': cfg.rope_implementation,
            'theta': cfg.rope_config.get('theta', 10000),
        }
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(out_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def save_model(model, cfg, out_root: Optional[str] = None) -> str:
    out_root = out_root or cfg.output_dir
    tag = _sanitize(cfg.dataset_tag)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    fp = config_fingerprint(cfg)[:8]
    run_dir_name = "__".join(filter(None, [tag, f"v={fp}", ts]))
    run_dir = os.path.join(out_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(run_dir, "model.pt")
    import torch
    torch.save({
        "state_dict": model.state_dict(),
    }, model_path)

    # Save manifest
    save_run_manifest(cfg, run_dir)
    return run_dir


# ---- W&B helpers ----

def init_wandb(cfg: Any, config_dict: Optional[Dict[str, Any]] = None) -> None:
    """Minimal W&B init with UX improvements for experiment tracking."""
    import wandb
    from pathlib import Path
    
    # Extract tokenizer name for better run naming
    tokenizer_name = getattr(cfg, 'tokenizer_name', 'unknown')
    is_custom = Path(tokenizer_name).exists() and tokenizer_name.endswith('.json')
    
    # Create descriptive run name and tags
    if is_custom:
        if "roneneldan_TinyStories" in tokenizer_name:
            name_suffix = "custom_tinystories"
        else:
            name_suffix = "custom"
        tags = ["custom_tokenizer"]
    else:
        if "gpt-neo" in tokenizer_name:
            name_suffix = "gpt_neo"
        else:
            name_suffix = tokenizer_name.split("/")[-1] if "/" in tokenizer_name else tokenizer_name
        tags = ["hf_tokenizer"]
    
    wandb.init(
        project=cfg.wandb_project_name, 
        config=config_dict, 
        reinit=True,
        name=f"sweep_{name_suffix}",
        tags=tags
    )

# ---- Config helpers ----
def flatten_cfg(cfg: Any) -> Any:
    """Flatten namespaced Hydra config groups (model, tokenizer, logging) into a flat object.

    Returns an attribute-accessible object (SimpleNamespace) with merged keys.
    """
    from types import SimpleNamespace
    try:
        from omegaconf import OmegaConf
        to_dict = lambda x: (OmegaConf.to_container(x, resolve=True) if x is not None else {})
    except Exception:
        to_dict = lambda x: dict(x) if x is not None else {}

    merged: Dict[str, Any] = {}
    for group in ("model", "tokenizer", "logging"):
        try:
            sub = cfg.get(group) if hasattr(cfg, 'get') else getattr(cfg, group, None)
        except Exception:
            sub = getattr(cfg, group, None)
        if sub is not None:
            d = to_dict(sub)
            if isinstance(d, dict):
                merged.update(d)
    return SimpleNamespace(**merged)


def log_dataset_artifacts(tok_dir: str, pack_dir: str, cfg: Any) -> None:
    """Disabled - no longer logging dataset artifacts to wandb."""
    pass


def log_model_artifact(model_path: str, cfg: Any) -> None:
    """Disabled - no longer logging model artifacts to wandb."""
    pass
