import torch
import gc
import os
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

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

def resolve_device(cfg) -> None:
    """Resolve cfg.device when set to 'auto'.

    Sets cfg.device to 'cuda' if a CUDA device is available, else 'cpu'.
    """
    try:
        if getattr(cfg, 'device', 'auto') == 'auto':
            import torch
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        cfg.device = 'cpu'


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
    except Exception:
        return {}


def save_run_manifest(cfg, out_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "time": datetime.utcnow().isoformat() + "Z",
        "dataset": _cfg_subset(cfg),
        "optimizer": {
            "primary": getattr(cfg, 'primary_optimizer', 'adamuon'),
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
        "dataset_tag": getattr(cfg, 'dataset_tag', ''),
        "fingerprint": config_fingerprint(cfg),
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(out_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def save_model(model, cfg, out_root: Optional[str] = None) -> str:
    out_root = out_root or getattr(cfg, 'output_dir', 'outputs')
    tag = _sanitize(getattr(cfg, 'dataset_tag', ''))
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    fp = config_fingerprint(cfg)[:8]
    run_dir_name = "__".join(filter(None, [tag, f"v={fp}", ts]))
    run_dir = os.path.join(out_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(run_dir, "model.pt")
    try:
        import torch
        torch.save({
            "state_dict": model.state_dict(),
            "config": {k: getattr(cfg, k) for k in vars(cfg).keys()} if hasattr(cfg, "__dict__") else {},
        }, model_path)
    except Exception as e:
        print(f"Warning: model save failed: {e}")

    # Save manifest
    save_run_manifest(cfg, run_dir)
    return run_dir


# ---- W&B helpers ----

def init_wandb(cfg: Any, config_dict: Optional[Dict[str, Any]] = None) -> None:
    """Initialize W&B with consistent metrics.

    Args:
        cfg: Hydra config or simple object with attributes used below.
        config_dict: Optional resolved config (Python dict) to store in W&B.
    """
    import wandb

    wandb.init(project=getattr(cfg, 'wandb_project_name', 'scaletraining'),
               entity=os.environ.get('WANDB_ENTITY', None),
               config=config_dict)
    try:
        wandb.define_metric("used tokens")
        wandb.define_metric("train_per_token_loss", step_metric="used tokens")
    except Exception:
        pass


def log_dataset_artifacts(tok_dir: str, pack_dir: str, cfg: Any) -> None:
    """Log tokenized/packed dataset directories as W&B Artifacts.

    Args:
        tok_dir: Filesystem path to tokenized dataset root (contains train/ and optional val/).
        pack_dir: Filesystem path to packed dataset root (contains train/ and optional val/).
        cfg: Config-like object; stored as artifact metadata.
    """
    import wandb
    meta = {k: getattr(cfg, k) for k in getattr(cfg, 'keys', lambda: [])()} if hasattr(cfg, 'keys') else vars(cfg) if hasattr(cfg, '__dict__') else {}
    art_tok = wandb.Artifact("tokenized", type="dataset", metadata=meta)
    art_tok.add_dir(tok_dir)
    wandb.log_artifact(art_tok)

    art_pack = wandb.Artifact("packed", type="dataset", metadata=meta)
    art_pack.add_dir(pack_dir)
    wandb.log_artifact(art_pack)


def log_model_artifact(model_path: str, cfg: Any) -> None:
    """Log a saved model checkpoint file as a W&B Artifact.

    Args:
        model_path: Filesystem path to the saved model file (e.g., model.pt).
        cfg: Config-like object; stored as artifact metadata.
    """
    import wandb
    meta = {k: getattr(cfg, k) for k in getattr(cfg, 'keys', lambda: [])()} if hasattr(cfg, 'keys') else vars(cfg) if hasattr(cfg, '__dict__') else {}
    art = wandb.Artifact("model", type="model", metadata=meta)
    art.add_file(model_path)
    wandb.log_artifact(art)
