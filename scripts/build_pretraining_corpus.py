"""Thin wrapper around `build_mixed_corpus` for ad-hoc runs from the repo root."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hydra import compose, initialize

from scaletraining.data_processing.corpus_builder import build_mixed_corpus
from scaletraining.util import flatten_cfg


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Assemble and cache a mixed pretraining corpus.")
    parser.add_argument("--dataset-id", help="Identifier to store under hf_dataset_names (defaults to config).")
    parser.add_argument("--tokenizer", help="Tokenizer name/path (defaults to config tokenizer_name).")
    parser.add_argument("--max-seq-len", type=int, help="Packing sequence length (defaults to config max_seq_len).")
    parser.add_argument("--output-root", type=Path, default=Path("data/pretrain/mixed"))
    parser.add_argument("--dataset-tag", help="Optional dataset_tag override.")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--summaries-path", type=Path, help="Optional JSON file capturing per-source stats.")
    parser.add_argument("--hf-token", help="Hugging Face access token for gated models/datasets.")
    parser.add_argument("--reuse-raw", action="store_true", help="Skip re-streaming if raw jsonl exists.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")

    flat = flatten_cfg(cfg)

    if args.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    dataset_id = args.dataset_id
    cfg_dataset = getattr(flat, "hf_dataset_names", None)
    if dataset_id is None:
        if isinstance(cfg_dataset, list):
            if len(cfg_dataset) == 1:
                dataset_id = cfg_dataset[0]
            else:
                parser.error("Config hf_dataset_names is a list; supply --dataset-id.")
        else:
            dataset_id = cfg_dataset
    if not dataset_id:
        parser.error("Unable to infer dataset id; pass --dataset-id.")

    tokenizer_name = args.tokenizer or getattr(flat, "tokenizer_name", None)
    if not tokenizer_name:
        parser.error("Tokenizer not provided and config has no tokenizer_name.")

    max_seq_len = args.max_seq_len or getattr(flat, "max_seq_len", None)
    if not max_seq_len:
        parser.error("Max seq len missing; supply --max-seq-len or set in config.")
    max_seq_len = int(max_seq_len)

    flat.hf_dataset_names = dataset_id
    flat.dataset_tag = args.dataset_tag or getattr(flat, "dataset_tag", "") or dataset_id
    flat.tokenizer_name = tokenizer_name
    flat.max_seq_len = max_seq_len

    tok_dir, pk_dir, summaries = build_mixed_corpus(
        flat_cfg=flat,
        dataset_id=dataset_id,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        num_proc=args.num_proc,
        seed=args.seed,
        reuse_raw=args.reuse_raw,
    )

    if args.summaries_path:
        args.summaries_path.parent.mkdir(parents=True, exist_ok=True)
        args.summaries_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))

    print("\nDone!")
    print(f"Tokenized shards: {tok_dir}")
    print(f"Packed shards:    {pk_dir}")
    print(f"Use hf_dataset_names: {dataset_id}")


if __name__ == "__main__":
    main()
