"""Utilities for assembling mixed pretraining corpora."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None  # type: ignore

from datasets import IterableDataset, load_dataset
from transformers import AutoTokenizer

from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.data_processing.tokenization import get_tokenizer
from scaletraining.util import _cfg_subset, get_packed_directory, read_metadata, get_tokenized_directory, write_metadata


TOKENS_PER_GB = 250_000_000

FilterFn = Callable[[dict], bool]
CleanerFn = Callable[[str], str]


def _default_clean(value: str) -> str:
    return value.strip()


def _concat_fields(example: dict, fields: Iterable[str], separator: str) -> str:
    parts: list[str] = []
    for field in fields:
        value = example.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for key in ("title", "body", "text"):
                item = value.get(key)
                if isinstance(item, str):
                    parts.append(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in ("text", "body", "content", "answer"):
                        maybe = item.get(key)
                        if isinstance(maybe, str):
                            parts.append(maybe)
    return separator.join(part for part in parts if part)


@dataclass
class SourceSpec:
    """Configuration describing a single source dataset."""

    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    text_fields: tuple[str, ...] = ("text",)
    separator: str = "\n"
    target_gb: float = 1.0
    tokens_per_gb: int = TOKENS_PER_GB
    min_chars: int = 200
    shuffle_buffer: int = 10_000
    filter_fn: Optional[FilterFn] = None
    cleaner: CleanerFn = _default_clean
    description: str = ""
    keep_probability: float = 1.0
    use_tokenizer_for_count: bool = False
    tokens_per_char: float = 0.25

    def target_tokens(self) -> int:
        return int(self.target_gb * self.tokens_per_gb)

    def extract_text(self, example: dict) -> str:
        if self.text_fields == ("__concat__",):
            return _concat_fields(example, example.keys(), self.separator)
        return _concat_fields(example, self.text_fields, self.separator)


# Edit this list directly to control which datasets are streamed.
SOURCES: list[SourceSpec] = [
    SourceSpec(
        dataset="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        text_fields=("text",),
        target_gb=0.1,
        description="FineWeb EDU sample",
    ),
    SourceSpec(
        dataset="wikimedia/wikipedia",
        subset="20231101.en",
        text_fields=("text",),
        target_gb=0.1,
        description="English Wikipedia",
    ),
    SourceSpec(
        dataset="lvwerra/stack-exchange-paired",
        split="test",
        text_fields=("question", "response_j"),
        separator="\n\n",
        target_gb=0.1,
        description="StackExchange (multi-domain)",
    ),
    SourceSpec(
        dataset="gfissore/arxiv-abstracts-2021",
        subset=None,
        split="train",
        text_fields=("title", "abstract"),
        separator="\n\n",
        target_gb=0.1,
        description="arXiv titles + abstracts",
    ),
]


class JsonlTokenWriter:
    """Helper for accumulating raw text snippets."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.examples = 0
        self.tokens = 0

    def write(self, text: str, token_count: int) -> None:
        self._handle.write(json.dumps({"text": text}) + "\n")
        self.examples += 1
        self.tokens += token_count

    def close(self) -> None:
        self._handle.close()


def stream_source(
    spec: SourceSpec,
    tokenizer,
    train_writer: JsonlTokenWriter,
    val_writer: JsonlTokenWriter,
    val_ratio: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    total_tokens = 0
    target_tokens = spec.target_tokens()

    dataset_kwargs = {"streaming": True, "split": spec.split}
    if spec.subset is not None:
        dataset_kwargs["name"] = spec.subset

    stream = load_dataset(spec.dataset, **dataset_kwargs)
    if isinstance(stream, IterableDataset) and spec.shuffle_buffer:
        stream = stream.shuffle(seed=seed, buffer_size=spec.shuffle_buffer)

    train_before = train_writer.examples
    val_before = val_writer.examples

    iterator = stream
    progress_bar = None
    if tqdm is not None:
        desc = f"Streaming {spec.description or spec.dataset}"
        progress_bar = tqdm(iterator, desc=desc, unit="ex", dynamic_ncols=True)
        iterator = progress_bar

    for example in iterator:
        if spec.filter_fn and not spec.filter_fn(example):
            continue
        if rng.random() > spec.keep_probability:
            continue
        text = spec.cleaner(spec.extract_text(example))
        if not text or len(text) < spec.min_chars:
            continue
        if spec.use_tokenizer_for_count:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded.get("input_ids")
            if not ids:
                continue
            count = len(ids)
        else:
            count = max(1, int(len(text) * spec.tokens_per_char))
        total_tokens += count
        if rng.random() < val_ratio:
            val_writer.write(text, count)
        else:
            train_writer.write(text, count)
        if progress_bar is not None:
            progress_bar.set_postfix(tokens=f"{total_tokens/1e6:.1f}M", refresh=False)
        if total_tokens >= target_tokens:
            break

    if progress_bar is not None:
        progress_bar.close()

    return {
        "dataset": spec.dataset,
        "subset": spec.subset,
        "target_tokens": target_tokens,
        "collected_tokens": total_tokens,
        "train_examples": train_writer.examples - train_before,
        "val_examples": val_writer.examples - val_before,
        "token_count_method": "tokenizer" if spec.use_tokenizer_for_count else "estimate",
    }


def tokenize_and_pack(raw_dir: Path, flat_cfg, tokenizer_name: str, max_seq_len: int, num_proc: int) -> tuple[str, str]:
    data_files = {"train": str(raw_dir / "train.jsonl")}
    val_path = raw_dir / "val.jsonl"
    if val_path.exists() and val_path.stat().st_size > 0:
        data_files["validation"] = str(val_path)

    dataset = load_dataset("json", data_files=data_files)
    tokenizer, eos_id = get_tokenizer(tokenizer_name)

    tok_dir = Path(get_tokenized_directory(flat_cfg, for_training=True))
    pk_dir = Path(get_packed_directory(flat_cfg))
    tok_dir.mkdir(parents=True, exist_ok=True)
    pk_dir.mkdir(parents=True, exist_ok=True)

    def tokenize_batch(batch: dict) -> dict:
        output = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len - 1,
            padding=False,
        )
        input_ids = [ids + [eos_id] for ids in output["input_ids"]]
        return {"input_ids": input_ids}

    dataset["train"].map(
        tokenize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing train",
    ).save_to_disk(str(tok_dir / "train"))

    if "validation" in dataset:
        dataset["validation"].map(
            tokenize_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"],
            desc="Tokenizing val",
        ).save_to_disk(str(tok_dir / "val"))

    write_metadata(str(tok_dir), {"config": _cfg_subset(flat_cfg), "tokenizer_name": tokenizer_name})

    pack_and_save(
        tokenized_path=str(tok_dir),
        packed_path=str(pk_dir),
        block_size=int(flat_cfg.max_seq_len),
        num_proc=int(flat_cfg.pack_num_proc),
        map_batch_size=int(flat_cfg.pack_map_batch_size),
        writer_batch_size=int(flat_cfg.pack_writer_batch_size),
        metadata={"config": _cfg_subset(flat_cfg), "tokenizer_name": tokenizer_name},
    )

    return str(tok_dir), str(pk_dir)


def build_mixed_corpus(
    flat_cfg,
    dataset_id: str,
    tokenizer_name: str,
    max_seq_len: int,
    output_root: Path,
    val_ratio: float,
    num_proc: int,
    seed: int,
    reuse_raw: bool,
) -> tuple[str, str, list[dict]]:
    raw_dir = output_root / dataset_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []

    if not reuse_raw:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        tokenizer.model_max_length = 1_000_000_000
        try:
            tokenizer.init_kwargs["model_max_length"] = 1_000_000_000
        except Exception:
            pass

        train_writer = JsonlTokenWriter(raw_dir / "train.jsonl")
        val_writer = JsonlTokenWriter(raw_dir / "val.jsonl")

        for spec in SOURCES:
            print(f"Streaming {spec.description or spec.dataset} -> ~{spec.target_tokens():,} tokens")
            summary = stream_source(
                spec,
                tokenizer,
                train_writer,
                val_writer,
                val_ratio=val_ratio,
                seed=seed,
            )
            summaries.append(summary)
            print(
                f"  collected {summary['collected_tokens']:,} tokens"
                f" across {summary['train_examples']} train / {summary['val_examples']} val examples"
            )

        train_writer.close()
        val_writer.close()
    else:
        print("Reusing existing raw jsonl files; skipping streaming.")

    tok_dir, pk_dir = tokenize_and_pack(
        raw_dir=raw_dir,
        flat_cfg=flat_cfg,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        num_proc=num_proc,
    )

    meta = read_metadata(pk_dir) or {}
    meta.update(
        {
            "config": _cfg_subset(flat_cfg),
            "sources": summaries,
            "tokenizer_name": tokenizer_name,
            "max_seq_len": max_seq_len,
        }
    )
    write_metadata(pk_dir, meta)

    return tok_dir, pk_dir, summaries


__all__ = [
    "SourceSpec",
    "TOKENS_PER_GB",
    "SOURCES",
    "build_mixed_corpus",
]
