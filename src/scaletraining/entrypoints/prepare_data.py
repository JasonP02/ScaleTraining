"""
Hydra entrypoint to fully prepare data: tokenize and pack.

Supports a single dataset or a list of datasets in cfg.hf_dataset_names.
Each dataset is processed into its own fingerprinted directories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import hydra
from omegaconf import DictConfig

from scaletraining.data_processing.tokenization import tokenize_dataset
from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.util import _cfg_subset, flatten_cfg, get_packed_directory, get_tokenized_directory, write_metadata


def _as_list(x: Any) -> list:
    return x if isinstance(x, list) else [x]


def _clone_flat(flat):
    from types import SimpleNamespace
    return SimpleNamespace(**vars(flat))


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Prepare datasets by tokenizing and packing them.

    Behavior:
      - If cfg.hf_dataset_names is a single spec, prepares that dataset.
      - If it's a list, prepares each dataset independently.
    """
    flat = flatten_cfg(cfg)
    specs: Iterable[Any] = _as_list(flat.hf_dataset_names)

    for spec in specs:
        sub = _clone_flat(flat)
        sub.hf_dataset_names = spec

        # Tokenize
        tokenize_dataset(sub)

        # Pack
        tok_dir = get_tokenized_directory(sub)
        pk_dir = get_packed_directory(sub)
        pack_and_save(
            tokenized_path=tok_dir,
            packed_path=pk_dir,
            block_size=int(sub.max_seq_len),
            num_proc=int(sub.pack_num_proc),
            map_batch_size=int(sub.pack_map_batch_size),
            writer_batch_size=int(sub.pack_writer_batch_size),
            metadata={"config": _cfg_subset(sub)},
        )

        # Also write a top-level metadata file for quick inspection (redundant but handy)
        try:
            write_metadata(pk_dir, {"config": _cfg_subset(sub)})
        except Exception:
            pass


if __name__ == "__main__":
    main()
