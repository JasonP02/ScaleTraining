"""Utilities for working with HuggingFace dataset specifications."""
from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

from omegaconf import DictConfig
from datasets import DatasetDict, concatenate_datasets, load_dataset
from pathlib import Path

def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def get_dataset_text_files(cfg: DictConfig) -> list[str]:
    """Get or create text files for the specified HF dataset(s).
    
    Args:
        cfg: Hydra config with hf_dataset_names
        
    Returns:
        List of paths to text files for training
    """
    # Prefer configured path; fallback to project-local data/train/raw
    data_dir = Path(getattr(cfg, "tokenizer_train_data", "data/train/raw"))
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle single dataset or list
    dataset_specs = normalize_dataset_specs(
        names=cfg.tokenizer.hf_dataset_names,
        configs=cfg.tokenizer.dataset_tag
    )

    text_files = []

    for dataset_path, config_name in dataset_specs:
        label = dataset_path if not config_name else f"{dataset_path}:{config_name}"
        print(label)
        # Create a safe filename from the dataset spec
        safe_name = dataset_safe_name(dataset_path, config_name)
        text_file = data_dir / f"{safe_name}.txt"
        
        if text_file.exists():
            print(f"Found existing text file: {text_file}")
            text_files.append(str(text_file))
        else:
            print(f"Downloading and converting dataset: {label}")
            try:
                # Load the dataset
                ds = load_hf_dataset(dataset_path, config_name)
                
                # Get the training split (or first available split)
                split_name = "train" if "train" in ds else list(ds.keys())[0]
                dataset = ds[split_name]
                
                # Extract text and write to file
                with open(text_file, 'w', encoding='utf-8') as f:
                    for example in dataset:
                        if "text" in example:
                            f.write(example["text"] + "\n")
                        else:
                            # Handle different column names
                            text_col = None
                            for col in example.keys():
                                if isinstance(example[col], str) and len(example[col]) > 100:
                                    text_col = col
                                    break
                            if text_col:
                                f.write(example[text_col] + "\n")
                
                print(f"Created text file: {text_file}")
                text_files.append(str(text_file))
                
            except Exception as e:
                print(f"Error processing dataset {label}: {e}")
                raise
    
    return text_files

def normalize_dataset_specs(names: Any, configs: Any | None = None) -> List[Tuple[str, str | None]]:
    """Return a list of (dataset_path, config_name) pairs."""

    name_list = _ensure_list(names)
    if not name_list:
        raise ValueError("At least one dataset name must be provided")

    config_list = _ensure_list(configs)
    if not config_list:
        config_list = [None] * len(name_list)
    elif len(config_list) == 1 and len(name_list) > 1:
        config_list = config_list * len(name_list)
    elif len(config_list) != len(name_list):
        raise ValueError("hf_dataset_config_name must align with hf_dataset_names")

    return [
        (str(dataset_path), str(config_name) if config_name not in (None, "") else None)
        for dataset_path, config_name in zip(name_list, config_list)
    ]


def dataset_label(names: Any, configs: Any | None = None) -> str:
    """Return a canonical label for the dataset+config selection."""

    parts = []
    for dataset_path, config_name in normalize_dataset_specs(names, configs):
        if config_name:
            parts.append(f"{dataset_path}:{config_name}")
        else:
            parts.append(dataset_path)
    return ",".join(parts)


def dataset_safe_name(names: Any, configs: Any | None = None) -> str:
    """
    Return a filesystem-safe representation of the dataset selection.
    names: the name of the dataset page
        example: roneneldan/TinyStories
    configs: the (optional) configuration of the split
        example: 
    
    """
    label = dataset_label(names, configs)
    return (
        label.replace("/", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace(" ", "_")
    )


def load_hf_dataset(names: Any, configs: Any | None = None, **kwargs: Any) -> DatasetDict:
    """Load one or more HuggingFace datasets, concatenating splits when necessary."""

    specs = normalize_dataset_specs(names, configs)
    datasets = []
    for dataset_path, config_name in specs:
        load_kwargs = dict(kwargs)
        if config_name:
            load_kwargs["name"] = config_name
        else:
            load_kwargs.pop("name", None)
        datasets.append(load_dataset(dataset_path, **load_kwargs))

    if not datasets:
        raise ValueError("No datasets resolved from specification")

    if len(datasets) == 1:
        return datasets[0]

    # Concatenate splits across dataset dicts, keeping only common splits
    shared_splits = set(datasets[0].keys())
    for ds in datasets[1:]:
        shared_splits &= set(ds.keys())

    if not shared_splits:
        raise ValueError("Datasets do not share any common splits to concatenate")

    result = {}
    for split in shared_splits:
        pieces = [ds[split] for ds in datasets]
        if len(pieces) == 1:
            result[split] = pieces[0]
        else:
            result[split] = concatenate_datasets(pieces)

    return DatasetDict(result)


__all__ = [
    "normalize_dataset_specs",
    "dataset_label",
    "dataset_safe_name",
    "load_hf_dataset",
]
