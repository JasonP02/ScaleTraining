from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from omegaconf import DictConfig
import hydra
from datasets import load_dataset
from pathlib import Path
import os

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
    dataset_specs = cfg.hf_dataset_names if isinstance(cfg.hf_dataset_names, list) else [cfg.hf_dataset_names]
    
    text_files = []
    
    for spec in dataset_specs:
        # Create a safe filename from the dataset spec
        safe_name = spec.replace("/", "_").replace("-", "_")
        text_file = data_dir / f"{safe_name}.txt"
        
        if text_file.exists():
            print(f"Found existing text file: {text_file}")
            text_files.append(str(text_file))
        else:
            print(f"Downloading and converting dataset: {spec}")
            try:
                # Load the dataset
                ds = load_dataset(spec)
                
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
                                if isinstance(example[col], str) and len(example[col]) > 10:
                                    text_col = col
                                    break
                            if text_col:
                                f.write(example[text_col] + "\n")
                
                print(f"Created text file: {text_file}")
                text_files.append(str(text_file))
                
            except Exception as e:
                print(f"Error processing dataset {spec}: {e}")
                raise
    
    return text_files

from scaletraining.util import flatten_cfg

def train_tokenizer_from_cfg(cfg: DictConfig) -> str:
    """Train a dataset-specific tokenizer in-process and return its save path.

    This function avoids spawning a subprocess and prevents Hydra re-initialization.
    """
    flat = flatten_cfg(cfg)
    # Get text files for the dataset(s)
    files = get_dataset_text_files(flat)
    if not files:
        raise ValueError("No text files found or created for training")

    # Create tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=flat.tokenizer_vocab_size,
        show_progress=True,
        max_token_length=10,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()

    print(f"Training tokenizer on {len(files)} file(s): {files}")
    tokenizer.train(files, trainer)

    # Generate dataset-based save path
    dataset_specs = flat.hf_dataset_names if isinstance(flat.hf_dataset_names, list) else [flat.hf_dataset_names]
    dataset_name = "_".join([spec.replace("/", "_").replace("-", "_") for spec in dataset_specs])
    # Save under project-local tokenizers/
    out_dir = Path.cwd() / "tokenizers"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = int(getattr(flat, "tokenizer_vocab_size", 0) or 0)
    suffix = f"_v{vocab_size}" if vocab_size > 0 else ""
    save_path = out_dir / f"tokenizer_{dataset_name}{suffix}.json"
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to: {save_path}")
    return str(save_path)

@hydra.main(version_base=None, config_path='../../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    """Hydra console script entrypoint for tokenization."""
    train_tokenizer_from_cfg(cfg)

if __name__ == "__main__":
    main()
