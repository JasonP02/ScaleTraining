from transformers import AutoTokenizer, PreTrainedTokenizerFast
from pathlib import Path
from scaletraining.data_processing.dataset_utils import dataset_safe_name
from scaletraining.data_processing.train_tokenizer import train_tokenizer_from_cfg

class Tokenizer:
    def __init__(self, cfg):
        self.dataset_names: list[str] | str = cfg.tokenizer_dataset_names
        self.custom_tokenizer_vocab_size: int = cfg.tokenizer.custom_tokenizer_vocab_size

        if cfg.tokenizer.is_pretrained: 
            self.tok, self.vocab_size, self.eos_id, self.pad_token = self.get_pretrained_tokenizer(cfg.tokenizer.pretrained_tokenizer_name)
        else:
            self.tok, self.vocab_size = self.get_custom_tokenizer()

    @staticmethod
    def _get_pretrained_tokenizer(tok_name):
        # Use standard AutoTokenizer for HF models
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        if tok.eos_token_id is None:
            print(f"Warning, eos token does not exist, using '' as eos token")
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        vocab_size = len(tok) if hasattr(tok, "__len__") else tok.vocab_size
        return tok, vocab_size, tok.eos_id, tok.pad_token

    @staticmethod
    def get_tokenizer_name_from_dataset(
        dataset_specs,
        vocab_size: int | None = None,
        dataset_configs=None,
    ):
        """Generate tokenizer name based on dataset specifications.
        Args:
            dataset_specs: Single dataset name or list of dataset names
            vocab_size: Defined tokenizer vocab size from config
            dataset_configs: Configuration for huggingface dataset (optional)
        Returns:
            Path to the corresponding tokenizer file
        """
        # Handle single dataset or list
        specs = dataset_specs if isinstance(dataset_specs, list) else [dataset_specs]
        configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs] if dataset_configs else [None] * len(specs)
        if len(configs) == 1 and len(specs) > 1:
            configs = configs * len(specs)
        safe_name = dataset_safe_name(specs, configs)
        base_dir = Path.cwd() / "tokenizers"
        base_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_v{int(vocab_size)}" if vocab_size is not None else ""
        tokenizer_path = base_dir / f"tokenizer_{safe_name}{suffix}.json"
        return str(tokenizer_path)
        
    def get_custom_tokenizer(self):
        """Loads a pretrained tokenizer"""
        
        tok_name = self._get_custom_tokenizer_path(
            self.dataset_names,
            self.custom_tokenizer_vocab_size, "custom_tokenizer_vocab_size"
        )
        # Check if it's a local .json tokenizer file
        if Path(tok_name).exists() and tok_name.endswith('.json'):
            pass
        else:
            train_tokenizer_from_cfg()

        
        print(f"Loading local tokenizer file: {tok_name}")
        # Load the tokenizer using tokenizers library
        tokenizer_obj = Tokenizer.from_file(tok_name)
        
        # Wrap it in a minimal AutoTokenizer-compatible object
        tok = LocalTokenizer(tokenizer_obj)
        vocab_size = len(tok) if hasattr(tok, "__len__") else tok.vocab_size


        return tok, vocab_size, tok.eos_token_id, tok.pad_token_id

class LocalTokenizer:
    def __init__(self, tokenizer_obj):
        self._tokenizer = tokenizer_obj
        self.vocab_size = tokenizer_obj.get_vocab_size()
        # Set EOS token (assume it's one of the special tokens)
        vocab = tokenizer_obj.get_vocab()
        self.eos_token_id = vocab.get("[SEP]", vocab.get("</s>", vocab.get("<|endoftext|>", 2)))
        self.pad_token_id = vocab.get("[PAD]", self.eos_token_id)
        
    def __call__(self, text, **kwargs):
        # Handle batch tokenization
        if isinstance(text, list):
            results = [self._tokenizer.encode(t) for t in text]
            input_ids = [r.ids for r in results]
        else:
            result = self._tokenizer.encode(text)
            input_ids = result.ids

        return {"input_ids": input_ids}
