
from dataclasses import dataclass
from functools import partial
import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Tokenizer

from scaletraining.config import Config



class Tokenization():
    def __init__(self, cfg):
        self.tokenizer, self.eos_token = self.get_tok_and_eos(cfg.tokenizer_type, cfg.tokenizer)
        self.max_length = cfg.max_seq_len
        self.save_path = cfg.tokenized_path

        try:
            self.dataset = load_dataset(cfg.hf_dataset_names)
        except Exception as e:
            print(f"Could not load dataset: {e}")

    def get_tok_and_eos(self, tok_type, tok_name):
        if tok_type == 'hf':
            tokenizer = AutoTokenizer.from_pretrained(tok_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif tok_type == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained(tok_name)
        elif tok_type == 'custom':
            raise NotImplementedError("Custom tokenizer type not implemented")
        else:
            raise ValueError(f"Tokenizer type {tok_type} not supported")
        return tokenizer, tokenizer.eos_token

    def tokenize_dataset(self):
        """
        Take in huggingface dataset, and tokenize it with a chosen tokenizer. 
        This is currently not generalizable to datasets with different columns.

        Args:
            dataset_name: Name of the dataset to load
            tokenizer: Tokenizer to use
            save_path: Path to save the tokenized dataset
            tok_type: Type of tokenizer to use
        """
        def tokenize_function(examples):
            """
            A tokenization function which is used to map inputs -> outputs based on desired parameters
            
            """
            return self.tokenizer(examples['text'], truncation=True, max_length=self.max_length, padding=False)

        
        # Tokenize both train and validation splits directly
        tokenized_train_dataset = self.dataset['train'].map(tokenize_function, remove_columns=['text'], batched=True)
        tokenized_val_dataset = self.dataset['validation'].map(tokenize_function, remove_columns=['text'], batched=True)
        
        # Save train and validation separately
        tokenized_train_dataset.save_to_disk(f'{self.save_path}/train')
        tokenized_val_dataset.save_to_disk(f'{self.save_path}/val')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="HF dataset name to override config")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer name to override config")
    parser.add_argument("--out", default=None, help="Output dir to override config")
    args = parser.parse_args()

    cfg = Config()
    if args.dataset: cfg.hf_dataset_names = args.dataset
    if args.tokenizer: cfg.tokenizer = args.tokenizer
    if args.out: cfg.tokenized_path = args.out

    tokenizer = Tokenization(cfg)
    tokenizer.tokenize_dataset()

if __name__ == '__main__':
    main()
