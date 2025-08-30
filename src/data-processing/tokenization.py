
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, GPT2Tokenizer, dynamic_rope_update
import torch
from config import Config
from functools import partial



class Tokenization():
    def __init__(self, cfg):
        self.tokenizer, self.eos_token = self.get_tok_and_eos(cfg.tok_type)
        self.max_length = cfg.max_length
        self.save_path = cfg.data_path

        try:
            self.dataset = load_dataset(cfg.dataset_name)
        except Exception as e:
            print(f"Could not load dataset: {e}")

    def get_tok_and_eos(self, tok_type):
        if tok_type == 'hf':
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif tok_type == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
        elif tok_type == 'custom':
            pass
        else:
            raise ValueError(f"Tokenizer type {tok_type} not supported")

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


if __name__ == '__main__':
    cfg = Config()
    tokenizer = Tokenization(cfg)

    # Tokenize and split the dataset
    tokenizer.tokenize_dataset(
        dataset_name='roneneldan/TinyStories',
        tokenizer='EleutherAI/gpt-neo-125M',
        save_path=cfg.data_path,
        tok_type='hf'
    )

