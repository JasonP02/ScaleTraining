from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, GPT2Tokenizer
import torch
from config import cfg

# Loading tinystories
def load_tokenized_dataset(cfg):
    train_loader = DataLoader(torch.load(f'{cfg.data_path}/train.pt'), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(torch.load(f'{cfg.data_path}/val.pt'), batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def tokenize_dataset(split_name, dataset_name, tokenizer, save_path, tok_type='hf'):
    dataset = load_dataset(dataset_name, split_name) # Load data
    if tok_type == 'hf':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    elif tok_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
    else:
        raise ValueError(f"Tokenizer type {tok_type} not supported")
 
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=['text'], batched=True)
    tokenized_dataset.save_to_disk(f'{save_path}/{split_name}.pt')

    return tokenized_dataset

if __name__ == '__main__':
    tokenize_dataset('train', 'roneneldan/TinyStories', 'gpt2', 'datasets', 'gpt2')
    tokenize_dataset('validation', 'roneneldan/TinyStories', 'gpt2', 'datasets', 'gpt2')
    train_loader, val_loader = load_tokenized_dataset(cfg)
    print(train_loader[0])
    print(val_loader[0])


