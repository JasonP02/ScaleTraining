from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, GPT2Tokenizer, dynamic_rope_update
import torch
from config import Config
from functools import partial

def dynamic_collate_fn(batch, pad_token):
    max_len = max(len(item['input_ids']) for item in batch)

    padded_batch = []
    for item in batch:
        seq = item['input_ids']
        padding_needed = max_len - len(seq)
        if padding_needed > 0:
            padded_seq = seq + [pad_token] * padding_needed
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)
        
    return {'input_ids': torch.tensor(padded_batch)}
# Loading tinystories

def load_tokenized_dataset(cfg):
    """
    Load the tokenized dataset from the disk
    Args:
        cfg: Config object
    Returns:
        train_loader: DataLoader object for the train dataset
        val_loader: DataLoader object for the validation dataset
    """
    from datasets import load_from_disk
    
    # Load the tokenizer to get pad_token_id
    # TODO: Make tokenizer more generalized... cfg based.
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token_id
    
    # Load the saved datasets
    train_dataset = load_from_disk(f'{cfg.data_path}/train')
    val_dataset = load_from_disk(f'{cfg.data_path}/val')
    
    # Convert to PyTorch datasets
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Create collate function with the correct pad_token
    collate_fn = partial(dynamic_collate_fn, pad_token=pad_token)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def tokenize_dataset(dataset_name, tokenizer, save_path, tok_type='hf', max_length=256):
    """
    Tokenize the dataset and save it to the disk
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use
        save_path: Path to save the tokenized dataset
        tok_type: Type of tokenizer to use
    """
    # Load the full dataset (default split)
    dataset = load_dataset(dataset_name)
    
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
 
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length, padding=False)

    
    # Tokenize both train and validation splits directly
    tokenized_train_dataset = dataset['train'].map(tokenize_function, remove_columns=['text'], batched=True)
    tokenized_val_dataset = dataset['validation'].map(tokenize_function, remove_columns=['text'], batched=True)
    
    # Save train and validation separately
    tokenized_train_dataset.save_to_disk(f'{save_path}/train')
    tokenized_val_dataset.save_to_disk(f'{save_path}/val')


if __name__ == '__main__':
    cfg = Config()

    # Tokenize and split the dataset
    tokenize_dataset(
        dataset_name='roneneldan/TinyStories',
        tokenizer='EleutherAI/gpt-neo-125M',
        save_path=cfg.data_path,
        tok_type='hf'
    )
                     
    train_loader, val_loader = load_tokenized_dataset(cfg)
    print("Train batch example:")
    print(next(iter(train_loader)))
    print("\nValidation batch example:")
    print(next(iter(val_loader)))


