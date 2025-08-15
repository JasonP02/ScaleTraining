from main import AttentionBlock, Config
from dataload import load_tiny_stories
import torch



# Test passed: the dataset properly loads a text sample (print inspection)
def test_tiny_stories_loader(loader):
    for i, batch in enumerate(loader):
        print(f"Batch {i}")
        print(f"Sample: \n {batch['text'][0][:10]}")
        if i>3:
            break

def flash_attention_test(cfg):
    test_cases = [
        (2,10,cfg.n_embed),
        (1,1,cfg.n_embed),
        (500,500,cfg.n_embed)
    ]
    for batch, seq, embed in test_cases:
        x = torch.randn(batch, seq, embed)
        attention_block = AttentionBlock(cfg)
        out = attention_block.forward(x)
        assert x.shape == out.shape, "Output shape should match input"

# Load config
cfg = Config()

# Test the dataloader
train_loader, _ = load_tiny_stories(cfg)
test_tiny_stories_loader(train_loader)

# Test the flash attention block
flash_attention_test(cfg)