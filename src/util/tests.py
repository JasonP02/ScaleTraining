from model import AttentionBlock
from dataload import load_tokenized_dataset
from config import Config
import torch



# Test passed: the dataset properly loads a text sample (print inspection)
def test_tiny_stories_loader(cfg):
    train_loader, val_loader = load_tokenized_dataset(cfg)
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}")
        print(f"Sample: \n {batch['input_ids'][0][:10]}")
        if i>3:
            break

def rope_and_flash_attention_test(cfg):
    test_cases = [
        (2,10,cfg.n_embed),
        (1,1,cfg.n_embed),
        (500,500,cfg.n_embed)
    ]
    for batch, seq, embed in test_cases:
        x = torch.randn(batch, seq, embed)

        print(f"x input shape {x.shape}")
        attention_block = AttentionBlock(cfg)
        out = attention_block.forward(x)
        assert x.shape == out.shape, "Output shape should match input"

def run_all_tests():
    # Load config
    cfg = Config()

    # Test the dataloader
    print("== Testing tinystories loader ==")
    test_tiny_stories_loader(cfg)

    print(f'\n == Testing Attention Block ==')
    try:
        rope_and_flash_attention_test(cfg)
        print("Attention test passed")
    except Exception as e:
        print(f"Attention test failed: {e}")
        return False

    

if __name__ == "__main__":
    if run_all_tests():
        print("All tests passed!")
    else:
        print("Some tests failed")