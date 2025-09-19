"""
File for evaluating trained models on various evals. 
Currently supported: Arc-easy, hellaswag, MMLU, GSM8K, WikiTest-103

Functionality: 
1. Load model
2. Load val dataset via hf 
3. Evaluate function
"""
import hydra
from omegaconf import DictConfig
from scaletraining.util import flatten_cfg, resolve_device
from scaletraining.data_processing import build_loaders




@hydra.main(version_base=None, config_path='/../../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    flat = flatten_cfg(cfg)
    resolve_device(flat)

    _, val_loader = build_loaders(flat)
    if val_loader is None:
        raise RuntimeError("No validation split found")
    
    model = Trans

if __name__ == "__main__":
    main()