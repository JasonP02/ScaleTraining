# About
This project provides a harness for training a dense or MoE transformer on a single GPU. It is a personal project for experimenting with LLM training.

# Usage
The project uses hydra for configuration management

The configuration options include:

- Grid sweeping for testing different hyperparameter configurations:
- Logging for weights and biases, controlling various knobs for file naming across sweeps
- Generation for control of model temperature, top_k, etc. on a model checkpoint
- MOE for controlling the amount of expert layers, number of experts, and other knobs.
- Optimizer for controlling lr, optimizer type, scheduling, decay, etc.
- Train for bsz, kernel optimizations, train tokens, etc.
- Tokenizer for selecting a huggingface tokenizer, and whether to re-pack the data

## Entrypoints
``` train.py ```
Creates train_loader with build_loaders and trains

``` run_evals.py ``` 
Performs evaluations for a model determined by ```load_pretrained_model_and_tokenizer``` on supported benchmarks. Currently Arc-Easy is the only supported benchmark

```prepare_data.py```
A nice-to-have way of tokenizing and packing data without requiring a training run. Good for testing different datasets etc.

```generate_from_pretrained.py```
For evaluating model on a configurable prompt. Visual inspection of model outputs


## Functions 
build_loaders sees if the current 'tokenized_train_path' or 'tokenized_eval_path' exists based on the current: dataset, tokenizer, sequence length (all from config). We also look to see if there is 'packed' data for the current config. The key idea is that we check if we have previously packed and/or tokenized for the current data,tokenizer,seq_len to avoid recomputation.





