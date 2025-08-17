# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands
- **Run tests**: `python tests.py` - Executes comprehensive test suite for dataload and attention components
- **Test dataloader**: `python -c "from dataload import load_tiny_stories, test_tiny_stories_loader; test_tiny_stories_loader(load_tiny_stories(Config())[0])"`
- **Run attention tests**: `python -c "from tests import flash_attention_test; flash_attention_test(Config())"`

## Project Overview
ScaleTrain is a project focused on scaling up LLM training, starting with a 10M transformer baseline trained on tinystories and progressing to a 7B model. The project emphasizes einsum operations and follows a phased approach to implementation.

## Project Architecture
- **Architecture**: Decoder-only transformer 
- **Baseline config**: d_model: 512, nlayers: 3, nhead: 4, d_hidden: 512*4
- **Key technologies**: RoPE embeddings, MoE (Mixture of Experts) variants, muon optimization
- **Primary dataset**: tiny-stories
- **Secondary dataset**: math/logical reasoning datasets (Phase 2)

## Key Technical Components
- **Token embedding**: `nn.Embedding(cfg.vocab_size, cfg.n_embed)` with linear projections `W_e` and `W_ue`
- **Flash attention**: Uses `torch.nn.functional.scaled_dot_product_attention` with causal masking
- **MLP block**: Two linear layers with ReLU activation and dropout
- **Transformer block**: Standard pre-norm architecture with attention and MLP components
- **RoPE integration**: Fully implemented with vectorized operations using stack/reshape pattern for efficient rotation
- **Dataloading**: Uses HuggingFace `datasets` library with TinyStories dataset

## Development Workflow
- Implementation follows a phased approach:
  - Phase 1: Small transformer baseline with RoPE and W&B logging
  - Phase 2: MoE improvements and ablation studies
  - Phase X: Scaling to more data and model size
- Test-driven development with comprehensive test coverage in `tests.py`

## TODO Management
- All work should be centered around TODOS in the TODO.md file
- TODO.md entries should be high-level, big picture items rather than specific bug fixes
- Use minimal, 1-line entries for upcoming work items
- Focus on phase-based development milestones rather than granular tasks

## Dependencies
- **Core**: PyTorch, HuggingFace datasets
- **Architecture**: No configuration files found - dependencies managed by environment
- **Testing**: Uses PyTorch for tensor operations and assertions