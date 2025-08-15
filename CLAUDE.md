# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
ScaleTrain is a project focused on scaling up LLM training, starting with a 10M transformer baseline trained on tinystories and progressing to a 7B model. The project emphasizes einsum operations and follows a phased approach to implementation.

## Project Architecture
- **Architecture**: Decoder-only transformer 
- **Baseline config**: d_model: 512, nlayers: 3, nhead: 4, d_hidden: 512*4
- **Key technologies**: RoPE embeddings, MoE (Mixture of Experts) variants, muon optimization
- **Primary dataset**: tiny-stories
- **Secondary dataset**: math/logical reasoning datasets (Phase 2)

## Development Workflow
- All work should be centered around TODOS in the TODO.md file
- Implementation follows a phased approach:
  - Phase 1: Small transformer baseline with RoPE and W&B logging
  - Phase 2: MoE improvements and ablation studies
  - Phase X: Scaling to more data and model size

## Key Technical Components
- **Tokenizer**: Pretrained tokenizer integration with optional training capability
- **Training loop**: Optimized with muon or similar techniques
- **Evaluation**: Basic evaluation pipeline for larger models
- **Logging**: Detailed W&B integration for analysis and sweeps
- **MoE implementation**: Top-2 MoE in MLP blocks with 3 experts total

## Model Variants
The project supports easy argparse swapping between:
- Base transformer model
- MoE-enhanced model variants
- Different dataset combinations

## Metrics and Monitoring
- Primary metric: Loss over total tokens during training
- W&B sweeps for hyperparameter optimization
- Expert allocation inspection and performance ablations