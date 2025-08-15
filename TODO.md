# TODO

## Phase 1: Small transformer
- [ ]Load tiny-stories dataset
- [ ]Configure a basic encoder only transformer (d_model: 512, nlayers: 3, nhead: 4, d_hidden: 512*4)
- [ ]Apply RoPE embeddings
- [ ]Choose a pretrained tokenizer.
- [ ]Train tokenizer (optional).
- [ ]Set up training loop (muon or similar)
    - [ ]Research optimizations that can be done in the training loop
- [ ]Integrate detailed W&B logging for analysis and sweeps
    - [ ]Use loss over total tokens during training as primary metric
- [ ]Integrate basic evaluation pipeline for larger models
- [ ]Training
    - [ ]

## Phase 2: Improving small transformer
- [ ]Add top-2 MoE in MLP block (3 experts total) variant
- [ ]Ensure easy argparse 'swap' between model variants
- [ ]Add (second) math or similar logical dataset
- [ ]Train, and inspect proper allocation of experts
- [ ]Perform ablations of experts to visualize performance
- [ ]Optional: Do an interpretability analysis of the experts in comparison to a base model

## Phase x: Scaling to more data
