As of now, the project is in a good spot. I can essentially train an arbitrary transformer on any dataset. 

# Stuff on my mind
- The custom optimizers are not working as intended. This should be fixed
- There is code that I don't really understand, which is an issue
- I really should setup a tokenizer training pipeline
- I have not yet implemeneted MoE
- I have no eval pipline at all (requires research)
- There is nothing for post-training RL
- - Should I even include this in the repo scope? 

In all, there is a major problem (optimizer issues), and a lot of future directions. More concretely:
- Create tokenizer trainer
- - Requires learning about tokenizers
- Implement MoE
- Implement model evals
- Consider how post-training fits in

On the optimizer issue, I don't know how to proceed. I could likely compare to an existing muon/adamuon implementation. They diverge similarly, likely an issue in the base optimizer class.

I'd like to write a simple MLP trainer on mnist digits for debugging the optimizer in this case.