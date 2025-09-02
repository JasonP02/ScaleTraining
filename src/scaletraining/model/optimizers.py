import torch
from typing import Optional, Callable


def newton_shulz(momentum, steps=5, eps=1e-8) -> torch.Tensor:
    """
    Newton Shulz iteration for creating orthogonal matrix; default 5 steps
    params:
        momentum: the momentum matrix of size 2
        steps: number of convergence steps
        eps: epsilon value for non-zero division
    outputs:
        X: approximate orthogonalized matrix of the momentum for muon loss function
    """
    assert momentum.ndim == 2, f"Input momentum matrix size (and the weight matrix) is not 2"
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Initialize from the tensor itself (not sign). Use bf16 only on CUDA; keep f32 on CPU.
    if momentum.is_cuda:
        X = momentum.to(torch.bfloat16)
    else:
        X = momentum.to(torch.float32)
    # Ensure spectral norm is at most 1 (Frobenius-based bound is sufficient)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    if momentum.size(-2) > momentum.size(-1):
        X = X.mT

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if momentum.size(-2) > momentum.size(-1):
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, beta: float = 0.9, beta2: float = 0.999,
                 weight_decay: float = 0.0, ns_iters: int = 5, eps: float = 1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, ns_iters=ns_iters, beta=beta, beta2=beta2)
        super().__init__(params, defaults)

    def update_weights(self, param, NS):
        """
        Update the weights of the parameters
        Args:
            param: Parameter to update
            NS: Newton Shulz matrix
        """
        n, m = param.shape
        gamma = 0.2 * (max(n, m)) ** 0.5
        return NS * gamma

    def second_moment_update(self, param, NS, beta, beta2, eps) -> torch.Tensor:
        """Identity function for Muon - no second moment update"""
        return NS

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = group['beta']; beta2 = group['beta2']; lr = group['lr']; eps = group['eps']; wd = group['weight_decay']; steps = group['ns_iters']
            for param in group['params']:
                state = self.state[param]
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                param_dtype = param.dtype

                momentum = state.get('momentum')
                if momentum is None:
                    state['momentum'] = momentum = torch.zeros_like(param, 
                    dtype=grad.dtype, device=param.device)
                momentum.mul_(beta).add_(grad)
                    
                # Apply NS to momentum
                NS = newton_shulz(momentum, eps=eps, steps=steps) # Apply N-S for 5 iterations for weight update

                # Apply second moment update (identity for Muon)
                NS = self.second_moment_update(param, NS, beta, beta2, eps)

                if wd:
                    param.add_(param, alpha=-lr*wd)
                
                # Weight update is specific to muon / adamuon
                direction = self.update_weights(param, NS)
                direction = direction.to(param_dtype)
                param.add_(direction, alpha=-lr) # Update the weights by subtracting the gradient + muon params 

                state['step'] = state.get('step', 0) + 1
        
        return loss if loss is not None else 0.0  # type: ignore


class AdaMuon(torch.optim.Optimizer):
    def __init__(self, params, lr, beta: float = 0.9, beta2: float = 0.999,
                 weight_decay: float = 0.0, ns_iters: int = 5, eps: float = 1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, ns_iters=ns_iters, beta=beta, beta2=beta2)
        super().__init__(params, defaults)

    def second_moment_update(self, param, NS, beta, beta2, eps) -> torch.Tensor:
        '''
        Second moment update using exponential moving average (v)
        params:
            newton-shulz matrix of the momentum
        outputs:
            variance normalized version of n-s matrix
        '''
        state = self.state[param]
        v = state.get('v')
        if v is None:
            v = torch.zeros_like(param, dtype=NS.dtype, device=NS.device)
            state['v'] = v

        v = v.mul_(beta2).addcmul_(NS, NS, value=(1.0-beta2))

        NS = NS / (v.sqrt().add_(eps))

        state['v'] = v
        return NS

    def update_weights(self, param, NS):
        """
        Update the weights of the parameters
        Args:
            param: Parameter to update
            NS: Newton Shulz matrix
        """
        n, m = param.shape
        denom = NS.norm(p='fro').clamp_min(1e-12)
        gamma = 0.2 * (n*m) ** 0.5 / denom
        return NS * gamma

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = group['beta']; beta2 = group['beta2']; lr = group['lr']; eps = group['eps']; wd = group['weight_decay']; steps = group['ns_iters']
            for param in group['params']:
                state = self.state[param]
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                param_dtype = param.dtype

                momentum = state.get('momentum')
                if momentum is None:
                    state['momentum'] = momentum = torch.zeros_like(param, 
                    dtype=grad.dtype, device=param.device)
                momentum.mul_(beta).add_(grad)
                    
                # Apply NS to momentum
                NS = newton_shulz(momentum, eps=eps, steps=steps) # Apply N-S for 5 iterations for weight update

                # Apply second moment update (variance normalization for AdaMuon)
                NS = self.second_moment_update(param, NS, beta, beta2, eps)

                if wd:
                    param.add_(param, alpha=-lr*wd)
                
                # Weight update is specific to muon / adamuon
                direction = self.update_weights(param, NS)
                direction = direction.to(param_dtype)
                param.add_(direction, alpha=-lr) # Update the weights by subtracting the gradient + muon params 

                state['step'] = state.get('step', 0) + 1
        
        return loss if loss is not None else 0.0  # type: ignore