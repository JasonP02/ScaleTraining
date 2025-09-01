import torch
import torch.nn as nn
from scaletraining.model.optimizers import Muon, AdaMuon


def test_zero_grad_no_decay_no_update():
    torch.manual_seed(0)
    p0 = torch.randn(4, 8, dtype=torch.float64)
    p = p0.clone().detach().requires_grad_(True)

    opt = Muon([p], lr=1e-3, weight_decay=0.0)
    p.grad = torch.zeros_like(p)
    opt.step()

    assert torch.allclose(p.detach(), p0, atol=0, rtol=0)


def test_weight_decay_only_scales_params():
    torch.manual_seed(0)
    lr = 1e-2
    wd = 0.1

    p0 = torch.randn(4, 8, dtype=torch.float64)
    p = p0.clone().detach().requires_grad_(True)

    opt = Muon([p], lr=lr, weight_decay=wd)
    p.grad = torch.zeros_like(p)
    opt.step()

    expected = (1.0 - lr * wd) * p0
    assert torch.allclose(p.detach(), expected, atol=1e-12, rtol=0)


def test_tiny_overfit_linear_with_muon_and_adamuon():
    torch.manual_seed(0)

    x = torch.randn(64, 8, dtype=torch.float64)
    true_w = torch.randn(8, 4, dtype=torch.float64)
    y = x @ true_w + 0.05 * torch.randn(64, 4, dtype=torch.float64)

    def run(opt_cls):
        model = nn.Linear(8, 4, bias=False).to(dtype=torch.float64)
        # Initialize away from truth
        with torch.no_grad():
            model.weight.copy_(torch.randn_like(model.weight))
        opt = opt_cls(model.parameters(), lr=5e-3, weight_decay=0.0)
        crit = nn.MSELoss()

        losses = []
        for _ in range(60):
            opt.zero_grad(set_to_none=True) if hasattr(opt, "zero_grad") else None
            pred = model(x)
            loss = crit(pred, y)
            losses.append(loss.item())
            loss.backward()
            opt.step()
        return losses

    losses_muon = run(Muon)
    losses_adamuon = run(AdaMuon)

    # Expect meaningful decrease
    assert losses_muon[-1] <= 0.5 * losses_muon[0]
    assert losses_adamuon[-1] <= 0.5 * losses_adamuon[0]


def test_adamuon_first_step_scale_invariance():
    torch.manual_seed(0)
    p0 = torch.randn(6, 6, dtype=torch.float64)
    g = torch.randn_like(p0)

    def step_with_grad(grad):
        p = p0.clone().detach().requires_grad_(True)
        opt = AdaMuon([p], lr=1e-2, weight_decay=0.0)
        p.grad = grad.clone()
        opt.step()
        return p.detach()

    p1 = step_with_grad(g)
    p2 = step_with_grad(10.0 * g)

    # AdaMuon should be approximately scale-invariant on the first step
    max_abs_diff = (p1 - p2).abs().max().item()
    assert max_abs_diff < 1e-6
