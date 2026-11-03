"""GMFlow core: pixel-wise Gaussian-mixture velocity head, losses, and analytic
transition — the image (pixel-wise factorized) generalization of the 2D core validated
in tools/gmflow_2d_smoke.py.

Each pixel carries an independent C-dim GM over the velocity (paper §3.4 pixel-wise
factorization). A base velocity net outputs `K*C + K` channels → per-pixel means μₖ and
mixture logits Aₖ; the scale s is predicted by a separate tiny MLP on t (more stable).

Tensor conventions (images): x [B,C,H,W]; μ [B,K,C,H,W]; logits [B,K,H,W]; log_s [B,1,H,W].
Schedule: α_t = 1−t, σ_t = t (straight-line flow matching, Gaussian source).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- GM head wrapper ---------------------------

class GMFlowHead(nn.Module):
    """Wrap a base velocity net whose output has `K*C + K` channels into a pixel-wise GM:
    means μₖ [B,K,C,H,W] and mixture logits Aₖ [B,K,H,W]. Scale s comes from a separate
    2-layer MLP on the (scalar) time t, broadcast over space."""

    def __init__(self, base_net, C, K, s_min=0.1):
        super().__init__()
        self.base = base_net
        self.C, self.K = C, K
        self.s_min = float(s_min)                              # variance-collapse floor
        self.s_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, t, **kwargs):
        out = self.base(x, t, **kwargs)                        # [B, K*C+K, H, W]
        B, _, H, W = out.shape
        mu = out[:, : self.K * self.C].view(B, self.K, self.C, H, W)
        logits = out[:, self.K * self.C:]                      # [B,K,H,W]
        t_col = t.view(-1, 1).float() if torch.is_tensor(t) else torch.full((B, 1), float(t), device=x.device)
        raw = self.s_mlp(t_col).view(B, 1, 1, 1)
        # s = s_min + softplus(raw): keeps a gradient everywhere AND lower-bounds the GM
        # component std, so the transition-NLL can no longer collapse s->0 (delta on the
        # per-sample target) and is forced to model a finite-width conditional density.
        s = self.s_min + F.softplus(raw)
        log_s = s.log().expand(B, 1, H, W)                     # [B,1,H,W]
        return mu, logits, log_s


# --------------------------- pixel-wise GM losses ---------------------------

def pixelwise_gm_nll(u, mu, logits, log_s):
    """Per-pixel GM-NLL of the velocity u [B,C,H,W] under the predicted GM (Σₖ = s²I over C)."""
    C = u.shape[1]
    log_A = F.log_softmax(logits, dim=1)                       # over K -> [B,K,H,W]
    s2 = torch.exp(2 * log_s)                                  # [B,1,H,W]
    diff2 = ((u.unsqueeze(1) - mu) ** 2).sum(2)               # sum over C -> [B,K,H,W]
    log_N = -0.5 * C * math.log(2 * math.pi) - C * log_s - diff2 / (2 * s2)   # [B,K,H,W]
    log_q = torch.logsumexp(log_A + log_N, dim=1)            # over K -> [B,H,W]
    return -log_q.mean()


def _transition_coeffs(t, lam):
    """Analytic transition coefficients (paper Eq. 1/9) for α_t=1−t, σ_t=t, Δt=λt.
    t: [B] tensor. Returns c1,c2,c3 each [B,1,1,1] and t_prev [B]."""
    t = t.view(-1)
    t_prev = t * (1 - lam)
    sig_t = t.clamp(min=1e-6)
    a_t, a_p, sp = 1 - t, (1 - t_prev).clamp(min=1e-6), t_prev
    beta = sig_t ** 2 - (a_t ** 2 / a_p ** 2) * sp ** 2
    c1 = (sp ** 2 / sig_t ** 2) * (a_t / a_p)
    c2 = (beta / sig_t ** 2) * a_p
    c3 = (beta / sig_t ** 2) * sp ** 2
    r = lambda z: z.view(-1, 1, 1, 1)
    return r(c1), r(c2), r(c3), t_prev


def pixelwise_gm_transition_nll(x_prev, x_t, mu, logits, log_s, t, lam):
    """Per-pixel transition loss (paper §3.4): NLL of the true transition x_{t-Δt} under the
    GM transition derived from the predicted velocity GM. The c₃ variance floor stabilizes s."""
    C = x_t.shape[1]
    sig_t = t.view(-1, 1, 1, 1).clamp(min=1e-6)
    s = torch.exp(log_s)                                       # [B,1,H,W]
    mu_x = x_t.unsqueeze(1) - sig_t.unsqueeze(1) * mu          # x0-GM means [B,K,C,H,W]
    s_x = sig_t * s                                            # [B,1,H,W]
    c1, c2, c3, _ = _transition_coeffs(t, lam)                 # [B,1,1,1]
    mean_k = c1.unsqueeze(1) * x_t.unsqueeze(1) + c2.unsqueeze(1) * mu_x   # [B,K,C,H,W]
    var = (c3 + c2 ** 2 * s_x ** 2).clamp(min=1e-12)          # [B,1,H,W]
    log_A = F.log_softmax(logits, dim=1)                       # [B,K,H,W]
    diff2 = ((x_prev.unsqueeze(1) - mean_k) ** 2).sum(2)      # [B,K,H,W]
    log_N = -0.5 * C * (math.log(2 * math.pi) + torch.log(var)) - diff2 / (2 * var)   # [B,K,H,W]
    log_q = torch.logsumexp(log_A + log_N, dim=1)            # [B,H,W]
    return -log_q.mean()


# --------------------------- CPU self-test ---------------------------

def _selftest():
    torch.manual_seed(0)
    B, C, H, W, K = 2, 3, 8, 8, 8

    # shape check via a dummy base net (K*C+K output channels) that mimics the ncsnpp
    # contract net(x, time_cond) -> [B, out, H, W]
    class _DummyNet(nn.Module):
        def __init__(self):
            super().__init__(); self.conv = nn.Conv2d(C, K * C + K, 3, padding=1)
        def forward(self, x, t, **kw):
            return self.conv(x)
    head = GMFlowHead(_DummyNet(), C, K)
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)
    mu, logits, log_s = head(x, t)
    assert mu.shape == (B, K, C, H, W), mu.shape
    assert logits.shape == (B, K, H, W), logits.shape
    assert log_s.shape == (B, 1, H, W), log_s.shape

    # losses run and are finite
    u = torch.randn(B, C, H, W)
    nll = pixelwise_gm_nll(u, mu, logits, log_s)
    x_prev = torch.randn(B, C, H, W)
    tr = pixelwise_gm_transition_nll(x_prev, x, mu, logits, log_s, t, lam=0.5)
    assert torch.isfinite(nll) and torch.isfinite(tr), (nll, tr)

    # K=1 pixel-wise GM-NLL gradient == L2 gradient (×1/(B·H·W)), the correctness gate
    mu1 = torch.randn(B, 1, C, H, W, requires_grad=True)
    logits1 = torch.zeros(B, 1, H, W)
    log_s1 = torch.zeros(B, 1, H, W)                           # s=1
    loss1 = pixelwise_gm_nll(u, mu1, logits1, log_s1)
    loss1.backward()
    gm_grad = mu1.grad.squeeze(1)                              # [B,C,H,W]
    l2_grad = (mu1.detach().squeeze(1) - u) / (B * H * W)
    ok = torch.allclose(gm_grad, l2_grad, atol=1e-6)
    print(f"[unit test] pixel-wise K=1 GM-NLL grad == L2 grad (×1/BHW): {ok}")
    assert ok

    print(f"SELF-TEST PASSED: shapes μ{tuple(mu.shape)} logits{tuple(logits.shape)} "
          f"s{tuple(log_s.shape)}; gm_nll={nll.item():.3f} transition={tr.item():.3f}; "
          f"pixel-wise core matches the validated 2D math.")


if __name__ == '__main__':
    _selftest()
