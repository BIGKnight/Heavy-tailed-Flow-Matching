"""GMFlow image sampler: analytic pixel-wise GM-SDE reverse steps + spectral sampling.

The reverse transition is a per-pixel GM (paper Eq. 9). Sampling each pixel independently
neglects spatial correlation and yields artifacts; spectral sampling injects a spatially
correlated Gaussian field (from a learnable power spectrum) as the transition noise.

NOTE (to validate against official code): the paper maps the correlated Gaussian to the GM
via Knothe–Rosenblatt transport. Here we use per-pixel component selection + the correlated
field as the within-component noise — a faithful-in-spirit, runnable version; the exact KR
transport is the one piece flagged for a cross-check with github.com/Lakonik/GMFlow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralField(nn.Module):
    """Learnable power spectrum → spatially-correlated Gaussian field, marginally N(0,1)
    per pixel. `log_psd` starts at 0 (→ white noise = independent pixels)."""

    def __init__(self, H, W):
        super().__init__()
        self.log_psd = nn.Parameter(torch.zeros(H, W))

    def forward(self, B, C, device):
        H, W = self.log_psd.shape
        white = torch.randn(B, C, H, W, device=device)
        amp = torch.exp(0.5 * self.log_psd).clamp(min=1e-6)          # sqrt(PSD) [H,W]
        Z = torch.fft.ifft2(torch.fft.fft2(white) * amp).real         # correlated field
        Z = Z / (Z.std(dim=(-2, -1), keepdim=True) + 1e-8)            # unit per-pixel variance
        return Z


def _coeffs(t, tp):
    """Scalar transition coefficients (α_t=1-t, σ_t=t)."""
    sig_t = max(t, 1e-6)
    a_t, a_p, sp = 1 - t, max(1 - tp, 1e-6), tp
    beta = sig_t ** 2 - (a_t ** 2 / a_p ** 2) * sp ** 2
    c1 = (sp ** 2 / sig_t ** 2) * (a_t / a_p)
    c2 = (beta / sig_t ** 2) * a_p
    c3 = (beta / sig_t ** 2) * sp ** 2
    return sig_t, c1, c2, c3


def gm_sde_step(x_t, mu, logits, log_s, t, tp, noise):
    """One analytic GM-SDE reverse step. x_t,noise [B,C,H,W]; mu [B,K,C,H,W]; logits [B,K,H,W];
    log_s [B,1,H,W]. Samples a per-pixel component ~ A_k, then the Gaussian with `noise`."""
    B, C, H, W = x_t.shape
    K = mu.shape[1]
    sig_t, c1, c2, c3 = _coeffs(t, tp)
    s = torch.exp(log_s)                                              # [B,1,H,W]
    mu_x = x_t.unsqueeze(1) - sig_t * mu                              # x0-GM means [B,K,C,H,W]
    s_x = sig_t * s
    mean_k = c1 * x_t.unsqueeze(1) + c2 * mu_x                        # [B,K,C,H,W]
    var = (c3 + c2 ** 2 * s_x ** 2).clamp(min=0.0)                   # [B,1,H,W]
    A = F.softmax(logits, dim=1)                                      # [B,K,H,W]
    k = torch.multinomial(A.permute(0, 2, 3, 1).reshape(-1, K), 1).reshape(B, 1, 1, H, W)
    mean_sel = torch.gather(mean_k, 1, k.expand(B, 1, C, H, W)).squeeze(1)   # [B,C,H,W]
    return mean_sel + torch.sqrt(var) * noise


@torch.no_grad()
def gm_sde_sample(head, shape, steps, device, spectral=None):
    """Full GM-SDE sampling loop. `head(x,t)->(mu,logits,log_s)`; `spectral` optional
    SpectralField for correlated noise (else independent Gaussian)."""
    B, C, H, W = shape
    x = torch.randn(B, C, H, W, device=device)                       # t=1 noise
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t, tp = float(ts[i]), float(ts[i + 1])
        mu, logits, log_s = head(x, torch.full((B,), t, device=device))
        noise = spectral(B, C, device) if spectral is not None else torch.randn(B, C, H, W, device=device)
        x = gm_sde_step(x, mu, logits, log_s, t, tp, noise)
    return x


@torch.no_grad()
def gm_ode_sample(head, shape, steps, device):
    """Deterministic GM-ODE sampling: integrate the mixture-mean velocity E[u]=Σ_k A_k μ_k
    with explicit Euler (dt<0). No per-pixel component selection and no injected noise, so
    the sample quality reflects the learned velocity field directly (the fair GMFlow-ODE)."""
    B, C, H, W = shape
    x = torch.randn(B, C, H, W, device=device)                       # t=1 noise
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t, tp = float(ts[i]), float(ts[i + 1])
        mu, logits, _ = head(x, torch.full((B,), t, device=device))
        A = F.softmax(logits, dim=1).unsqueeze(2)                    # [B,K,1,H,W]
        Eu = (A * mu).sum(1)                                          # E[velocity] [B,C,H,W]
        x = x + (tp - t) * Eu                                         # Euler step
    return x


def _selftest():
    import torch.nn as nn
    torch.manual_seed(0)
    B, C, H, W, K = 2, 3, 8, 8, 8
    from htfm.methods.gmflow_core import GMFlowHead

    class _DummyNet(nn.Module):
        def __init__(self):
            super().__init__(); self.conv = nn.Conv2d(C, K * C + K, 3, padding=1)
        def forward(self, x, t, **kw):
            return self.conv(x)
    head = GMFlowHead(_DummyNet(), C, K)
    spec = SpectralField(H, W)

    # spectral field: shape + marginal ~ unit variance
    Z = spec(B, C, 'cpu')
    assert Z.shape == (B, C, H, W)
    assert abs(Z.var().item() - 1.0) < 0.3, Z.var().item()

    # few-step GM-SDE sampling runs, shape-correct, finite
    for steps in (4, 8):
        x = gm_sde_sample(head, (B, C, H, W), steps, 'cpu', spectral=spec)
        assert x.shape == (B, C, H, W) and torch.isfinite(x).all(), (steps, x.shape)
    print(f"SELF-TEST PASSED: spectral field var {Z.var().item():.3f}; GM-SDE sampling "
          f"(4 & 8 steps) shape-correct and finite. Component selection + correlated-noise "
          f"injection work end-to-end.")


if __name__ == '__main__':
    _selftest()
