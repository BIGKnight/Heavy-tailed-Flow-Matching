"""Heavy-tailed noise generators.

Thin factory wrappers around bem.datasets.Data.Generator. Pulled into a
separate subpackage so methods (FM / DLPM / LIM / future) can hold these
directly instead of going through DLPM's container.

  build_skewed_levy(alpha, device, ...)  -> α/2 -stable subordinator A_t
  build_sas(alpha, device, ...)          -> SAS Gaussian noise eps_t

Adding a new generator (e.g. tempered Lévy, Linnik) is one file +
one register-style addition (or just expose another bem.datasets factory).
"""

from bem.datasets.Data import Generator


def build_skewed_levy(alpha, device, isotropic=True, clamp_a=None):
    """α/2-stable totally-skewed positive subordinator (controls tail-heaviness)."""
    return Generator(
        'skewed_levy',
        alpha=alpha, device=device,
        isotropic=isotropic, clamp_a=clamp_a,
    )


def build_sas(alpha, device, isotropic=True, clamp_eps=None):
    """Symmetric α-stable Gaussian for the noise endpoint G."""
    return Generator(
        'sas',
        alpha=alpha, device=device,
        isotropic=isotropic, clamp_eps=clamp_eps,
    )


__all__ = ['build_skewed_levy', 'build_sas']
