"""FM ODE solver registry.

Adding a new solver:
  1. Create <name>.py in this dir with a class subclassing ODESolver, setting name.
  2. Import and register below.

Solvers are chosen from yaml via eval.fm.solver (or --solver CLI flag).
"""
from .base import ODESolver
from .euler import EulerSolver
from .heun import HeunSolver
from .midpoint import MidpointSolver
from .rk4 import RK4Solver
from .ab2 import AB2Solver
from .dopri5 import DopriSolver

_REGISTRY = {}


def register(cls):
    if not issubclass(cls, ODESolver):
        raise TypeError(f'{cls} must subclass ODESolver')
    if cls.name in _REGISTRY:
        raise ValueError(f"solver name '{cls.name}' already registered")
    _REGISTRY[cls.name] = cls
    return cls


register(EulerSolver)
register(HeunSolver)
register(MidpointSolver)
register(RK4Solver)
register(AB2Solver)
register(DopriSolver)


def build_solver(name: str) -> ODESolver:
    if name is None:
        name = 'euler'
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown solver '{name}'; available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def available_solvers():
    return sorted(_REGISTRY.keys())
