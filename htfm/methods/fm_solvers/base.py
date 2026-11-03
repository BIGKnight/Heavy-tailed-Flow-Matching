"""ODE solver abstraction for FM sampling.

Solvers integrate dx/dt = v(x, t) backward from t=t_start (noise) to t=t_end (data).
The integration loop calls step() repeatedly; each solver is self-contained and only
interacts with a pure velocity function v_fn(x, t_scalar) -> Tensor.

Conventions
-----------
- dt > 0 is the step size in t; we advance from t to t - dt each step.
- v_fn(x, t_scalar) takes a float-valued time scalar and returns velocity of the same
  shape as x. The caller is responsible for building t -> batched time tensor internally
  (done once in flow_matching.sample()).
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional


class ODESolver(ABC):
    name: str = 'abstract'
    nfe_per_step: int = 1

    @abstractmethod
    def step(self, v_fn: Callable, x, t: float, dt: float):
        """One integration step. Returns x at t - dt."""

    def integrate(self,
                  v_fn: Callable,
                  x0,
                  N: int,
                  t_start: float = 1.0,
                  t_end: float = 0.0,
                  step_hook: Optional[Callable] = None,
                  **solver_kwargs):
        """Default fixed-step integration. Solvers with variable-step logic override this.

        step_hook(x, i) -> x is applied after each step (used for clip/history/tqdm).
        **solver_kwargs: ignored by default; adaptive solvers use them (rtol/atol/etc).
        """
        dt = (t_start - t_end) / N
        x = x0
        for i in range(N):
            t = t_start - i * dt
            x = self.step(v_fn, x, t, dt)
            if step_hook is not None:
                x = step_hook(x, i)
        return x
