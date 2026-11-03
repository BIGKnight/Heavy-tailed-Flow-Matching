"""Noise-schedule registry.

Adding a new schedule:
  1. Create htfm/methods/noise_schedule/<name>.py with a class subclassing
     NoiseSchedule and overriding _compute(timesteps, diffusion_steps, alpha).
  2. Register it in SCHEDULE_REGISTRY below with the string used in YAML's
     `<method>.scale`.

Consumed by DLPM.__init__ via build_schedule().
"""

from .base import NoiseSchedule
from .scale_preserving import ScalePreservingSchedule
from .scale_exploding import ScaleExplodingSchedule

# Trajectory-interpolation schedule (NOT a sigma/gamma producer; intentionally
# absent from SCHEDULE_REGISTRY). Used by FM with flow_type=signature_linear_ve.
from .signature_linear_ve import SignatureLinearVESchedule


SCHEDULE_REGISTRY = {
    'scale_preserving': ScalePreservingSchedule,
    'scale_exploding': ScaleExplodingSchedule,
}


def build_schedule(scale, **kwargs):
    if scale not in SCHEDULE_REGISTRY:
        raise ValueError(
            f'Unknown scale={scale!r}. Known: {list(SCHEDULE_REGISTRY.keys())}')
    return SCHEDULE_REGISTRY[scale](**kwargs)


def available_schedules():
    return list(SCHEDULE_REGISTRY.keys())
