"""t-Flow: heavy-tailed flow matching with a multivariate Student-t source.

Re-implementation of the t-Flow construction from Pandey et al. 2024
(arXiv:2410.14171), Appendix B, applied to image generation. The original
paper evaluates t-Flow only on weather data; this project extends the same
construction to CIFAR10-LT for an apples-to-apples comparison with HTFM.
"""

from .source import StudentTSource, sample_student_t  # noqa: F401
from .flow import flow_matching_loss, euler_sample  # noqa: F401
