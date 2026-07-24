"""Standalone stochastic noise models for normalized S/TEM images.

PyTorch and SciPy are optional dependencies. Install them with
``denoise-learn[noise]``.
"""

from .gaussian_noise import GAUSSIAN_SIGMA_RANGE, add_gaussian_noise
from .poisson_noise import (
    POISSON_PEAK_SIGMA_RANGE,
    add_poisson_noise_peak_sigma,
)
from .scan_noise import SCAN_JITTER_SIGMA_RANGE, add_scan_noise

__all__ = [
    "GAUSSIAN_SIGMA_RANGE",
    "POISSON_PEAK_SIGMA_RANGE",
    "SCAN_JITTER_SIGMA_RANGE",
    "add_gaussian_noise",
    "add_poisson_noise_peak_sigma",
    "add_scan_noise",
]
