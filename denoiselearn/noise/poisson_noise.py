"""Peak-sigma Poisson noise for normalized images."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ._common import sample_poisson_with_gain, validate_poisson_inputs


POISSON_PEAK_SIGMA_RANGE: tuple[float, float] = (0.12, 0.35)


def add_poisson_noise_peak_sigma(
    clean: torch.Tensor,
    *,
    peak_sigma: float,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample Poisson noise with fixed sigma at reference intensity one.

    The input is expected in ``[0, 1]``. For requested ``peak_sigma = s``, the
    function uses ``gain = 1 / s**2``. Therefore the noise standard deviation
    at intensity ``x`` is ``s * sqrt(x)`` and does not depend on how much zero
    background or how many other signal pixels the image contains.

    ``peak_sigma`` names the sigma at reference intensity one. Metadata also
    records the observed clean maximum, so callers can distinguish this
    reference from inputs whose actual maximum is below one. The caller owns
    ``rng``. The raw noisy tensor is returned without normalization.
    """

    peak_sigma = float(peak_sigma)
    eps = float(eps)
    if not np.isfinite(peak_sigma) or peak_sigma <= 0.0:
        raise ValueError("peak_sigma must be finite and > 0")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0")

    clean_nonnegative, clean_mean, clean_max = validate_poisson_inputs(clean, rng)
    gain = 1.0 / (peak_sigma**2)
    noisy, realized_rmse = sample_poisson_with_gain(
        clean_nonnegative, gain=gain, rng=rng
    )
    metadata: dict[str, Any] = {
        "poisson_parameterization": "peak_sigma_v1",
        "poisson_peak_sigma": peak_sigma,
        "poisson_gain": gain,
        "clean_mean": clean_mean,
        "clean_max": clean_max,
        "expected_mean_counts_per_pixel": clean_mean * gain,
        "expected_peak_counts": clean_max * gain,
        "expected_global_rmse_raw": peak_sigma * np.sqrt(clean_mean),
        "expected_observed_peak_sigma": peak_sigma * np.sqrt(clean_max),
        "realized_global_rmse_raw": realized_rmse,
        "poisson_zero_signal": clean_max <= eps,
        "normalization_status": "not_applied",
    }
    return noisy, metadata
