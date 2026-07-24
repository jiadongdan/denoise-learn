"""Independent zero-mean Gaussian noise for normalized images."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ._common import validate_positive_finite, validate_tensor


GAUSSIAN_SIGMA_RANGE: tuple[float, float] = (0.01, 0.25)


def _sample_gaussian(
    image: torch.Tensor, *, sigma: float, rng: np.random.Generator
) -> tuple[torch.Tensor, dict[str, Any]]:
    gaussian = rng.normal(0.0, sigma, size=tuple(image.shape))
    residual = torch.as_tensor(gaussian, dtype=image.dtype, device=image.device)
    noisy = image + residual
    metadata: dict[str, Any] = {
        "gaussian_sigma": sigma,
        "gaussian_realized_mean": float(residual.mean()),
        "gaussian_realized_std": float(residual.std(unbiased=False)),
        "pre_gaussian_min": float(image.min()),
        "pre_gaussian_max": float(image.max()),
        "post_gaussian_min": float(noisy.min()),
        "post_gaussian_max": float(noisy.max()),
    }
    return noisy, metadata


def add_gaussian_noise(
    clean: torch.Tensor,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Add zero-mean independent Gaussian noise in normalized intensity units.

    ``clean`` must be a finite floating-point tensor in ``[0, 1]``. The caller
    owns ``rng`` so the function composes with a larger deterministic sampling
    pipeline. The raw noisy tensor is returned without clipping or
    normalization.
    """

    sigma = validate_positive_finite(sigma, "sigma")
    validate_tensor(clean, rng, name="clean", require_unit_range=True)
    noisy, metadata = _sample_gaussian(clean, sigma=sigma, rng=rng)
    metadata.update(
        {
            "noise_type": "gaussian",
            "normalization_status": "not_applied",
        }
    )
    return noisy, metadata
