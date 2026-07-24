"""Shared validation and sampling helpers for standalone noise functions."""

from __future__ import annotations

import numpy as np
import torch


def validate_tensor(
    image: torch.Tensor,
    rng: np.random.Generator,
    *,
    name: str,
    require_unit_range: bool,
) -> tuple[float, float]:
    """Validate one floating-point tensor and return its minimum and maximum."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not image.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if image.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f"{name} must contain only finite values")

    minimum = float(image.min())
    maximum = float(image.max())
    if require_unit_range:
        range_tolerance = 1e-7
        if minimum < -range_tolerance or maximum > 1.0 + range_tolerance:
            raise ValueError(f"{name} must be normalized to [0, 1] before noise")
    return minimum, maximum


def validate_poisson_inputs(
    clean: torch.Tensor, rng: np.random.Generator
) -> tuple[torch.Tensor, float, float]:
    """Validate and return non-negative clean data plus its mean and maximum."""

    _, clean_max = validate_tensor(
        clean, rng, name="clean", require_unit_range=True
    )
    clean_nonnegative = clean.clamp_min(0.0)
    return clean_nonnegative, float(clean_nonnegative.mean()), clean_max


def sample_poisson_with_gain(
    clean: torch.Tensor, *, gain: float, rng: np.random.Generator
) -> tuple[torch.Tensor, float]:
    """Sample a non-negative clean tensor with one fixed Poisson gain."""

    expected_counts = clean.detach().to(torch.float64).cpu().numpy() * gain
    sampled_counts = rng.poisson(expected_counts)
    noisy = torch.as_tensor(
        sampled_counts / gain,
        dtype=clean.dtype,
        device=clean.device,
    )
    realized_rmse = float(torch.sqrt(torch.mean((noisy - clean) ** 2)))
    return noisy, realized_rmse


def validate_positive_finite(value: float, name: str) -> float:
    """Return a finite positive scalar or raise ``ValueError``."""

    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def validate_nonnegative_finite(value: float, name: str) -> float:
    """Return a finite non-negative scalar or raise ``ValueError``."""

    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value
