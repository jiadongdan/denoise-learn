"""Frequency-domain denoising for two-dimensional scientific images."""

from __future__ import annotations

import numpy as np


def denoise_fft(image: np.ndarray, p: float) -> np.ndarray:
    """Retain the fraction ``p`` of Fourier coefficients with highest power."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError("Input image must be a two-dimensional array.")
    if not np.isfinite(image).all():
        raise ValueError("Input image must contain only finite values.")
    if not 0.0 < p <= 1.0:
        raise ValueError("Fraction p must be in (0, 1].")

    spectrum = np.fft.fft2(image)
    power = np.abs(spectrum) ** 2
    number_to_keep = int(np.ceil(float(p) * power.size))
    selected = np.argpartition(power.ravel(), -number_to_keep)[-number_to_keep:]
    mask = np.zeros(power.size, dtype=bool)
    mask[selected] = True
    filtered = spectrum * mask.reshape(power.shape)
    return np.real(np.fft.ifft2(filtered))
