"""Full-reference metrics for noisy and denoised images.

All metrics compare an image with a ground-truth image of the same shape. Images
are converted to ``float64`` for calculation but are otherwise left unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage.metrics import structural_similarity as _skimage_ssim


def _validated_pair(
    image: ArrayLike, ground_truth: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    image_array = np.asarray(image, dtype=np.float64)
    truth_array = np.asarray(ground_truth, dtype=np.float64)

    if image_array.shape != truth_array.shape:
        raise ValueError(
            "image and ground_truth must have the same shape; "
            f"got {image_array.shape} and {truth_array.shape}"
        )
    if image_array.size == 0:
        raise ValueError("images must not be empty")
    if not np.all(np.isfinite(image_array)) or not np.all(np.isfinite(truth_array)):
        raise ValueError("images must contain only finite values")

    return image_array, truth_array


def _resolve_data_range(ground_truth: NDArray[np.float64], data_range: float | None) -> float:
    if data_range is None:
        data_range = float(np.ptp(ground_truth))
        if data_range == 0:
            raise ValueError(
                "data_range cannot be inferred from a constant ground-truth image; "
                "provide it explicitly"
            )
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be a finite positive number")
    return float(data_range)


def mean_squared_error(image: ArrayLike, ground_truth: ArrayLike) -> float:
    """Return the mean squared error (MSE); lower is better."""
    image_array, truth_array = _validated_pair(image, ground_truth)
    return float(np.mean(np.square(image_array - truth_array)))


def root_mean_squared_error(image: ArrayLike, ground_truth: ArrayLike) -> float:
    """Return the root mean squared error (RMSE); lower is better."""
    return float(np.sqrt(mean_squared_error(image, ground_truth)))


def mean_absolute_error(image: ArrayLike, ground_truth: ArrayLike) -> float:
    """Return the mean absolute error (MAE); lower is better."""
    image_array, truth_array = _validated_pair(image, ground_truth)
    return float(np.mean(np.abs(image_array - truth_array)))


def peak_signal_noise_ratio(
    image: ArrayLike,
    ground_truth: ArrayLike,
    *,
    data_range: float | None = None,
) -> float:
    """Return PSNR in decibels; higher is better.

    ``data_range`` is inferred from the ground-truth peak-to-peak range when it
    is omitted. Pass it explicitly (for example, ``1.0`` or ``255``) when the
    physical image range is known.
    """
    image_array, truth_array = _validated_pair(image, ground_truth)
    resolved_range = _resolve_data_range(truth_array, data_range)
    mse = float(np.mean(np.square(image_array - truth_array)))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(resolved_range**2 / mse))


def structural_similarity(
    image: ArrayLike,
    ground_truth: ArrayLike,
    *,
    data_range: float | None = None,
    channel_axis: int | None = None,
    win_size: int | None = None,
) -> float:
    """Return the mean structural similarity index (SSIM); higher is better.

    Set ``channel_axis`` explicitly for multichannel images. Its default is
    ``None`` so a 3-D image stack is evaluated as spatial data rather than being
    silently interpreted as a color image.
    """
    image_array, truth_array = _validated_pair(image, ground_truth)
    resolved_range = _resolve_data_range(truth_array, data_range)
    return float(
        _skimage_ssim(
            truth_array,
            image_array,
            data_range=resolved_range,
            channel_axis=channel_axis,
            win_size=win_size,
        )
    )


def evaluate_denoising(
    noisy: ArrayLike,
    denoised: ArrayLike,
    ground_truth: ArrayLike,
    *,
    data_range: float | None = None,
    channel_axis: int | None = None,
    win_size: int | None = None,
) -> Mapping[str, Mapping[str, float]]:
    """Compare noisy and denoised images against the same ground truth.

    The returned ``improvement`` values are oriented so positive is always
    better: error reductions for MSE/RMSE/MAE and gains for PSNR/SSIM.
    """
    noisy_array, truth_array = _validated_pair(noisy, ground_truth)
    denoised_array, _ = _validated_pair(denoised, truth_array)
    resolved_range = _resolve_data_range(truth_array, data_range)

    def metrics(image: NDArray[np.float64]) -> dict[str, float]:
        return {
            "mse": mean_squared_error(image, truth_array),
            "rmse": root_mean_squared_error(image, truth_array),
            "mae": mean_absolute_error(image, truth_array),
            "psnr": peak_signal_noise_ratio(
                image, truth_array, data_range=resolved_range
            ),
            "ssim": structural_similarity(
                image,
                truth_array,
                data_range=resolved_range,
                channel_axis=channel_axis,
                win_size=win_size,
            ),
        }

    noisy_metrics = metrics(noisy_array)
    denoised_metrics = metrics(denoised_array)
    improvement = {
        "mse": noisy_metrics["mse"] - denoised_metrics["mse"],
        "rmse": noisy_metrics["rmse"] - denoised_metrics["rmse"],
        "mae": noisy_metrics["mae"] - denoised_metrics["mae"],
        "psnr": denoised_metrics["psnr"] - noisy_metrics["psnr"],
        "ssim": denoised_metrics["ssim"] - noisy_metrics["ssim"],
    }
    return {
        "noisy": noisy_metrics,
        "denoised": denoised_metrics,
        "improvement": improvement,
    }
