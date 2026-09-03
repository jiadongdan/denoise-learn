"""Patch-based low-rank denoising for two-dimensional scientific images."""

from __future__ import annotations

from itertools import product
from numbers import Integral

import numpy as np
from sklearn.utils.extmath import randomized_svd


def _validate_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _patch_start_indices(
    image_extent: int, patch_extent: int, step: int
) -> np.ndarray:
    if patch_extent >= image_extent:
        raise ValueError("patch_size must be smaller than both image dimensions.")
    last_start = image_extent - patch_extent
    indices = np.arange(0, last_start, step, dtype=np.intp)
    if indices.size == 0 or indices[-1] != last_start:
        indices = np.append(indices, last_start)
    return indices


def _extract_patches(
    image: np.ndarray, patch_size: int, extraction_step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_starts = _patch_start_indices(
        image.shape[0], patch_size, extraction_step
    )
    column_starts = _patch_start_indices(
        image.shape[1], patch_size, extraction_step
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        image, (patch_size, patch_size)
    )
    patches = windows[np.ix_(row_starts, column_starts)].reshape(
        -1, patch_size, patch_size
    )
    return patches, row_starts, column_starts


def _reconstruct_patches(
    patches: np.ndarray,
    image_shape: tuple[int, int],
    row_starts: np.ndarray,
    column_starts: np.ndarray,
) -> np.ndarray:
    patch_size = int(patches.shape[1])
    reconstructed = np.zeros(image_shape, dtype=patches.dtype)
    overlap_count = np.zeros(image_shape, dtype=np.int32)
    for patch, (row, column) in zip(
        patches, product(row_starts, column_starts)
    ):
        row_slice = slice(int(row), int(row) + patch_size)
        column_slice = slice(int(column), int(column) + patch_size)
        reconstructed[row_slice, column_slice] += patch
        overlap_count[row_slice, column_slice] += 1
    if np.any(overlap_count == 0):
        raise RuntimeError("Patch reconstruction left uncovered image pixels.")
    return reconstructed / overlap_count


def denoise_svd(
    image: np.ndarray,
    patch_size: int,
    n_components: int,
    *,
    extraction_step: int | None = None,
    random_state: int | np.random.RandomState | None = 0,
) -> np.ndarray:
    """Denoise an image by low-rank reconstruction of sampled image patches."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError("Input image must be a two-dimensional array.")
    if not np.isfinite(image).all():
        raise ValueError("Input image must contain only finite values.")

    patch_size = _validate_positive_integer(patch_size, "patch_size")
    n_components = _validate_positive_integer(n_components, "n_components")
    if patch_size < 2:
        raise ValueError("patch_size must be at least 2.")
    if patch_size >= min(image.shape):
        raise ValueError("patch_size must be smaller than both image dimensions.")
    if extraction_step is None:
        extraction_step = max(1, patch_size // 4)
    extraction_step = _validate_positive_integer(
        extraction_step, "extraction_step"
    )

    patches, row_starts, column_starts = _extract_patches(
        image, patch_size, extraction_step
    )
    patch_matrix = patches.reshape(patches.shape[0], -1)
    maximum_rank = min(patch_matrix.shape)
    if n_components > maximum_rank:
        raise ValueError(
            "n_components must not exceed the smaller patch-matrix dimension "
            f"({maximum_rank})."
        )
    left, singular_values, right = randomized_svd(
        patch_matrix,
        n_components=n_components,
        random_state=random_state,
    )
    reconstructed_patches = (
        (left * singular_values) @ right
    ).reshape(-1, patch_size, patch_size)
    return _reconstruct_patches(
        reconstructed_patches,
        image.shape,
        row_starts,
        column_starts,
    )
