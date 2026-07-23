"""Shared data contracts for offline clean-image generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TARGET_PATCH_SIZE = 512
MIN_ROTATION_SAFE_SIZE = int(np.ceil(TARGET_PATCH_SIZE * np.sqrt(2.0)))
DEFAULT_IMAGE_SIZE = 768
INT16_SCALE = np.int16(32767)
SCHEMA_VERSION = "clean-image-h5-v1"


@dataclass(frozen=True)
class ImageContract:
    """Shape and quantization contract for a clean-image file."""

    image_size: int = DEFAULT_IMAGE_SIZE
    target_patch_size: int = TARGET_PATCH_SIZE

    def validate(self) -> None:
        if self.image_size < int(np.ceil(self.target_patch_size * np.sqrt(2.0))):
            raise ValueError(
                "image_size is too small for a padding-free arbitrary-angle "
                f"{self.target_patch_size}x{self.target_patch_size} crop"
            )


def validate_2d_image(image: np.ndarray, *, allow_constant: bool = False) -> np.ndarray:
    """Return a finite float32 2-D image or raise a descriptive error."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"image is too small: {array.shape}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinity")
    if not allow_constant and float(np.ptp(array)) <= 0.0:
        raise ValueError("constant images are not valid clean targets")
    return array


def normalize_clean_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Min-max normalize one generated clean image and report its source range."""

    array = validate_2d_image(image)
    source_min = float(array.min())
    source_max = float(array.max())
    normalized = (array - source_min) / (source_max - source_min)
    return normalized.astype(np.float32), {
        "pre_normalization_min": source_min,
        "pre_normalization_max": source_max,
    }


def quantize_int16(image_01: np.ndarray) -> np.ndarray:
    """Map a normalized image to the non-negative range of signed int16."""

    array = validate_2d_image(image_01, allow_constant=True)
    if float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6:
        raise ValueError("quantize_int16 expects values in [0, 1]")
    return np.rint(np.clip(array, 0.0, 1.0) * int(INT16_SCALE)).astype(np.int16)


def dequantize_int16(image: np.ndarray) -> np.ndarray:
    """Restore an int16 clean image to float32 [0, 1]."""

    array = np.asarray(image)
    if array.dtype != np.int16:
        raise TypeError(f"expected int16, got {array.dtype}")
    if int(array.min()) < 0:
        raise ValueError("clean int16 images must use the non-negative range")
    return array.astype(np.float32) / float(INT16_SCALE)
