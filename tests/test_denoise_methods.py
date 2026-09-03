from __future__ import annotations

import numpy as np
import pytest

from denoiselearn.methods import denoise_fft, denoise_svd


def _image(size: int = 24) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    return (np.sin(x / 3.0) + np.cos(y / 5.0)).astype(np.float32)


def test_fft_retaining_all_coefficients_reconstructs_input():
    image = _image()

    result = denoise_fft(image, p=1.0)

    np.testing.assert_allclose(result, image, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("p", [0.0, -0.1, 1.1])
def test_fft_rejects_invalid_fraction(p: float):
    with pytest.raises(ValueError, match="p must be"):
        denoise_fft(_image(), p=p)


def test_svd_is_deterministic_without_mutating_numpy_global_rng():
    image = _image()
    np.random.seed(123)
    state_before = np.random.get_state()

    first = denoise_svd(image, 6, 4, random_state=11)
    state_after = np.random.get_state()
    second = denoise_svd(image, 6, 4, random_state=11)

    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape
    assert np.isfinite(first).all()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_svd_rejects_rank_larger_than_patch_matrix():
    with pytest.raises(ValueError, match="patch-matrix dimension"):
        denoise_svd(_image(10), 8, 10, extraction_step=8)
