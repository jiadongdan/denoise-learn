import numpy as np
import pytest

from denoiselearn.metrics import (
    evaluate_denoising,
    mean_absolute_error,
    mean_squared_error,
    peak_signal_noise_ratio,
    root_mean_squared_error,
    structural_similarity,
)


def test_error_metrics_have_expected_values():
    truth = np.zeros((8, 8))
    image = np.full((8, 8), 0.5)

    assert mean_squared_error(image, truth) == pytest.approx(0.25)
    assert root_mean_squared_error(image, truth) == pytest.approx(0.5)
    assert mean_absolute_error(image, truth) == pytest.approx(0.5)
    assert peak_signal_noise_ratio(image, truth, data_range=1.0) == pytest.approx(
        10 * np.log10(4)
    )


def test_identical_images_have_perfect_full_reference_scores():
    truth = np.arange(64, dtype=float).reshape(8, 8) / 63

    assert peak_signal_noise_ratio(truth, truth) == float("inf")
    assert structural_similarity(truth, truth) == pytest.approx(1.0)


def test_evaluate_denoising_reports_positive_improvement():
    truth = np.linspace(0, 1, 64).reshape(8, 8)
    noisy = np.clip(truth + 0.2, 0, 1)
    denoised = np.clip(truth + 0.05, 0, 1)

    result = evaluate_denoising(noisy, denoised, truth, data_range=1.0)

    assert set(result) == {"noisy", "denoised", "improvement"}
    assert all(value > 0 for value in result["improvement"].values())


def test_multichannel_image_requires_explicit_channel_axis():
    truth = np.linspace(0, 1, 8 * 8 * 3).reshape(8, 8, 3)

    assert structural_similarity(
        truth, truth, data_range=1.0, channel_axis=-1
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("image", "truth", "message"),
    [
        (np.zeros((8, 8)), np.zeros((7, 8)), "same shape"),
        (np.array([]), np.array([]), "must not be empty"),
        (np.full((8, 8), np.nan), np.zeros((8, 8)), "finite values"),
    ],
)
def test_invalid_images_raise_clear_errors(image, truth, message):
    with pytest.raises(ValueError, match=message):
        mean_squared_error(image, truth)


def test_constant_truth_requires_explicit_data_range():
    image = np.ones((8, 8))
    truth = np.zeros((8, 8))

    with pytest.raises(ValueError, match="cannot be inferred"):
        peak_signal_noise_ratio(image, truth)
