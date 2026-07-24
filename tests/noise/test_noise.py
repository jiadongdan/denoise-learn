from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import map_coordinates

from denoiselearn import noise


def _clean_image(size: int = 48) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, size * size, dtype=torch.float32)
    return values.reshape(size, size)


def test_public_api_and_frozen_ranges() -> None:
    assert noise.GAUSSIAN_SIGMA_RANGE == (0.01, 0.25)
    assert noise.POISSON_PEAK_SIGMA_RANGE == (0.12, 0.35)
    assert noise.SCAN_JITTER_SIGMA_RANGE == (0.0, 0.8)
    assert callable(noise.add_gaussian_noise)
    assert callable(noise.add_poisson_noise_peak_sigma)
    assert callable(noise.add_scan_noise)


def test_gaussian_noise_is_exactly_reproducible() -> None:
    clean = _clean_image()
    first, first_metadata = noise.add_gaussian_noise(
        clean,
        sigma=0.17,
        rng=np.random.default_rng(20260724),
    )
    second, second_metadata = noise.add_gaussian_noise(
        clean,
        sigma=0.17,
        rng=np.random.default_rng(20260724),
    )
    assert torch.equal(first, second)
    assert first_metadata == second_metadata
    assert first_metadata["noise_type"] == "gaussian"
    assert first_metadata["normalization_status"] == "not_applied"


def test_gaussian_returns_raw_unclipped_values() -> None:
    clean = torch.zeros((64, 64), dtype=torch.float32)
    noisy, metadata = noise.add_gaussian_noise(
        clean,
        sigma=0.25,
        rng=np.random.default_rng(5),
    )
    assert float(noisy.min()) < 0.0
    assert metadata["post_gaussian_min"] == float(noisy.min())


def test_peak_sigma_poisson_is_exactly_reproducible() -> None:
    clean = _clean_image()
    first, first_metadata = noise.add_poisson_noise_peak_sigma(
        clean,
        peak_sigma=0.25,
        rng=np.random.default_rng(20260724),
    )
    second, second_metadata = noise.add_poisson_noise_peak_sigma(
        clean,
        peak_sigma=0.25,
        rng=np.random.default_rng(20260724),
    )
    assert torch.equal(first, second)
    assert first_metadata == second_metadata
    assert first_metadata["poisson_gain"] == 16.0
    assert first_metadata["poisson_parameterization"] == "peak_sigma_v1"


def test_peak_sigma_poisson_zero_signal_is_well_defined() -> None:
    clean = torch.zeros((32, 32), dtype=torch.float32)
    noisy, metadata = noise.add_poisson_noise_peak_sigma(
        clean,
        peak_sigma=0.35,
        rng=np.random.default_rng(17),
    )
    assert torch.equal(noisy, clean)
    assert metadata["poisson_zero_signal"] is True
    assert metadata["realized_global_rmse_raw"] == 0.0


@pytest.mark.parametrize(
    ("jx", "jy", "expected_axis"),
    ((0.4, 0.0, "x"), (0.0, 0.4, "y"), (0.0, 0.0, "none")),
)
def test_scan_noise_matches_historical_coordinate_equation(
    jx: float,
    jy: float,
    expected_axis: str,
) -> None:
    clean = _clean_image(32)
    seed = 20260724
    actual, metadata = noise.add_scan_noise(
        clean,
        jx=jx,
        jy=jy,
        rng=np.random.default_rng(seed),
    )

    rng = np.random.default_rng(seed)
    coordinates_1d = range(32)
    x_coordinates, y_coordinates = np.meshgrid(
        coordinates_1d,
        coordinates_1d,
    )
    dx = rng.normal(0.0, 1.0, (32, 1)) * jx
    dy = rng.normal(0.0, 1.0, (1, 32)) * jy
    expected = map_coordinates(
        clean.numpy(),
        np.array([y_coordinates + dy, x_coordinates + dx]),
    )

    np.testing.assert_array_equal(actual.numpy(), expected)
    assert metadata["scan_axis_inferred"] == expected_axis
    assert metadata["normalization_status"] == "not_applied"


def test_scan_noise_preserves_single_channel_shape() -> None:
    clean = _clean_image(32).unsqueeze(0)
    noisy, _ = noise.add_scan_noise(
        clean,
        jx=0.3,
        jy=0.0,
        rng=np.random.default_rng(8),
    )
    assert noisy.shape == clean.shape
    assert noisy.dtype == clean.dtype


@pytest.mark.parametrize(
    ("function_name", "keyword"),
    (
        ("add_gaussian_noise", {"sigma": 0.1}),
        ("add_poisson_noise_peak_sigma", {"peak_sigma": 0.2}),
        ("add_scan_noise", {"jx": 0.2, "jy": 0.0}),
    ),
)
def test_noise_functions_require_caller_owned_generator(
    function_name: str,
    keyword: dict[str, float],
) -> None:
    function = getattr(noise, function_name)
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        function(_clean_image(), rng=None, **keyword)


@pytest.mark.parametrize(
    ("function_name", "keyword"),
    (
        ("add_gaussian_noise", {"sigma": 0.1}),
        ("add_poisson_noise_peak_sigma", {"peak_sigma": 0.2}),
        ("add_scan_noise", {"jx": 0.2, "jy": 0.0}),
    ),
)
def test_noise_functions_reject_out_of_range_clean_input(
    function_name: str,
    keyword: dict[str, float],
) -> None:
    function = getattr(noise, function_name)
    with pytest.raises(ValueError, match=r"normalized to \[0, 1\]"):
        function(
            torch.full((32, 32), 1.1, dtype=torch.float32),
            rng=np.random.default_rng(3),
            **keyword,
        )


def test_scan_noise_rejects_non_square_or_multichannel_input() -> None:
    with pytest.raises(ValueError, match="square"):
        noise.add_scan_noise(
            torch.zeros((16, 24), dtype=torch.float32),
            jx=0.2,
            jy=0.0,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match=r"\[H,W\]"):
        noise.add_scan_noise(
            torch.zeros((2, 16, 16), dtype=torch.float32),
            jx=0.2,
            jy=0.0,
            rng=np.random.default_rng(1),
        )


def test_noise_source_has_no_machine_specific_paths() -> None:
    package_root = Path(noise.__file__).resolve().parent
    forbidden = ("C:\\Users\\", "D:\\work\\", "Denoise benchmark")
    for path in package_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path
