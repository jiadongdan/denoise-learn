"""Historical single-axis scan-coordinate jitter for square images."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from ._common import validate_nonnegative_finite, validate_tensor


SCAN_JITTER_SIGMA_RANGE: tuple[float, float] = (0.0, 0.8)


def add_scan_noise(
    clean: torch.Tensor,
    *,
    jx: float,
    jy: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the historical single-axis scan-coordinate jitter model.

    ``jx`` is the standard deviation in pixels of one horizontal displacement
    sampled per image row. ``jy`` is the standard deviation in pixels of one
    vertical displacement sampled per image column. The historical
    ``map_coordinates`` defaults are preserved: cubic interpolation,
    constant-zero boundaries, and spline prefiltering.

    The caller owns ``rng``. The function returns the raw interpolated result
    without clipping or normalization. Cubic interpolation can overshoot the
    normalized clean range, so callers must apply any range guard required by
    a later Poisson stage explicitly.
    """

    jx = validate_nonnegative_finite(jx, "jx")
    jy = validate_nonnegative_finite(jy, "jy")
    clean_min, clean_max = validate_tensor(
        clean,
        rng,
        name="clean",
        require_unit_range=True,
    )
    if clean.ndim == 2:
        plane = clean
        restore_channel = False
    elif clean.ndim == 3 and clean.shape[0] == 1:
        plane = clean[0]
        restore_channel = True
    else:
        raise ValueError("clean must have shape [H,W] or [1,H,W]")
    height, width = (int(value) for value in plane.shape)
    if height != width:
        raise ValueError("the historical scan-noise model requires a square image")

    coordinates_1d = range(height)
    x_coordinates, y_coordinates = np.meshgrid(
        coordinates_1d,
        coordinates_1d,
    )
    dx = rng.normal(0.0, 1.0, (height, 1)) * jx
    dy = rng.normal(0.0, 1.0, (1, width)) * jy
    coordinates = np.array(
        [y_coordinates + dy, x_coordinates + dx]
    )
    warped_np = map_coordinates(
        plane.detach().to(torch.float32).cpu().numpy(),
        coordinates,
    )
    warped_plane = torch.as_tensor(
        warped_np,
        dtype=clean.dtype,
        device=clean.device,
    )
    warped = warped_plane.unsqueeze(0) if restore_channel else warped_plane

    if jx > 0.0 and jy == 0.0:
        inferred_axis = "x"
    elif jy > 0.0 and jx == 0.0:
        inferred_axis = "y"
    elif jx == 0.0 and jy == 0.0:
        inferred_axis = "none"
    else:
        inferred_axis = "xy"

    metadata: dict[str, Any] = {
        "scan_noise_model": "historical_line_jitter_map_coordinates_v1",
        "scan_axis_inferred": inferred_axis,
        "scan_jitter_x_sigma_px": jx,
        "scan_jitter_y_sigma_px": jy,
        "scan_interpolation_order": 3,
        "scan_boundary_mode": "constant",
        "scan_boundary_cval": 0.0,
        "scan_prefilter": True,
        "pre_scan_min": clean_min,
        "pre_scan_max": clean_max,
        "post_scan_raw_min": float(warped.min()),
        "post_scan_raw_max": float(warped.max()),
        "normalization_status": "not_applied",
    }
    return warped, metadata
