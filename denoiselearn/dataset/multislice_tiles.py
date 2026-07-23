"""Tileability QC and single-source tiling for multislice TIFF images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import tifffile

from .contracts import DEFAULT_IMAGE_SIZE, ImageContract, validate_2d_image


@dataclass(frozen=True)
class TileabilityReport:
    source_path: str
    height: int
    width: int
    dtype: str
    finite: bool
    nonconstant: bool
    left_right_nrmse: float
    top_bottom_nrmse: float
    left_right_gradient_ratio: float
    top_bottom_gradient_ratio: float
    status: str
    threshold_nrmse: float
    threshold_gradient_ratio: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tiff_2d(path: str | Path) -> np.ndarray:
    """Read one finite rectangular grayscale TIFF without changing its values."""

    array = tifffile.imread(path)
    return validate_2d_image(array)


def inspect_tileability(
    image: np.ndarray,
    *,
    source_path: str = "",
    max_nrmse: float = 0.05,
    max_gradient_ratio: float = 3.0,
) -> TileabilityReport:
    """Measure opposite-edge seams relative to intensity and internal gradients."""

    array = validate_2d_image(image).astype(np.float64)
    span = float(np.ptp(array))
    lr_jump = np.sqrt(np.mean((array[:, 0] - array[:, -1]) ** 2))
    tb_jump = np.sqrt(np.mean((array[0, :] - array[-1, :]) ** 2))
    x_gradient = float(np.sqrt(np.mean(np.diff(array, axis=1) ** 2)))
    y_gradient = float(np.sqrt(np.mean(np.diff(array, axis=0) ** 2)))
    epsilon = np.finfo(np.float64).eps * max(span, 1.0)
    lr_nrmse = float(lr_jump / span)
    tb_nrmse = float(tb_jump / span)
    lr_ratio = float(lr_jump / max(x_gradient, epsilon))
    tb_ratio = float(tb_jump / max(y_gradient, epsilon))
    passed = (
        lr_nrmse <= max_nrmse
        and tb_nrmse <= max_nrmse
        and lr_ratio <= max_gradient_ratio
        and tb_ratio <= max_gradient_ratio
    )
    return TileabilityReport(
        source_path=source_path,
        height=int(array.shape[0]),
        width=int(array.shape[1]),
        dtype=str(np.asarray(image).dtype),
        finite=True,
        nonconstant=True,
        left_right_nrmse=lr_nrmse,
        top_bottom_nrmse=tb_nrmse,
        left_right_gradient_ratio=lr_ratio,
        top_bottom_gradient_ratio=tb_ratio,
        status="PASS" if passed else "REVIEW",
        threshold_nrmse=max_nrmse,
        threshold_gradient_ratio=max_gradient_ratio,
    )


def self_tile_image(
    image: np.ndarray,
    *,
    output_size: int = DEFAULT_IMAGE_SIZE,
    offset_yx: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, dict]:
    """Build one square image by repeating one source image only."""

    ImageContract(image_size=output_size).validate()
    source = validate_2d_image(image)
    height, width = source.shape
    offset_y = int(offset_yx[0]) % height
    offset_x = int(offset_yx[1]) % width
    tiles_y = int(np.ceil((output_size + offset_y) / height))
    tiles_x = int(np.ceil((output_size + offset_x) / width))
    canvas = np.tile(source, (tiles_y, tiles_x))
    output = canvas[offset_y : offset_y + output_size, offset_x : offset_x + output_size]
    if output.shape != (output_size, output_size):
        raise RuntimeError(f"self-tiling produced unexpected shape {output.shape}")
    return output, {
        "source_height": height,
        "source_width": width,
        "tiles_y": tiles_y,
        "tiles_x": tiles_x,
        "offset_y": offset_y,
        "offset_x": offset_x,
        "tiling_policy": "single_source_self_tile_only",
    }


def generate_from_tiff(
    path: str | Path,
    *,
    output_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 0,
    require_pass: bool = True,
) -> tuple[np.ndarray, dict, TileabilityReport]:
    """QC and self-tile one TIFF; never combines it with another source."""

    path = Path(path)
    source = load_tiff_2d(path)
    report = inspect_tileability(source, source_path=str(path))
    if require_pass and report.status != "PASS":
        raise ValueError(f"tileability QC did not pass for {path}: {asdict(report)}")
    rng = np.random.default_rng(seed)
    offset = (int(rng.integers(0, source.shape[0])), int(rng.integers(0, source.shape[1])))
    image, tile_metadata = self_tile_image(source, output_size=output_size, offset_yx=offset)
    metadata = {
        "source_id": "multislice_2d",
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "seed": int(seed),
        "output_image_size": output_size,
        "tileability": asdict(report),
        **tile_metadata,
    }
    return image, metadata, report
