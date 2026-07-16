"""Metadata for pretrained AtomSegNet denoising checkpoints.

Download, cache, and loading operations live in :mod:`.pretrained`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata tying a pretrained checkpoint to its model architecture."""

    filename: str
    architecture: str
    url: str
    source_commit: str
    output_range: tuple[float, float]
    sha256: str
    state_dict_prefix: str = ""


_SOURCE_COMMIT = "fe317bab38d9ecee7762c60d98c3b986ab51be01"
_RAW_ROOT = (
    "https://raw.githubusercontent.com/xinhuolin/AtomSegNet/"
    f"{_SOURCE_COMMIT}/model_weights"
)

ATOMSEGNET_CHECKPOINTS = {
    "unet_denoise": CheckpointInfo(
        filename="denoise.pth",
        architecture="AtomSegNetUNet",
        url=f"{_RAW_ROOT}/denoise.pth",
        source_commit=_SOURCE_COMMIT,
        output_range=(0.0, 1.0),
        sha256="a9e677353b48532fbb948f35ed2912a8806901cd0aaa416cf32701619f699a42",
    ),
    "nested_unet_denoise": CheckpointInfo(
        filename="Gen1-noNoise.pth",
        architecture="AtomSegNetNestedUNet",
        url=f"{_RAW_ROOT}/Gen1-noNoise.pth",
        source_commit=_SOURCE_COMMIT,
        output_range=(-1.0, 1.0),
        sha256="5dfd9abdaf5a80579cad49df900c22454ee403bc4747d032afcd42951bb3dabf",
        state_dict_prefix="module.",
    ),
}
