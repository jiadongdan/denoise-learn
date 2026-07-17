"""Metadata for pretrained denoising checkpoints.

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
    output_range: tuple[float, float] | None
    sha256: str
    state_dict_prefix: str = ""
    state_dict_key: str | None = None
    cache_subdir: str = "atomsegnet"
    allow_legacy_pickle: bool = False


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


_SFIN_SOURCE_COMMIT = "8aa3442e59cab26ac7328b7ad3aec5aaf9c67b93"
_SFIN_RELEASE_ROOT = (
    "https://github.com/jiadongdan/denoise-learn/releases/download/"
    "sfin-checkpoints-v1"
)

SFIN_CHECKPOINTS = {
    "sfin_bf": CheckpointInfo(
        filename="sfin_enhance_bf_500.pth",
        architecture="SFIN",
        url=f"{_SFIN_RELEASE_ROOT}/sfin_enhance_bf_500.pth",
        source_commit=_SFIN_SOURCE_COMMIT,
        output_range=None,
        sha256="6f16af3084470ec11c3e9479d5c216057a2f820ffe1a2ca373bbf9fb9822c65a",
        state_dict_prefix="module.",
        state_dict_key="model_state_dict",
        cache_subdir="sfin",
        allow_legacy_pickle=True,
    ),
    "sfin_haadf": CheckpointInfo(
        filename="sfin_enhance_haadf_500.pth",
        architecture="SFIN",
        url=f"{_SFIN_RELEASE_ROOT}/sfin_enhance_haadf_500.pth",
        source_commit=_SFIN_SOURCE_COMMIT,
        output_range=None,
        sha256="fd9ae1f112ad83a5783023266f594f4502b32640678bd24a0d394110c3af8212",
        state_dict_prefix="module.",
        state_dict_key="model_state_dict",
        cache_subdir="sfin",
        allow_legacy_pickle=True,
    ),
}

PRETRAINED_CHECKPOINTS = {**ATOMSEGNET_CHECKPOINTS, **SFIN_CHECKPOINTS}
