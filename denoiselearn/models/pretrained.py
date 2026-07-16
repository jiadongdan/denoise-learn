"""Download, cache, verify, and load pretrained denoising models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Union
from urllib.request import urlopen

from platformdirs import user_cache_path
import torch
from torch import nn

from .atomsegnet import AtomSegNetNestedUNet, AtomSegNetUNet
from .checkpoints import (
    ATOMSEGNET_CHECKPOINTS,
    PRETRAINED_CHECKPOINTS,
    SFIN_CHECKPOINTS,
    CheckpointInfo,
)
from .sfin import SFIN

PathLike = Union[str, os.PathLike[str]]


class CheckpointError(RuntimeError):
    """Base error for checkpoint resolution and loading."""


class CheckpointChecksumError(CheckpointError):
    """Raised when a checkpoint does not match its registered SHA-256."""


def _checkpoint_info(name: str) -> CheckpointInfo:
    for registry in (ATOMSEGNET_CHECKPOINTS, SFIN_CHECKPOINTS):
        if name in registry:
            return registry[name]
    available = ", ".join(sorted(PRETRAINED_CHECKPOINTS))
    raise KeyError(
        f"unknown checkpoint {name!r}; available checkpoints: {available}"
    )


def _checkpoint_dir(
    info: CheckpointInfo, cache_dir: PathLike | None = None
) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    configured = os.environ.get("DENOISELEARN_CHECKPOINT_DIR")
    if configured:
        return Path(configured).expanduser()

    return user_cache_path("denoiselearn") / "checkpoints" / info.cache_subdir


def get_checkpoint_path(name: str, *, cache_dir: PathLike | None = None) -> Path:
    """Return the expected local cache path without downloading the file."""
    info = _checkpoint_info(name)
    return _checkpoint_dir(info, cache_dir) / info.filename


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path, info: CheckpointInfo) -> None:
    actual = _sha256(path)
    if actual.lower() != info.sha256.lower():
        raise CheckpointChecksumError(
            f"checksum mismatch for {path}: expected {info.sha256}, got {actual}. "
            "Remove the file or use force=True to replace it explicitly."
        )


def download_checkpoint(
    name: str,
    *,
    cache_dir: PathLike | None = None,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download and verify a registered checkpoint into the user cache.

    Existing valid files are reused. Existing invalid files raise unless
    ``force=True`` is supplied. Downloads are written to a temporary file and
    atomically moved into place only after checksum verification.
    """
    info = _checkpoint_info(name)
    destination = get_checkpoint_path(name, cache_dir=cache_dir)

    if destination.exists() and not force:
        _verify_checkpoint(destination, info)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{info.filename}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(info.url, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)

        _verify_checkpoint(temporary_path, info)
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return destination


def clear_checkpoint_cache(
    name: str | None = None, *, cache_dir: PathLike | None = None
) -> None:
    """Remove one registered checkpoint, or all registered checkpoints."""
    names = [name] if name is not None else list(PRETRAINED_CHECKPOINTS)
    for checkpoint_name in names:
        get_checkpoint_path(checkpoint_name, cache_dir=cache_dir).unlink(
            missing_ok=True
        )


def load_pretrained(
    name: str,
    *,
    checkpoint_path: PathLike | None = None,
    cache_dir: PathLike | None = None,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Construct a model, safely load verified weights, and set eval mode.

    When ``checkpoint_path`` is omitted, the checkpoint is downloaded on
    demand and cached. A manually supplied checkpoint is still verified
    against the registered SHA-256 checksum.
    """
    info = _checkpoint_info(name)
    path = (
        Path(checkpoint_path).expanduser()
        if checkpoint_path is not None
        else download_checkpoint(name, cache_dir=cache_dir)
    )
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    _verify_checkpoint(path, info)

    model_classes: dict[str, type[nn.Module]] = {
        "AtomSegNetUNet": AtomSegNetUNet,
        "AtomSegNetNestedUNet": AtomSegNetNestedUNet,
        "SFIN": SFIN,
    }
    try:
        model = model_classes[info.architecture]()
    except KeyError as exc:
        raise CheckpointError(
            f"unsupported checkpoint architecture: {info.architecture}"
        ) from exc

    state_dict = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state_dict, dict):
        raise CheckpointError(f"checkpoint does not contain a state dict: {path}")
    if info.state_dict_key:
        try:
            state_dict = state_dict[info.state_dict_key]
        except KeyError as exc:
            raise CheckpointError(
                f"checkpoint does not contain {info.state_dict_key!r}: {path}"
            ) from exc
        if not isinstance(state_dict, dict):
            raise CheckpointError(
                f"checkpoint entry {info.state_dict_key!r} is not a state dict: "
                f"{path}"
            )
    if info.state_dict_prefix:
        prefix = info.state_dict_prefix
        state_dict = {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
