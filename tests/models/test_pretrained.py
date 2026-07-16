from dataclasses import replace
import hashlib

import pytest

torch = pytest.importorskip("torch")

from denoiselearn.models import (
    ATOMSEGNET_CHECKPOINTS,
    AtomSegNetNestedUNet,
    AtomSegNetUNet,
    CheckpointChecksumError,
    clear_checkpoint_cache,
    download_checkpoint,
    get_checkpoint_path,
    load_pretrained,
)


def _register_local_checkpoint(monkeypatch, name, path, architecture, prefix=""):
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    original = ATOMSEGNET_CHECKPOINTS[name]
    monkeypatch.setitem(
        ATOMSEGNET_CHECKPOINTS,
        name,
        replace(
            original,
            architecture=architecture,
            url=path.as_uri(),
            sha256=checksum,
            state_dict_prefix=prefix,
        ),
    )


def test_download_checkpoint_caches_and_reuses_verified_file(tmp_path, monkeypatch):
    source = tmp_path / "source.pth"
    source.write_bytes(b"checkpoint contents")
    _register_local_checkpoint(
        monkeypatch, "unet_denoise", source, "AtomSegNetUNet"
    )
    cache = tmp_path / "cache"

    downloaded = download_checkpoint("unet_denoise", cache_dir=cache)
    source.unlink()
    reused = download_checkpoint("unet_denoise", cache_dir=cache)

    assert downloaded == reused
    assert reused.read_bytes() == b"checkpoint contents"


def test_corrupt_cached_checkpoint_is_not_replaced_silently(tmp_path):
    path = get_checkpoint_path("unet_denoise", cache_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"corrupt")

    with pytest.raises(CheckpointChecksumError, match="checksum mismatch"):
        download_checkpoint("unet_denoise", cache_dir=tmp_path)


def test_clear_checkpoint_cache(tmp_path):
    path = get_checkpoint_path("unet_denoise", cache_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"cached")

    clear_checkpoint_cache("unet_denoise", cache_dir=tmp_path)

    assert not path.exists()


@pytest.mark.parametrize(
    ("name", "model", "architecture", "prefix"),
    [
        ("unet_denoise", AtomSegNetUNet(), "AtomSegNetUNet", ""),
        (
            "nested_unet_denoise",
            AtomSegNetNestedUNet(),
            "AtomSegNetNestedUNet",
            "module.",
        ),
    ],
)
def test_load_pretrained_with_verified_manual_checkpoint(
    tmp_path, monkeypatch, name, model, architecture, prefix
):
    checkpoint = tmp_path / f"{name}.pth"
    state_dict = model.state_dict()
    if prefix:
        state_dict = {f"{prefix}{key}": value for key, value in state_dict.items()}
    torch.save(state_dict, checkpoint)
    _register_local_checkpoint(monkeypatch, name, checkpoint, architecture, prefix)

    loaded = load_pretrained(name, checkpoint_path=checkpoint)

    assert isinstance(loaded, type(model))
    assert not loaded.training


def test_environment_variable_controls_cache_location(tmp_path, monkeypatch):
    monkeypatch.setenv("DENOISELEARN_CHECKPOINT_DIR", str(tmp_path))

    assert get_checkpoint_path("unet_denoise").parent == tmp_path
