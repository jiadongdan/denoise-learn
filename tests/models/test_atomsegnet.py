import pytest

torch = pytest.importorskip("torch")

from denoiselearn.models import (
    ATOMSEGNET_CHECKPOINTS,
    AtomSegNetNestedUNet,
    AtomSegNetUNet,
)


def test_shallow_unet_preserves_shape_and_has_sigmoid_range():
    model = AtomSegNetUNet().eval()
    image = torch.randn(1, 1, 20, 24)

    with torch.no_grad():
        output = model(image)

    assert output.shape == image.shape
    assert torch.all((output >= 0) & (output <= 1))


def test_nested_unet_preserves_shape_and_has_tanh_range():
    model = AtomSegNetNestedUNet().eval()
    image = torch.randn(1, 1, 16, 32)

    with torch.no_grad():
        output = model(image)

    assert output.shape == image.shape
    assert torch.all((output >= -1) & (output <= 1))


def test_shallow_unet_rejects_incompatible_spatial_size():
    model = AtomSegNetUNet().eval()

    with pytest.raises(RuntimeError):
        model(torch.randn(1, 1, 19, 20))


def test_nested_unet_rejects_incompatible_spatial_size():
    model = AtomSegNetNestedUNet().eval()

    with pytest.raises(RuntimeError):
        model(torch.randn(1, 1, 17, 32))


def test_checkpoint_registry_maps_models_and_state_dict_formats():
    shallow = ATOMSEGNET_CHECKPOINTS["unet_denoise"]
    nested = ATOMSEGNET_CHECKPOINTS["nested_unet_denoise"]

    assert shallow.architecture == "AtomSegNetUNet"
    assert shallow.filename == "denoise.pth"
    assert shallow.state_dict_prefix == ""
    assert len(shallow.sha256) == 64

    assert nested.architecture == "AtomSegNetNestedUNet"
    assert nested.filename == "Gen1-noNoise.pth"
    assert nested.state_dict_prefix == "module."
    assert len(nested.sha256) == 64
