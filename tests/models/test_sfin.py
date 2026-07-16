import pytest

torch = pytest.importorskip("torch")

from denoiselearn.models import SFIN


@pytest.mark.parametrize("height,width", [(8, 8), (9, 10)])
def test_sfin_preserves_shape(height, width):
    model = SFIN().eval()
    image = torch.randn(1, 1, height, width)

    with torch.no_grad():
        output = model(image)

    assert output.shape == image.shape
    assert torch.isfinite(output).all()


def test_sfin_uses_upstream_checkpoint_parameter_names():
    state_dict = SFIN().state_dict()

    assert "head_conv.weight" in state_dict
    assert "body.0.conv1.ffc.convl2l.weight" in state_dict
    assert "body.7.conv2.ffc.convg2g.fu.conv_layer.weight" in state_dict
    assert "tail_conv.weight" in state_dict
