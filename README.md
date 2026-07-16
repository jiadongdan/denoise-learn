# denoise-learn

Deep-learning models and image-quality metrics for denoising STEM images.

## Installation

Install the package with its optional PyTorch dependency:

```powershell
pip install -e ".[torch]"
```

For development and testing:

```powershell
pip install -e ".[dev]"
```

## AtomSegNet denoising models

The package includes two model architectures adapted from
[AtomSegNet](https://github.com/xinhuolin/AtomSegNet):

| Model | Checkpoint | Input size | Output range |
|---|---|---|---|
| `AtomSegNetUNet` | `denoise.pth` | Height and width divisible by 4 | `[0, 1]` |
| `AtomSegNetNestedUNet` | `Gen1-noNoise.pth` | Height and width divisible by 16 | `[-1, 1]` |

Both models expect grayscale tensors with shape `(batch, 1, height, width)`.
Image normalization and padding are intentionally left to the caller.

### Construct a model

```python
from denoiselearn.models import AtomSegNetUNet

model = AtomSegNetUNet()
model.eval()
```

### Checkpoint locations

Checkpoint metadata, including pinned download URLs and SHA-256 checksums, is
available through `ATOMSEGNET_CHECKPOINTS`:

```python
from denoiselearn.models import ATOMSEGNET_CHECKPOINTS

info = ATOMSEGNET_CHECKPOINTS["unet_denoise"]
print(info.filename)
print(info.url)
print(info.sha256)
```

Locally downloaded weights can be stored in:

```text
checkpoints/
└── atomsegnet/
    ├── denoise.pth
    └── Gen1-noNoise.pth
```

This directory is ignored by Git and is not included in the installed package.

### Load the shallow U-Net checkpoint

```python
from pathlib import Path

import torch

from denoiselearn.models import AtomSegNetUNet

checkpoint = Path("checkpoints/atomsegnet/denoise.pth")
state_dict = torch.load(
    checkpoint,
    map_location="cpu",
    weights_only=True,
)

model = AtomSegNetUNet()
model.load_state_dict(state_dict, strict=True)
model.eval()
```

### Load the nested U-Net checkpoint

`Gen1-noNoise.pth` was saved from `torch.nn.DataParallel`, so its state-dict
keys contain a `module.` prefix:

```python
from pathlib import Path

import torch

from denoiselearn.models import AtomSegNetNestedUNet

checkpoint = Path("checkpoints/atomsegnet/Gen1-noNoise.pth")
state_dict = torch.load(
    checkpoint,
    map_location="cpu",
    weights_only=True,
)
state_dict = {
    key.removeprefix("module."): value
    for key, value in state_dict.items()
}

model = AtomSegNetNestedUNet()
model.load_state_dict(state_dict, strict=True)
model.eval()
```

Only load checkpoints from trusted sources. PyTorch checkpoint files can use
pickle-based serialization; `weights_only=True` reduces the loading risk for
state-dict checkpoints.

### Run the model on a prepared tensor

```python
image = torch.rand(1, 1, 256, 256)

with torch.inference_mode():
    denoised = model(image)

print(denoised.shape)
```

The model output is an estimate produced from its training distribution. Keep
the raw STEM image unchanged and evaluate whether atomic features and intensity
relationships are preserved.

## Image-quality metrics

Compare noisy and denoised images with a ground-truth image:

```python
from denoiselearn.metrics import evaluate_denoising

results = evaluate_denoising(
    noisy,
    denoised,
    ground_truth,
    data_range=1.0,
)

print(results["noisy"])
print(results["denoised"])
print(results["improvement"])
```

Positive values under `results["improvement"]` indicate improved performance.

## Attribution

The AtomSegNet model definitions are adapted under the MIT License. The
original license is included in `licenses/ATOMSEGNET_LICENSE.txt`.
