# denoise-learn

Deep-learning models and full-reference image-quality metrics for denoising
scanning transmission electron microscopy (STEM) images.

The package provides PyTorch implementations adapted from AtomSegNet and SFIN,
verified pretrained-checkpoint loading, and NumPy-based metrics for comparing
denoised images with ground truth.

## Installation

From a local clone, install the package with PyTorch support:

```powershell
pip install -e ".[torch]"
```

Install development dependencies to run the tests:

```powershell
pip install -e ".[dev]"
pytest
```

PyTorch is optional. The image-quality metrics can be used without it.

## Pretrained models

All models expect grayscale PyTorch tensors with shape
`(batch, 1, height, width)`.

| Identifier | Architecture | Intended data | Spatial requirement | Output activation |
|---|---|---|---|---|
| `unet_denoise` | `AtomSegNetUNet` | AtomSegNet denoising | Height and width divisible by 4 | Sigmoid (`[0, 1]`) |
| `nested_unet_denoise` | `AtomSegNetNestedUNet` | AtomSegNet Generation 1 | Height and width divisible by 16 | Tanh (`[-1, 1]`) |
| `sfin_bf` | `SFIN` | Bright-field STEM | Any positive height and width | None |
| `sfin_haadf` | `SFIN` | HAADF-STEM | Any positive height and width | None |

Load a model by identifier:

```python
from denoiselearn.models import load_pretrained

model = load_pretrained("sfin_haadf", device="cpu")
```

`load_pretrained` downloads the registered checkpoint when needed, verifies
its SHA-256 checksum, constructs the matching architecture, loads its weights
strictly, moves it to the requested device, and returns it in evaluation mode.
Later calls reuse the verified cached file.

The SFIN assets are currently attached to a private GitHub release. Because
the downloader does not send GitHub credentials, automatic SFIN downloads
will work only after this repository is made public. Until then, download the
files through GitHub and use `checkpoint_path` as shown below. AtomSegNet
downloads are unaffected.

## Run inference

Input preparation is deliberately left to the caller because the correct
normalization depends on the checkpoint's training distribution. Do not infer
a normalization rule from an architecture's output activation.

```python
import torch

from denoiselearn.models import load_pretrained

image = torch.rand(1, 1, 256, 256)
model = load_pretrained("sfin_haadf", device="cpu")

with torch.inference_mode():
    denoised = model(image)

print(denoised.shape)
```

Preserve the raw image and evaluate whether denoising retains atomic features
and meaningful intensity relationships.

## Construct architectures without weights

The architecture classes can be instantiated directly for training or custom
checkpoint workflows:

```python
from denoiselearn.models import (
    AtomSegNetNestedUNet,
    AtomSegNetUNet,
    SFIN,
)

unet = AtomSegNetUNet()
nested_unet = AtomSegNetNestedUNet()
sfin = SFIN()
```

No checkpoint is downloaded when an architecture is constructed directly.

## Checkpoint files and cache

To load a manually downloaded checkpoint, pass its path. The file is still
checked against the registered SHA-256 digest:

```python
from pathlib import Path

from denoiselearn.models import load_pretrained

model = load_pretrained(
    "sfin_bf",
    checkpoint_path=Path("checkpoints/sfin/sfin_enhance_bf_500.pth"),
)
```

By default, downloaded weights are stored in the operating system's user
cache. Set `DENOISELEARN_CHECKPOINT_DIR` to choose a different location, such
as shared storage on an HPC system:

```powershell
$env:DENOISELEARN_CHECKPOINT_DIR = "D:\model-cache\denoiselearn"
```

The cache can also be managed explicitly:

```python
from denoiselearn.models import (
    clear_checkpoint_cache,
    download_checkpoint,
    get_checkpoint_path,
)

path = download_checkpoint("sfin_bf")
print(path)
print(get_checkpoint_path("sfin_bf"))
clear_checkpoint_cache("sfin_bf")
```

Downloads use temporary files and enter the cache only after verification. An
invalid cached file raises `CheckpointChecksumError`; it is not silently
replaced. Call `download_checkpoint(name, force=True)` to replace it
explicitly.

Checkpoint metadata is available through a combined registry or through the
source-specific registries:

```python
from denoiselearn.models import (
    ATOMSEGNET_CHECKPOINTS,
    PRETRAINED_CHECKPOINTS,
    SFIN_CHECKPOINTS,
)

info = PRETRAINED_CHECKPOINTS["sfin_haadf"]
print(info.filename)
print(info.url)
print(info.sha256)
```

## Image-quality metrics

Compare noisy and denoised images against the same ground-truth image:

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

The result includes MSE, RMSE, MAE, PSNR, and SSIM. Positive values in
`results["improvement"]` indicate improvement after denoising.

## Attribution and licenses

- AtomSegNet model definitions are adapted from
  [xinhuolin/AtomSegNet](https://github.com/xinhuolin/AtomSegNet) under the MIT
  License. The upstream license is included in
  `licenses/ATOMSEGNET_LICENSE.txt`.

- The SFIN model definition is adapted from
  [HeasonLee/SFIN](https://github.com/HeasonLee/SFIN) under the Apache License
  2.0. The upstream license is included in `licenses/SFIN_LICENSE.txt`.

The pretrained checkpoints remain subject to any rights and restrictions
associated with their original training data and upstream distribution.
