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

## SFIN architecture

The package also includes the grayscale `SFIN` architecture adapted from
[SFIN](https://github.com/HeasonLee/SFIN). It preserves arbitrary spatial
dimensions and has no output activation. Pretrained SFIN checkpoint handling
is not included yet.

```python
from denoiselearn.models import SFIN

model = SFIN()
model.eval()
```

### Construct a model

```python
from denoiselearn.models import AtomSegNetUNet

model = AtomSegNetUNet()
model.eval()
```

### Load pretrained weights on demand

`load_pretrained` downloads a missing checkpoint, verifies its SHA-256
checksum, caches it, constructs the matching architecture, and loads it in
evaluation mode:

```python
from denoiselearn.models import load_pretrained

model = load_pretrained("unet_denoise", device="cpu")
```

Use the Generation 1 nested U-Net with:

```python
model = load_pretrained("nested_unet_denoise", device="cpu")
```

The nested checkpoint's `module.` DataParallel prefix is handled
automatically. The first call requires network access; later calls reuse the
verified cached file.

### Checkpoint metadata and cache

Checkpoint metadata, including pinned download URLs and SHA-256 checksums, is
available through `ATOMSEGNET_CHECKPOINTS`:

```python
from denoiselearn.models import ATOMSEGNET_CHECKPOINTS

info = ATOMSEGNET_CHECKPOINTS["unet_denoise"]
print(info.filename)
print(info.url)
print(info.sha256)
```

By default, weights are stored in the operating system's user cache. Set
`DENOISELEARN_CHECKPOINT_DIR` to use a custom location, such as shared storage
on an HPC system:

```powershell
$env:DENOISELEARN_CHECKPOINT_DIR = "D:\model-cache\denoiselearn"
```

The cache can also be controlled explicitly:

```python
from denoiselearn.models import (
    clear_checkpoint_cache,
    download_checkpoint,
    get_checkpoint_path,
)

path = download_checkpoint("unet_denoise")
print(get_checkpoint_path("unet_denoise"))
clear_checkpoint_cache("unet_denoise")
```

To use a checkpoint downloaded manually, supply its path. It is still verified
against the registered checksum:

```python
from pathlib import Path

from denoiselearn.models import load_pretrained

model = load_pretrained(
    "unet_denoise",
    checkpoint_path=Path("checkpoints/atomsegnet/denoise.pth"),
)
```

Downloads are written to temporary files and moved into the cache only after
verification. A corrupted cached file raises an error instead of being
silently replaced; delete it or explicitly download it with `force=True`.

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

The SFIN model definition is adapted under the Apache License 2.0. The
original license is included in `licenses/SFIN_LICENSE.txt`.
