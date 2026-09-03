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

## Offline clean-image generation

Install the optional scientific-data dependencies to use the clean-generation
package:

```powershell
pip install -e ".[data]"
```

The reusable implementation lives under `denoiselearn.dataset`. It provides:

- SymmLearn image-channel-only generation without symmetry maps;
- single-source multislice tileability checks and self-tiling;
- seeded twisted homo- and heterobilayer projected-column images;
- column-mask vacancy, substitution, and vacuum variants;
- non-negative int16 HDF5 I/O; and
- group-aware `train`/`valid`/`test` clean-master assembly.

All stochastic functions accept explicit seeds. SymmLearn generation also
requires the caller to provide the symmetry-learn repository path explicitly,
so the installed package contains no machine-specific source location:

```python
from pathlib import Path

from denoiselearn.dataset import (
    generate_image_only,
    load_seed_registry,
)

records = load_seed_registry("seeds.xlsx")
image, metadata = generate_image_only(
    records[0],
    repo_path=Path("external/symmetry-learn"),
)
```

The package generates clean images offline. Training Dataset workers should
read versioned HDF5 data and perform only augmentation, valid cropping, and
noise synthesis.

## Standalone noise synthesis

Install the optional noise dependencies:

```powershell
pip install -e ".[noise]"
```

The `denoiselearn.noise` package provides three standalone functions for
floating-point PyTorch tensors normalized to `[0, 1]`:

```python
import numpy as np
import torch

from denoiselearn.noise import (
    add_gaussian_noise,
    add_poisson_noise_peak_sigma,
    add_scan_noise,
)

clean = torch.rand(256, 256)
rng = np.random.default_rng(20260724)

gaussian, gaussian_metadata = add_gaussian_noise(
    clean, sigma=0.10, rng=rng
)
poisson, poisson_metadata = add_poisson_noise_peak_sigma(
    clean, peak_sigma=0.20, rng=rng
)
scan, scan_metadata = add_scan_noise(
    clean, jx=0.40, jy=0.0, rng=rng
)
```

The caller owns the NumPy generator. Each function returns raw noise output
and metadata; it does not clip or normalize the result. The validated project
ranges are Gaussian sigma `[0.01, 0.25]`, Poisson peak sigma `[0.12, 0.35]`,
and single-axis scan jitter sigma `[0.0, 0.8]` pixels.

Three compact executed notebooks show increasing standalone noise strength:

- `notebooks/noise/gaussian_noise_sweep.ipynb`
- `notebooks/noise/poisson_peak_sigma_sweep.ipynb`
- `notebooks/noise/scan_noise_sweep.ipynb`

Install `.[notebooks]` and run them from anywhere inside the repository. They
use a deterministic synthetic atomic lattice, fixed display limits, ordinary
PNG output, and no external dataset or machine-specific path. The scan
notebook presents x-axis row-wise and y-axis column-wise jitter separately.

## Pretrained models

All models expect grayscale PyTorch tensors with shape
`(batch, 1, height, width)`.

| Identifier | Architecture | Intended data | Model input | Spatial requirement | Output activation |
|---|---|---|---|---|---|
| `unet_denoise` | `AtomSegNetUNet` | AtomSegNet denoising | Float `[0, 1]` | Height and width divisible by 4 | Sigmoid (`[0, 1]`) |
| `nested_unet_denoise` | `AtomSegNetNestedUNet` | AtomSegNet Generation 1 | Float `[0, 1]` | Height and width divisible by 16 | Tanh (`[-1, 1]`) |
| `sfin_bf` | `SFIN` | Bright-field STEM | Float `[0, 255]` | Any positive height and width | None |
| `sfin_haadf` | `SFIN` | HAADF-STEM | Float `[0, 255]` | Any positive height and width | None |

Both AtomSegNet input ranges follow the released GUI inference utility, which
uses `ToTensor()` without further normalization. The nested U-Net training
pipeline does normalize to `[-1, 1]`, so the upstream training and inference
code are internally inconsistent.

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

`load_pretrained()` loads the architecture and weights only. It does not
normalize model inputs, clip model outputs, or perform display conversion.
Prepare inputs according to the `Model input` column above.

```python
import torch

from denoiselearn.models import load_pretrained

image = torch.rand(1, 1, 256, 256) * 255.0
model = load_pretrained("sfin_haadf", device="cpu")

with torch.inference_mode():
    denoised = model(image)

print(denoised.shape)
```

Preserve the raw image and evaluate whether denoising retains atomic features
and meaningful intensity relationships.

## Provider interface

`denoiselearn.provider` is the stable execution boundary for external tools such
as denoise-agent. It keeps model construction, checkpoint validation, method-specific
padding, numeric range conversion, and pixel inference inside denoise-learn.

Inspect provider capabilities without loading a model:

```powershell
denoise-learn-provider --capabilities
```

The direct array API accepts finite two-dimensional float arrays in `[0, 1]`:

```python
from denoiselearn.provider import denoise

result = denoise(image, method="fft")
print(result.options)
print(result.raw.shape)
print(result.comparison.shape)
```

FFT defaults to `p=0.01`. FFT and SVD are implemented directly in
`denoiselearn.methods`; they do not require `motif-learn`. Their provider identifiers
are `fft` and `svd`. Neural methods require PyTorch plus an explicit local checkpoint
path and SHA-256 value. Project-specific retrained checkpoints are not downloaded or
guessed by the provider.

External orchestrators should use the versioned file-job worker rather than import
private model modules:

```powershell
denoise-learn-provider --probe job.json
denoise-learn-provider --job job.json
```

The worker preserves raw and comparison arrays separately and writes provider,
checkpoint, transform, runtime, and output-range provenance to its JSON record.

## Twisted-graphene HAADF example

The repository includes clean and light-, medium-, and heavy-noise
`512 × 512` HAADF-STEM arrays under `data/twisted_graphene_30deg`. Run the
SFIN HAADF checkpoint on all three noise levels with:

```powershell
pip install -e ".[examples]"
python examples/evaluate_twisted_graphene_haadf.py
```

The example automatically uses
`checkpoints/sfin/sfin_enhance_haadf_500.pth` when that local file exists.
Specify a different checkpoint or device when needed:

```powershell
python examples/evaluate_twisted_graphene_haadf.py `
    --checkpoint "D:\checkpoints\sfin_enhance_haadf_500.pth" `
    --device cuda `
    --output-dir "D:\results\twisted-graphene"
```

For each noise level, the script reports noisy, denoised, and improvement
values for MSE, PSNR, and SSIM against the clean image. It also saves one
three-panel PNG for each noise level, with clean, noisy, and denoised images in
a row. Matplotlib autoscales the grayscale display range independently for
each panel. By default, figures are written under
`outputs/twisted_graphene_30deg`; pass `--show` to display them interactively.
The example sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing its scientific
libraries to work around the duplicate OpenMP runtimes present in the tested
Windows Conda environment.

The included arrays use the `[0, 1]` range, while the upstream SFIN inference
convention uses float intensities in `[0, 255]`. The script therefore scales
each input by 255, clips the model output to `[0, 255]`, and converts it back
to `[0, 1]` before calculating metrics and creating figures.

### Compare both AtomSegNet models

Run the sigmoid U-Net and nested U-Net on the same HAADF arrays with:

```powershell
python examples/evaluate_twisted_graphene_atomsegnet.py
```

The example uses `checkpoints/atomsegnet/denoise.pth` and
`checkpoints/atomsegnet/Gen1-noNoise.pth` when present, otherwise it downloads
the registered checkpoints. Custom paths can be supplied with
`--unet-checkpoint` and `--nested-checkpoint`.

The `[0, 1]` arrays are passed directly to both models, matching the upstream
GUI inference utility. AtomSegNet's nested U-Net training pipeline contains a
`[-1, 1]` normalization step, but the released `Gen1-noNoise.pth` inference
path does not apply it. This upstream inconsistency is documented here rather
than resolved by changing the published inference behavior. The nested
model's tanh output is converted to `[0, 1]` before metrics are calculated;
this output conversion is part of the example, not `load_pretrained()`.

The script reports MSE, PSNR, and SSIM and saves a clean/noisy/denoised figure
for every model and noise level under
`outputs/twisted_graphene_30deg/atomsegnet`. Matplotlib autoscales each panel
independently for display.

### Interactive notebook

For an interactive comparison of SFIN HAADF and both AtomSegNet models, open
`notebooks/twisted_graphene_denoising.ipynb`:

```powershell
pip install -e ".[notebooks]"
jupyter lab notebooks/twisted_graphene_denoising.ipynb
```

The notebook documents the model-specific ranges, reports metrics in a pandas
table, and displays all clean/noisy/denoised comparisons. The committed
notebook includes executed outputs so the results render directly on GitHub;
run the cells locally to reproduce or update them.

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

The SFIN release files are legacy training bundles containing optimizer and
scheduler state in addition to model weights. Some older PyTorch releases
cannot read these bundles with the restricted `weights_only=True` loader.
After verifying the file against its registered SHA-256, `load_pretrained()`
may use legacy pickle loading for these two explicitly registered SFIN files
and emit a warning. It never uses that fallback for an unknown or
checksum-mismatched file.

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
