"""Evaluate pretrained AtomSegNet models on twisted-graphene HAADF images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from denoiselearn.metrics import evaluate_denoising
from denoiselearn.models import load_pretrained


DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "twisted_graphene_30deg"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs" / "twisted_graphene_30deg" / "atomsegnet"
)
DEFAULT_UNET_CHECKPOINT = (
    REPOSITORY_ROOT / "checkpoints" / "atomsegnet" / "denoise.pth"
)
DEFAULT_NESTED_CHECKPOINT = (
    REPOSITORY_ROOT / "checkpoints" / "atomsegnet" / "Gen1-noNoise.pth"
)
NOISE_LEVELS = ("light", "medium", "heavy")
MODEL_CONFIGS = (
    ("unet_denoise", "AtomSegNet U-Net"),
    ("nested_unet_denoise", "AtomSegNet nested U-Net"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Denoise the twisted-graphene HAADF examples with both pretrained "
            "AtomSegNet models and report MSE, PSNR, and SSIM."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"directory containing the .npy images (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        help=(
            "local denoise.pth path; when omitted, the repository-local file "
            "is used if present, otherwise it is downloaded"
        ),
    )
    parser.add_argument(
        "--nested-checkpoint",
        type=Path,
        help=(
            "local Gen1-noNoise.pth path; when omitted, the repository-local "
            "file is used if present, otherwise it is downloaded"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='PyTorch device such as "cpu", "cuda", or "cuda:0" (default: cpu)',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for comparison figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also display each comparison figure interactively",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    image = np.load(path, allow_pickle=False)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image at {path}, got {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"expected numeric image data at {path}, got {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError(f"image contains non-finite values: {path}")

    image = np.asarray(image, dtype=np.float32)
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"expected image values in [0, 1]: {path}")
    return image


def denoise(
    model: torch.nn.Module,
    model_name: str,
    image: np.ndarray,
    device: str,
) -> np.ndarray:
    tensor = torch.from_numpy(image.copy()).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(tensor)

    if model_name == "nested_unet_denoise":
        output = (output + 1.0) / 2.0
    output = output.clamp(0.0, 1.0)
    return output.squeeze(0).squeeze(0).cpu().numpy()


def format_row(noise_level: str, stage: str, metrics: dict[str, float]) -> str:
    return (
        f"{noise_level:<8} {stage:<9} "
        f"{metrics['mse']:>12.6f} "
        f"{metrics['psnr']:>12.3f} "
        f"{metrics['ssim']:>12.6f}"
    )


def save_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    model_name: str,
    model_label: str,
    noise_level: str,
    output_dir: Path,
    *,
    show: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{noise_level}_comparison.png"

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        ("Clean", clean),
        (f"Noisy ({noise_level})", noisy),
        (f"{model_label} denoised", denoised),
    )
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.set_axis_off()

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def local_checkpoint(requested: Path | None, default: Path) -> Path | None:
    if requested is not None:
        return requested
    return default if default.is_file() else None


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    clean = load_image(data_dir / "twisted_graphene_30deg_clean.npy")
    noisy_images = {
        level: load_image(
            data_dir / f"twisted_graphene_30deg_{level}_noisy.npy"
        )
        for level in NOISE_LEVELS
    }
    for noise_level, noisy in noisy_images.items():
        if noisy.shape != clean.shape:
            raise ValueError(
                f"{noise_level} image shape {noisy.shape} does not match "
                f"clean image shape {clean.shape}"
            )

    checkpoints = {
        "unet_denoise": local_checkpoint(
            args.unet_checkpoint, DEFAULT_UNET_CHECKPOINT
        ),
        "nested_unet_denoise": local_checkpoint(
            args.nested_checkpoint, DEFAULT_NESTED_CHECKPOINT
        ),
    }

    print(f"Data:    {data_dir}")
    print(f"Device:  {args.device}")
    print(f"Figures: {output_dir}")

    for model_name, model_label in MODEL_CONFIGS:
        checkpoint = checkpoints[model_name]
        model = load_pretrained(
            model_name,
            checkpoint_path=checkpoint,
            device=args.device,
        )

        print()
        print(model_label)
        print(f"Checkpoint: {checkpoint if checkpoint is not None else 'download/cache'}")
        print(f"{'Noise':<8} {'Image':<9} {'MSE':>12} {'PSNR (dB)':>12} {'SSIM':>12}")
        print("-" * 57)

        for noise_level, noisy in noisy_images.items():
            denoised = denoise(model, model_name, noisy, args.device)
            results = evaluate_denoising(
                noisy,
                denoised,
                clean,
                data_range=1.0,
            )
            print(format_row(noise_level, "noisy", dict(results["noisy"])))
            print(format_row(noise_level, "denoised", dict(results["denoised"])))
            print(format_row(noise_level, "improved", dict(results["improvement"])))
            figure_path = save_comparison(
                clean,
                noisy,
                denoised,
                model_name,
                model_label,
                noise_level,
                output_dir,
                show=args.show,
            )
            print(f"         figure: {figure_path}")
            print()


if __name__ == "__main__":
    main()
