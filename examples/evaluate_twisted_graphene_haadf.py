"""Evaluate the pretrained SFIN HAADF model on twisted-graphene examples."""

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
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "twisted_graphene_30deg"
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "checkpoints"
    / "sfin"
    / "sfin_enhance_haadf_500.pth"
)
NOISE_LEVELS = ("light", "medium", "heavy")
MODEL_INTENSITY_SCALE = 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Denoise the twisted-graphene HAADF examples with SFIN and report "
            "MSE, PSNR, and SSIM against the clean image."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"directory containing the .npy images (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "local SFIN HAADF checkpoint; when omitted, the repository-local "
            "checkpoint is used if present, otherwise it is downloaded"
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
    return np.asarray(image, dtype=np.float32)


def denoise(model: torch.nn.Module, image: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    tensor = tensor * MODEL_INTENSITY_SCALE
    with torch.inference_mode():
        output = model(tensor).clamp(0.0, MODEL_INTENSITY_SCALE)
    output = output / MODEL_INTENSITY_SCALE
    return output.squeeze(0).squeeze(0).cpu().numpy()


def format_row(
    noise_level: str, stage: str, metrics: dict[str, float]
) -> str:
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
    noise_level: str,
    output_dir: Path,
    *,
    show: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"twisted_graphene_30deg_{noise_level}_comparison.png"

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        ("Clean", clean),
        (f"Noisy ({noise_level})", noisy),
        ("SFIN denoised", denoised),
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


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    checkpoint = args.checkpoint
    if checkpoint is None and DEFAULT_CHECKPOINT.is_file():
        checkpoint = DEFAULT_CHECKPOINT

    model = load_pretrained(
        "sfin_haadf",
        checkpoint_path=checkpoint,
        device=args.device,
    )
    clean = load_image(data_dir / "twisted_graphene_30deg_clean.npy")
    output_dir = args.output_dir.resolve()

    print(f"Data:       {data_dir}")
    print(f"Checkpoint: {checkpoint if checkpoint is not None else 'download/cache'}")
    print(f"Device:     {args.device}")
    print(f"Figures:    {output_dir}")
    print()
    print(f"{'Noise':<8} {'Image':<9} {'MSE':>12} {'PSNR (dB)':>12} {'SSIM':>12}")
    print("-" * 57)

    for noise_level in NOISE_LEVELS:
        noisy = load_image(
            data_dir / f"twisted_graphene_30deg_{noise_level}_noisy.npy"
        )
        if noisy.shape != clean.shape:
            raise ValueError(
                f"{noise_level} image shape {noisy.shape} does not match "
                f"clean image shape {clean.shape}"
            )

        denoised = denoise(model, noisy, args.device)
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
            noise_level,
            output_dir,
            show=args.show,
        )
        print(f"         figure: {figure_path}")
        print()


if __name__ == "__main__":
    main()
