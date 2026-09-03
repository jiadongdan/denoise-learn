"""Execute one versioned denoising job for an external orchestrator."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .contracts import (
    COMPARISON_POLICIES,
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
)
from .registry import build_registered_model, provider_capabilities


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_job(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != WORKER_SCHEMA_VERSION:
        raise ValueError("Unsupported worker job schema version.")
    return payload


def _load_input(path: Path) -> np.ndarray:
    image = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Worker input must be a finite float32 2D array.")
    if float(image.min()) < -1e-6 or float(image.max()) > 1.000001:
        raise ValueError("Worker input must lie in the shared [0, 1] space.")
    return image


def _comparison_output(unit_output: np.ndarray, policy: str) -> np.ndarray:
    if policy not in COMPARISON_POLICIES:
        raise ValueError(f"Unsupported comparison policy: {policy}")
    values = np.asarray(unit_output, dtype=np.float32)
    if policy == "raw_unit_space":
        return values.copy()
    if policy == "clipped_0_1":
        return np.clip(values, 0.0, 1.0)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        raise ValueError("A constant method output cannot use min-max normalization.")
    return (values - minimum) / (maximum - minimum)


def _run_fft(
    image: np.ndarray, options: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from denoiselearn.methods import denoise_fft

    p = float(options.get("p", 0.01))
    if not 0.0 < p <= 1.0:
        raise ValueError("FFT p must be in (0, 1].")
    return np.asarray(denoise_fft(image, p=p), dtype=np.float32), []


def _run_svd(
    image: np.ndarray, options: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from denoiselearn.methods import denoise_svd

    patch_size = int(options.get("patch_size", 32))
    n_components = int(options.get("n_components", 64))
    seed = int(options.get("random_seed", 0))
    if patch_size < 2 or patch_size >= min(image.shape):
        raise ValueError(
            "SVD patch_size must be at least 2 and smaller than both dimensions."
        )
    if n_components < 1 or n_components > patch_size * patch_size:
        raise ValueError("SVD n_components must be in [1, patch_size squared].")
    raw = denoise_svd(
        image,
        patch_size=patch_size,
        n_components=n_components,
        random_state=seed,
    )
    return np.asarray(raw, dtype=np.float32), []


def _extract_state_dict(
    checkpoint: Any, options: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Neural checkpoint must decode to a mapping.")
    expected_schema = options.get("checkpoint_schema_version")
    if expected_schema is not None and checkpoint.get("schema_version") != expected_schema:
        raise ValueError("Neural checkpoint schema version mismatch.")
    expected_model = options.get("checkpoint_model_name")
    if expected_model is not None and checkpoint.get("model_name") != expected_model:
        raise ValueError("Neural checkpoint model name mismatch.")
    state_key = options.get("checkpoint_state_key")
    state = checkpoint if state_key is None else checkpoint.get(str(state_key))
    if not isinstance(state, dict):
        raise TypeError("Neural checkpoint state dict is missing or invalid.")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def _pad_to_multiple(image, multiple: int):
    import torch.nn.functional as functional

    height, width = image.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height == 0 and pad_width == 0:
        return image, None
    if pad_height >= height or pad_width >= width:
        raise ValueError("Input is too small for reflection padding to the model multiple.")
    padded = functional.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    return padded, {
        "name": "reflection_pad_then_crop",
        "original_shape": [height, width],
        "padded_shape": list(padded.shape[-2:]),
        "pad_bottom": pad_height,
        "pad_right": pad_width,
    }


def _run_neural(
    image: np.ndarray, method: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    import torch

    requested_device = str(method.get("device", method.get("device_kind", "cuda")))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable.")
    checkpoint_path = Path(method["checkpoint_path"])
    actual_sha256 = _file_sha256(checkpoint_path)
    expected_sha256 = str(method["checkpoint_sha256"]).lower()
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Neural checkpoint SHA-256 mismatch.")
    adapter_options = dict(method.get("adapter_options", {}))
    model_name = str(method["model_name"])
    model = build_registered_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(checkpoint, adapter_options)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict neural checkpoint load returned incompatible keys.")
    device = torch.device(requested_device)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tensor = torch.from_numpy(image[None, None]).to(device=device, dtype=torch.float32)
    tensor, padding_transform = _pad_to_multiple(
        tensor, int(method["size_multiple"])
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    with torch.inference_mode():
        raw_tensor = model(tensor)
    inference_seconds = perf_counter() - started
    raw_tensor = raw_tensor[..., : image.shape[0], : image.shape[1]]
    if str(method["postprocess"]) == "signed_to_unit":
        unit_tensor = raw_tensor.add(1.0).mul(0.5)
    elif str(method["postprocess"]) == "native_unit":
        unit_tensor = raw_tensor
    else:
        raise ValueError("Unsupported neural postprocess policy.")
    raw = raw_tensor[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    unit = unit_tensor[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    transforms = [] if padding_transform is None else [padding_transform]
    runtime = {
        "torch_version": str(torch.__version__),
        "device": str(device),
        "inference_seconds": inference_seconds,
        "checkpoint_sha256": actual_sha256,
        "checkpoint_strict_load": True,
    }
    if device.type == "cuda":
        runtime.update(
            {
                "cuda_device": torch.cuda.get_device_name(device),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    return raw, unit, transforms, runtime


def run_worker(job_path: Path) -> dict[str, Any]:
    """Execute one method job and persist arrays plus provider provenance."""
    job = _load_job(job_path)
    image = _load_input(Path(job["input_path"]))
    method = dict(job["method"])
    options = dict(job.get("options", {}))
    kind = str(method["kind"])
    started = perf_counter()
    runtime_details: dict[str, Any] = {}
    if kind == "fft":
        raw, transforms = _run_fft(image, options)
        unit = raw
    elif kind == "svd":
        raw, transforms = _run_svd(image, options)
        unit = raw
    elif kind == "asn":
        raw, unit, transforms, runtime_details = _run_neural(image, method)
    else:
        raise ValueError(f"Unsupported provider method kind: {kind}")
    runtime_seconds = perf_counter() - started
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != image.shape or not np.isfinite(raw).all():
        raise RuntimeError("Method raw output shape or finite check failed.")
    comparison = _comparison_output(unit, str(job["comparison_policy"]))
    if comparison.shape != image.shape or not np.isfinite(comparison).all():
        raise RuntimeError("Method comparison output shape or finite check failed.")
    output_path = Path(job["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, raw=raw, comparison=comparison)
    record = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "denoise-learn",
        "identifier": str(method["identifier"]),
        "kind": kind,
        "options": options,
        "runtime_seconds": runtime_seconds,
        "raw_output_range": [float(raw.min()), float(raw.max())],
        "comparison_output_range": [
            float(comparison.min()),
            float(comparison.max()),
        ],
        "transforms": transforms,
        "runtime": runtime_details,
        "output_path": str(output_path.resolve()),
    }
    record_path = Path(job["record_path"])
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def probe_worker(job_path: Path) -> dict[str, Any]:
    """Check one declared method without executing image inference."""
    job = _load_job(job_path)
    method = dict(job["method"])
    kind = str(method["kind"])
    details: dict[str, Any] = {}
    if kind == "fft":
        from denoiselearn.methods import denoise_fft

        details["callable"] = denoise_fft.__module__ + "." + denoise_fft.__name__
    elif kind == "svd":
        from denoiselearn.methods import denoise_svd

        details["callable"] = denoise_svd.__module__ + "." + denoise_svd.__name__
    elif kind == "asn":
        import torch

        requested_device = str(method.get("device", method.get("device_kind", "cuda")))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA probe was requested but CUDA is unavailable.")
        checkpoint_path = Path(method["checkpoint_path"])
        actual_sha256 = _file_sha256(checkpoint_path)
        if actual_sha256 != str(method["checkpoint_sha256"]).lower():
            raise RuntimeError("Neural checkpoint SHA-256 mismatch during probe.")
        model = build_registered_model(str(method["model_name"]))
        details.update(
            {
                "torch_version": str(torch.__version__),
                "device": requested_device,
                "checkpoint_sha256": actual_sha256,
                "model_class": model.__class__.__module__
                + "."
                + model.__class__.__name__,
            }
        )
        if requested_device.startswith("cuda"):
            details["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        raise ValueError(f"Unsupported provider method kind: {kind}")
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "denoise-learn",
        "identifier": str(method["identifier"]),
        "available": True,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", type=Path)
    group.add_argument("--probe", type=Path)
    group.add_argument("--capabilities", action="store_true")
    arguments = parser.parse_args()
    if arguments.capabilities:
        result = provider_capabilities()
    elif arguments.job is not None:
        result = run_worker(arguments.job.resolve())
    else:
        result = probe_worker(arguments.probe.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
