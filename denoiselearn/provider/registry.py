"""Stable capability registry and model construction for provider clients."""

from __future__ import annotations

from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .contracts import PROVIDER_CONTRACT_VERSION


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def method_capabilities() -> list[dict[str, Any]]:
    """Return method metadata without importing optional numeric runtimes."""
    sklearn_available = _module_available("sklearn")
    torch_available = _module_available("torch")
    return [
        {
            "identifier": "fft",
            "kind": "fft",
            "family": "classical_fft",
            "available": True,
            "device_kind": "cpu",
            "defaults": {"p": 0.01},
            "parameter_schema": {
                "p": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 1.0,
                    "candidates": [0.01, 0.02, 0.05, 0.1, 0.2],
                }
            },
            "internal_input_range": [0.0, 1.0],
            "raw_output_range": "unbounded",
            "size_multiple": 1,
            "postprocess": "native_unit",
        },
        {
            "identifier": "svd",
            "kind": "svd",
            "family": "classical_svd",
            "available": sklearn_available,
            "device_kind": "cpu",
            "defaults": {
                "patch_size": 32,
                "n_components": 64,
                "random_seed": 0,
            },
            "parameter_schema": {
                "patch_size": {"type": "integer", "minimum": 2},
                "n_components": {"type": "integer", "minimum": 1},
                "random_seed": {"type": "integer", "minimum": 0},
                "candidate_pairs": [
                    {"patch_size": 8, "n_components": 4},
                    {"patch_size": 8, "n_components": 8},
                    {"patch_size": 16, "n_components": 8},
                    {"patch_size": 16, "n_components": 16},
                    {"patch_size": 32, "n_components": 16},
                    {"patch_size": 32, "n_components": 32},
                    {"patch_size": 32, "n_components": 64},
                    {"patch_size": 64, "n_components": 64},
                ],
            },
            "internal_input_range": [0.0, 1.0],
            "raw_output_range": "unbounded",
            "size_multiple": 1,
            "postprocess": "native_unit",
        },
        {
            "identifier": "asn_gen1",
            "kind": "asn",
            "family": "deep_learning",
            "available": torch_available,
            "device_kind": "cuda",
            "defaults": {},
            "parameter_schema": {},
            "internal_input_range": [0.0, 1.0],
            "raw_output_range": "[-1,1] tanh",
            "size_multiple": 16,
            "postprocess": "signed_to_unit",
            "model_name": "asn_gen1",
            "checkpoint_required": True,
        },
        {
            "identifier": "asn_denoise",
            "kind": "asn",
            "family": "deep_learning",
            "available": torch_available,
            "device_kind": "cuda",
            "defaults": {},
            "parameter_schema": {},
            "internal_input_range": [0.0, 1.0],
            "raw_output_range": "[0,1] sigmoid",
            "size_multiple": 4,
            "postprocess": "native_unit",
            "model_name": "asn_denoise",
            "checkpoint_required": True,
        },
    ]


def provider_capabilities() -> dict[str, Any]:
    """Return the complete provider capability document."""
    try:
        package_version = version("denoise-learn")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "denoise-learn",
        "provider_version": package_version,
        "methods": method_capabilities(),
    }


def build_registered_model(name: str):
    """Construct a supported neural architecture without loading weights."""
    from denoiselearn.models import AtomSegNetNestedUNet, AtomSegNetUNet

    factories = {
        "asn_gen1": AtomSegNetNestedUNet,
        "asn_denoise": AtomSegNetUNet,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        available = ", ".join(sorted(factories))
        raise ValueError(
            f"Unsupported provider model {name!r}; available models: {available}"
        ) from exc
