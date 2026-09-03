"""Direct array API backed by the same code used by provider workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .registry import method_capabilities
from .worker import _comparison_output, _run_fft, _run_neural, _run_svd


@dataclass(frozen=True)
class ProviderResult:
    """In-memory result from one denoise-learn method execution."""

    identifier: str
    raw: np.ndarray
    comparison: np.ndarray
    options: dict[str, Any]
    transforms: list[dict[str, Any]]
    runtime: dict[str, Any]


def _method_definition(identifier: str) -> dict[str, Any]:
    for method in method_capabilities():
        if method["identifier"] == identifier:
            return dict(method)
    available = ", ".join(
        sorted(method["identifier"] for method in method_capabilities())
    )
    raise ValueError(
        f"Unknown denoise-learn method {identifier!r}; available methods: {available}"
    )


def denoise(
    image: np.ndarray,
    *,
    method: str,
    options: dict[str, Any] | None = None,
    comparison_policy: str = "clipped_0_1",
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_options: dict[str, Any] | None = None,
    device: str = "cpu",
) -> ProviderResult:
    """Denoise one finite 2D unit-range array with a registered method.

    Neural methods require an explicit verified checkpoint. This function does
    not download project-specific retrained checkpoints or guess their identity.
    """
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Input must be a finite two-dimensional array.")
    if float(values.min()) < -1e-6 or float(values.max()) > 1.000001:
        raise ValueError("Input must lie in the shared [0, 1] space.")
    definition = _method_definition(method)
    resolved_options = dict(definition.get("defaults", {}))
    resolved_options.update(options or {})
    kind = str(definition["kind"])
    runtime: dict[str, Any] = {}
    if kind == "fft":
        raw, transforms = _run_fft(values, resolved_options)
        unit = raw
    elif kind == "svd":
        raw, transforms = _run_svd(values, resolved_options)
        unit = raw
    else:
        if checkpoint_path is None or checkpoint_sha256 is None:
            raise ValueError(
                f"Method {method} requires checkpoint_path and checkpoint_sha256."
            )
        execution_definition = {
            **definition,
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "adapter_options": dict(checkpoint_options or {}),
            "device": device,
        }
        raw, unit, transforms, runtime = _run_neural(
            values, execution_definition
        )
    comparison = _comparison_output(unit, comparison_policy)
    return ProviderResult(
        identifier=method,
        raw=np.asarray(raw, dtype=np.float32),
        comparison=np.asarray(comparison, dtype=np.float32),
        options=resolved_options,
        transforms=transforms,
        runtime=runtime,
    )
