from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from denoiselearn.provider import (
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
    denoise,
    method_capabilities,
)
from denoiselearn.provider.worker import probe_worker, run_worker


def _unit_image(size: int = 32) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    image = np.sin(x / 4.0) + np.cos(y / 5.0) + 0.02 * x
    return ((image - image.min()) / (image.max() - image.min())).astype(
        np.float32
    )


def test_capabilities_have_unique_identifiers_and_versioned_contract():
    methods = method_capabilities()
    identifiers = [method["identifier"] for method in methods]

    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == {
        "fft",
        "svd",
        "asn_gen1",
        "asn_denoise",
    }
    assert PROVIDER_CONTRACT_VERSION == "denoise-learn-provider-v1"


def test_direct_fft_api_uses_registered_default():
    result = denoise(_unit_image(), method="fft")

    assert result.identifier == "fft"
    assert result.options == {"p": 0.01}
    assert result.raw.shape == (32, 32)
    assert result.comparison.shape == (32, 32)


def test_worker_writes_versioned_fft_artifacts(tmp_path: Path):
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "output.npz"
    record_path = tmp_path / "record.json"
    np.save(input_path, _unit_image())
    job = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "record_path": str(record_path),
        "comparison_policy": "clipped_0_1",
        "options": {"p": 0.01},
        "method": {
            "identifier": "fft",
            "kind": "fft",
        },
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    record = run_worker(job_path)

    assert record["provider"] == "denoise-learn"
    assert record["provider_contract_version"] == PROVIDER_CONTRACT_VERSION
    assert output_path.is_file()
    assert record_path.is_file()


def test_direct_svd_api_is_deterministic():
    options = {"patch_size": 8, "n_components": 4, "random_seed": 7}

    first = denoise(_unit_image(), method="svd", options=options)
    second = denoise(_unit_image(), method="svd", options=options)

    np.testing.assert_array_equal(first.raw, second.raw)
    assert first.raw.shape == (32, 32)
    assert np.isfinite(first.raw).all()


def test_classical_methods_do_not_import_mtflearn(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mtflearn" or name.startswith("mtflearn."):
            raise AssertionError("Classical methods must not import mtflearn.")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    image = _unit_image()

    fft = denoise(image, method="fft")
    svd = denoise(
        image,
        method="svd",
        options={"patch_size": 8, "n_components": 4, "random_seed": 3},
    )

    assert fft.raw.shape == image.shape
    assert svd.raw.shape == image.shape


@pytest.mark.parametrize(
    ("identifier", "kind", "callable_name"),
    [
        ("fft", "fft", "denoiselearn.methods.fft.denoise_fft"),
        ("svd", "svd", "denoiselearn.methods.svd.denoise_svd"),
    ],
)
def test_classical_probe_uses_native_implementation(
    tmp_path: Path, identifier: str, kind: str, callable_name: str
):
    job_path = tmp_path / f"{identifier}.json"
    job_path.write_text(
        json.dumps(
            {
                "schema_version": WORKER_SCHEMA_VERSION,
                "method": {"identifier": identifier, "kind": kind},
            }
        ),
        encoding="utf-8",
    )

    result = probe_worker(job_path)

    assert result["available"] is True
    assert result["details"]["callable"] == callable_name


@pytest.mark.parametrize(
    ("method_name", "size"),
    [("asn_gen1", 16), ("asn_denoise", 8)],
)
def test_direct_neural_api_strict_loads_local_checkpoint(
    tmp_path: Path, method_name: str, size: int
):
    torch = pytest.importorskip("torch")
    from denoiselearn.provider.registry import build_registered_model

    model = build_registered_model(method_name)
    checkpoint = tmp_path / f"{method_name}.pt"
    payload = {
        "schema_version": "provider-test-v1",
        "model_name": method_name,
        "model": model.state_dict(),
    }
    torch.save(payload, checkpoint)
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    result = denoise(
        _unit_image(size),
        method=method_name,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checksum,
        checkpoint_options={
            "checkpoint_schema_version": "provider-test-v1",
            "checkpoint_model_name": method_name,
            "checkpoint_state_key": "model",
        },
        device="cpu",
    )

    assert result.raw.shape == (size, size)
    assert result.comparison.shape == (size, size)
    assert result.runtime["checkpoint_strict_load"] is True
