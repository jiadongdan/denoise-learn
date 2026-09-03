"""Versioned denoising provider API for external orchestrators."""

from .contracts import PROVIDER_CONTRACT_VERSION, WORKER_SCHEMA_VERSION
from .registry import method_capabilities

__all__ = [
    "PROVIDER_CONTRACT_VERSION",
    "ProviderResult",
    "WORKER_SCHEMA_VERSION",
    "denoise",
    "method_capabilities",
]


def __getattr__(name: str):
    if name in {"ProviderResult", "denoise"}:
        from .api import ProviderResult, denoise

        return {"ProviderResult": ProviderResult, "denoise": denoise}[name]
    raise AttributeError(name)
