"""Runtime capability gates for the MixLLM CUDA backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuntimeCapability:
    major: int
    minor: int
    requested_backend: str = "auto"
    sm75_available: bool = False

    @property
    def compute_capability(self) -> float:
        return self.major + self.minor / 10

    @property
    def supports_ampere_mixllm(self) -> bool:
        return self.major >= 8

    @property
    def supports_sm75_backend(self) -> bool:
        return self.sm75_available and (self.major, self.minor) == (7, 5)

    def select_backend(self) -> str:
        if self.requested_backend not in {"auto", "sm75", "ampere", "reference"}:
            raise ValueError(f"unknown MixLLM backend: {self.requested_backend}")
        if self.requested_backend == "reference":
            return "reference"
        if self.requested_backend == "ampere":
            if not self.supports_ampere_mixllm:
                raise RuntimeError("Ampere MixLLM backend requires compute capability >= 8.0")
            return "ampere"
        if self.requested_backend == "sm75":
            if not self.supports_sm75_backend:
                raise RuntimeError(
                    "SM75 backend requires compute capability 7.5 and a validated SM75 build"
                )
            return "sm75"
        if self.supports_ampere_mixllm:
            return "ampere"
        if self.supports_sm75_backend:
            return "sm75"
        return "reference"


def detect_runtime(torch_module) -> Optional[RuntimeCapability]:
    """Detect CUDA capability without importing torch at module import time."""
    if not torch_module.cuda.is_available():
        return None
    major, minor = torch_module.cuda.get_device_capability()
    return RuntimeCapability(major, minor)