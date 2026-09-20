"""Runtime environment probes for dual-backend demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class EnvInfo:
    is_cuda: bool
    has_triton: bool
    compute_capability: Optional[Tuple[int, int]]
    is_sm120: bool
    is_sm121: bool
    is_sm100: bool
    device_name: Optional[str]
    torch_version: str
    torch_cuda_version: Optional[str]


def probe() -> EnvInfo:
    import torch

    is_cuda = bool(torch.cuda.is_available())
    cc: Optional[Tuple[int, int]] = None
    device_name: Optional[str] = None
    if is_cuda:
        cc = tuple(torch.cuda.get_device_capability(0))  # type: ignore[assignment]
        device_name = torch.cuda.get_device_name(0)

    has_triton = False
    try:
        import triton  # noqa: F401

        has_triton = True
    except ImportError:
        pass

    major = cc[0] if cc else -1
    minor = cc[1] if cc else -1
    return EnvInfo(
        is_cuda=is_cuda,
        has_triton=has_triton,
        compute_capability=cc,
        is_sm120=is_cuda and major == 12 and minor == 0,
        is_sm121=is_cuda and major == 12 and minor == 1,
        is_sm100=is_cuda and major == 10,
        device_name=device_name,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
    )


# Lazy module-level aliases; refreshed by bootstrap().
IS_CUDA = False
HAS_TRITON = False
IS_SM120 = False
IS_SM121 = False
IS_SM100 = False
COMPUTE_CAPABILITY: Optional[Tuple[int, int]] = None
DEVICE_NAME: Optional[str] = None
_ENV: Optional[EnvInfo] = None


def refresh() -> EnvInfo:
    global IS_CUDA, HAS_TRITON, IS_SM120, IS_SM121, IS_SM100
    global COMPUTE_CAPABILITY, DEVICE_NAME, _ENV
    _ENV = probe()
    IS_CUDA = _ENV.is_cuda
    HAS_TRITON = _ENV.has_triton
    IS_SM120 = _ENV.is_sm120
    IS_SM121 = _ENV.is_sm121
    IS_SM100 = _ENV.is_sm100
    COMPUTE_CAPABILITY = _ENV.compute_capability
    DEVICE_NAME = _ENV.device_name
    return _ENV


def get_env() -> EnvInfo:
    if _ENV is None:
        return refresh()
    return _ENV


def default_backend() -> str:
    """Prefer sm120 when the machine can actually run those kernels."""
    env = get_env()
    if env.is_sm120 and env.has_triton:
        return "sm120"
    return "cpu_ref"


def require_sm120() -> None:
    env = get_env()
    if not env.is_cuda:
        raise RuntimeError(
            "sm120 backend requires CUDA, but torch.cuda.is_available() is False. "
            f"torch={env.torch_version}, torch.version.cuda={env.torch_cuda_version}."
        )
    if not env.is_sm120:
        raise RuntimeError(
            "sm120 backend requires compute capability 12.0 (RTX 5070 Ti / RTX 50-series). "
            f"Got device={env.device_name!r} capability={env.compute_capability}."
        )
    if not env.has_triton:
        raise RuntimeError(
            "sm120 backend requires triton. Install a CUDA-enabled triton "
            "(Linux) or triton-windows (Windows)."
        )
