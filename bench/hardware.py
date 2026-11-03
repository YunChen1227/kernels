"""Hardware and library-version probes recorded on every trace."""

from __future__ import annotations

import functools
import importlib
import subprocess
from typing import Dict, Optional, Tuple

from bench.schema import Hardware

_LIBS = (
    "torch",
    "triton",
    "flashinfer",
    "flash_attn",
    "flash_attn_interface",
    "flash_mla",
    "sglang",
    "sgl_kernel",
    "cupti",
)


def _version(mod: str) -> Optional[str]:
    try:
        m = importlib.import_module(mod)
    except Exception:
        return None
    return str(getattr(m, "__version__", "present"))


@functools.lru_cache(maxsize=1)
def lib_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in _LIBS:
        v = _version(name)
        if v is not None:
            out[name] = v
    return out


@functools.lru_cache(maxsize=1)
def probe() -> Hardware:
    import torch

    if not torch.cuda.is_available():
        return Hardware(gpu="cpu", cuda=str(torch.version.cuda or ""))
    props = torch.cuda.get_device_properties(0)
    driver = ""
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except Exception:
        pass
    return Hardware(
        gpu=props.name,
        sm=f"{props.major}{props.minor}",
        num_gpus=torch.cuda.device_count(),
        total_mem_gb=round(props.total_memory / 2**30, 1),
        driver=driver,
        cuda=str(torch.version.cuda or ""),
    )


def sm() -> Optional[Tuple[int, int]]:
    hw = probe()
    if not hw.sm:
        return None
    return int(hw.sm[:-1]), int(hw.sm[-1])


def free_mem_bytes() -> int:
    import torch

    if not torch.cuda.is_available():
        return 8 * 2**30
    free, _ = torch.cuda.mem_get_info()
    return int(free)


def has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False
