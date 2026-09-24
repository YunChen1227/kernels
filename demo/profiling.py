"""NVTX instrumentation + Nsight Systems helpers for the attention demos.

Goal: give Nsight Systems (``nsys``) a clean, named view of the attention
"data-flow pipeline" (Q path -> KV path -> compressor -> indexer -> topk ->
attention -> output) so the timeline shows each stage as its own band instead
of one anonymous blob of CUDA kernels.

Design notes
------------
* Ranges use ``torch.cuda.nvtx`` so **no extra pip package** is required and it
  works with the existing ``torch`` install.
* Everything degrades to a near-zero-cost no-op when:
    - NVTX is unavailable (CPU-only torch build), or
    - it is disabled via ``KERNELS_NVTX=0``.
  That means the demos still run unchanged on the current CPU box; the
  annotations only "light up" once you run them under ``nsys`` on the SM120
  GPU machine.
* ``cuda_profiler()`` brackets the region of interest so you can launch
  ``nsys profile --capture-range=cudaProfilerApi`` and skip warmup / setup.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch


def _env_disabled() -> bool:
    return os.environ.get("KERNELS_NVTX", "1").lower() in ("0", "false", "no", "off")


def _detect_nvtx() -> bool:
    """Probe whether torch.cuda.nvtx is callable on this build.

    On CPU-only wheels ``range_push`` raises, so we detect once with a matched
    push/pop and cache the result.
    """
    if _env_disabled():
        return False
    try:
        torch.cuda.nvtx.range_push("__nvtx_probe__")
        torch.cuda.nvtx.range_pop()
        return True
    except Exception:
        return False


_NVTX_OK = _detect_nvtx()


def nvtx_enabled() -> bool:
    """True when NVTX ranges will actually be emitted."""
    return _NVTX_OK


@contextlib.contextmanager
def nvtx_range(msg: str) -> Iterator[None]:
    """Named NVTX range that shows up as a band on the Nsight timeline.

    Usage::

        with nvtx_range("dsv4/indexer"):
            ...  # ops here are grouped under this label

    Use ``/``-separated names to get a readable hierarchy in the NVTX row.
    """
    if not _NVTX_OK:
        yield
        return
    torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def mark(msg: str) -> None:
    """Emit an instantaneous NVTX marker (a vertical line on the timeline)."""
    if _NVTX_OK:
        torch.cuda.nvtx.mark(msg)


def sync() -> None:
    """Synchronize CUDA so async kernel time is attributed to the open range."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def cuda_profiler(active: bool = True, *, sync_edges: bool = True):
    """Bracket the region of interest for ``nsys --capture-range=cudaProfilerApi``.

    ``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop``
    will only record between ``cudaProfilerStart`` and ``cudaProfilerStop``,
    which lets us exclude warmup / model construction from the report.

    No-op on CPU-only builds so the runner stays portable.
    """
    use = active and torch.cuda.is_available()
    if use and sync_edges:
        torch.cuda.synchronize()
    if use:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        if use and sync_edges:
            torch.cuda.synchronize()
        if use:
            torch.cuda.cudart().cudaProfilerStop()


__all__ = [
    "nvtx_enabled",
    "nvtx_range",
    "mark",
    "sync",
    "cuda_profiler",
]
