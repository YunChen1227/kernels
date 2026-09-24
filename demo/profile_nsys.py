"""Nsight Systems capture driver for the DSV4 / QSA attention demos.

This is the script you run *under* ``nsys profile``. It:

1. Builds the model + inputs once (outside the capture region).
2. Runs a few **warmup** iterations (JIT / autotune / allocator warmup) that
   are *not* recorded.
3. Opens a ``cudaProfilerStart`` / ``cudaProfilerStop`` region and runs the
   steady-state iterations, each wrapped in a per-iter NVTX range.

Combined with ``nsys profile --capture-range=cudaProfilerApi
--capture-range-end=stop`` this yields a clean timeline containing only the
steady-state forward passes, with each pipeline stage shown as its own NVTX
band (Q path -> KV path -> compressor -> indexer -> topk -> attention -> out).

Run it directly (no profiler) to sanity-check it still works on CPU::

    D:\\conda\\python.exe D:\\workspace\\kernels\\demo\\profile_nsys.py --only qsa --tiny --iters 3

Under Nsight Systems, prefer the ``profile.ps1`` launcher in the repo root.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

_DEMO = Path(__file__).resolve().parent
_ROOT = _DEMO.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from _compat import bootstrap  # noqa: E402

bootstrap()

import torch  # noqa: E402

from backends import resolve_backend  # noqa: E402
from profiling import cuda_profiler, mark, nvtx_enabled, nvtx_range, sync  # noqa: E402


def _run_once(which: str, cfg_dsv4, cfg_qsa, backend) -> None:
    """One forward pass of the selected demo(s)."""
    import dsv4_attention_demo as dsv4
    import qsa_attention_demo as qsa

    if which in ("dsv4", "all"):
        with nvtx_range("DSV4/forward"):
            dsv4.run_forward(cfg_dsv4, backend, seed=0)
    if which in ("qsa", "all"):
        with nvtx_range("QSA/forward"):
            qsa.run_forward(cfg_qsa, backend, seed=0)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["cpu_ref", "sm120", "auto"], default="auto")
    p.add_argument("--only", choices=["dsv4", "qsa", "all"], default="all")
    p.add_argument("--tiny", action="store_true", help="use tiny shapes")
    p.add_argument("--warmup", type=int, default=3, help="unrecorded warmup iters")
    p.add_argument("--iters", type=int, default=10, help="recorded steady-state iters")
    args = p.parse_args(argv)

    backend = resolve_backend(None if args.backend == "auto" else args.backend)

    # Import here so shapes resolve after bootstrap.
    from shapes import dsv4_shapes, qsa_shapes

    preset = "tiny" if args.tiny else "full"
    is_sm120 = backend.name == "sm120"
    cfg_dsv4 = dsv4_shapes(preset, sm120=is_sm120)
    cfg_qsa = qsa_shapes(preset, sm120=is_sm120)
    # Keep CPU 'full' runs light (mirrors the demos' own auto-shrink).
    if backend.name == "cpu_ref" and not args.tiny:
        from shapes import DSV4Shapes, QSAShapes

        if cfg_dsv4.context_len > 256:
            cfg_dsv4 = DSV4Shapes(**{**cfg_dsv4.__dict__, "context_len": 256})
        cfg_qsa = QSAShapes(
            **{**cfg_qsa.__dict__, "context_len": 256, "indexer_budget": 128}
        )

    print("=== nsys capture driver ===")
    print(
        f"backend={backend.name} device={backend.device} preset={preset} "
        f"only={args.only} warmup={args.warmup} iters={args.iters} "
        f"nvtx={'on' if nvtx_enabled() else 'OFF (cpu build or KERNELS_NVTX=0)'}"
    )
    if not torch.cuda.is_available():
        print(
            "  [note] CUDA not available: this runs fine but nsys GPU rows will "
            "be empty. Run on the SM120 box for kernel-level timelines."
        )

    # --- warmup (not recorded) ---
    with nvtx_range("warmup"):
        for _ in range(max(0, args.warmup)):
            _run_once(args.only, cfg_dsv4, cfg_qsa, backend)
    sync()

    # --- recorded region ---
    mark("capture-start")
    t0 = time.perf_counter()
    with cuda_profiler(active=True):
        for i in range(max(1, args.iters)):
            with nvtx_range(f"iter{i}"):
                _run_once(args.only, cfg_dsv4, cfg_qsa, backend)
            sync()
    dt = time.perf_counter() - t0
    mark("capture-stop")

    n = max(1, args.iters)
    print(f"  [ok] captured {n} iters in {dt * 1e3:.1f} ms ({dt / n * 1e3:.2f} ms/iter)")
    print("done. Open the .nsys-rep in Nsight Systems and look at the NVTX row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
