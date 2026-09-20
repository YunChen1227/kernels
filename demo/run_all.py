"""Run DSV4 + QSA demos with automatic backend selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent
_ROOT = _DEMO.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DEMO))

from _compat import bootstrap
from _compat import env as envmod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["cpu_ref", "sm120", "auto"], default="auto")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--check-parity", action="store_true")
    p.add_argument("--only", choices=["dsv4", "qsa", "all"], default="all")
    args = p.parse_args(argv)

    info = bootstrap()
    print("=== kernels dual-backend demo ===")
    print(
        f"torch={info.torch_version} cuda={info.torch_cuda_version} "
        f"device={info.device_name} cc={info.compute_capability} "
        f"triton={info.has_triton}"
    )
    if args.backend == "auto":
        chosen = envmod.default_backend()
        print(f"auto backend -> {chosen}")
    else:
        chosen = args.backend
        print(f"forced backend -> {chosen}")

    forward = []
    if args.tiny:
        forward.append("--tiny")
    if args.backend != "auto":
        forward.extend(["--backend", args.backend])
    else:
        forward.extend(["--backend", chosen])
    if args.check_parity:
        forward.append("--check-parity")

    rc = 0
    if args.only in ("dsv4", "all"):
        import dsv4_attention_demo as dsv4

        print()
        rc = dsv4.main(forward) or rc
    if args.only in ("qsa", "all"):
        import qsa_attention_demo as qsa

        print()
        rc = qsa.main(forward) or rc

    if rc == 0:
        print("\nALL DEMOS PASSED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
