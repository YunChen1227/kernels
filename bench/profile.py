"""Timeline capture for one benchmark case.

``python -m bench profile --model qwen3-8b --op attn_decode --backend flashinfer \
    --batch 64 --input-len 2048 --output-len 1024 --mode both --tool both``

* nsys: re-launches this process under ``nsys profile`` (capture range =
  cudaProfilerApi, so only the measured iterations are recorded; CUDA graph
  replays are expanded with ``--cuda-graph-trace=node``) and exports
  ``nsys stats`` CSVs (NVTX ranges, GPU kernels, CUDA API).
* torch: torch.profiler chrome trace (open in chrome://tracing or Perfetto)
  plus a key_averages CSV.

Every capture appends a ``kind="profile"`` trace whose key matches the
``run`` traces, so the report links the artifacts next to the timing rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _case_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True)
    p.add_argument("--op", default="attn_decode")
    p.add_argument("--backend", required=True)
    p.add_argument("--variant", default=None, help="attention variant (default: first)")
    p.add_argument("--phase", choices=["decode", "prefill"], default=None)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--input-len", type=int, default=2048)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--kv-point", choices=["start", "mid", "end"], default="end",
                   help="decode KV length: input, input+out/2, input+out")
    p.add_argument("--spec-tokens", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--kv-dtype", default=None)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--moe-expert-cap", type=int, default=0)
    p.add_argument("--block-layers", type=int, default=1)
    p.add_argument("--block-impl", nargs="*", metavar="K=V")


def add_args(p: argparse.ArgumentParser) -> None:
    _case_args(p)
    p.add_argument("--mode", choices=["eager", "cudagraph", "both"], default="both")
    p.add_argument("--tool", choices=["nsys", "torch", "both"], default="both")
    p.add_argument("--out-dir", default="reports/profiles")
    p.add_argument("--traces", default="traces/profiles.jsonl",
                   help="JSONL to append kind=profile traces to")
    p.add_argument("--nsys", default=os.environ.get("NSYS", "nsys"), help="nsys executable")
    p.add_argument("--nsys-extra", default="", help="extra nsys profile flags")


def add_target_args(p: argparse.ArgumentParser) -> None:
    _case_args(p)
    p.add_argument("--mode", choices=["eager", "cudagraph"], required=True)


# -- case construction --------------------------------------------------------
def _workload(args: argparse.Namespace):
    from bench.workloads import ATTN_OPS, BLOCK_OP, Grid, expand

    phase = args.phase or ("prefill" if args.op == "attn_prefill" else "decode")
    grid = Grid(
        name="profile",
        phases=[phase],
        batch_sizes=[args.batch],
        input_lens=[args.input_len],
        output_lens=[args.output_len],
        decode_kv_points="start_mid_end",
        spec_tokens=args.spec_tokens,
        dtype=args.dtype,
        kv_dtype=args.kv_dtype or args.dtype,
        page_size=args.page_size,
        max_prefill_tokens=1 << 62,
    )
    wls = expand(grid, args.op)
    if not wls:
        raise SystemExit(f"no {phase} workload for op {args.op}")
    if args.op in ATTN_OPS + (BLOCK_OP,) and phase == "decode":
        wls.sort(key=lambda w: w.kv_len)
        return wls[{"start": 0, "mid": len(wls) // 2, "end": -1}[args.kv_point]]
    return wls[0]


def _case(args: argparse.Namespace):
    import torch

    from bench.cli import _kv
    from bench.models import REGISTRY
    from bench.ops import adapters_for
    from bench.run import _variants, build_problem, make_ctx, runner_factory

    model = REGISTRY.get(args.model)
    vs = _variants(model, args.op, [args.variant] if args.variant else None)
    variant = vs[0]
    wl = _workload(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ctx = make_ctx(model, variant, args.op, wl, device, args.moe_expert_cap)
    adapters = adapters_for(args.op, [args.backend])
    if not adapters:
        raise SystemExit(f"backend {args.backend!r} does not implement {args.op}")
    a = adapters[0]
    reason = a.check(ctx)
    if reason:
        raise SystemExit(f"cannot profile: {reason}")
    p = build_problem(ctx, args.seed)
    make = runner_factory(ctx, a, p, args.block_layers,
                          {k: str(v) for k, v in _kv(args.block_impl).items()}, args.seed)
    return ctx, a, p, make


def _case_name(ctx, backend: str, mode: str) -> str:
    return f"{ctx.model.name}.{ctx.op}.{ctx.variant.name}.{backend}.{ctx.wl.key()}.{mode}".replace(
        "/", "_"
    ).replace(" ", "")


def _callable(ctx, make, mode: str, warmup: int):
    """Return the per-iteration callable (eager fn or graph.replay) after warmup."""
    import torch

    from bench.timing import TimingConfig, capture_graph

    runner = make(mode == "cudagraph")
    if mode == "cudagraph":
        if not runner.graph_safe:
            raise SystemExit(f"GRAPH_UNSUPPORTED: {runner.graph_reason}")
        graph, _, info = capture_graph(runner.fn, TimingConfig(graph_warmup=3), ctx.device)
        fn = graph.replay
        keep = graph
    else:
        fn, keep, info = runner.fn, None, None
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)
    return fn, keep, info


# -- in-process target (runs under nsys) --------------------------------------
def target_main(args: argparse.Namespace) -> int:
    import torch

    ctx, a, p, make = _case(args)
    fn, keep, _ = _callable(ctx, make, args.mode, args.warmup)
    name = _case_name(ctx, a.backend, args.mode)
    cuda = ctx.device.type == "cuda"
    if cuda:
        torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        for i in range(args.iters):
            if cuda:
                torch.cuda.nvtx.range_push(f"{args.mode}/iter{i}")
            fn()
            if cuda:
                torch.cuda.nvtx.range_pop()
    if cuda:
        torch.cuda.synchronize(ctx.device)
        torch.cuda.cudart().cudaProfilerStop()
    print(f"profiled {name} x{args.iters}")
    del keep
    return 0


def _target_argv(args: argparse.Namespace, mode: str) -> List[str]:
    out = ["_profile-target", "--mode", mode]
    for k in ("model", "op", "backend", "variant", "phase", "batch", "input_len", "output_len",
              "kv_point", "spec_tokens", "dtype", "kv_dtype", "page_size", "device", "iters",
              "warmup", "seed", "moe_expert_cap", "block_layers"):
        v = getattr(args, k)
        if v is not None:
            out += [f"--{k.replace('_', '-')}", str(v)]
    if args.block_impl:
        out += ["--block-impl", *args.block_impl]
    return out


# -- nsys ---------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v: Any) -> float:
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return 0.0


def summarize_nsys(base: Path, iters: int) -> Dict[str, Any]:
    kern = _read_csv(Path(f"{base}_cuda_gpu_kern_sum.csv"))
    nvtx = _read_csv(Path(f"{base}_nvtx_sum.csv"))
    api = _read_csv(Path(f"{base}_cuda_api_sum.csv"))
    top = []
    total_ns = 0.0
    launches = 0.0
    for r in kern:
        t = _num(r.get("Total Time (ns)"))
        n = _num(r.get("Instances"))
        total_ns += t
        launches += n
        top.append({"name": r.get("Name", "")[:200], "total_ns": t, "instances": int(n),
                    "avg_ns": _num(r.get("Avg (ns)")), "pct": _num(r.get("Time (%)"))})
    top.sort(key=lambda x: -x["total_ns"])
    iter_ranges = [r for r in nvtx if "iter" in r.get("Range", "")]
    nvtx_avg = (sum(_num(r.get("Avg (ns)")) * _num(r.get("Instances")) for r in iter_ranges)
                / max(1.0, sum(_num(r.get("Instances")) for r in iter_ranges))) if iter_ranges else 0.0
    api_top = sorted(
        ({"name": r.get("Name", ""), "total_ns": _num(r.get("Total Time (ns)")),
          "calls": int(_num(r.get("Num Calls")))} for r in api),
        key=lambda x: -x["total_ns"],
    )[:10]
    return {
        "metrics": {
            "kernels_per_iter": launches / max(1, iters),
            "gpu_kernel_us_per_iter": total_ns / max(1, iters) / 1e3,
            "nvtx_iter_us": nvtx_avg / 1e3,
        },
        "top_kernels": top[:15],
        "top_cuda_api": api_top,
    }


def _run_nsys(args: argparse.Namespace, mode: str, out_dir: Path, name: str) -> Tuple[bool, Dict]:
    nsys = shutil.which(args.nsys) or (args.nsys if Path(args.nsys).exists() else None)
    if not nsys:
        return False, {"message": f"nsys not found ({args.nsys}); install Nsight Systems or set NSYS"}
    base = out_dir / name
    cmd = [
        nsys, "profile", "-o", str(base), "--force-overwrite", "true",
        "--trace=cuda,nvtx,osrt,cudnn,cublas",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--sample=none", "--cpuctxsw=none",
    ]
    if mode == "cudagraph":
        cmd.append("--cuda-graph-trace=node")
    cmd += shlex.split(args.nsys_extra)
    cmd += [sys.executable, "-m", "bench", *_target_argv(args, mode)]
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    rep = Path(f"{base}.nsys-rep")
    if r.returncode != 0 or not rep.exists():
        return False, {"message": f"nsys profile failed (exit {r.returncode})"}
    stats = [nsys, "stats", "--report", "nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum",
             "--format", "csv", "--output", str(base), "--force-export=true", str(rep)]
    print("$", " ".join(shlex.quote(c) for c in stats), flush=True)
    subprocess.run(stats, cwd=ROOT, stdout=subprocess.DEVNULL)
    summary = summarize_nsys(base, args.iters)
    artifacts = {"nsys_rep": str(rep)}
    for rpt in ("nvtx_sum", "cuda_gpu_kern_sum", "cuda_api_sum"):
        pth = Path(f"{base}_{rpt}.csv")
        if pth.exists():
            artifacts[f"nsys_{rpt}"] = str(pth)
    return True, {"artifacts": artifacts, **summary}


# -- torch.profiler -----------------------------------------------------------
def _run_torch(args: argparse.Namespace, mode: str, out_dir: Path, name: str) -> Tuple[bool, Dict]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    ctx, a, p, make = _case(args)
    fn, keep, _ = _callable(ctx, make, mode, args.warmup)
    acts = [ProfilerActivity.CPU]
    if ctx.device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with torch.no_grad(), profile(activities=acts, record_shapes=True, with_stack=False) as prof:
        for i in range(args.iters):
            with record_function(f"{mode}/iter{i}"):
                fn()
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
    trace = out_dir / f"{name}.torch.json"
    prof.export_chrome_trace(str(trace))
    rows = []
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", None)
        if dev is None:
            dev = getattr(e, "self_cuda_time_total", 0.0)
        rows.append({
            "name": e.key,
            "calls": e.count,
            "self_cpu_us": e.self_cpu_time_total,
            "cpu_total_us": e.cpu_time_total,
            "self_device_us": dev,
        })
    rows.sort(key=lambda r: (-r["self_device_us"], -r["self_cpu_us"]))
    table = out_dir / f"{name}.torch_ops.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    del keep
    dev_rows = [r for r in rows if r["self_device_us"] > 0 and "iter" not in r["name"]]
    return True, {
        "artifacts": {"torch_trace": str(trace), "torch_ops_csv": str(table)},
        "metrics": {
            "device_us_per_iter": sum(r["self_device_us"] for r in dev_rows) / max(1, args.iters),
            "kernels_per_iter": sum(r["calls"] for r in dev_rows) / max(1, args.iters),
        },
        "top_kernels": [
            {"name": r["name"][:200], "total_us": r["self_device_us"], "instances": r["calls"]}
            for r in (dev_rows or rows)[:15]
        ],
    }


def main(args: argparse.Namespace) -> int:
    from bench import hardware
    from bench.schema import Status, Trace, append_traces

    ctx, a, _, _ = _case(args)  # validate early (unsupported -> exit)
    modes = ["eager", "cudagraph"] if args.mode == "both" else [args.mode]
    if ctx.device.type != "cuda":
        modes = [m for m in modes if m == "eager"]
    tools = ["nsys", "torch"] if args.tool == "both" else [args.tool]
    out_dir = Path(args.out_dir) / hardware.probe().gpu.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    hw, libs = hardware.probe(), hardware.lib_versions()
    rc = 0
    for mode in modes:
        name = _case_name(ctx, a.backend, mode)
        artifacts: Dict[str, str] = {}
        meta: Dict[str, Any] = {"iters": args.iters}
        mets: Dict[str, float] = {}
        msgs = []
        for tool in tools:
            if tool == "nsys" and ctx.device.type != "cuda":
                msgs.append("nsys skipped (no CUDA)")
                continue
            try:
                ok, res = (_run_nsys if tool == "nsys" else _run_torch)(args, mode, out_dir, name)
            except SystemExit as e:
                ok, res = False, {"message": str(e)}
            if not ok:
                msgs.append(f"{tool}: {res.get('message')}")
                rc = 1
                continue
            artifacts.update(res.get("artifacts", {}))
            mets.update({f"{tool}_{k}": v for k, v in res.get("metrics", {}).items()})
            meta[f"{tool}_top_kernels"] = res.get("top_kernels", [])
            if "top_cuda_api" in res:
                meta["nsys_top_cuda_api"] = res["top_cuda_api"]
        t = Trace(
            level="block" if ctx.op == "decoder_layer" else "op",
            model=ctx.model.name, op=ctx.op, variant=ctx.variant.name, backend=a.backend,
            mode=mode, workload=ctx.wl, status=Status.PASSED if artifacts else Status.RUNTIME_ERROR,
            hardware=hw, libs=libs, kind="profile", metrics=mets, artifacts=artifacts, meta=meta,
            message="; ".join(msgs),
        )
        append_traces(Path(args.traces), [t])
        print(f"[{mode}] artifacts:")
        for k, v in artifacts.items():
            print(f"  {k:<22} {v}")
        if mets:
            print("  " + json.dumps({k: round(v, 3) for k, v in mets.items()}))
        for m in msgs:
            print(f"  ! {m}")
    return rc
