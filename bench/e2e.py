"""End-to-end eager vs CUDA graph comparison through sglang's one-batch benchmark.

For every (model, attention backend, TP) the benchmark runs twice -- once with
``--disable-cuda-graph`` (eager) and once with CUDA graphs -- over the
batch_size x input_len x output_len product. Results become ``level="e2e"``
traces in the same JSONL format as op/block runs; ``graph_speedup`` is the
eager / cudagraph ratio of the median decode step latency.

Hardware-specific model lists live in YAML (``bench/configs/<gpu>.yaml``)::

    e2e:
      batch_sizes: [1, 16, 64]
      input_lens: [1024, 4096]
      output_lens: [256]
      modes: [eager, cudagraph]
      dummy_weights: true
      models:
        - name: qwen3-8b
          model_path: Qwen/Qwen3-8B
          tp: 1
          attention_backends: [flashinfer, fa3, triton]
          extra_args: ["--mem-fraction-static", "0.8"]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
_MODULES = ("sglang.benchmark.one_batch", "sglang.bench_one_batch")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML/JSON with an `e2e:` section")
    p.add_argument("--model", help="registry name used for grouping in the report")
    p.add_argument("--model-path", help="HF id or local path passed to sglang")
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--attention-backend", help="comma list, e.g. flashinfer,fa3,triton")
    p.add_argument("--batch", help="comma list of batch sizes")
    p.add_argument("--input-len", help="comma list")
    p.add_argument("--output-len", help="comma list")
    p.add_argument("--modes", help="eager,cudagraph (default both)")
    p.add_argument("--real-weights", action="store_true",
                   help="load checkpoint weights (default: --load-format dummy)")
    p.add_argument("--nsys", action="store_true", help="wrap each run in nsys profile")
    p.add_argument("--torch-profile", action="store_true", help="pass --profile to sglang")
    p.add_argument("--extra", default="", help="extra args appended to every sglang command")
    p.add_argument("--out", help="trace JSONL (default traces/<gpu>_e2e_<time>.jsonl)")
    p.add_argument("--out-dir", default="reports/e2e", help="logs / raw results / profiles")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--dry-run", action="store_true")


def _ints(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, str):
        return [int(x) for x in v.split(",") if x.strip()]
    if isinstance(v, int):
        return [v]
    return [int(x) for x in v]


def _strs(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [str(x) for x in v]


def sglang_module() -> Optional[str]:
    for m in _MODULES:
        try:
            if importlib.util.find_spec(m) is not None:
                return m
        except (ImportError, ValueError):
            continue
    return None


def _jobs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from bench.workloads import load_config_file

    c = dict(load_config_file(Path(args.config)).get("e2e", {}) or {}) if args.config else {}
    common = {
        "batch_sizes": _ints(args.batch) or _ints(c.get("batch_sizes")) or [1, 16, 64],
        "input_lens": _ints(args.input_len) or _ints(c.get("input_lens")) or [1024],
        "output_lens": _ints(args.output_len) or _ints(c.get("output_lens")) or [256],
        "modes": _strs(args.modes) or _strs(c.get("modes")) or ["eager", "cudagraph"],
        "dummy": not args.real_weights and c.get("dummy_weights", True),
        "extra": shlex.split(args.extra) + [str(x) for x in c.get("extra_args", []) or []],
        "dtype": c.get("dtype", "bfloat16"),
    }
    models: List[Dict[str, Any]] = []
    if args.model_path or args.model:
        from bench.models import REGISTRY

        name = args.model or Path(args.model_path).name
        path = args.model_path
        if path is None:
            spec = REGISTRY.get(name)
            path = spec.hf_id or name
        models.append({
            "name": name,
            "model_path": path,
            "tp": args.tp or 1,
            "attention_backends": _strs(args.attention_backend) or ["flashinfer"],
            "extra_args": [],
        })
    else:
        for m in c.get("models", []) or []:
            m = dict(m)
            if args.attention_backend:
                m["attention_backends"] = _strs(args.attention_backend)
            if args.tp:
                m["tp"] = args.tp
            models.append(m)
    if not models:
        raise SystemExit("no models: pass --model/--model-path or a --config with e2e.models")
    return models, common


def _command(module: str, m: Dict[str, Any], backend: str, mode: str, common: Dict[str, Any],
             result_file: Path, run_name: str) -> List[str]:
    cmd = [
        sys.executable, "-m", module,
        "--model-path", str(m["model_path"]),
        "--tp-size", str(m.get("tp", 1)),
        "--attention-backend", backend,
        "--batch-size", *map(str, common["batch_sizes"]),
        "--input-len", *map(str, common["input_lens"]),
        "--output-len", *map(str, common["output_lens"]),
        "--result-filename", str(result_file),
        "--run-name", run_name,
        "--trust-remote-code",
    ]
    if common["dummy"]:
        cmd += ["--load-format", "dummy"]
    if mode == "eager":
        cmd.append("--disable-cuda-graph")
    else:
        cmd += ["--cuda-graph-max-bs", str(max(common["batch_sizes"]))]
    cmd += [str(x) for x in m.get("extra_args", []) or []]
    cmd += common["extra"]
    return cmd


def _first(d: Dict[str, Any], *names: str) -> Optional[float]:
    for n in names:
        v = d.get(n)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def parse_results(path: Path, run_name: str) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("run_name", run_name) != run_name:
            continue
        out.append({
            "batch_size": int(d.get("batch_size", 0)),
            "input_len": int(d.get("input_len", 0)),
            "output_len": int(d.get("output_len", 0)),
            "prefill_ms": (_first(d, "prefill_latency") or 0.0) * 1e3,
            "prefill_tok_s": _first(d, "prefill_throughput"),
            "decode_ms": (_first(d, "median_decode_latency", "decode_latency") or 0.0) * 1e3,
            "decode_tok_s": _first(d, "median_decode_throughput", "decode_throughput"),
            "total_s": _first(d, "total_latency", "overall_latency"),
            "overall_tok_s": _first(d, "overall_throughput", "total_throughput"),
            "raw": d,
        })
    return out


def main(args: argparse.Namespace) -> int:
    from bench import hardware
    from bench.schema import Latency, Level, Status, Trace, WorkloadRecord, append_traces

    models, common = _jobs(args)
    module = sglang_module()
    hw, libs = hardware.probe(), hardware.lib_versions()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu = hw.gpu.replace(" ", "_")
    out = Path(args.out or f"traces/{gpu}_e2e_{stamp}.jsonl")
    work = Path(args.out_dir) / gpu / stamp
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
    if module is None and not args.dry_run:
        print("sglang is not installed (looked for " + ", ".join(_MODULES) + ")", file=sys.stderr)
        return 2
    module = module or _MODULES[0]
    nsys = shutil.which(os.environ.get("NSYS", "nsys")) if args.nsys else None
    if args.nsys and not nsys:
        print("warning: nsys not found, running without it", file=sys.stderr)
    rc = 0
    for m in models:
        for backend in m.get("attention_backends", ["flashinfer"]):
            per_mode: Dict[str, Dict[Tuple[int, int, int], Dict[str, Any]]] = {}
            info: Dict[str, Dict[str, Any]] = {}
            for mode in common["modes"]:
                run_name = f"{m['name']}.tp{m.get('tp', 1)}.{backend}.{mode}"
                result_file = work / f"{run_name}.jsonl"
                log = work / f"{run_name}.log"
                cmd = _command(module, m, backend, mode, common, result_file, run_name)
                if args.torch_profile:
                    cmd.append("--profile")
                artifacts = {"log": str(log), "raw_results": str(result_file)}
                if nsys:
                    rep = work / run_name
                    cmd = [nsys, "profile", "-o", str(rep), "--force-overwrite", "true",
                           "--trace=cuda,nvtx,osrt,cublas", "--sample=none",
                           *(["--cuda-graph-trace=node"] if mode == "cudagraph" else []), *cmd]
                    artifacts["nsys_rep"] = f"{rep}.nsys-rep"
                print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
                if args.dry_run:
                    continue
                env = dict(os.environ)
                if args.torch_profile:
                    env.setdefault("SGLANG_TORCH_PROFILER_DIR", str(work / f"{run_name}.torch"))
                    artifacts["torch_trace_dir"] = env["SGLANG_TORCH_PROFILER_DIR"]
                status, msg = Status.PASSED, ""
                try:
                    with log.open("w", encoding="utf-8") as lf:
                        r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                           env=env, timeout=args.timeout)
                    if r.returncode != 0:
                        status = Status.RUNTIME_ERROR
                        msg = f"exit {r.returncode}: " + _tail(log)
                except subprocess.TimeoutExpired:
                    status, msg = Status.TIMEOUT, f"timeout after {args.timeout}s"
                rows = parse_results(result_file, run_name)
                if status == Status.PASSED and not rows:
                    status, msg = Status.RUNTIME_ERROR, "no results written: " + _tail(log)
                if "out of memory" in msg.lower():
                    status = Status.OOM
                info[mode] = {"status": status, "message": msg, "artifacts": artifacts,
                              "cmd": " ".join(shlex.quote(c) for c in cmd)}
                per_mode[mode] = {(r["batch_size"], r["input_len"], r["output_len"]): r for r in rows}
            if args.dry_run:
                continue
            keys = sorted({k for d in per_mode.values() for k in d} or {
                (b, i, o) for b in common["batch_sizes"] for i in common["input_lens"]
                for o in common["output_lens"]})
            traces = []
            for mode in common["modes"]:
                inf = info.get(mode, {})
                for b, il, ol in keys:
                    r = per_mode.get(mode, {}).get((b, il, ol))
                    wl = WorkloadRecord(phase="e2e", batch_size=b, input_len=il, output_len=ol,
                                        kv_len=il + ol, q_len=1, dtype=common["dtype"],
                                        kv_dtype=common["dtype"], page_size=0)
                    t = Trace(level=Level.E2E, model=m["name"], op="e2e",
                              variant=f"tp{m.get('tp', 1)}", backend=backend, mode=mode,
                              workload=wl, status=inf.get("status", Status.RUNTIME_ERROR),
                              hardware=hw, libs=libs, artifacts=inf.get("artifacts", {}),
                              meta={"cmd": inf.get("cmd", ""), "model_path": str(m["model_path"])},
                              message=inf.get("message", ""))
                    if r is None:
                        if t.status == Status.PASSED:
                            t.status, t.message = Status.RUNTIME_ERROR, "missing result row"
                    else:
                        t.latency = Latency.from_samples([r["decode_ms"]], "sglang")
                        t.metrics = {k: v for k, v in r.items()
                                     if k not in ("raw", "batch_size", "input_len", "output_len")
                                     and isinstance(v, (int, float))}
                        t.meta["raw"] = r["raw"]
                        other = per_mode.get("eager", {}).get((b, il, ol))
                        if mode == "cudagraph" and other and r["decode_ms"] > 0:
                            t.metrics["graph_speedup"] = other["decode_ms"] / r["decode_ms"]
                    if t.status != Status.PASSED:
                        rc = 1
                    traces.append(t)
                    lat = f"decode {t.latency.median_ms:8.3f} ms" if t.latency else ""
                    gs = t.metrics.get("graph_speedup")
                    print(f"[{t.status:<15}] {m['name']:<20} {backend:<12} {mode:<9} "
                          f"b{b} in{il} out{ol}  {lat}{f'  graph x{gs:.2f}' if gs else ''}"
                          f"{'  ' + t.message[:120] if t.message else ''}")
            append_traces(out, traces)
    if not args.dry_run:
        print(f"traces: {out}\nreport: python -m bench report {out}")
    return rc


def _tail(path: Path, n: int = 400) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-n:].replace("\n", " | ")
    except OSError:
        return ""
