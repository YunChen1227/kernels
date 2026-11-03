"""Command line: ``python -m bench {run,profile,e2e,report,list}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.schema import Mode


def _csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def _kv(items: Optional[List[str]]) -> Dict[str, Any]:
    """``a=1 b=x,y`` -> {"a": 1, "b": ["x", "y"]} (ints/floats/lists parsed)."""
    out: Dict[str, Any] = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = _parse_value(v.strip())
    return out


def _parse_value(v: str) -> Any:
    if "," in v:
        return [_parse_value(x) for x in v.split(",") if x]
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _section(cfg_path: Optional[str], name: str) -> Dict[str, Any]:
    if not cfg_path:
        return {}
    from bench.workloads import load_config_file

    return dict(load_config_file(Path(cfg_path)).get(name, {}) or {})


def _listish(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        return _csv(v)
    return [str(x) for x in v]


# -- run ----------------------------------------------------------------------
def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML/JSON with a `run:` section (per-hardware defaults)")
    p.add_argument("--model", help="comma list of registry names or HF config.json paths; 'all'")
    p.add_argument("--op", help="comma list; 'all' (default)")
    p.add_argument("--backend", help="comma list of backends (default: every applicable one)")
    p.add_argument("--variant", help="attention variants to keep, e.g. full,sliding")
    p.add_argument("--grid", help="builtin grid (tiny/smoke/decode/full) or YAML with `grid:`")
    p.add_argument("--set", nargs="*", metavar="K=V", help="grid overrides, e.g. batch_sizes=1,8")
    p.add_argument("--modes", help="eager,cudagraph (default both)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", help="trace JSONL (default traces/<gpu>_<time>.jsonl)")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warmup", type=int)
    p.add_argument("--iters", type=int)
    p.add_argument("--trials", type=int)
    p.add_argument("--no-cupti", action="store_true", help="use CUDA events instead of CUPTI")
    p.add_argument("--no-cold-l2", action="store_true")
    p.add_argument("--atol", type=float)
    p.add_argument("--rtol", type=float)
    p.add_argument("--matched-ratio", type=float)
    p.add_argument("--moe-expert-cap", type=int, help="limit #experts to fit memory (0 = all)")
    p.add_argument("--block-layers", type=int, help="decoder layers per block measurement")
    p.add_argument("--block-impl", nargs="*", metavar="K=V",
                   help="decoder_layer sub-kernel impls, e.g. rmsnorm=flashinfer moe=sglang_triton")
    p.add_argument("--dry-run", action="store_true", help="print the matrix without running")


def run_main(args: argparse.Namespace) -> int:
    from bench.models import REGISTRY
    from bench.ops import Tol
    from bench.run import ALL_OPS, RunConfig, default_out, run
    from bench.timing import TimingConfig
    from bench.workloads import load_grid

    c = _section(args.config, "run")
    models = _listish(args.model) or _listish(c.get("models")) or ["qwen3-8b"]
    if models == ["all"]:
        models = [n for n in REGISTRY.names() if not n.startswith("tiny")]
    models = [REGISTRY.get(m).name for m in models]  # registers config.json paths
    ops = _listish(args.op) or _listish(c.get("ops")) or ["all"]
    if ops == ["all"]:
        ops = list(ALL_OPS)
    unknown = [o for o in ops if o not in ALL_OPS]
    if unknown:
        print(f"unknown ops {unknown}; choose from {list(ALL_OPS)}", file=sys.stderr)
        return 2
    grid_overrides = dict(c.get("grid_overrides", {}) or {})
    grid_overrides.update(_kv(args.set))
    grid = load_grid(args.grid or c.get("grid", "smoke"), grid_overrides)
    t = dict(c.get("timing", {}) or {})
    for k in ("warmup", "iters", "trials"):
        if getattr(args, k) is not None:
            t[k] = getattr(args, k)
    if args.no_cupti:
        t["use_cupti"] = False
    if args.no_cold_l2:
        t["cold_l2"] = False
    tol = None
    if args.atol is not None or args.rtol is not None:
        tol = Tol(args.atol if args.atol is not None else 1e-2,
                  args.rtol if args.rtol is not None else 1e-2,
                  args.matched_ratio if args.matched_ratio is not None else 1.0)
    block_impl = dict(c.get("block_impl", {}) or {})
    block_impl.update(_kv(args.block_impl))
    cfg = RunConfig(
        models=models,
        ops=ops,
        grid=grid,
        out=Path(args.out or c.get("out") or default_out(args.tag or c.get("tag", ""))),
        backends=_listish(args.backend) or _listish(c.get("backends")),
        modes=_listish(args.modes) or _listish(c.get("modes")) or list(Mode.ALL),
        variants=_listish(args.variant) or _listish(c.get("variants")),
        timing=TimingConfig(**t),
        device=args.device or c.get("device", "cuda"),
        seed=args.seed if args.seed is not None else int(c.get("seed", 0)),
        moe_expert_cap=args.moe_expert_cap if args.moe_expert_cap is not None
        else int(c.get("moe_expert_cap", 0)),
        block_layers=args.block_layers or int(c.get("block_layers", 1)),
        block_impl={k: str(v) for k, v in block_impl.items()},
        tol=tol,
        dry_run=args.dry_run,
        tag=args.tag or c.get("tag", ""),
    )
    counts = run(cfg)
    if not cfg.dry_run:
        print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing ran")
        print(f"traces: {cfg.out}")
        print(f"report: python -m bench report {cfg.out}")
    return 0


# -- list ---------------------------------------------------------------------
def add_list_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("what", nargs="?", default="models", choices=["models", "backends", "support"])
    p.add_argument("--model", help="for `support`: models to check (default: all)")
    p.add_argument("--device", default="cuda")


def list_main(args: argparse.Namespace) -> int:
    import torch

    from bench import hardware
    from bench.models import REGISTRY
    from bench.ops import Ctx, all_adapters, attn_geom
    from bench.run import ALL_OPS, _variants, ops_for
    from bench.schema import WorkloadRecord

    if args.what == "models":
        for n in REGISTRY.names():
            m = REGISTRY.get(n)
            vs = ", ".join(f"{v.name}{sorted(v.features) or ''}" for v in m.attn_variants())
            moe = f" moe={m.moe.num_experts}x top{m.moe.topk}" if m.moe else ""
            print(f"{n:<26} {m.family:<9} {m.attn_type:<4} hq={m.num_q_heads:<3} "
                  f"hkv={m.kv_heads:<3} d={m.qk_head_dim}/{m.vo_head_dim}{moe}  [{vs}]")
        return 0
    if args.what == "backends":
        for a in all_adapters():
            attn = any(o.startswith("attn") for o in a.ops)
            kinds = f"kinds={','.join(a.kinds):<8} features={','.join(sorted(a.features)) or '-'} " if attn else ""
            print(f"{a.backend:<18} ops={','.join(a.ops):<28} {kinds}"
                  f"requires={','.join(a.requires) or '-'}")
        return 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sm = hardware.sm()
    hw = hardware.probe()
    print(f"support matrix on {hw.label} (device={device})")
    names = _csv(args.model) or [n for n in REGISTRY.names() if not n.startswith("tiny")]
    for name in names:
        m = REGISTRY.get(name)
        print(f"\n== {m.name}")
        for op in ops_for(m, list(ALL_OPS)):
            for v in _variants(m, op, None):
                for phase in (["decode"] if op != "attn_prefill" else ["prefill"]):
                    wl = WorkloadRecord(phase, 1, 1024, 128, 1024, 1)
                    geom = attn_geom(m, v, phase) if op.startswith("attn") or op == "decoder_layer" else None
                    ctx = Ctx(m, v, op, wl, device, sm, geom)
                    cells = []
                    for a in all_adapters():
                        tgt = "attn_decode" if op == "decoder_layer" else op
                        if tgt not in a.ops:
                            continue
                        r = a.check(ctx)
                        cells.append(f"{a.backend}:{'OK' if r is None else 'no'}")
                        if r is not None:
                            cells[-1] += f"({r})"
                    print(f"  {op:<13} {v.name:<12} " + "  ".join(cells))
    return 0


# -- entry --------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    from bench import e2e, profile, report

    ap = argparse.ArgumentParser(prog="python -m bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="op/block-level correctness + eager/cudagraph timing"))
    profile.add_args(sub.add_parser("profile", help="nsys / torch.profiler capture of one case"))
    profile.add_target_args(sub.add_parser("_profile-target"))
    e2e.add_args(sub.add_parser("e2e", help="end-to-end sglang bench_one_batch eager vs cudagraph"))
    report.add_args(sub.add_parser("report", help="HTML/Markdown report per hardware"))
    add_list_args(sub.add_parser("list", help="models / backends / support matrix"))
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run_main(args)
    if args.cmd == "profile":
        return profile.main(args)
    if args.cmd == "_profile-target":
        return profile.target_main(args)
    if args.cmd == "e2e":
        return e2e.main(args)
    if args.cmd == "report":
        return report.main(args)
    return list_main(args)


if __name__ == "__main__":
    sys.exit(main())
