"""HTML / Markdown reports from trace JSONL, one per hardware plus a cross-hardware index.

``python -m bench report traces/ [--baseline torch_ref] [--out-dir reports/bench]``

Per hardware:
  * backend comparison tables per (model, op, variant): eager and CUDA graph
    latency for every workload, speedup vs the baseline backend, best marked;
  * CUDA graph speedup (eager / graph) vs batch size, inline SVG;
  * end-to-end (sglang) eager vs CUDA graph tables;
  * support matrix with the SKIPPED / GRAPH_UNSUPPORTED / error reasons;
  * links to nsys / torch.profiler artifacts captured by ``bench profile``.
"""

from __future__ import annotations

import argparse
import html
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bench.schema import Mode, Status, Trace, iter_traces

ARTIFACT_LABELS = {
    "nsys_rep": "nsys",
    "nsys_cuda_gpu_kern_sum": "kern.csv",
    "nsys_nvtx_sum": "nvtx.csv",
    "nsys_cuda_api_sum": "api.csv",
    "torch_trace": "torch",
    "torch_ops_csv": "ops.csv",
    "torch_trace_dir": "torch",
    "log": "log",
}
COLORS = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2",
          "#97bbf5", "#9c6b4e", "#9498a0"]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="*", default=["traces"], help="JSONL files or directories")
    p.add_argument("--out-dir", default="reports/bench")
    p.add_argument("--baseline", nargs="*", default=["torch_ref"],
                   help="baseline backend, or per-op OP=BACKEND (e.g. attn_decode=flashinfer)")
    p.add_argument("--hardware", help="substring filter on the hardware label")
    p.add_argument("--model", help="comma list of models to include")
    p.add_argument("--format", default="html,md")
    p.add_argument("--all-runs", action="store_true",
                   help="keep every trace instead of the latest per key")


# -- loading ------------------------------------------------------------------
class Data:
    def __init__(self, traces: Iterable[Trace], latest: bool = True):
        self.measure: List[Trace] = []
        self.profiles: Dict[str, Dict[str, str]] = {}
        self.profile_meta: Dict[str, Dict[str, Any]] = {}
        best: Dict[str, Trace] = {}
        for t in traces:
            if t.kind == "profile":
                k = t.key()
                self.profiles.setdefault(k, {}).update(t.artifacts)
                self.profile_meta[k] = {**self.profile_meta.get(k, {}), **t.metrics}
                continue
            if not latest:
                self.measure.append(t)
                continue
            k = t.key()
            if k not in best or t.timestamp >= best[k].timestamp:
                best[k] = t
        if latest:
            self.measure = list(best.values())

    def by_hardware(self) -> Dict[str, List[Trace]]:
        out: Dict[str, List[Trace]] = defaultdict(list)
        for t in self.measure:
            out[t.hardware.label].append(t)
        return dict(sorted(out.items()))


def _baselines(items: List[str]) -> Tuple[str, Dict[str, str]]:
    default, per_op = "torch_ref", {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            per_op[k] = v
        else:
            default = it
    return default, per_op


# -- formatting helpers -------------------------------------------------------
def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "x"


def wl_label(t: Trace) -> str:
    w = t.workload
    if t.op == "e2e":
        return f"b{w.batch_size} in{w.input_len} out{w.output_len}"
    if w.phase == "prefill":
        if w.kv_len:
            return f"prefill b{w.batch_size} len{w.input_len}"
        return f"prefill T={w.batch_size * w.input_len}"
    if w.kv_len:
        q = f" q{w.q_len}" if w.q_len != 1 else ""
        return f"decode b{w.batch_size} kv{w.kv_len}{q}"
    return f"decode T={w.batch_size * w.q_len}"


def wl_sort(t: Trace) -> Tuple:
    w = t.workload
    return (w.phase, w.batch_size, w.input_len, w.kv_len, w.output_len, w.q_len, w.dtype)


def fmt_ms(ms: float) -> str:
    if ms >= 100:
        return f"{ms:.1f}ms"
    if ms >= 1:
        return f"{ms:.3f}ms"
    return f"{ms * 1e3:.1f}us"


_SHORT = {
    Status.SKIPPED: "skip",
    Status.GRAPH_UNSUPPORTED: "no-graph",
    Status.INCORRECT_NUMERICAL: "WRONG",
    Status.INCORRECT_SHAPE: "WRONG-shape",
    Status.INCORRECT_DTYPE: "WRONG-dtype",
    Status.RUNTIME_ERROR: "error",
    Status.OOM: "OOM",
    Status.TIMEOUT: "timeout",
}


def short_status_name(st: str) -> str:
    return _SHORT.get(st, st)


def short_status(t: Trace) -> str:
    return short_status_name(t.status)


def _ms(t: Optional[Trace]) -> Optional[float]:
    if t is None or t.status != Status.PASSED or not t.latency or not t.latency.n:
        return None
    return t.latency.median_ms


# -- table model --------------------------------------------------------------
class Group:
    """All traces for (model, level, op, variant) on one hardware."""

    def __init__(self, model: str, level: str, op: str, variant: str):
        self.model, self.level, self.op, self.variant = model, level, op, variant
        self.cells: Dict[Tuple[str, str, str], Trace] = {}  # (wl_key, backend, mode)
        self.wls: Dict[str, Trace] = {}
        self.backends: List[str] = []

    def add(self, t: Trace) -> None:
        wk = t.workload.key()
        self.cells[(wk, t.backend, t.mode)] = t
        if wk not in self.wls:
            self.wls[wk] = t
        if t.backend not in self.backends:
            self.backends.append(t.backend)

    def rows(self) -> List[str]:
        return [k for k, _ in sorted(self.wls.items(), key=lambda kv: wl_sort(kv[1]))]

    def ordered_backends(self, baseline: str) -> List[str]:
        """Backends with at least one non-SKIPPED trace (all-skipped ones live in the support matrix)."""
        ran = {b for (_, b, _), t in self.cells.items() if t.status != Status.SKIPPED}
        bs = sorted(ran) or sorted(self.backends)
        if baseline in bs:
            bs.remove(baseline)
            bs.insert(0, baseline)
        return bs

    def title(self) -> str:
        v = "" if self.variant in ("-", "") else f" / {self.variant}"
        return f"{self.model} · {self.op}{v}"


def build_groups(traces: List[Trace]) -> Dict[str, List[Group]]:
    groups: Dict[Tuple[str, str, str, str], Group] = {}
    for t in traces:
        k = (t.model, t.level, t.op, t.variant)
        if k not in groups:
            groups[k] = Group(*k)
        groups[k].add(t)
    order = {"op": 0, "block": 1, "e2e": 2}
    op_order = ["attn_decode", "attn_prefill", "decoder_layer", "rmsnorm", "rope", "gemm_qkv",
                "gemm_o", "mlp", "moe", "e2e"]
    by_model: Dict[str, List[Group]] = defaultdict(list)
    for k in sorted(groups, key=lambda k: (k[0], order.get(k[1], 9),
                                           op_order.index(k[2]) if k[2] in op_order else 99, k[3])):
        by_model[k[0]].append(groups[k])
    return dict(by_model)


def best_backends(g: Group, wk: str, mode: str) -> Optional[str]:
    best, best_ms = None, None
    for b in g.backends:
        ms = _ms(g.cells.get((wk, b, mode)))
        if ms is not None and (best_ms is None or ms < best_ms):
            best, best_ms = b, ms
    return best


# -- HTML ---------------------------------------------------------------------
CSS = """
body{font:13px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1b1f24}
h1{font-size:22px}h2{font-size:18px;margin-top:32px;border-bottom:1px solid #ddd}
h3{font-size:15px;margin:22px 0 6px}
table{border-collapse:collapse;margin:6px 0 14px}td,th{border:1px solid #d8dee4;padding:3px 7px;
text-align:right;vertical-align:top;white-space:nowrap}th{background:#f6f8fa}td.l,th.l{text-align:left}
.best{background:#dafbe1;font-weight:600}.bad{background:#ffebe9}.skip{color:#8c959f}
.ng{color:#9a6700}.mut{color:#57606a;font-size:11px}.g{color:#0969da}
.small{font-size:11px}code{background:#f6f8fa;padding:0 3px}
details{margin:4px 0}summary{cursor:pointer;color:#57606a}
"""


def _link(path: str, label: str, base: Path) -> str:
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        rel = path
    return f'<a href="{html.escape(rel)}">{html.escape(label)}</a>'


def _artifacts_html(t: Trace, data: Data, base: Path) -> str:
    arts = dict(data.profiles.get(t.key(), {}))
    arts.update(t.artifacts)
    links = [_link(v, ARTIFACT_LABELS.get(k, k), base) for k, v in arts.items() if v]
    return f'<div class="mut">{" ".join(links)}</div>' if links else ""


def _cell_html(t: Optional[Trace], best: bool, base_ms: Optional[float], data: Data,
               out_dir: Path, graph: bool) -> str:
    if t is None:
        return '<td class="skip">-</td>'
    if t.status != Status.PASSED or _ms(t) is None:
        cls = "skip" if t.status == Status.SKIPPED else (
            "ng" if t.status == Status.GRAPH_UNSUPPORTED else "bad")
        return (f'<td class="{cls}" title="{html.escape(t.message or t.status)}">'
                f"{short_status(t)}</td>")
    ms = _ms(t)
    tip = [f"median {ms:.4f} ms", f"p10 {t.latency.p10_ms:.4f}", f"p90 {t.latency.p90_ms:.4f}",
           f"n={t.latency.n} {t.latency.timer}"]
    if t.host:
        tip.append(f"host enqueue {t.host.enqueue_ms:.4f} ms, wall {t.host.wall_ms:.4f} ms")
    if t.graph:
        tip.append(f"capture {t.graph.capture_ms:.1f} ms, +{t.graph.mem_mb:.1f} MB, "
                   f"{t.graph.num_nodes} nodes")
    for k in ("tflops", "gbps"):
        if k in t.metrics:
            tip.append(f"{k} {t.metrics[k]:.2f}")
    if t.correctness:
        tip.append(f"max abs err {t.correctness.max_abs_error:.2e}")
    for k, v in sorted(data.profile_meta.get(t.key(), {}).items()):
        tip.append(f"{k} {v:.2f}")
    parts = [fmt_ms(ms)]
    if base_ms:
        parts.append(f'<span class="mut">×{base_ms / ms:.2f}</span>')
    if graph and "graph_speedup" in t.metrics:
        parts.append(f'<span class="g">g×{t.metrics["graph_speedup"]:.2f}</span>')
    body = " ".join(parts) + _artifacts_html(t, data, out_dir)
    return f'<td class="{"best" if best else ""}" title="{html.escape(chr(10).join(tip))}">{body}</td>'


def group_table_html(g: Group, baseline: str, data: Data, out_dir: Path) -> str:
    backends = g.ordered_backends(baseline)
    modes = [m for m in Mode.ALL if any(k[2] == m for k in g.cells)]
    head = '<tr><th class="l" rowspan="2">workload</th>' + "".join(
        f'<th colspan="{len(backends)}">{m}</th>' for m in modes) + "</tr>"
    head += "<tr>" + "".join(f"<th>{html.escape(b)}</th>" for _ in modes for b in backends) + "</tr>"
    rows = []
    for wk in g.rows():
        cells = [f'<td class="l">{html.escape(wl_label(g.wls[wk]))}</td>']
        for m in modes:
            best = best_backends(g, wk, m)
            base_ms = _ms(g.cells.get((wk, baseline, Mode.EAGER)))
            for b in backends:
                cells.append(_cell_html(g.cells.get((wk, b, m)), b == best, base_ms, data,
                                        out_dir, m == Mode.CUDAGRAPH))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def svg_lines(series: Dict[str, List[Tuple[float, float]]], *, xlabel: str, ylabel: str,
              logx: bool = True, ref_y: Optional[float] = 1.0, w: int = 560, h: int = 260) -> str:
    import math

    pts = [p for s in series.values() for p in s]
    if not pts:
        return ""
    fx = (lambda x: math.log2(max(x, 1e-9))) if logx else (lambda x: x)
    xs = [fx(x) for x, _ in pts]
    ys = [y for _, y in pts] + ([ref_y] if ref_y is not None else [])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(0.0, min(ys)), max(ys) * 1.1
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    L, R, T, B = 48, 140, 14, 34

    def sx(x):
        return L + (fx(x) - x0) / (x1 - x0) * (w - L - R)

    def sy(y):
        return h - B - (y - y0) / (y1 - y0) * (h - T - B)

    out = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" '
           f'style="font:11px sans-serif">']
    out.append(f'<line x1="{L}" y1="{h - B}" x2="{w - R}" y2="{h - B}" stroke="#888"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h - B}" stroke="#888"/>')
    for x in sorted({x for x, _ in pts}):
        out.append(f'<text x="{sx(x):.1f}" y="{h - B + 14}" text-anchor="middle">{x:g}</text>')
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        out.append(f'<text x="{L - 4}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.2f}</text>')
        out.append(f'<line x1="{L}" y1="{sy(y):.1f}" x2="{w - R}" y2="{sy(y):.1f}" stroke="#eee"/>')
    if ref_y is not None:
        out.append(f'<line x1="{L}" y1="{sy(ref_y):.1f}" x2="{w - R}" y2="{sy(ref_y):.1f}" '
                   f'stroke="#999" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{(L + w - R) / 2}" y="{h - 4}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="12" y="{T + (h - T - B) / 2}" transform="rotate(-90 12 {T + (h - T - B) / 2})"'
               f' text-anchor="middle">{ylabel}</text>')
    for i, (name, s) in enumerate(sorted(series.items())):
        c = COLORS[i % len(COLORS)]
        s = sorted(s)
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(s))
        out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
        for x, y in s:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{c}">'
                       f"<title>{html.escape(name)} x={x:g} y={y:.3f}</title></circle>")
        ly = T + 6 + i * 15
        out.append(f'<rect x="{w - R + 10}" y="{ly - 8}" width="10" height="10" fill="{c}"/>')
        out.append(f'<text x="{w - R + 24}" y="{ly + 1}">{html.escape(name[:18])}</text>')
    out.append("</svg>")
    return "".join(out)


def graph_speedup_series(g: Group) -> Dict[str, List[Tuple[float, float]]]:
    """backend -> [(batch, mean graph_speedup over the other workload dims)]."""
    acc: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (wk, b, m), t in g.cells.items():
        if m == Mode.CUDAGRAPH and t.status == Status.PASSED and "graph_speedup" in t.metrics:
            acc[b][t.workload.batch_size].append(t.metrics["graph_speedup"])
    return {b: [(x, sum(v) / len(v)) for x, v in d.items()] for b, d in acc.items()}


def support_rows(traces: List[Trace]) -> Tuple[List[str], List[Tuple[str, Dict[str, Tuple]]]]:
    """[(row label, {backend: (n_pass, n_total, worst_status, message)})]."""
    acc: Dict[str, Dict[str, List[Trace]]] = defaultdict(lambda: defaultdict(list))
    backends = set()
    for t in traces:
        v = "" if t.variant in ("-", "") else f" / {t.variant}"
        acc[f"{t.model} · {t.op}{v}"][t.backend].append(t)
        backends.add(t.backend)
    rows = []
    for label in sorted(acc):
        d = {}
        for b, ts in acc[label].items():
            ok = sum(1 for t in ts if t.status == Status.PASSED)
            bad = [t for t in ts if t.status != Status.PASSED]
            if bad:
                st, n = Counter(t.status for t in bad).most_common(1)[0]
                msg = next((t.message for t in bad if t.status == st and t.message), "")
            else:
                st, msg = Status.PASSED, ""
            d[b] = (ok, len(ts), st, msg)
        rows.append((label, d))
    return sorted(backends), rows


def render_html(hw: str, traces: List[Trace], data: Data, baseline: Tuple[str, Dict[str, str]],
                out_dir: Path) -> str:
    default_base, per_op = baseline
    libs = {}
    for t in traces:
        libs.update(t.libs)
    counts = Counter(t.status for t in traces)
    by_model = build_groups([t for t in traces if t.level != "e2e"])
    e2e = build_groups([t for t in traces if t.level == "e2e"])
    h = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(hw)} kernel bench</title>"
         f"<style>{CSS}</style><h1>{html.escape(hw)}</h1>"]
    h.append('<p class="mut">' + " · ".join(f"{k} {v}" for k, v in sorted(libs.items())) + "</p>")
    h.append("<p>" + ", ".join(f"<b>{k}</b> {v}" for k, v in sorted(counts.items())) + "</p>")
    h.append(f'<p class="mut">Cells: median latency · ×speedup vs <code>{html.escape(default_base)}'
             "</code> eager (per-op overrides: "
             f"{html.escape(str(per_op) if per_op else 'none')}) · g× = eager/cudagraph for the "
             "same backend. Green = fastest in that mode. Hover for p10/p90, host overhead, "
             "graph capture cost, TFLOP/s, GB/s. Block (decoder_layer) correctness is "
             "CUDA-graph-vs-eager consistency.</p>")
    toc = "".join(f'<li><a href="#{slug(m)}">{html.escape(m)}</a></li>' for m in by_model)
    if e2e:
        toc += '<li><a href="#e2e">end-to-end (sglang)</a></li>'
    toc += '<li><a href="#support">support matrix</a></li>'
    h.append(f"<ul>{toc}</ul>")
    for model, groups in by_model.items():
        h.append(f'<h2 id="{slug(model)}">{html.escape(model)}</h2>')
        for g in groups:
            base = per_op.get(g.op, default_base)
            h.append(f"<h3>{html.escape(g.title())}</h3>")
            h.append(group_table_html(g, base, data, out_dir))
            series = graph_speedup_series(g)
            if series:
                h.append("<details open><summary>CUDA graph speedup vs batch size</summary>"
                         + svg_lines(series, xlabel="batch size", ylabel="eager / graph") +
                         "</details>")
    if e2e:
        h.append('<h2 id="e2e">End-to-end (sglang one-batch)</h2>')
        h.append('<p class="mut">Latency = median decode step. Hover for prefill latency and '
                 "throughput. Logs and raw results are linked.</p>")
        for model, groups in e2e.items():
            for g in groups:
                h.append(f"<h3>{html.escape(g.title())}</h3>")
                h.append(group_table_html(g, "", data, out_dir))
                h.append(_e2e_detail_table(g))
                series = graph_speedup_series(g)
                if series:
                    h.append(svg_lines(series, xlabel="batch size", ylabel="decode eager / graph"))
    backends, rows = support_rows(traces)
    h.append('<h2 id="support">Support matrix</h2><p class="mut">passed / total traces; '
             "otherwise the most common failure and its reason.</p><table><tr><th class='l'>case</th>"
             + "".join(f"<th>{html.escape(b)}</th>" for b in backends) + "</tr>")
    for label, d in rows:
        cells = []
        for b in backends:
            if b not in d:
                cells.append('<td class="skip">-</td>')
                continue
            ok, n, st, msg = d[b]
            if ok == n:
                cells.append(f'<td class="best">{ok}/{n}</td>')
            else:
                cls = "skip" if st == Status.SKIPPED else ("ng" if st == Status.GRAPH_UNSUPPORTED else "bad")
                cells.append(f'<td class="{cls} l small" title="{html.escape(msg)}">{ok}/{n} '
                             f"{short_status_name(st)}<br>{html.escape(msg[:70])}</td>")
        h.append(f'<tr><td class="l">{html.escape(label)}</td>{"".join(cells)}</tr>')
    h.append("</table>")
    return "".join(h)


def _e2e_detail_table(g: Group) -> str:
    cols = [("prefill_ms", "prefill ms"), ("prefill_tok_s", "prefill tok/s"),
            ("decode_ms", "decode ms"), ("decode_tok_s", "decode tok/s"),
            ("overall_tok_s", "overall tok/s")]
    rows = []
    for wk in g.rows():
        for b in sorted(g.backends):
            for m in Mode.ALL:
                t = g.cells.get((wk, b, m))
                if t is None or t.status != Status.PASSED:
                    continue
                vals = "".join(
                    f"<td>{t.metrics[k]:.2f}</td>" if isinstance(t.metrics.get(k), (int, float))
                    else "<td>-</td>" for k, _ in cols)
                rows.append(f'<tr><td class="l">{html.escape(wl_label(t))}</td><td class="l">{b}</td>'
                            f'<td class="l">{m}</td>{vals}</tr>')
    if not rows:
        return ""
    head = "".join(f"<th>{c}</th>" for _, c in cols)
    return ("<details><summary>throughput details</summary><table><tr><th class='l'>workload</th>"
            f"<th class='l'>backend</th><th class='l'>mode</th>{head}</tr>{''.join(rows)}</table></details>")


# -- Markdown -----------------------------------------------------------------
def _cell_md(t: Optional[Trace], best: bool, base_ms: Optional[float], graph: bool) -> str:
    if t is None:
        return "-"
    ms = _ms(t)
    if ms is None:
        return short_status(t)
    s = fmt_ms(ms)
    if base_ms:
        s += f" ×{base_ms / ms:.2f}"
    if graph and "graph_speedup" in t.metrics:
        s += f" (g×{t.metrics['graph_speedup']:.2f})"
    return f"**{s}**" if best else s


def render_md(hw: str, traces: List[Trace], data: Data, baseline: Tuple[str, Dict[str, str]],
              out_dir: Path) -> str:
    default_base, per_op = baseline
    counts = Counter(t.status for t in traces)
    lines = [f"# {hw}", "", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), "",
             f"Cells: median latency, ×speedup vs `{default_base}` eager, (g×) eager/cudagraph. "
             "**Bold** = fastest in that mode. Charts and profiler links are in report.html.", ""]
    for level_name, subset in (("", [t for t in traces if t.level != "e2e"]),
                               ("End-to-end (sglang) · ", [t for t in traces if t.level == "e2e"])):
        for model, groups in build_groups(subset).items():
            lines += [f"## {level_name}{model}", ""]
            for g in groups:
                base = per_op.get(g.op, default_base) if not level_name else ""
                backends = g.ordered_backends(base)
                modes = [m for m in Mode.ALL if any(k[2] == m for k in g.cells)]
                lines += [f"### {g.title()}", ""]
                hdr = ["workload"] + [f"{b} ({m})" for m in modes for b in backends]
                lines.append("| " + " | ".join(hdr) + " |")
                lines.append("|" + "---|" * len(hdr))
                for wk in g.rows():
                    row = [wl_label(g.wls[wk])]
                    for m in modes:
                        best = best_backends(g, wk, m)
                        base_ms = _ms(g.cells.get((wk, base, Mode.EAGER))) if base else None
                        row += [_cell_md(g.cells.get((wk, b, m)), b == best, base_ms,
                                         m == Mode.CUDAGRAPH) for b in backends]
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
    backends, rows = support_rows(traces)
    lines += ["## Support matrix", "", "| case | " + " | ".join(backends) + " |",
              "|" + "---|" * (len(backends) + 1)]
    for label, d in rows:
        cells = []
        for b in backends:
            if b not in d:
                cells.append("-")
                continue
            ok, n, st, msg = d[b]
            cells.append(f"{ok}/{n}" if ok == n else
                         f"{ok}/{n} {short_status_name(st)}: {msg[:60].replace('|', '/')}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# -- cross hardware -------------------------------------------------------------
def render_index(per_hw: Dict[str, List[Trace]], out_dir: Path, fmt: List[str]) -> Tuple[str, str]:
    """Best backend + latency per hardware for every (model, op, variant, workload, mode)."""
    hws = list(per_hw)
    best: Dict[Tuple, Dict[str, Tuple[str, float]]] = defaultdict(dict)
    label: Dict[Tuple, Trace] = {}
    for hw, ts in per_hw.items():
        for t in ts:
            ms = _ms(t)
            if ms is None:
                continue
            k = (t.model, t.op, t.variant, t.workload.key(), t.mode)
            label.setdefault(k, t)
            cur = best[k].get(hw)
            if cur is None or ms < cur[1]:
                best[k][hw] = (t.backend, ms)
    h = [f"<!doctype html><meta charset='utf-8'><title>kernel bench</title><style>{CSS}</style>"
         "<h1>Kernel benchmark</h1><ul>"]
    md = ["# Kernel benchmark", ""]
    for hw in hws:
        d = slug(hw)
        links = []
        if "html" in fmt:
            links.append(f'<a href="{d}/report.html">html</a>')
        if "md" in fmt:
            links.append(f'<a href="{d}/report.md">md</a>')
        h.append(f"<li>{html.escape(hw)}: {' '.join(links)} ({len(per_hw[hw])} traces)</li>")
        md.append(f"- {hw}: [report]({d}/report.md) ({len(per_hw[hw])} traces)")
    h.append("</ul>")
    md.append("")
    if len(hws) > 1:
        h.append("<h2>Fastest backend per hardware</h2><table><tr><th class='l'>case</th>"
                 "<th class='l'>workload</th><th>mode</th>"
                 + "".join(f"<th>{html.escape(x)}</th>" for x in hws) + "</tr>")
        md += ["## Fastest backend per hardware", "",
               "| case | workload | mode | " + " | ".join(hws) + " |", "|" + "---|" * (3 + len(hws))]
        for k in sorted(best, key=lambda k: (k[0], k[1], k[2], wl_sort(label[k]), k[4])):
            t = label[k]
            v = "" if t.variant in ("-", "") else f" / {t.variant}"
            vals = [best[k].get(hw) for hw in hws]
            fastest = min((x[1] for x in vals if x), default=None)
            cells, mcells = [], []
            for x in vals:
                if x is None:
                    cells.append('<td class="skip">-</td>')
                    mcells.append("-")
                    continue
                b, ms = x
                cls = "best" if ms == fastest and len([y for y in vals if y]) > 1 else ""
                cells.append(f'<td class="{cls}">{fmt_ms(ms)} <span class="mut">{html.escape(b)}</span></td>')
                mcells.append(f"{fmt_ms(ms)} {b}")
            h.append(f'<tr><td class="l">{html.escape(t.model + " · " + t.op + v)}</td>'
                     f'<td class="l">{html.escape(wl_label(t))}</td><td>{k[4]}</td>{"".join(cells)}</tr>')
            md.append(f"| {t.model} · {t.op}{v} | {wl_label(t)} | {k[4]} | " + " | ".join(mcells) + " |")
        h.append("</table>")
    return "".join(h), "\n".join(md) + "\n"


def main(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"no such trace path: {', '.join(map(str, missing))}")
        return 2
    data = Data(iter_traces(paths), latest=not args.all_runs)
    models = set(args.model.split(",")) if args.model else None
    per_hw = {}
    for hw, ts in data.by_hardware().items():
        if args.hardware and args.hardware.lower() not in hw.lower():
            continue
        if models:
            ts = [t for t in ts if t.model in models]
        if ts:
            per_hw[hw] = ts
    if not per_hw:
        print("no traces matched")
        return 1
    fmt = [f.strip() for f in args.format.split(",")]
    out_root = Path(args.out_dir)
    baseline = _baselines(args.baseline)
    for hw, ts in per_hw.items():
        d = out_root / slug(hw)
        d.mkdir(parents=True, exist_ok=True)
        if "html" in fmt:
            (d / "report.html").write_text(render_html(hw, ts, data, baseline, d), encoding="utf-8")
        if "md" in fmt:
            (d / "report.md").write_text(render_md(hw, ts, data, baseline, d), encoding="utf-8")
        print(f"{hw}: {len(ts)} traces -> {d}")
    ih, im = render_index(per_hw, out_root, fmt)
    if "html" in fmt:
        (out_root / "index.html").write_text(ih, encoding="utf-8")
    if "md" in fmt:
        (out_root / "index.md").write_text(im, encoding="utf-8")
    print(f"index: {out_root / ('index.html' if 'html' in fmt else 'index.md')}")
    return 0
