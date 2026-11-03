"""Trace records written one-per-line to ``traces/*.jsonl``.

One trace = one (hardware, level, model, op, variant, backend, workload, mode)
measurement. Profiling runs append traces with ``kind="profile"`` whose
``artifacts`` are joined back onto measurement traces by ``Trace.key()``.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


class Status:
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    GRAPH_UNSUPPORTED = "GRAPH_UNSUPPORTED"
    OOM = "OOM"
    TIMEOUT = "TIMEOUT"

    FAILED = (
        INCORRECT_SHAPE,
        INCORRECT_DTYPE,
        INCORRECT_NUMERICAL,
        RUNTIME_ERROR,
        GRAPH_UNSUPPORTED,
        OOM,
        TIMEOUT,
    )


class Mode:
    EAGER = "eager"
    CUDAGRAPH = "cudagraph"
    ALL = (EAGER, CUDAGRAPH)


class Level:
    OP = "op"
    BLOCK = "block"
    E2E = "e2e"


@dataclass
class Hardware:
    gpu: str = "cpu"
    sm: str = ""
    num_gpus: int = 0
    total_mem_gb: float = 0.0
    driver: str = ""
    cuda: str = ""

    @property
    def label(self) -> str:
        return f"{self.gpu} (sm_{self.sm})" if self.sm else self.gpu


@dataclass
class WorkloadRecord:
    phase: str
    batch_size: int
    input_len: int
    output_len: int
    kv_len: int
    q_len: int
    dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"
    page_size: int = 16
    varlen: bool = False

    def key(self) -> str:
        return (
            f"{self.phase}/b{self.batch_size}/in{self.input_len}/out{self.output_len}"
            f"/kv{self.kv_len}/q{self.q_len}/{self.dtype}/{self.kv_dtype}/p{self.page_size}"
            f"{'/varlen' if self.varlen else ''}"
        )


@dataclass
class Correctness:
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    matched_ratio: float = 1.0
    atol: float = 0.0
    rtol: float = 0.0


@dataclass
class Latency:
    """Milliseconds. ``samples`` are per-iteration GPU (or wall, on CPU) times."""

    median_ms: float = 0.0
    mean_ms: float = 0.0
    p10_ms: float = 0.0
    p90_ms: float = 0.0
    std_ms: float = 0.0
    n: int = 0
    timer: str = ""

    @classmethod
    def from_samples(cls, samples: List[float], timer: str) -> "Latency":
        if not samples:
            return cls(timer=timer)
        s = sorted(samples)

        def pct(p: float) -> float:
            idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
            return s[idx]

        return cls(
            median_ms=statistics.median(s),
            mean_ms=statistics.fmean(s),
            p10_ms=pct(0.10),
            p90_ms=pct(0.90),
            std_ms=statistics.pstdev(s) if len(s) > 1 else 0.0,
            n=len(s),
            timer=timer,
        )


@dataclass
class HostOverhead:
    """CPU-side cost per call, the part CUDA graphs are meant to remove.

    ``enqueue_ms``: time to return from the call without synchronizing.
    ``wall_ms``: call + ``cuda.synchronize`` end-to-end wall time.
    """

    enqueue_ms: float = 0.0
    wall_ms: float = 0.0


@dataclass
class GraphInfo:
    capture_ms: float = 0.0
    mem_mb: float = 0.0
    num_nodes: int = 0


@dataclass
class Trace:
    level: str
    model: str
    op: str
    backend: str
    mode: str
    workload: WorkloadRecord
    status: str
    hardware: Hardware = field(default_factory=Hardware)
    libs: Dict[str, str] = field(default_factory=dict)
    variant: str = ""
    kind: str = "measure"
    correctness: Optional[Correctness] = None
    latency: Optional[Latency] = None
    host: Optional[HostOverhead] = None
    graph: Optional[GraphInfo] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def key(self, *, with_mode: bool = True, with_backend: bool = True) -> str:
        parts = [self.hardware.label, self.level, self.model, self.op, self.variant]
        if with_backend:
            parts.append(self.backend)
        parts.append(self.workload.key())
        if with_mode:
            parts.append(self.mode)
        return "|".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trace":
        d = dict(d)
        d["workload"] = WorkloadRecord(**d["workload"])
        d["hardware"] = Hardware(**d.get("hardware") or {})
        for name, typ in (
            ("correctness", Correctness),
            ("latency", Latency),
            ("host", HostOverhead),
            ("graph", GraphInfo),
        ):
            if d.get(name) is not None:
                d[name] = typ(**d[name])
        return cls(**d)


def _json_default(o: Any) -> Any:
    if isinstance(o, float):
        return o
    return str(o)


def _sanitize(v: Any) -> Any:
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return str(v)
    if isinstance(v, dict):
        return {k: _sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_sanitize(x) for x in v]
    return v


def _restore(v: Any) -> Any:
    if v in ("nan", "inf", "-inf"):
        return float(v)
    if isinstance(v, dict):
        return {k: _restore(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_restore(x) for x in v]
    return v


def append_traces(path: Path, traces: Iterable[Trace]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(_sanitize(t.to_dict()), default=_json_default) + "\n")
            n += 1
    return n


def iter_traces(paths: Iterable[Path]) -> Iterator[Trace]:
    for p in paths:
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield Trace.from_dict(_restore(json.loads(line)))
