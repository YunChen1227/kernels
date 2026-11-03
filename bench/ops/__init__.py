"""Importing this package registers every adapter."""

from bench.ops import (  # noqa: F401
    attention_fa,
    attention_flashinfer,
    attention_mla,
    attention_torch,
    attention_triton,
    dense,
)
from bench.ops.base import (  # noqa: F401
    AttnGeom,
    Ctx,
    OpAdapter,
    Problem,
    Runner,
    Tol,
    adapters_for,
    all_adapters,
    attn_geom,
)
