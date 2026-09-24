# kernels — QSA / DSV4 / DSA extract + dual-backend attention demos

Upstream: [`sgl-project/sglang`](https://github.com/sgl-project/sglang) @ `f9c2791460` (2026-09-20)

This tree mirrors the QSA, DeepSeek-V4 (dsv4), and DSA attention kernel sources from SGLang, plus two runnable demos:

| Backend | When | What runs |
|---------|------|-----------|
| **`cpu_ref`** | Any machine (current default: torch CPU, no CUDA) | Pure-torch reference implementations (golden) |
| **`sm120`** | RTX 5070 Ti / consumer Blackwell (`sm_120`) | SGLang SM120 paths (`flash_mla_sm120`, flashinfer trtllm sparse decode, FA2 / `flash_attention_v4_sm120`) |

**SM120 ≠ SM100.** Datacenter Blackwell (B200, `sm_100`) kernels that use `tcgen05` + TMEM do **not** load on a 5070 Ti. See [MANIFEST.md](MANIFEST.md).

## Layout

```
kernels/
  _compat/          # triton stub + sglang.* import aliases + env probes
  qsa/              # extracted QSA sources
  dsv4/             # extracted DSV4 + SM120 helpers
  dsa/              # extracted DSA sources (manifest only; no demo)
  demo/             # shapes, ref_ops, backends, dsv4/qsa demos, run_all
  MANIFEST.md
  README.md
```

## Quick start (CPU — works now)

```powershell
D:\conda\python.exe D:\workspace\kernels\demo\run_all.py --tiny
D:\conda\python.exe D:\workspace\kernels\demo\run_all.py
```

No extra packages beyond the existing `torch 2.10.0+cpu` (and `numpy`).

## RTX 5070 Ti (SM120) setup

1. CUDA Toolkit ≥ 12.8
2. PyTorch 2.7+ **cu128** (must include `sm_120`)
3. Real triton (Linux) or `triton-windows`
4. flashinfer build with SM120 support
5. Rebuild `sgl_kernel` with `-gencode=arch=compute_120,code=sm_120`
6. Prefer FA2 or SGLang `flash_attention_v4_sm120` (FA4 cute is SM100/110-only)

Then:

```powershell
python D:\workspace\kernels\demo\run_all.py --backend sm120
python D:\workspace\kernels\demo\run_all.py --backend sm120 --check-parity
```

`--check-parity` compares `sm120` vs `cpu_ref` outputs (loose tol for bf16).

### Common failure

`no kernel image is available for execution on the device` → a package still ships SM100 (or older) cubins. Rebuild for `sm_120`.

### What SM120 can / cannot run

| Path | 5070 Ti |
|------|---------|
| QSA flashinfer `trtllm_batch_decode_with_kv_cache` | Yes (validated SM100+SM120 in SGLang) |
| QSA `qwen38_qsa_sm121` | **No** (DGX Spark SM121 only) |
| DSV4 `flash_mla_sm120` decode | Yes |
| DSV4 `decode_attention_sm100` / `sparse_prefill_fwd` | **No** |
| DeepGEMM SM100 cubins | **No** (use SM120 fallbacks / torch) |

## Demo dimensions

**DeepSeek-V4.1-Flash:** hidden 4096, 64 heads, head_dim 512 (448 nope + 64 rope), q_lora 1280, MQA latent 512, o_groups 8 × o_lora 1024, index 64×128 topk 512, window 128, compress_ratio 2.

**Qwen3.8-Flash-Next:** hidden 2560, 24Q/2KV × 256, indexer 4×128, budget 2048, ratio 4 → block_topk 512, expand width 2051, page 64.

`--tiny` shrinks heads/hidden for a seconds-scale smoke test.

## Assertions

1. **Degeneracy:** QSA with full-context slots equals dense GQA; DSV4 with full window and no compressor/indexer equals dense causal MLA.
2. **Cross-backend:** `sm120` vs `cpu_ref` (`--check-parity`).
3. **fp64 baseline:** reports fp32 max relative error on CPU.

## Nsight Systems profiling (pipeline / data-flow timeline)

The demos are annotated with **NVTX ranges** so each attention pipeline stage
shows up as its own named band on the Nsight Systems timeline instead of one
anonymous blob of CUDA kernels:

- **DSV4:** `q_path` → `kv_path` → `compressor` → `indexer` → `gather_keys` →
  `combined_attention` → `forward_out`
- **QSA:** `project_main` / `project_index` → `compress_index_keys` →
  `mqa_logits_prefill` → `fast_topk` → `expand_block_indices` →
  `sparse_attention` → `output`

Instrumentation lives in [`demo/profiling.py`](demo/profiling.py) and is a
zero-cost no-op on the current CPU-only torch build (NVTX simply doesn't emit),
so it only "lights up" under `nsys` on the SM120 GPU box. Disable it anytime
with `KERNELS_NVTX=0`.

### Quick start

```powershell
# Fast smoke capture (tiny shapes). Works on CPU (NVTX+CUDA-API rows only).
.\profile.ps1 -Only qsa -Tiny

# Full capture on the RTX 5070 Ti with a stats summary:
.\profile.ps1 -Backend sm120 -Only all -Iters 20 -Stats
```

`profile.ps1` requires the Nsight Systems CLI (`nsys`) on PATH (ships with the
CUDA Toolkit, or install standalone). Reports land in `reports/*.nsys-rep`;
open them with `nsys-ui`.

### How it works

- [`demo/profile_nsys.py`](demo/profile_nsys.py) builds the model once, runs
  unrecorded **warmup** iters, then a `cudaProfilerStart/Stop` region of
  steady-state iters (each wrapped in an `iterN` NVTX range).
- `profile.ps1` invokes `nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas
  --capture-range=cudaProfilerApi --capture-range-end=stop`, so the report
  contains **only** the steady-state passes — warmup and model construction are
  excluded.

Run the driver directly (no profiler) to sanity-check it:

```powershell
D:\conda\python.exe D:\workspace\kernels\demo\profile_nsys.py --only qsa --tiny --iters 3
```

## Import shim

```python
import sys
sys.path.insert(0, r"D:\workspace\kernels")
from _compat import bootstrap
bootstrap()
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference
```

On hosts without triton, `_compat` installs a stub so modules import and expose torch reference functions. If a real triton is installed, the stub is skipped.

## Re-sync from upstream

Re-copy the paths listed in [MANIFEST.md](MANIFEST.md) after `git pull` in `sglang`. Do not edit extracted sources in place if you want clean diffs.
