# bench — model × op × backend inference benchmark

Answers, per GPU: *for this model's shapes, which kernel library is fastest for each op, how much
does CUDA graph help, and what exactly runs on the GPU?*

```
models (bench/models)          ops (bench/ops)                 backends
  qwen3-8b, qwen3-30b-a3b        attn_decode, attn_prefill       flashinfer, fa2, fa3, flashmla,
  qwen3.8-flash-next-qsa         rmsnorm, rope, gemm_qkv,        flashmla_sm120, triton, trtllm,
  deepseek-v3 / v3.2-dsa / v4.1  gemm_o, mlp, moe                sdpa, torch_ref, torch,
  glm-4.5-air, glm-4-9b          decoder_layer (block level)     sgl_kernel, sglang_triton,
  gpt-oss-20b/120b, gemma-2/3                                    flashinfer_cutlass
          │                              │                              │
          └──── workload grid: phase × batch × input_len × output_len ──┘
                                          │
               correctness vs fp32 reference → eager CUPTI timing → CUDA graph capture,
               replay correctness, graph timing → traces/<gpu>_*.jsonl
                                          │
              bench profile (nsys / torch.profiler)   bench e2e (sglang one-batch)
                                          │
                       bench report → reports/bench/<gpu>/report.{html,md}
```

## Commands

```bash
python -m bench list models | backends | support      # what exists / what runs on this GPU (and why not)
python -m bench run   --config bench/configs/h100.yaml # per-hardware matrix
python -m bench run   --model qwen3-8b --op attn_decode --backend flashinfer,fa2 --grid decode
python -m bench run   ... --dry-run                    # print the matrix only
python -m bench profile --model qwen3-8b --op attn_decode --backend flashinfer \
                        --batch 64 --input-len 2048 --output-len 1024 --mode both --tool both
python -m bench e2e   --config bench/configs/h100.yaml # sglang eager vs --disable-cuda-graph
python -m bench report traces/ [--baseline torch_ref | attn_decode=flashinfer]
```

`run` flags worth knowing: `--set batch_sizes=1,16 input_lens=4096` (grid overrides),
`--modes eager,cudagraph`, `--variant local,global`, `--moe-expert-cap 64` (fit big MoEs),
`--block-layers N`, `--block-impl rmsnorm=flashinfer moe=sglang_triton`, `--no-cupti`,
`--atol/--rtol/--matched-ratio`, `--model path/to/config.json` (any HF config).

## Workload dimensions

Yes, batch size, input length and output length are all dimensions, but each op only sees the ones
it depends on (duplicates are removed):

| op | depends on |
|---|---|
| attn_prefill | batch, input_len (q = kv = input_len); output_len irrelevant |
| attn_decode, decoder_layer | batch, kv_len ∈ {input, input+out/2, input+out} (`decode_kv_points`), q_len = spec tokens |
| rmsnorm, rope, GEMMs, mlp, moe | number of tokens only (batch × q_len for decode, batch × input_len for prefill) |
| e2e | batch × input_len × output_len (sglang runs the real prefill + decode loop) |

Builtin grids: `tiny` (CPU tests), `smoke`, `decode`, `full`; or a YAML with `grid:`. Cases whose
estimated memory exceeds `mem_fraction` of free VRAM are recorded as SKIPPED, not attempted.
CUDA graph mode runs for `graph_phases` (decode by default, as in serving engines).

## What is measured

* **Correctness first** (flashinfer-bench rules): shape, dtype, NaN/Inf, then an element fails only
  if it exceeds both atol and rtol; the case passes if ≥ `matched_ratio` elements pass (1.0, or
  0.999 for half-precision rope/GEMM/MLP/MoE). Failing cases are not timed.
* **Eager latency**: CUPTI kernel time (cold L2, warmup 10, 50 iters × 3 trials; falls back to
  CUDA events + L2 flush), plus host overhead (`enqueue_ms` without sync, `wall_ms` with sync).
* **CUDA graph**: fresh runner prepared with graph-safe buffers (FlashInfer `use_cuda_graph`
  wrappers, `plan()` outside capture), side-stream warmup, capture (`capture_ms`, memory delta,
  node count), replay output re-checked, then replay latency. `graph_speedup` = eager / graph.
  Data-dependent kernels report GRAPH_UNSUPPORTED with the reason.
* **Block level** (`decoder_layer`): norm → QKV → RoPE → KV write → attention → O → norm → MLP/MoE
  around each attention backend; its correctness check compares the graph output with eager.
* **Throughput**: TFLOP/s and GB/s from problem sizes (hover in the HTML report).

Statuses: PASSED, SKIPPED (unsupported or doesn't fit, with the reason), INCORRECT_*, RUNTIME_ERROR,
GRAPH_UNSUPPORTED, OOM, TIMEOUT.

## Profiling

`bench profile` re-runs one case under `nsys profile --capture-range=cudaProfilerApi` (warmup
excluded, one NVTX range per iteration, `--cuda-graph-trace=node` in graph mode so graph replays
are expanded into kernels), exports `nsys stats` CSVs (NVTX, GPU kernels, CUDA API), and/or records
a torch.profiler chrome trace + per-op CSV. It appends a `kind="profile"` trace, so the report shows
`nsys` / `torch` links and kernels-per-iteration next to the timing cell. `./profile.sh bench ...` is
the same thing; `./profile.sh demo` / `profile.ps1` capture the DSV4/QSA demos.

## Reports

`reports/bench/<gpu>/report.html`: per model and op, a workload × backend table for eager and CUDA
graph (fastest highlighted, ×speedup vs baseline, g× graph speedup), graph speedup vs batch charts,
end-to-end tables, and a support matrix with skip and failure reasons. `reports/bench/index.html`
compares the fastest backend across GPUs when traces from several machines are merged. Copy each
machine's `traces/*.jsonl` into one directory and run `bench report` on it.

## Adding things

* **Model**: `REGISTRY.add(ModelSpec(...))` in `models/__init__.py` (attention variants carry window,
  sink, softcap, sparse_topk), or pass an HF `config.json` path as `--model`.
* **Backend**: subclass `OpAdapter` (`ops`, `kinds`, `features`, `requires`, `sm_allow`/`sm_min`),
  implement `prepare(problem, graph) -> Runner` and decorate with `@register`. Do layout conversion
  in `prepare` (untimed); `Runner.fn` is the timed call.
* **Op**: a builder in `ops/problems.py` with an fp32 reference, then adapters.

## Known gaps

The sizes for DSV4-flash and QSA (layer counts and MoE dimensions) are placeholders. Two pieces are
written but have not been checked on a GPU: the flash_mla_sm120 FP8 page packing and the
cutlass MoE weight order. The correctness gate will report INCORRECT if either is wrong.
