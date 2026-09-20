# Kernel Source Manifest

Upstream: `sglang` @ `f9c2791460` (2026-09-20)

Legend for runnability columns:

| Column | Meaning |
|--------|---------|
| CPU | Pure-torch / reference path on any host (incl. SM75 CPU-only torch) |
| SM120 | RTX 5070 Ti / consumer Blackwell (`sm_120`) with CUDA torch + triton |
| SM100 | B200 / datacenter Blackwell (`sm_100`) production path |

Dependency tags: `pure-torch`, `triton`, `jit-cuda`, `deep_gemm`, `flashinfer`, `sgl_kernel`, `tilelang`, `cutedsl`, `flash_mla`, `hip`

---

## QSA (Qwen Sparse Attention)

| Local path | Upstream | Deps | CPU | SM120 | SM100 |
|---|---|---|---|---|---|
| `qsa/layers/config.py` | `srt/layers/attention/qsa/config.py` | pure-torch | Y | Y | Y |
| `qsa/layers/glue.py` | `.../qsa/glue.py` | pure-torch | Y | Y | Y |
| `qsa/layers/__init__.py` | `.../qsa/__init__.py` | pure-torch | Y | Y | Y |
| `qsa/layers/kernel.py` | `.../qsa/kernel.py` | triton (stub OK), sgl_kernel lazy | **Y (torch refs)** | Y | Y |
| `qsa/layers/mqa.py` | `.../qsa/mqa.py` | tilelang optional | **Y (torch refs)** | Y | Y |
| `qsa/layers/metadata.py` | `.../qsa/metadata.py` | via kernel | Y | Y | Y |
| `qsa/layers/sparse_attn.py` | `.../qsa/sparse_attn.py` | triton | N | Y* | Y |
| `qsa/layers/graph_metadata.py` | `.../qsa/graph_metadata.py` | triton | N | Y* | Y |
| `qsa/layers/qsa_indexer.py` | `.../qsa/qsa_indexer.py` | nn + fused CUDA | N | Y* | Y |
| `qsa/backend/qwen_sparse_attn_backend.py` | `srt/layers/attention/qwen_sparse_attn_backend.py` | flashinfer / FA2 | N | **Y (trtllm decode validated)** | Y |
| `qsa/ops/qsa_indexer.py` | `kernels/ops/attention/qsa_indexer.py` | jit-cuda | N | needs sm_120 rebuild | Y |
| `qsa/sm121/*` | `kernels/kda_kernels/qwen38_qsa_sm121/` | triton | N | **N (SM121 only)** | N |
| `qsa/csrc/qsa_indexer.cuh` | `kernels/jit/csrc/attention/qsa_indexer.cuh` | jit-cuda | N | rebuild | Y |

\* Requires real triton + CUDA torch on SM120.

Torch golden symbols used by demos: `average_pool_qsa_keys`, `qsa_fast_topk` (CPU branch), `torch_expand_qsa_block_indices`, `qsa_sparse_attention_reference`, `torch_qsa_mqa_prefill`, `torch_qsa_mqa_decode`.

---

## DSV4 (DeepSeek-V4 / V4.1)

| Local path | Upstream | Deps | CPU | SM120 | SM100 |
|---|---|---|---|---|---|
| `dsv4/layers/metadata.py` | `srt/layers/attention/dsv4/metadata.py` | pure-torch | Y | Y | Y |
| `dsv4/layers/dsv41_sparse.py` | `.../dsv41_sparse.py` | pure-torch (+ optional fp4) | **Y (rope_tail, pool_pairs, scores)** | Y | Y |
| `dsv4/layers/candidate_indexer.py` | `.../candidate_indexer.py` | deep_gemm lazy | partial | partial | Y |
| `dsv4/layers/indexer.py` | `.../indexer.py` | deep_gemm / flashinfer / sgl_kernel | N | partial (torch_sm120 logits) | Y |
| `dsv4/layers/compressor*.py` | `.../compressor*.py` | jit-cuda / triton | N | needs rebuild | Y |
| `dsv4/backend/deepseek_v4_backend.py` | `srt/layers/attention/deepseek_v4_backend.py` | flash_mla / deep_gemm | N | **decode via flash_mla_sm120**; sparse_prefill **N** | Y |
| `dsv4/backend/deepseek_v4_trtllm_backend.py` | `.../deepseek_v4_trtllm_backend.py` | flashinfer | N | partial | Y |
| `dsv4/backend/deepseek_v4_backend_hip_radix.py` | `.../deepseek_v4_backend_hip_radix.py` | hip / flash_mla | N | N | N (HIP) |
| `dsv4/ops/torch_quant.py` | `kernels/ops/attention/dsv4/torch_quant.py` | pure-torch | Y | Y | Y |
| `dsv4/ops/kv_layout.py` / `utils.py` | `.../dsv4/` | pure-torch | Y | Y | Y |
| `dsv4/ops/decode_attention_sm100*.py` | `.../dsv4/` | triton / gluon | N | **N** | Y |
| `dsv4/ops/sparse_prefill_kernels.py` | `.../dsv4/` | triton | N | **N** | Y |
| `dsv4/ops/attn.py` / `compress.py` / `wo_a.py` / `topk.py` / … | `.../dsv4/` | jit-cuda / sgl_kernel | N | rebuild sm_120 | Y |
| `dsv4/ops/unified_kv_kernels/*` | `.../dsv4/unified_kv_kernels/` | triton | N | Y* | Y |
| `dsv4/ops/deepseek_v4_rope.py` | `kernels/ops/attention/deepseek_v4_rope.py` | triton | partial (precompute) | Y* | Y |
| `dsv4/ops/dsv4_attn_metadata_kernels.py` | `.../dsv4_attn_metadata_kernels.py` | triton + torch CPU branch | Y | Y | Y |
| `dsv4/sm120/flash_mla_sm120.py` | `kernels/ops/attention/flash_mla_sm120.py` | triton / torch fallback | N | **Y (decode)** | N/A |
| `dsv4/sm120/flash_mla_sm120_triton.py` | `.../flash_mla_sm120_triton.py` | triton | N | **Y** | N/A |
| `dsv4/sm120/flash_attention_v4_sm120.py` | `.../flash_attention_v4_sm120.py` | triton | N | **Y** | N/A |
| `dsv4/sm120/deep_gemm_sm120.py` | `srt/layers/moe/moe_runner/deep_gemm_sm120.py` | deep_gemm / cutlass | N | Y* | N/A |
| `dsv4/csrc/*.cu` | `kernels/aot/csrc/elementwise/` | AOT CUDA | N | rebuild | Y |

Demo golden math mirrors `_pure_torch_dsv4_combined_reference` in SGLang test kits (SWA ∪ compressed + attn_sink softmax).

---

## DSA (DeepSeek Sparse Attention / NSA)

| Local path | Upstream | Deps | CPU | SM120 | SM100 |
|---|---|---|---|---|---|
| `dsa/layers/dsa_indexer_metadata.py` | `srt/layers/attention/dsa/` | pure-torch | Y | Y | Y |
| `dsa/layers/dsa_topk_backend.py` | `.../` | torch topk + lazy CUDA | **Y (TORCH backend)** | Y | Y |
| `dsa/layers/paged_mqa_logits_backend.py` | `.../` | pure-torch enum | Y | Y | Y |
| `dsa/layers/dsa_backend_kpool.py` | `.../` | pure-torch | Y | Y | Y |
| `dsa/layers/dsa_indexer.py` | `.../` | deep_gemm / triton / tilelang / cutedsl | N | partial | Y |
| `dsa/backend/dsa_backend.py` | `srt/layers/attention/dsa_backend.py` | flash_mla / flashinfer / triton | N | decode via flash_mla_sm120 | Y |
| `dsa/ops/dsa/triton_sparse_mla*.py` | `kernels/ops/attention/dsa/` | triton (bf16/fp8) | N | Y* | Y |
| `dsa/ops/dsa/tilelang_kernel.py` | `.../` | tilelang | N | ? | Y |
| `dsa/ops/dsa/cutedsl_paged_mqa_logits.py` | `.../` | cutedsl | N | ? | Y |
| `dsa/ops/dsa/paged_mqa_logits.py` | `.../` | pure-torch wrapper | Y | Y | Y |
| `dsa/ops/nsa_triton_decode/*` | `kernels/ops/attention/nsa_triton_decode/` | triton | N | Y* | Y |
| `dsa/ops/dsa_kpool_metadata/*` | `.../dsa_kpool_metadata/` | triton (scan.py pure) | partial | Y* | Y |
| `dsa/ops/dsa_metadata.py` | `.../dsa_metadata.py` | triton | N | Y* | Y |
| `dsa/ops/hip/dsa_topk_coop.cuh` | `kernels/aot/include/hip/` | hip | N | N | N |

DSA demos are **not** shipped in this drop (extract + manifest only).

---

## SM120 software requirements (5070 Ti)

1. CUDA Toolkit ≥ 12.8
2. PyTorch 2.7+ with cu128 (`sm_120` in binary)
3. Real triton (or `triton-windows`)
4. flashinfer build with SM120 support
5. Rebuild `sgl_kernel` with `-gencode=arch=compute_120,code=sm_120`
6. Prefer FA2 or `flash_attention_v4_sm120` over FA4 cute (SM100-only)

**Do not** expect SM100 cubins (`tcgen05` / TMEM / WGMMA) to load on SM120.
