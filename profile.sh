#!/usr/bin/env bash
# Nsight Systems / torch.profiler capture (Linux/macOS counterpart of profile.ps1).
#
#   ./profile.sh bench --model qwen3-8b --op attn_decode --backend flashinfer \
#       --batch 64 --input-len 2048 --output-len 1024 [--mode both] [--tool both]
#       -> python -m bench profile ...: nsys (+ --cuda-graph-trace=node for graph mode)
#          and torch.profiler artifacts under reports/profiles/<gpu>/, plus a
#          kind=profile trace that `python -m bench report` links into the report.
#
#   ./profile.sh demo [--backend auto|cpu_ref|sm120] [--only dsv4|qsa|all] [--tiny]
#       [--warmup 3] [--iters 10] [--stats] [--output reports/name]
#       -> demo/profile_nsys.py under nsys (NVTX stage bands of the DSV4/QSA demos).
#
# Env: PYTHON (default: python3), NSYS (default: nsys).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
NSYS="${NSYS:-nsys}"
cmd="${1:-bench}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
  bench)
    cd "$ROOT"
    exec "$PYTHON" -m bench profile --nsys "$NSYS" "$@"
    ;;
  demo)
    if ! command -v "$NSYS" >/dev/null 2>&1; then
      echo "nsys (Nsight Systems CLI) not found; install it or set NSYS=/path/to/nsys" >&2
      exit 1
    fi
    stats=0
    output=""
    args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --stats) stats=1 ;;
        --output) output="$2"; shift ;;
        *) args+=("$1") ;;
      esac
      shift
    done
    mkdir -p "$ROOT/reports"
    output="${output:-$ROOT/reports/attn_$(date +%Y%m%d_%H%M%S)}"
    set -x
    "$NSYS" profile \
      --trace=cuda,nvtx,osrt,cudnn,cublas \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --cuda-memory-usage=true --cuda-graph-trace=node \
      --force-overwrite=true --output "$output" \
      "$PYTHON" "$ROOT/demo/profile_nsys.py" ${args[@]+"${args[@]}"}
    set +x
    echo "[ok] report written: $output.nsys-rep"
    if [[ $stats -eq 1 ]]; then
      "$NSYS" stats --report nvtx_sum --report cuda_gpu_kern_sum "$output.nsys-rep"
      "$NSYS" stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --format csv \
        --output "$output" --force-export=true "$output.nsys-rep" >/dev/null
      echo "[ok] stats CSV: ${output}_{nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum}.csv"
    fi
    echo "Open in GUI: nsys-ui \"$output.nsys-rep\""
    ;;
  *)
    sed -n '2,16p' "$0"
    exit 2
    ;;
esac
