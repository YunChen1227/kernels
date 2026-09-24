<#
.SYNOPSIS
  Launch Nsight Systems (nsys) on the attention demos to capture a
  data-flow / pipeline timeline with NVTX stage bands.

.DESCRIPTION
  Wraps `demo/profile_nsys.py` with a `cudaProfilerApi` capture range so the
  report contains only the steady-state forward passes (warmup excluded).
  Each pipeline stage (Q path, KV path, compressor, indexer, topk, attention,
  output) shows up as its own NVTX band in the timeline.

  Requires the NVIDIA Nsight Systems CLI (`nsys`) on PATH. Meaningful GPU rows
  only appear on the SM120 (RTX 5070 Ti) box with a CUDA torch build; on the
  current CPU box it still records NVTX + CUDA-API rows.

.PARAMETER Backend
  cpu_ref | sm120 | auto  (default: auto)

.PARAMETER Only
  dsv4 | qsa | all  (default: all)

.PARAMETER Tiny
  Use tiny shapes for a fast smoke capture.

.PARAMETER Warmup
  Unrecorded warmup iterations (default: 3).

.PARAMETER Iters
  Recorded steady-state iterations (default: 10).

.PARAMETER Python
  Python interpreter to use (default: D:\conda\python.exe).

.PARAMETER Output
  Output report basename (default: reports\attn_<only>_<backend>_<timestamp>).

.PARAMETER Stats
  Also print an `nsys stats` summary (NVTX + CUDA kernel/gpukernsum) after capture.

.EXAMPLE
  # Fast CPU smoke capture
  .\profile.ps1 -Only qsa -Tiny

.EXAMPLE
  # Full SM120 GPU capture with a stats summary
  .\profile.ps1 -Backend sm120 -Only all -Iters 20 -Stats
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu_ref', 'sm120', 'auto')]
    [string]$Backend = 'auto',

    [ValidateSet('dsv4', 'qsa', 'all')]
    [string]$Only = 'all',

    [switch]$Tiny,

    [int]$Warmup = 3,

    [int]$Iters = 10,

    [string]$Python = 'D:\conda\python.exe',

    [string]$Output,

    [switch]$Stats
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
$Driver = Join-Path $RepoRoot 'demo\profile_nsys.py'

# --- locate nsys ------------------------------------------------------------
$nsys = Get-Command nsys -ErrorAction SilentlyContinue
if (-not $nsys) {
    Write-Error @"
nsys (Nsight Systems CLI) not found on PATH.

Install Nsight Systems (bundled with the CUDA Toolkit, or standalone from
https://developer.nvidia.com/nsight-systems) and ensure 'nsys' is on PATH,
e.g. add: C:\Program Files\NVIDIA Corporation\Nsight Systems <ver>\target-windows-x64
"@
    exit 1
}

if (-not (Test-Path $Python)) {
    Write-Warning "Python '$Python' not found; falling back to 'python' on PATH."
    $Python = 'python'
}

# --- output path ------------------------------------------------------------
$ReportDir = Join-Path $RepoRoot 'reports'
if (-not (Test-Path $ReportDir)) { New-Item -ItemType Directory -Path $ReportDir | Out-Null }
if (-not $Output) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $Output = Join-Path $ReportDir "attn_${Only}_${Backend}_${stamp}"
}

# --- build arg lists --------------------------------------------------------
$driverArgs = @('--backend', $Backend, '--only', $Only, '--warmup', "$Warmup", '--iters', "$Iters")
if ($Tiny) { $driverArgs += '--tiny' }

$nsysArgs = @(
    'profile',
    '--trace=cuda,nvtx,osrt,cudnn,cublas',
    '--capture-range=cudaProfilerApi',
    '--capture-range-end=stop',
    '--cuda-memory-usage=true',
    '--force-overwrite=true',
    '--output', $Output,
    $Python, $Driver
) + $driverArgs

Write-Host '=== Nsight Systems capture ===' -ForegroundColor Cyan
Write-Host "nsys     : $($nsys.Source)"
Write-Host "python   : $Python"
Write-Host "driver   : $Driver"
Write-Host "output   : $Output.nsys-rep"
Write-Host "command  : nsys $($nsysArgs -join ' ')"
Write-Host ''

& $nsys.Source @nsysArgs
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Error "nsys exited with code $rc"
    exit $rc
}

$rep = "$Output.nsys-rep"
Write-Host ''
Write-Host "[ok] report written: $rep" -ForegroundColor Green

# --- optional stats summary -------------------------------------------------
if ($Stats -and (Test-Path $rep)) {
    Write-Host ''
    Write-Host '=== nsys stats (NVTX + GPU kernels) ===' -ForegroundColor Cyan
    & $nsys.Source stats --report nvtx_sum --report gpukernsum $rep
}

Write-Host ''
Write-Host "Open in GUI:  nsys-ui `"$rep`"" -ForegroundColor Yellow
