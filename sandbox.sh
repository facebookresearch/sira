#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# sandbox.sh — Set up the SIRA development environment.
# Usage: source sandbox.sh

CONDA_ENV="sira312"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_ok()   { echo "✓ $1"; }
_warn() { echo "⚠ $1"; }

# Secrets
if [ -f "$DIR/.private/token.json" ]; then
    while IFS='=' read -r key value; do
        export "$key=$value"
    done < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' "$DIR/.private/token.json")
fi

# HuggingFace
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="$HOME/cache/hf_datasets_cache"

# DeepGEMM cache
export SGLANG_DG_CACHE_DIR="$HOME/.cache/deep_gemm"
mkdir -p "$SGLANG_DG_CACHE_DIR/cache"

# Python / Debug
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS="ignore::FutureWarning"
export RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=true
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Weights & Biases
export WANDB_DIR="$HOME/.cache/wandb"

# Conda — locate conda if not already on PATH
if ! command -v conda &>/dev/null; then
    for _conda_bin in "$HOME/.conda/bin" "$HOME/miniconda3/bin" "$HOME/anaconda3/bin" \
                      "/opt/conda/bin" "$HOME/.miniforge3/bin"; do
        [ -x "$_conda_bin/conda" ] && export PATH="$_conda_bin:$PATH" && break
    done
    unset _conda_bin
fi
if ! declare -f conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)"
fi
conda activate "$CONDA_ENV" 2>/dev/null || { echo "ERROR: conda env '$CONDA_ENV' not found — see README.md for setup instructions"; return 1; }
export CUDA_HOME="$CONDA_PREFIX"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LIBRARY_PATH:-}"
[ -f "$CONDA_PREFIX/bin/ptxas" ] && export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"

# Rust toolchain — prefer rustup, fall back to conda's cargo (already on PATH)
_rustup_bin=$(ls -d "$HOME"/.rustup/toolchains/*/bin 2>/dev/null | head -1)
[ -n "$_rustup_bin" ] && export PATH="$_rustup_bin:$PATH"
unset _rustup_bin
command -v cargo &>/dev/null || _warn "cargo not found — Rust builds will fail"

# Project root (PYTHONPATH ensures current worktree's source wins)
export PYTHONPATH="$DIR:$PYTHONPATH"

# Rebuild bm25x from current worktree (compiled extension, ~1.5s).
_bm25x_log=$(mktemp -t bm25x-build.XXXXXX.log)
if ! RUSTFLAGS="${RUSTFLAGS:-} -C linker=/usr/bin/cc" PATH="/usr/bin:$PATH" \
    maturin develop --release --manifest-path "$DIR/src/sira/bm25x/python/Cargo.toml" -q \
    >"$_bm25x_log" 2>&1; then
    echo "⚠ bm25x build failed — see $_bm25x_log (tail below):"
    tail -20 "$_bm25x_log"
else
    rm -f "$_bm25x_log"
fi
unset _bm25x_log

cd "$DIR" || return 1
echo "SIRA sandbox ready ($(basename "$CONDA_PREFIX" 2>/dev/null || echo 'no conda'), HF_HOME=$HF_HOME)"

# Cleanup
unset -f _ok _warn
unset CONDA_ENV DIR
