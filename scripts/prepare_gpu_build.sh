#!/usr/bin/env bash

# CUDA architectures are configured in CMakeLists.txt (default: RAPIDS).
# For faster local iteration on the current GPU: make WITH_GPU=True CUDA_ARCHITECTURES=NATIVE
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found; GPU build may fail at runtime." >&2
fi
