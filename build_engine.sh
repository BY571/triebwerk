#!/bin/bash
# Build the Triebwerk C++ inference engine.
# Auto-detects GPU architecture from nvidia-smi.

set -e

# Auto-detect CUDA architecture
if command -v nvidia-smi &>/dev/null; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "Detected GPU architecture: sm_$ARCH"
else
    ARCH=89
    echo "nvidia-smi not found, defaulting to sm_$ARCH (RTX 40xx)"
fi

mkdir -p engine/build_local
cd engine/build_local
cmake .. -DCMAKE_CUDA_ARCHITECTURES=$ARCH
make -j$(nproc)
echo "Engine built: engine/build_local/jetson_engine$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' 2>/dev/null || echo '.so')"
