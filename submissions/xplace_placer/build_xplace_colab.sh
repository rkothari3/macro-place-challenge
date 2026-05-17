#!/bin/bash
# Build Xplace on Google Colab T4 (CUDA 12.x, GCC 11+, CMake 3.27+)
# Run this cell ONCE before evaluating the Xplace placer.
# Takes ~5-10 minutes to compile.

set -e

XPLACE_HOME=/opt/xplace

if [ -f "$XPLACE_HOME/main.py" ] && [ -f "$XPLACE_HOME/cpp_to_py/io_parser/io_parser.cpython"*".so" ]; then
    echo "[build_xplace] Already built at $XPLACE_HOME. Skipping."
    exit 0
fi

echo "[build_xplace] Installing system dependencies..."
apt-get install -y --no-install-recommends libboost-all-dev libcairo2-dev bison flex > /dev/null 2>&1

echo "[build_xplace] Cloning Xplace..."
git clone --depth 1 --recurse-submodules https://github.com/cuhk-eda/Xplace $XPLACE_HOME

echo "[build_xplace] Building Xplace (this takes ~5-10 min)..."
cd $XPLACE_HOME
mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native \
      -DPYTHON_EXECUTABLE=$(which python3) \
      .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install

echo "[build_xplace] Done! XPLACE_HOME=$XPLACE_HOME"
export XPLACE_HOME=$XPLACE_HOME
