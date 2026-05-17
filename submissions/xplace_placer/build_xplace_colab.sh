#!/bin/bash
# Build Xplace on Google Colab T4 (CUDA 12.x, GCC 11+, CMake 3.27+)
# Run this cell ONCE before evaluating the Xplace placer.
# Takes ~5-10 minutes to compile.

set -e

XPLACE_HOME=/opt/xplace

if [ -f "$XPLACE_HOME/main.py" ] && [ -f "$XPLACE_HOME/cpp_to_py/cpybin/io_parser.cpython"*".so" ]; then
    echo "[build_xplace] Already built at $XPLACE_HOME. Skipping."
    exit 0
fi

echo "[build_xplace] Installing system dependencies..."
apt-get install -y --no-install-recommends libboost-all-dev libcairo2-dev bison flex libfl-dev > /dev/null 2>&1

echo "[build_xplace] Cloning Xplace..."
rm -rf $XPLACE_HOME
git clone --depth 1 --recurse-submodules https://github.com/cuhk-eda/Xplace $XPLACE_HOME

echo "[build_xplace] Building Xplace (this takes ~5-10 min)..."
cd $XPLACE_HOME

# PyTorch ≥2.0 on Colab uses the C++11 ABI (compiled_with_cxx11_abi=True),
# but Xplace's cmake reads torch._C._GLIBCXX_USE_CXX11_ABI which returns 0
# (a legacy attribute that doesn't reflect the actual build). Force ABI=1 so
# the .so symbols match what PyTorch exports, avoiding undefined symbol errors.
TORCH_ABI=$(python3 -c "import torch; print(int(torch.compiled_with_cxx11_abi()))" 2>/dev/null || echo "1")
echo "[build_xplace] PyTorch compiled_with_cxx11_abi=$TORCH_ABI"

# Also patch Xplace's cmake so its own ABI detection doesn't override our flag.
# The cmake sets _GLIBCXX_USE_CXX11_ABI via torch._C._GLIBCXX_USE_CXX11_ABI;
# replace that with the correct runtime value.
sed -i "s/torch\._C\._GLIBCXX_USE_CXX11_ABI/torch.compiled_with_cxx11_abi()/g" \
    CMakeLists.txt 2>/dev/null || true

mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}" \
      -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}" \
      .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install

echo "[build_xplace] Done! XPLACE_HOME=$XPLACE_HOME"
export XPLACE_HOME=$XPLACE_HOME
