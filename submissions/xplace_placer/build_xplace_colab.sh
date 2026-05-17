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

# Patch ALL CMakeLists.txt files: replace every ${CMAKE_CXX_ABI} with hardcoded 1.
# Using Python str.replace (not sed) to avoid bash/regex escaping of cmake's ${} syntax.
# Xplace cmake reads torch._C._GLIBCXX_USE_CXX11_ABI (always returns 0) and then calls
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}), which comes AFTER any
# CMAKE_CXX_FLAGS in GCC's compile command — so it always wins. Direct file patch wins.
python3 << 'PYEOF'
import os
for root, dirs, files in os.walk('.'):
    for fname in files:
        if fname == 'CMakeLists.txt':
            path = os.path.join(root, fname)
            with open(path) as f:
                content = f.read()
            marker = '${CMAKE_CXX_ABI}'
            if marker in content:
                count = content.count(marker)
                with open(path, 'w') as f:
                    f.write(content.replace(marker, '1'))
                print(f"[build_xplace] Patched {count} occurrence(s) in {path}")
PYEOF
echo "[build_xplace] ABI patch done."

mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
      -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
      .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install

echo "[build_xplace] Installing Xplace Python dependencies..."
pip install igraph --quiet

echo "[build_xplace] Done! XPLACE_HOME=$XPLACE_HOME"
export XPLACE_HOME=$XPLACE_HOME
