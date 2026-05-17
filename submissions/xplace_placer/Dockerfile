# Xplace macro placer — Dockerfile
#
# Build: docker build -t xplace-placer -f submissions/xplace_placer/Dockerfile .
# Run:   docker run --gpus all xplace-placer /submission/placer.py --all
#
# Build-time: clones Xplace from GitHub, compiles CUDA extensions.
# Runtime: --network none (evaluation rules), GPU required.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive

# --- Build dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        gcc \
        g++ \
        make \
        bison \
        flex \
        libboost-all-dev \
        libcairo2-dev \
        && rm -rf /var/lib/apt/lists/*

# CMake 3.24+ (apt has older versions on Ubuntu 20.04)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh \
        -O /tmp/cmake.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake.sh

# Clone Xplace (shallow)
RUN git clone --depth 1 --recurse-submodules \
        https://github.com/cuhk-eda/Xplace /opt/xplace

# Build Xplace C++ / CUDA extensions
RUN cd /opt/xplace && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_CUDA_ARCHITECTURES=native \
          -DPYTHON_EXECUTABLE=$(which python) \
          .. && \
    make -j$(nproc) && \
    make install

# --- Runtime image ---
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-all-dev \
        libcairo2 \
        && rm -rf /var/lib/apt/lists/*

# Copy built Xplace from builder
COPY --from=builder /opt/xplace /opt/xplace
ENV XPLACE_HOME=/opt/xplace

# Challenge infrastructure
WORKDIR /challenge
COPY macro_place/ /challenge/macro_place/
COPY pyproject.toml /challenge/pyproject.toml
COPY requirements.txt /challenge/requirements.txt
COPY external/MacroPlacement/Testcases/ICCAD04/ /challenge/external/MacroPlacement/Testcases/ICCAD04/
COPY external/MacroPlacement/CodeElements/Plc_client/ /challenge/external/MacroPlacement/CodeElements/Plc_client/
COPY submissions/ /challenge/submissions/

# Install Python dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir scipy tqdm

ENTRYPOINT ["python", "-m", "macro_place.evaluate"]
