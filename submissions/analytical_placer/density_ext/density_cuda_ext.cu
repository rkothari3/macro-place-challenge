/*
 * density_cuda_ext.cu — Tiled CUDA Kernel for Differentiable Density Loss
 *
 * Replaces the chunked PyTorch loop in density_loss() with a single GPU kernel launch.
 *
 * Build (on Colab T4 or any CUDA-capable machine):
 *   cd submissions/analytical_placer/density_ext && pip install -e .
 *
 * === WHAT THIS COMPUTES ===
 *   Forward:  cell_density[g] = sum_i overlap_area(macro_i, cell_g) / cell_area
 *   Backward: grad_pos[i, :] = sum_g grad_density[g] * d(overlap_area)/d(pos[i]) / cell_area
 *
 * === WHY THE PYTORCH LOOP IS SLOW ===
 *   density_loss() processes macros in chunks of 256. For ibm01 (N=1140, G=1845):
 *     - 5 chunk iterations, each launches ~9 PyTorch ops (maximum, minimum, relu, mul, sum...)
 *     - 5 × 9 = 45 sequential GPU kernel launches for the forward
 *     - Each launch has ~5-15μs overhead → 45 launches × 10μs = 450μs just in dispatch
 *     - Plus backward: another ~45 launches
 *   For ibm17 (N=2604, G=2244, 11 chunks): ~110 kernel launches per forward+backward call
 *   At 300 optimization steps, that's 33,000 kernel launches from density alone.
 *
 *   This kernel: 1 forward launch + 1 backward launch per step. All N×G computations
 *   run in parallel, with shared memory caching macro positions across the cell dimension.
 *
 * === CUDA CONCEPTS (explained before each use) ===
 *
 * [1] SHARED MEMORY
 *   GPU memory hierarchy (fastest to slowest, with approximate latencies):
 *     Registers       ~1 cycle    — private to each thread, fastest
 *     Shared memory   ~5 cycles   — shared by all threads in a block, 48KB per SM
 *     L1 cache        ~20 cycles  — automatic, 32KB per SM
 *     L2 cache        ~100 cycles — chip-wide
 *     Global memory   ~200-800 cycles — where tensors live (off-chip HBM/GDDR)
 *
 *   Shared memory is ~100x faster than global memory. We use it to cache macro
 *   positions/sizes for the 16 macros in a tile. Each macro is read by BLOCK_G=16
 *   cell-threads in the block; without shared memory, that's 16 global reads per macro.
 *   With shared memory: 1 global read → cache → 16 threads reuse from shared mem.
 *   Savings: (BLOCK_G - 1) × (4 floats per macro) × N threads = 60 fewer global reads
 *   per block per tile.
 *
 * [2] __syncthreads()
 *   A barrier that ALL threads in a block must reach before ANY can proceed past it.
 *   Required after the shared memory load phase: only 16 threads (tx=0) load macro data
 *   and only 16 threads (ty=0) load cell data. The other 224 threads are idle and might
 *   rush ahead to read shared memory before it's populated — a race condition.
 *   __syncthreads() prevents this: it ensures the 16 loading threads finish writing
 *   before any of the 256 threads start reading.
 *
 * [3] atomicAdd
 *   Multiple macros can overlap the same grid cell. If two threads simultaneously do:
 *     cell_density[g] += contribution_from_macro_A  // reads old value, writes new
 *     cell_density[g] += contribution_from_macro_B  // reads OLD value (before A wrote!)
 *   The result is wrong (one update is lost). atomicAdd makes the read-modify-write
 *   indivisible: it completes A's update before B can read the value.
 *
 *   Performance note: atomicAdd serializes threads writing to the SAME address.
 *   Our thread layout (threadIdx.x = cell dimension) ensures that all threads in the
 *   SAME WARP write to DIFFERENT cell_density indices → no serialization, near-free.
 *   Only threads from DIFFERENT blocks (different macros, same cell) serialize.
 *
 * [4] THREAD INDEXING AND LAYOUT
 *   Forward grid: dim3 grid(ceil(G/BLOCK_G), ceil(N/BLOCK_N))
 *                 dim3 block(BLOCK_G, BLOCK_N)
 *   Thread (tx = threadIdx.x, ty = threadIdx.y) in block (bx, by):
 *     cell_idx  = bx * BLOCK_G + tx   (threadIdx.x = fastest-varying → coalesced)
 *     macro_idx = by * BLOCK_N + ty
 *
 * [5] MEMORY COALESCING
 *   A warp is 32 consecutive threads. Coalesced access = consecutive threads read
 *   consecutive addresses (one cache line = 128 bytes = 32 floats).
 *   With tx = cell dim (threadIdx.x fastest):
 *     cell_xy[cell_idx, :]: consecutive tx → consecutive cell_idx → COALESCED ✓
 *     atomicAdd(cell_density[cell_idx]): consecutive tx → consecutive addresses → ✓
 *   pos[macro_idx, :]: shared memory load by tx=0 threads (16 sequential macros) → coalesced ✓
 *
 * Arch targets: T4=sm_75, A100=sm_80, RTX6000Ada=sm_89
 * (torch BuildExtension auto-detects the GPU)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions. 16×16 = 256 threads/block (one full 8-warp CTA).
// Why 16: shared mem per block = 6 arrays × 16 floats × 4 bytes = 384 bytes
// (fits 128 such blocks per SM on a T4's 48KB shared mem budget).
// Why not 32×32=1024: shared mem would be 1536 bytes/block and register pressure rises.
// 16×16 hits the sweet spot: full warp utilization + minimal shared mem per block.
#define BLOCK_G 16   /* threads in x-dimension (cell tile) */
#define BLOCK_N 16   /* threads in y-dimension (macro tile) */

// ============================================================
// FORWARD KERNEL: cell_density[G]
// ============================================================

__global__ void density_forward_kernel(
    const float* __restrict__ pos,         // [N, 2] macro centers (x, y)
    const float* __restrict__ sizes,       // [N, 2] macro sizes (w, h)
    const float* __restrict__ cell_xy,     // [G, 2] cell centers (x, y)
    float* __restrict__ cell_density,      // [G] output (pre-zeroed by torch::zeros)
    int N, int G,
    float half_cw,      // half of cell width
    float half_ch,      // half of cell height
    float inv_cell_area // 1.0 / (cell_width * cell_height)
) {
    // --- Thread → (cell, macro) mapping ---
    // tx = cell dimension (fastest-varying, enables coalesced reads of cell_xy)
    // ty = macro dimension
    int tx = threadIdx.x;   // [0, BLOCK_G)
    int ty = threadIdx.y;   // [0, BLOCK_N)
    int cell_idx  = blockIdx.x * BLOCK_G + tx;
    int macro_idx = blockIdx.y * BLOCK_N + ty;

    // -------------------------------------------------------
    // [1] SHARED MEMORY — cache macro pos+sizes for this tile
    // Each of the BLOCK_N=16 macros in this block will be used by BLOCK_G=16 cell-threads.
    // Without shared memory: 16 threads each read pos[macro_idx, :] = 16 global loads.
    // With shared memory: 1 thread (tx==0) reads → 16 threads reuse from 5-cycle cache.
    // -------------------------------------------------------
    __shared__ float s_cx[BLOCK_N];   // macro center x
    __shared__ float s_cy[BLOCK_N];   // macro center y
    __shared__ float s_hw[BLOCK_N];   // macro half-width
    __shared__ float s_hh[BLOCK_N];   // macro half-height

    // Also cache cell centers — each cell is read by BLOCK_N=16 macro-threads.
    __shared__ float s_gx[BLOCK_G];   // cell center x
    __shared__ float s_gy[BLOCK_G];   // cell center y

    // Load macro data: tx=0 column (one thread per macro in this tile)
    if (tx == 0) {
        int mi = blockIdx.y * BLOCK_N + ty;
        if (mi < N) {
            s_cx[ty] = pos[mi * 2];
            s_cy[ty] = pos[mi * 2 + 1];
            s_hw[ty] = sizes[mi * 2]     * 0.5f;
            s_hh[ty] = sizes[mi * 2 + 1] * 0.5f;
        }
    }
    // Load cell data: ty=0 row (one thread per cell in this tile)
    if (ty == 0) {
        int gi = blockIdx.x * BLOCK_G + tx;
        if (gi < G) {
            s_gx[tx] = cell_xy[gi * 2];
            s_gy[tx] = cell_xy[gi * 2 + 1];
        }
    }

    // -------------------------------------------------------
    // [2] __syncthreads() — barrier before reading shared mem
    // Only 16 (tx=0) threads wrote to s_cx/cy/hw/hh.
    // Only 16 (ty=0) threads wrote to s_gx/gy.
    // The other 224 threads must wait here until all 32 loads are complete.
    // Without this: threads reading s_cx[ty] might see garbage (stale cache line).
    // -------------------------------------------------------
    __syncthreads();

    // --- Compute overlap for (macro_idx, cell_idx) ---
    if (macro_idx < N && cell_idx < G) {
        float cx = s_cx[ty];   // fast: shared memory (~5 cycles)
        float cy = s_cy[ty];
        float hw = s_hw[ty];
        float hh = s_hh[ty];
        float gx = s_gx[tx];
        float gy = s_gy[tx];

        // Exact rectangle overlap (differentiable via ReLU = fmaxf(0, ...)):
        //   overlap_x = relu(min(cx+hw, gx+half_cw) - max(cx-hw, gx-half_cw))
        float lo_x = fmaxf(cx - hw, gx - half_cw);
        float hi_x = fminf(cx + hw, gx + half_cw);
        float lo_y = fmaxf(cy - hh, gy - half_ch);
        float hi_y = fminf(cy + hh, gy + half_ch);

        float ov_x = fmaxf(0.0f, hi_x - lo_x);
        float ov_y = fmaxf(0.0f, hi_y - lo_y);
        float contrib = ov_x * ov_y * inv_cell_area;

        if (contrib > 0.0f) {
            // -------------------------------------------------------
            // [3] atomicAdd — prevent race when multiple macros overlap same cell
            // Consecutive tx threads write to consecutive cell_density[cell_idx] addresses
            // (cell_idx = bx*16 + tx → stride 1), so different threads in the same warp
            // target DIFFERENT addresses → no serialization within a warp.
            // Serialization only occurs between blocks that share the same cell_idx.
            // -------------------------------------------------------
            atomicAdd(&cell_density[cell_idx], contrib);
        }
    }
}

// ============================================================
// BACKWARD KERNEL: grad_pos[N, 2] from grad_density[G]
// ============================================================
//
// Why 1D, not 2D like the forward?
//   Each macro i owns its own grad_pos[i, :] — no atomicAdd needed.
//   One thread per macro loops over all G cells in a tight inner loop.
//   The inner loop (G ~1845-2244) fits in registers and L1 cache.
//   A 2D backward would require atomicAdd on grad_pos (multiple cells contribute
//   to same macro gradient), serializing threads — slower than a 1D loop.
//
// Gradient derivation:
//   density_g = sum_i overlap_x(i,g) * overlap_y(i,g) * inv_cell_area
//
//   d(density_g)/d(cx_i):
//     overlap_x = relu(hi_x - lo_x) where hi_x = min(cx+hw, gx+half_cw)
//                                          lo_x = max(cx-hw, gx-half_cw)
//     When overlap_x > 0:
//       d(hi_x)/d(cx_i) = 1 if (cx+hw < gx+half_cw) else 0  ← right edge inside cell
//       d(lo_x)/d(cx_i) = 1 if (cx-hw > gx-half_cw) else 0  ← left edge inside cell
//       d(overlap_x)/d(cx_i) = [right_inside] - [left_inside]
//     This is piecewise constant: 0, +1, or -1.
//
//   grad_pos[i, 0] = sum_g grad_density[g] * d(overlap_x)/d(cx_i) * overlap_y * inv_cell_area
//   grad_pos[i, 1] = sum_g grad_density[g] * overlap_x * d(overlap_y)/d(cy_i) * inv_cell_area

__global__ void density_backward_kernel(
    const float* __restrict__ grad_density, // [G] upstream gradient (from autograd)
    const float* __restrict__ pos,          // [N, 2] saved from forward
    const float* __restrict__ sizes,        // [N, 2] saved from forward
    const float* __restrict__ cell_xy,      // [G, 2] saved from forward
    float* __restrict__ grad_pos,           // [N, 2] output
    int N, int G,
    float half_cw, float half_ch, float inv_cell_area
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float cx = pos[2 * i];
    float cy = pos[2 * i + 1];
    float hw = sizes[2 * i]     * 0.5f;
    float hh = sizes[2 * i + 1] * 0.5f;

    float gx_grad = 0.0f;
    float gy_grad = 0.0f;

    // Inner loop: sweep all G cells for this macro.
    // Cell data is accessed sequentially → L1/L2 cache friendly (stride-2 reads of cell_xy).
    for (int g = 0; g < G; g++) {
        float gd = grad_density[g];
        if (gd == 0.0f) continue;   // no upstream gradient → skip (common after relu)

        float gx = cell_xy[2 * g];
        float gy = cell_xy[2 * g + 1];

        // Recompute overlap (same arithmetic as forward)
        float lo_x = fmaxf(cx - hw, gx - half_cw);
        float hi_x = fminf(cx + hw, gx + half_cw);
        float lo_y = fmaxf(cy - hh, gy - half_ch);
        float hi_y = fminf(cy + hh, gy + half_ch);

        float ov_x = fmaxf(0.0f, hi_x - lo_x);
        float ov_y = fmaxf(0.0f, hi_y - lo_y);
        if (ov_x == 0.0f || ov_y == 0.0f) continue;   // zero overlap → zero gradient

        // Piecewise-constant gradient of overlap_x w.r.t. cx:
        //   +1 if right macro edge is inside cell  (hi_x is the macro's right edge)
        //   -1 if left macro edge is inside cell   (lo_x is the macro's left edge)
        //    0 if cell is completely inside macro (both edges are cell edges, cancel)
        float d_ov_x_dcx = (cx + hw < gx + half_cw ? 1.0f : 0.0f)
                         - (cx - hw > gx - half_cw ? 1.0f : 0.0f);
        float d_ov_y_dcy = (cy + hh < gy + half_ch ? 1.0f : 0.0f)
                         - (cy - hh > gy - half_ch ? 1.0f : 0.0f);

        // Chain rule: d(loss)/d(cx_i) += grad_density[g] * d(ov_x*ov_y)/d(cx_i) / cell_area
        gx_grad += gd * d_ov_x_dcx * ov_y * inv_cell_area;
        gy_grad += gd * ov_x * d_ov_y_dcy * inv_cell_area;
    }

    grad_pos[2 * i]     = gx_grad;
    grad_pos[2 * i + 1] = gy_grad;
}

// ============================================================
// C++ wrapper functions (called from Python)
// ============================================================

torch::Tensor density_forward_cuda(
    torch::Tensor pos,        // [N, 2] float32 CUDA
    torch::Tensor sizes,      // [N, 2] float32 CUDA
    torch::Tensor cell_xy,    // [G, 2] float32 CUDA
    float half_cw,
    float half_ch,
    float inv_cell_area
) {
    TORCH_CHECK(pos.is_cuda(),      "pos must be a CUDA tensor");
    TORCH_CHECK(sizes.is_cuda(),    "sizes must be a CUDA tensor");
    TORCH_CHECK(cell_xy.is_cuda(),  "cell_xy must be a CUDA tensor");
    TORCH_CHECK(pos.dtype() == torch::kFloat32,     "pos must be float32");
    TORCH_CHECK(sizes.dtype() == torch::kFloat32,   "sizes must be float32");
    TORCH_CHECK(cell_xy.dtype() == torch::kFloat32, "cell_xy must be float32");

    pos     = pos.contiguous();
    sizes   = sizes.contiguous();
    cell_xy = cell_xy.contiguous();

    int N = pos.size(0);
    int G = cell_xy.size(0);

    // Pre-zero the output: atomicAdd accumulates into a zeroed buffer.
    auto cell_density = torch::zeros({G}, pos.options());

    // [4] Grid/block sizing:
    //   blockDim = (BLOCK_G=16, BLOCK_N=16) = 256 threads
    //   gridDim.x = ceil(G / BLOCK_G) — tiles covering all cells
    //   gridDim.y = ceil(N / BLOCK_N) — tiles covering all macros
    dim3 block(BLOCK_G, BLOCK_N);
    dim3 grid(
        (G + BLOCK_G - 1) / BLOCK_G,
        (N + BLOCK_N - 1) / BLOCK_N
    );

    density_forward_kernel<<<grid, block>>>(
        pos.data_ptr<float>(),
        sizes.data_ptr<float>(),
        cell_xy.data_ptr<float>(),
        cell_density.data_ptr<float>(),
        N, G, half_cw, half_ch, inv_cell_area
    );

    return cell_density;   // [G] — differentiable via _DensityKernel autograd.Function
}

torch::Tensor density_backward_cuda(
    torch::Tensor grad_density,   // [G] upstream gradient from autograd
    torch::Tensor pos,            // [N, 2] saved from forward
    torch::Tensor sizes,          // [N, 2] saved from forward
    torch::Tensor cell_xy,        // [G, 2] saved from forward
    float half_cw,
    float half_ch,
    float inv_cell_area
) {
    TORCH_CHECK(grad_density.is_cuda(), "grad_density must be a CUDA tensor");
    TORCH_CHECK(pos.is_cuda(),          "pos must be a CUDA tensor");

    grad_density = grad_density.contiguous();
    pos          = pos.contiguous();
    sizes        = sizes.contiguous();
    cell_xy      = cell_xy.contiguous();

    int N = pos.size(0);
    int G = cell_xy.size(0);

    auto grad_pos = torch::zeros({N, 2}, pos.options());

    // 1D grid: one thread per macro. Each thread loops over all G cells.
    // Block size 256: 8 warps, good occupancy. Backward is compute-bound (G iterations/thread).
    const int BLOCK = 256;
    density_backward_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(
        grad_density.data_ptr<float>(),
        pos.data_ptr<float>(),
        sizes.data_ptr<float>(),
        cell_xy.data_ptr<float>(),
        grad_pos.data_ptr<float>(),
        N, G, half_cw, half_ch, inv_cell_area
    );

    return grad_pos;   // [N, 2]
}

// ============================================================
// Standalone correctness + timing test (run as Python script,
// see density_ext/test_density_kernel.py)
// ============================================================
//
// Hardware test (10 macros, 9 cells) is implemented in Python for convenience.
// To run: python density_ext/test_density_kernel.py
//
// Expected output:
//   Max abs error vs PyTorch reference: < 1e-4
//   CUDA forward 100-iter avg: X.XX ms  (should be < 0.5ms for ibm17 scale)
//   CUDA backward 100-iter avg: X.XX ms

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &density_forward_cuda,
          "Tiled density forward kernel (cell_density = sum_i overlap_area/cell_area)");
    m.def("backward", &density_backward_cuda,
          "Density backward kernel (grad_pos from grad_density)");
}
