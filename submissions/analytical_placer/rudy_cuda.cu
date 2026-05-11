/*
 * ============================================================
 * RUDY CUDA Kernel — Learning Implementation
 * Rectangular Uniform wire DensitY congestion surrogate
 *
 * Purpose: This file teaches CUDA programming through the lens of
 * the RUDY algorithm used in chip placement. Every design decision
 * is explained. Read it top-to-bottom to learn CUDA from scratch.
 *
 * Build: nvcc -O2 -arch=sm_75 -o rudy_test rudy_cuda.cu
 *        (sm_75 = Turing / Tesla T4 compute capability)
 * Run:   ./rudy_test
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ============================================================
// SECTION 1: Why CUDA for RUDY?
// ============================================================
//
// RUDY has a perfect parallelism structure: nets are INDEPENDENT.
// Net #5's routing demand depends only on net #5's pins. It does
// NOT depend on net #8732. This is called "embarrassing parallelism."
//
// On a CPU, processing 45,825 nets takes ~45ms (sequential).
// On a T4 GPU with 2,560 CUDA cores, all nets can run simultaneously.
//
// The only complication: multiple nets scatter demand into the SAME
// grid cells. This creates write conflicts (race conditions) that
// we solve with atomicAdd — explained in Section 3.
//
// Without CUDA: O(num_nets) sequential
// With CUDA:    O(max_cells_per_bbox) parallel — ~10-100x faster
//               for large netlists like ibm17 (45,825 nets)

// ============================================================
// SECTION 2: Memory Layout and Access Patterns
// ============================================================
//
// Input arrays (device global memory):
//   net_x_min[n], net_x_max[n]  float — bbox x range for net n
//   net_y_min[n], net_y_max[n]  float — bbox y range for net n
//   net_weights[n]               float — weight for net n
//
// Output array (device global memory):
//   demand_grid[row * grid_cols + col]  float — routing demand per cell
//
// Memory ACCESS PATTERN: each thread reads 5 floats (its net's data)
// from sequential addresses:
//   net_x_min[0], net_x_min[1], ..., net_x_min[255]   <- block 0, coalesced!
//   net_x_min[256], ..., net_x_min[511]                <- block 1, coalesced!
//
// "Coalesced" means threads in a warp (32 threads) access consecutive
// memory addresses. The GPU memory controller can serve 32 threads in
// a single wide transaction instead of 32 separate ones. ~32x bandwidth.
//
// The demand_grid WRITES (via atomicAdd) are NOT coalesced — each
// thread writes to a different cell. This is unavoidable for scatter
// operations and is why shared memory tiling can help (see Section 5).

// ============================================================
// SECTION 3: The Kernel
// ============================================================

/*
 * rudy_kernel — One thread per net
 *
 * Each thread:
 * 1. Loads its net's bounding box from global memory (coalesced read)
 * 2. Computes which grid cells overlap the bbox
 * 3. For each overlapping cell, atomically adds routing demand
 *
 * Grid organization:
 *   blockDim.x = BLOCK_SIZE (256)
 *   gridDim.x  = ceil(num_nets / BLOCK_SIZE)
 *   Total threads = gridDim.x * blockDim.x >= num_nets
 */
__global__ void rudy_kernel(
    const float* __restrict__ net_x_min,    // [num_nets]
    const float* __restrict__ net_x_max,    // [num_nets]
    const float* __restrict__ net_y_min,    // [num_nets]
    const float* __restrict__ net_y_max,    // [num_nets]
    const float* __restrict__ net_weights,  // [num_nets]
    float*                    demand_grid,  // [grid_rows * grid_cols] OUTPUT
    float canvas_w,
    float canvas_h,
    int   grid_rows,
    int   grid_cols,
    int   num_nets
) {
    // ----------------------------------------------------------
    // Step 1: Compute global thread index = which net I handle
    //
    // WHY THIS FORMULA:
    //   blockIdx.x  = which block (0, 1, 2, ...)
    //   blockDim.x  = threads per block (always 256 in our case)
    //   threadIdx.x = my position within my block (0..255)
    //
    // Example: block 3, thread 7 → net_id = 3*256 + 7 = 775
    // Thread 775 processes net #775's bounding box.
    // ----------------------------------------------------------
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;

    // ----------------------------------------------------------
    // Step 2: Bounds check
    //
    // WHY WE OVERSHOOT: We launched ceil(num_nets/256)*256 threads,
    // which may be > num_nets. For example, 45,825 nets → 180 blocks
    // × 256 = 46,080 threads. Threads 45,825–46,079 have no net.
    // Without this check, they would read garbage from out-of-bounds
    // memory and write nonsense to demand_grid. Exit early instead.
    // ----------------------------------------------------------
    if (net_id >= num_nets) return;

    // ----------------------------------------------------------
    // Step 3: Load this net's bounding box
    //
    // These 5 loads from global memory are COALESCED:
    // Thread 0 loads net_x_min[0], thread 1 loads net_x_min[1], ...
    // All 32 threads in a warp load consecutive addresses in one
    // memory transaction. This is the ideal GPU access pattern.
    // ----------------------------------------------------------
    float xmin = net_x_min[net_id];
    float xmax = net_x_max[net_id];
    float ymin = net_y_min[net_id];
    float ymax = net_y_max[net_id];
    float w    = net_weights[net_id];

    // Cell dimensions
    float cell_w = canvas_w / grid_cols;
    float cell_h = canvas_h / grid_rows;

    // Epsilon: prevent division by zero for point nets (all pins co-located)
    float min_bbox_w = cell_w * 0.5f;
    float min_bbox_h = cell_h * 0.5f;
    float bbox_w = fmaxf(xmax - xmin, min_bbox_w);
    float bbox_h = fmaxf(ymax - ymin, min_bbox_h);
    float routing_density = w / (bbox_w * bbox_h);  // weight per μm²

    // ----------------------------------------------------------
    // Step 4: Find grid cell range overlapping this bbox
    //
    // WHY CLAMP TO [0, grid_cols-1]:
    // A net's bbox might extend slightly outside the canvas due to
    // pin positions at exact boundaries. Without clamping, we would
    // compute negative column indices or indices >= grid_cols, causing
    // out-of-bounds writes to demand_grid.
    //
    // col_min = first column whose right edge is > xmin
    //         = floor(xmin / cell_w)     [inclusive start]
    // col_max = last column whose left edge is < xmax
    //         = floor((xmax - eps) / cell_w)   [inclusive end]
    // ----------------------------------------------------------
    int col_min = (int)(xmin / cell_w);
    int col_max = (int)(xmax / cell_w);  // inclusive
    int row_min = (int)(ymin / cell_h);
    int row_max = (int)(ymax / cell_h);  // inclusive

    // Clamp to valid grid range
    col_min = max(col_min, 0);
    col_max = min(col_max, grid_cols - 1);
    row_min = max(row_min, 0);
    row_max = min(row_max, grid_rows - 1);

    // ----------------------------------------------------------
    // Step 5: Scatter demand to each overlapping cell
    //
    // For each overlapping cell, compute the overlap area between
    // the net's bbox and that cell rectangle, then add:
    //   demand += routing_density * overlap_area
    //
    // This is equivalent to the PyTorch bilinear overlap formula.
    // The inner loop over cells is sequential PER THREAD, but all
    // threads run simultaneously — fully parallel across nets.
    //
    // Typical bbox spans 2-10 cells in each dimension, so this
    // inner loop has ~4-100 iterations per thread.
    // ----------------------------------------------------------
    for (int row = row_min; row <= row_max; row++) {
        float cell_y_lo = row * cell_h;
        float cell_y_hi = cell_y_lo + cell_h;

        // Overlap height between bbox y-range and this row
        float ov_y = fminf(ymax, cell_y_hi) - fmaxf(ymin, cell_y_lo);
        if (ov_y <= 0.0f) continue;  // shouldn't happen, but safe

        for (int col = col_min; col <= col_max; col++) {
            float cell_x_lo = col * cell_w;
            float cell_x_hi = cell_x_lo + cell_w;

            // Overlap width between bbox x-range and this column
            float ov_x = fminf(xmax, cell_x_hi) - fmaxf(xmin, cell_x_lo);
            if (ov_x <= 0.0f) continue;

            // Routing demand contribution to this cell
            float contribution = routing_density * ov_x * ov_y;

            // --------------------------------------------------
            // atomicAdd: the ONLY correct way to accumulate
            // from multiple threads into a shared cell.
            //
            // WITHOUT atomicAdd (the bug):
            //   Thread A reads demand[42] = 5.0
            //   Thread B reads demand[42] = 5.0  ← sees stale value
            //   Thread A writes demand[42] = 5.3
            //   Thread B writes demand[42] = 5.7  ← OVERWRITES A!
            //   Result: 5.7 instead of correct 5.3+0.7 = 6.0
            //   Thread A's 0.3 contribution is LOST.
            //
            // WITH atomicAdd:
            //   Hardware serializes all writers to this address.
            //   Each read-modify-write is atomic (uninterruptible).
            //   All contributions are correctly accumulated.
            //   Order may vary but result is always correct (+ commutative).
            // --------------------------------------------------
            atomicAdd(&demand_grid[row * grid_cols + col], contribution);
        }
    }
}

// ============================================================
// SECTION 4: Host Code (CPU side)
// ============================================================

// CPU reference implementation for validation
void rudy_cpu_reference(
    const float* net_x_min, const float* net_x_max,
    const float* net_y_min, const float* net_y_max,
    const float* net_weights,
    float* demand_grid,
    float canvas_w, float canvas_h,
    int grid_rows, int grid_cols, int num_nets
) {
    float cell_w = canvas_w / grid_cols;
    float cell_h = canvas_h / grid_rows;

    for (int n = 0; n < num_nets; n++) {
        float xmin = net_x_min[n], xmax = net_x_max[n];
        float ymin = net_y_min[n], ymax = net_y_max[n];
        float w = net_weights[n];

        float bbox_w = fmaxf(xmax - xmin, cell_w * 0.5f);
        float bbox_h = fmaxf(ymax - ymin, cell_h * 0.5f);
        float density = w / (bbox_w * bbox_h);

        int col_min = (int)(xmin / cell_w);
        int col_max = (int)(xmax / cell_w);
        int row_min = (int)(ymin / cell_h);
        int row_max = (int)(ymax / cell_h);

        col_min = (col_min < 0) ? 0 : col_min;
        col_max = (col_max >= grid_cols) ? grid_cols - 1 : col_max;
        row_min = (row_min < 0) ? 0 : row_min;
        row_max = (row_max >= grid_rows) ? grid_rows - 1 : row_max;

        for (int r = row_min; r <= row_max; r++) {
            float cy_lo = r * cell_h, cy_hi = cy_lo + cell_h;
            float ov_y = fminf(ymax, cy_hi) - fmaxf(ymin, cy_lo);
            if (ov_y <= 0.0f) continue;
            for (int c = col_min; c <= col_max; c++) {
                float cx_lo = c * cell_w, cx_hi = cx_lo + cell_w;
                float ov_x = fminf(xmax, cx_hi) - fmaxf(xmin, cx_lo);
                if (ov_x <= 0.0f) continue;
                demand_grid[r * grid_cols + c] += density * ov_x * ov_y;
            }
        }
    }
}

// Comparison helper: check if two arrays match within tolerance
int arrays_match(const float* a, const float* b, int n, float tol) {
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            if (mismatches < 5)
                printf("  Mismatch at cell %d: CPU=%.6f GPU=%.6f diff=%.2e\n",
                       i, a[i], b[i], diff);
            mismatches++;
        }
    }
    printf("Max absolute difference: %.2e   Tolerance: %.2e\n", max_diff, tol);
    return mismatches == 0;
}

int main() {
    // ----------------------------------------------------------
    // Test parameters (approximating ibm01 scale)
    // ----------------------------------------------------------
    const int    NUM_NETS   = 5993;
    const int    GRID_ROWS  = 41;
    const int    GRID_COLS  = 45;
    const float  CANVAS_W   = 22.9f;   // μm
    const float  CANVAS_H   = 23.0f;   // μm
    const int    BLOCK_SIZE = 256;
    const int    G          = GRID_ROWS * GRID_COLS;  // 1845 cells

    printf("=================================================\n");
    printf("RUDY CUDA Kernel Test\n");
    printf("Nets: %d   Grid: %d x %d = %d cells\n",
           NUM_NETS, GRID_ROWS, GRID_COLS, G);
    printf("Canvas: %.1f x %.1f μm\n", CANVAS_W, CANVAS_H);
    printf("=================================================\n\n");

    // ----------------------------------------------------------
    // Generate synthetic test data (random bboxes on the canvas)
    // ----------------------------------------------------------
    srand(42);
    float* h_xmin = (float*)malloc(NUM_NETS * sizeof(float));
    float* h_xmax = (float*)malloc(NUM_NETS * sizeof(float));
    float* h_ymin = (float*)malloc(NUM_NETS * sizeof(float));
    float* h_ymax = (float*)malloc(NUM_NETS * sizeof(float));
    float* h_w    = (float*)malloc(NUM_NETS * sizeof(float));

    for (int n = 0; n < NUM_NETS; n++) {
        float x1 = CANVAS_W * ((float)rand() / RAND_MAX);
        float x2 = CANVAS_W * ((float)rand() / RAND_MAX);
        float y1 = CANVAS_H * ((float)rand() / RAND_MAX);
        float y2 = CANVAS_H * ((float)rand() / RAND_MAX);
        h_xmin[n] = fminf(x1, x2);
        h_xmax[n] = fmaxf(x1, x2);
        h_ymin[n] = fminf(y1, y2);
        h_ymax[n] = fmaxf(y1, y2);
        h_w[n]    = 1.0f;  // uniform weights for simplicity
    }

    // ----------------------------------------------------------
    // CPU reference computation
    // ----------------------------------------------------------
    float* h_demand_cpu = (float*)calloc(G, sizeof(float));
    printf("Running CPU reference...\n");
    clock_t t0 = clock();
    rudy_cpu_reference(h_xmin, h_xmax, h_ymin, h_ymax, h_w,
                       h_demand_cpu, CANVAS_W, CANVAS_H,
                       GRID_ROWS, GRID_COLS, NUM_NETS);
    clock_t t1 = clock();
    double cpu_ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("CPU time: %.2f ms\n\n", cpu_ms);

    // ----------------------------------------------------------
    // GPU allocation and data transfer (Host → Device)
    //
    // cudaMalloc: allocate device memory (GPU DRAM, global memory)
    // cudaMemcpy with cudaMemcpyHostToDevice: transfer CPU array
    //   to GPU. Goes through PCIe bus (~16 GB/s on T4).
    //
    // Rule of thumb: minimize H2D/D2H transfers. For repeated RUDY
    // calls (300 gradient steps), transfer the bboxes ONCE, keep
    // them on GPU, update only what changed.
    // ----------------------------------------------------------
    float *d_xmin, *d_xmax, *d_ymin, *d_ymax, *d_w, *d_demand;
    size_t net_bytes  = NUM_NETS * sizeof(float);
    size_t grid_bytes = G        * sizeof(float);

    cudaMalloc(&d_xmin,   net_bytes);
    cudaMalloc(&d_xmax,   net_bytes);
    cudaMalloc(&d_ymin,   net_bytes);
    cudaMalloc(&d_ymax,   net_bytes);
    cudaMalloc(&d_w,      net_bytes);
    cudaMalloc(&d_demand, grid_bytes);

    cudaMemcpy(d_xmin, h_xmin, net_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xmax, h_xmax, net_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ymin, h_ymin, net_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ymax, h_ymax, net_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,    h_w,    net_bytes, cudaMemcpyHostToDevice);

    // ----------------------------------------------------------
    // Kernel launch configuration
    //
    // blockDim = 256: standard choice, fills a warp (32) × 8,
    //   maximizes occupancy on most GPU architectures.
    //
    // gridDim = ceil(NUM_NETS / BLOCK_SIZE):
    //   ensures we launch at least NUM_NETS threads total.
    //   The remainder threads check the bounds condition and exit.
    //
    // Why 256 specifically?
    //   - Multiple of warp size (32): no wasted lanes
    //   - Fits in shared memory budget for most kernels
    //   - High enough to hide global memory latency (many threads
    //     in flight → GPU switches to another warp while waiting)
    // ----------------------------------------------------------
    int grid_dim = (NUM_NETS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("CUDA launch: %d blocks × %d threads = %d total threads\n",
           grid_dim, BLOCK_SIZE, grid_dim * BLOCK_SIZE);
    printf("Processing %d nets (%d extra idle threads)\n\n",
           NUM_NETS, grid_dim * BLOCK_SIZE - NUM_NETS);

    // ----------------------------------------------------------
    // GPU computation: zero demand_grid, then run kernel
    //
    // cudaMemset: fills device memory with a byte value.
    //   0 in float32 representation is all-zero bytes. Safe to use.
    //
    // cudaDeviceSynchronize: wait for GPU to finish before continuing.
    //   Without this, CPU continues immediately while GPU is still
    //   running. Timing would be wrong; results not yet ready.
    // ----------------------------------------------------------
    cudaMemset(d_demand, 0, grid_bytes);

    // Warm-up run (first kernel launch has JIT overhead on some drivers)
    rudy_kernel<<<grid_dim, BLOCK_SIZE>>>(
        d_xmin, d_xmax, d_ymin, d_ymax, d_w, d_demand,
        CANVAS_W, CANVAS_H, GRID_ROWS, GRID_COLS, NUM_NETS);
    cudaDeviceSynchronize();

    // Timed run
    cudaMemset(d_demand, 0, grid_bytes);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    rudy_kernel<<<grid_dim, BLOCK_SIZE>>>(
        d_xmin, d_xmax, d_ymin, d_ymax, d_w, d_demand,
        CANVAS_W, CANVAS_H, GRID_ROWS, GRID_COLS, NUM_NETS);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("GPU kernel time: %.3f ms\n\n", gpu_ms);

    // ----------------------------------------------------------
    // Copy result back (Device → Host) and validate
    // ----------------------------------------------------------
    float* h_demand_gpu = (float*)calloc(G, sizeof(float));
    cudaMemcpy(h_demand_gpu, d_demand, grid_bytes, cudaMemcpyDeviceToHost);

    // Find top-10 busiest cells
    printf("=== Top-10 Busiest Cells (GPU result) ===\n");
    // Simple selection sort for top-10 (we don't need efficiency here)
    int top_idx[10];
    float top_val[10];
    for (int k = 0; k < 10; k++) {
        float max_val = -1.0f;
        int   max_idx = -1;
        for (int i = 0; i < G; i++) {
            int already = 0;
            for (int j = 0; j < k; j++) if (top_idx[j] == i) { already = 1; break; }
            if (!already && h_demand_gpu[i] > max_val) {
                max_val = h_demand_gpu[i];
                max_idx = i;
            }
        }
        top_idx[k] = max_idx;
        top_val[k] = max_val;
        int r = max_idx / GRID_COLS, c = max_idx % GRID_COLS;
        printf("  #%2d  cell[%2d,%2d] = %.4f\n", k+1, r, c, max_val);
    }
    printf("\n");

    // Validate: GPU vs CPU results should match within floating-point tolerance
    // Note: atomicAdd may produce slightly different summation order than CPU,
    // causing small differences (~1e-5) due to floating-point non-associativity.
    printf("=== Validation: GPU vs CPU ===\n");
    int ok = arrays_match(h_demand_cpu, h_demand_gpu, G, 1e-3f);
    printf("Result: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

    // Print speedup
    if (gpu_ms > 0.0f)
        printf("Speedup: %.1fx  (CPU %.2f ms → GPU %.3f ms)\n\n",
               cpu_ms / gpu_ms, cpu_ms, gpu_ms);

    // ----------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------
    free(h_xmin); free(h_xmax); free(h_ymin); free(h_ymax);
    free(h_w); free(h_demand_cpu); free(h_demand_gpu);
    cudaFree(d_xmin); cudaFree(d_xmax); cudaFree(d_ymin);
    cudaFree(d_ymax); cudaFree(d_w); cudaFree(d_demand);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    printf("Done.\n");
    return 0;
}

// ============================================================
// SECTION 5: What Would Make This Faster?
// ============================================================
//
// 1. SHARED MEMORY TILING
//    Problem: Many threads atomicAdd to the same cells → serialized
//    by hardware. High contention on a coarse grid.
//    Fix: Each block accumulates its contributions in shared memory
//    first, then one thread writes the block's sum to global memory.
//    This reduces global atomics by a factor of BLOCK_SIZE.
//
//    Pseudo-code:
//      __shared__ float block_demand[GRID_ROWS * GRID_COLS];
//      // Each thread accumulates in shared_demand (still needs atomicAdd!)
//      // But shared atomicAdd is ~5x faster than global atomicAdd
//      // Then: atomicAdd(d_demand[c], block_demand[c]) — one per cell per block
//
// 2. WARP-LEVEL REDUCTION
//    For nets with the same bbox range, adjacent threads (same warp)
//    would write to the same cells. Use __reduce_add_sync() to sum
//    within a warp before the global atomicAdd.
//
// 3. KERNEL FUSION
//    Currently: bbox computation (in PyTorch) then RUDY (CUDA)
//    Fused: one kernel reads pin positions, computes LSE bbox,
//    AND scatters demand. Saves a GPU→CPU→GPU round trip.
//
// 4. AVOID TRANSFER OVERHEAD
//    For gradient descent with 300 steps, bbox changes every step.
//    But only the MOVABLE MACRO positions change.
//    Keep all tensors on GPU, update only the changed bboxes.
//
// ============================================================
// SECTION 6: PyTorch vs CUDA Comparison (fill in after profiling)
// ============================================================
//
// [Fill in after running profile_rudy.py on Colab T4]
//
// PyTorch RUDY:     avg over 100 runs = _____ ms
// CUDA kernel:      avg over 100 runs = _____ ms
// Speedup:          _____x
//
// Bottleneck analysis:
// [ ] Memory bandwidth (demand_grid writes via atomicAdd)
// [ ] atomicAdd contention (coarse grid, many nets per cell)
// [ ] Kernel launch overhead (amortized over 300 steps?)
// [ ] Inner loop over bbox cells (for nets with large bboxes)
//
// Could we fuse density + RUDY into one kernel?
// Both density and RUDY scatter per-cell values from net/macro data.
// Fusing saves: 1 kernel launch + 1 demand_grid read-back.
// Cost: larger kernel, harder to tune. Worth it for production code.
