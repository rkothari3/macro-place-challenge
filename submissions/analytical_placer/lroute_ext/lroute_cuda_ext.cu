/*
 * lroute_cuda_ext.cu — PyTorch CUDA Extension for L-route Routing Demand
 *
 * Implements differentiable L-route forward + backward as a custom
 * torch.autograd.Function. The forward is ~7x faster than the equivalent
 * PyTorch scatter_add approach because it avoids the [E, C] and [E, R]
 * intermediate matrices, replacing them with O(E × avg_span) CUDA kernels.
 *
 * Build:
 *   cd submissions/analytical_placer/lroute_ext
 *   pip install -e .
 *
 * Arch targets: T4=sm_75, A100=sm_80, RTX6000Ada=sm_89
 * (torch BuildExtension auto-detects the installed GPU)
 *
 * L-ROUTE SEMANTICS (matching plc_client_os.py):
 *   For each 2-pin edge (src at (sx,sy), snk at (kx,ky)):
 *     H segment: horizontal wire at row ≈ sy, spanning x from min(sx,kx) to max(sx,kx)
 *       → H_demand[bilinear_rows, c] += weight × col_overlap(c, x_min, x_max)
 *     V segment: vertical wire at col ≈ kx, spanning y from min(sy,ky) to max(sy,ky)
 *       → V_demand[r, bilinear_cols] += weight × row_overlap(r, y_min, y_max)
 *
 * GRADIENT DERIVATION (for the backward kernels):
 *   H_demand[row_lo,c] += w * (1-frac_y) * ov_c  where frac_y = frac(sy/ch)
 *   H_demand[row_hi,c] += w * frac_y * ov_c
 *
 *   d(H_demand)/d(sy) via bilinear row weights:
 *     g_sy_H = (w/ch) * sum_c ov_c * (gH[row_hi,c] - gH[row_lo,c])
 *
 *   d(H_demand)/d(x_min) via left-edge column overlap:
 *     At c_start: max(x_min, col_left) = x_min when x_min > col_left
 *     → d(ov)/d(x_min) = -1/cw (only at c_start, only when ov > 0)
 *
 *   d(H_demand)/d(x_max) via right-edge column overlap:
 *     At c_end: min(x_max, col_right) = x_max when x_max < col_right
 *     → d(ov)/d(x_max) = +1/cw (only at c_end, only when ov > 0)
 *
 *   V demand: symmetric (snk_x → col bilinear; src_y/snk_y → row overlap).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// Forward: H demand
// ============================================================

__global__ void h_fwd_kernel(
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ snk_x,
    const float* __restrict__ edge_wt,
    float* __restrict__ H_demand,       // [R, C] — must be pre-zeroed
    int E, int R, int C, float cw, float ch
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    float sx = src_x[e], sy = src_y[e], kx = snk_x[e], w = edge_wt[e];

    // Bilinear row assignment: H segment at row ≈ sy/ch
    float row_f = fmaxf(0.0f, fminf((float)R, sy / ch));
    int row_lo = max(0, min(R-1, (int)floorf(row_f)));
    int row_hi = min(R-1, row_lo + 1);
    float w_hi = fmaxf(0.0f, fminf(1.0f, row_f - (float)row_lo));
    float w_lo = 1.0f - w_hi;

    // Column range [x_min, x_max]
    float x_min = fminf(sx, kx), x_max = fmaxf(sx, kx);
    int c_start = max(0,   (int)floorf(x_min / cw));
    int c_end   = min(C-1, (int)floorf(x_max / cw));

    for (int c = c_start; c <= c_end; c++) {
        float col_left  = (float)c * cw;
        float col_right = col_left + cw;
        float ov = (fminf(x_max, col_right) - fmaxf(x_min, col_left)) / cw;
        if (ov <= 0.0f) continue;
        atomicAdd(&H_demand[row_lo * C + c], w * w_lo * ov);
        atomicAdd(&H_demand[row_hi * C + c], w * w_hi * ov);
    }
}

// ============================================================
// Forward: V demand
// ============================================================

__global__ void v_fwd_kernel(
    const float* __restrict__ src_y,
    const float* __restrict__ snk_x,
    const float* __restrict__ snk_y,
    const float* __restrict__ edge_wt,
    float* __restrict__ V_demand,       // [R, C] — must be pre-zeroed
    int E, int R, int C, float cw, float ch
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    float sy = src_y[e], kx = snk_x[e], ky = snk_y[e], w = edge_wt[e];

    // Bilinear column assignment: V segment at col ≈ kx/cw
    float col_f = fmaxf(0.0f, fminf((float)C, kx / cw));
    int col_lo = max(0,   (int)floorf(col_f));
    col_lo     = min(C-1, col_lo);
    int col_hi = min(C-1, col_lo + 1);
    float c_w_hi = fmaxf(0.0f, fminf(1.0f, col_f - (float)col_lo));
    float c_w_lo = 1.0f - c_w_hi;

    // Row range [y_min, y_max]
    float y_min = fminf(sy, ky), y_max = fmaxf(sy, ky);
    int r_start = max(0,   (int)floorf(y_min / ch));
    int r_end   = min(R-1, (int)floorf(y_max / ch));

    for (int r = r_start; r <= r_end; r++) {
        float row_bot = (float)r * ch;
        float row_top = row_bot + ch;
        float ov = (fminf(y_max, row_top) - fmaxf(y_min, row_bot)) / ch;
        if (ov <= 0.0f) continue;
        atomicAdd(&V_demand[r * C + col_lo], w * c_w_lo * ov);
        atomicAdd(&V_demand[r * C + col_hi], w * c_w_hi * ov);
    }
}

// ============================================================
// Backward: H demand — gradients w.r.t. src_x, src_y, snk_x
// ============================================================

__global__ void h_bwd_kernel(
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ snk_x,
    const float* __restrict__ edge_wt,
    const float* __restrict__ grad_H,  // [R, C] upstream gradient
    float* __restrict__ g_src_y_H,     // [E] out: grad via bilinear row weight
    float* __restrict__ g_src_x,       // [E] out: grad via x_min/x_max column span
    float* __restrict__ g_snk_x_H,     // [E] out: grad via x_min/x_max column span
    int E, int R, int C, float cw, float ch
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    float sx = src_x[e], sy = src_y[e], kx = snk_x[e], w = edge_wt[e];

    float row_f = fmaxf(0.0f, fminf((float)R, sy / ch));
    int row_lo = max(0, min(R-1, (int)floorf(row_f)));
    int row_hi = min(R-1, row_lo + 1);
    float w_hi = fmaxf(0.0f, fminf(1.0f, row_f - (float)row_lo));
    float w_lo = 1.0f - w_hi;

    float x_min = fminf(sx, kx), x_max = fmaxf(sx, kx);
    int c_start = max(0,   (int)floorf(x_min / cw));
    int c_end   = min(C-1, (int)floorf(x_max / cw));

    float g_sy  = 0.0f;
    float g_xmin = 0.0f, g_xmax = 0.0f;

    for (int c = c_start; c <= c_end; c++) {
        float col_left  = (float)c * cw;
        float col_right = col_left + cw;
        float ov = (fminf(x_max, col_right) - fmaxf(x_min, col_left)) / cw;
        if (ov <= 0.0f) continue;

        float gH_lo = grad_H[row_lo * C + c];
        float gH_hi = grad_H[row_hi * C + c];

        // Gradient via bilinear row weights: d(w_hi)/d(sy) = 1/ch
        g_sy += w * ov * (gH_hi - gH_lo);

        // Bilinear-weighted upstream gradient (used for edge-column grads)
        float blt = w_lo * gH_lo + w_hi * gH_hi;

        // d(ov)/d(x_min) = -1/cw at leftmost column (when x_min > col_left)
        if (c == c_start && x_min > col_left) {
            g_xmin += w * blt * (-1.0f / cw);
        }
        // d(ov)/d(x_max) = +1/cw at rightmost column (when x_max < col_right)
        if (c == c_end && x_max < col_right) {
            g_xmax += w * blt * (1.0f / cw);
        }
    }

    g_sy /= ch;

    // x_max = max(sx, kx): if sx >= kx, src_x is x_max
    float sx_is_xmax = (sx >= kx) ? 1.0f : 0.0f;
    g_src_y_H[e] = g_sy;
    g_src_x[e]   = g_xmax * sx_is_xmax       + g_xmin * (1.0f - sx_is_xmax);
    g_snk_x_H[e] = g_xmax * (1.0f - sx_is_xmax) + g_xmin * sx_is_xmax;
}

// ============================================================
// Backward: V demand — gradients w.r.t. snk_x, src_y, snk_y
// ============================================================

__global__ void v_bwd_kernel(
    const float* __restrict__ src_y,
    const float* __restrict__ snk_x,
    const float* __restrict__ snk_y,
    const float* __restrict__ edge_wt,
    const float* __restrict__ grad_V,  // [R, C] upstream gradient
    float* __restrict__ g_src_y_V,     // [E] out: grad via y_min/y_max row span
    float* __restrict__ g_snk_x_V,     // [E] out: grad via bilinear col weight
    float* __restrict__ g_snk_y,       // [E] out: grad via y_min/y_max row span
    int E, int R, int C, float cw, float ch
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    float sy = src_y[e], kx = snk_x[e], ky = snk_y[e], w = edge_wt[e];

    float col_f = fmaxf(0.0f, fminf((float)C, kx / cw));
    int col_lo = max(0,   (int)floorf(col_f));
    col_lo     = min(C-1, col_lo);
    int col_hi = min(C-1, col_lo + 1);
    float c_w_hi = fmaxf(0.0f, fminf(1.0f, col_f - (float)col_lo));
    float c_w_lo = 1.0f - c_w_hi;

    float y_min = fminf(sy, ky), y_max = fmaxf(sy, ky);
    int r_start = max(0,   (int)floorf(y_min / ch));
    int r_end   = min(R-1, (int)floorf(y_max / ch));

    float g_kx  = 0.0f;
    float g_ymin = 0.0f, g_ymax = 0.0f;

    for (int r = r_start; r <= r_end; r++) {
        float row_bot = (float)r * ch;
        float row_top = row_bot + ch;
        float ov = (fminf(y_max, row_top) - fmaxf(y_min, row_bot)) / ch;
        if (ov <= 0.0f) continue;

        float gV_lo = grad_V[r * C + col_lo];
        float gV_hi = grad_V[r * C + col_hi];

        // Gradient via bilinear col weights: d(c_w_hi)/d(kx) = 1/cw
        g_kx += w * ov * (gV_hi - gV_lo);

        float blt = c_w_lo * gV_lo + c_w_hi * gV_hi;

        // d(ov)/d(y_min) = -1/ch at bottom row (when y_min > row_bot)
        if (r == r_start && y_min > row_bot) {
            g_ymin += w * blt * (-1.0f / ch);
        }
        // d(ov)/d(y_max) = +1/ch at top row (when y_max < row_top)
        if (r == r_end && y_max < row_top) {
            g_ymax += w * blt * (1.0f / ch);
        }
    }

    g_kx /= cw;

    // y_min = min(sy, ky): if sy <= ky, src_y is y_min
    float sy_is_ymin = (sy <= ky) ? 1.0f : 0.0f;
    g_src_y_V[e] = g_ymin * sy_is_ymin       + g_ymax * (1.0f - sy_is_ymin);
    g_snk_x_V[e] = g_kx;
    g_snk_y[e]   = g_ymin * (1.0f - sy_is_ymin) + g_ymax * sy_is_ymin;
}

// ============================================================
// C++ interface: forward
// ============================================================

std::vector<torch::Tensor> lroute_forward(
    torch::Tensor src_x,
    torch::Tensor src_y,
    torch::Tensor snk_x,
    torch::Tensor snk_y,
    torch::Tensor edge_wt,
    int R, int C, float cw, float ch
) {
    TORCH_CHECK(src_x.is_cuda(),        "lroute_cuda_ext: tensors must be on CUDA");
    TORCH_CHECK(src_x.is_contiguous(),  "src_x must be contiguous");
    TORCH_CHECK(src_y.is_contiguous(),  "src_y must be contiguous");
    TORCH_CHECK(snk_x.is_contiguous(),  "snk_x must be contiguous");
    TORCH_CHECK(snk_y.is_contiguous(),  "snk_y must be contiguous");
    TORCH_CHECK(edge_wt.is_contiguous(),"edge_wt must be contiguous");

    int E = (int)src_x.size(0);
    auto opts = torch::TensorOptions().dtype(src_x.dtype()).device(src_x.device());
    auto H = torch::zeros({R, C}, opts);
    auto V = torch::zeros({R, C}, opts);

    if (E == 0) return {H, V};

    const int THREADS = 256;
    const int BLOCKS  = (E + THREADS - 1) / THREADS;

    h_fwd_kernel<<<BLOCKS, THREADS>>>(
        src_x.data_ptr<float>(), src_y.data_ptr<float>(), snk_x.data_ptr<float>(),
        edge_wt.data_ptr<float>(), H.data_ptr<float>(),
        E, R, C, cw, ch
    );
    v_fwd_kernel<<<BLOCKS, THREADS>>>(
        src_y.data_ptr<float>(), snk_x.data_ptr<float>(), snk_y.data_ptr<float>(),
        edge_wt.data_ptr<float>(), V.data_ptr<float>(),
        E, R, C, cw, ch
    );

    return {H, V};
}

// ============================================================
// C++ interface: backward
// ============================================================

std::vector<torch::Tensor> lroute_backward(
    torch::Tensor src_x,
    torch::Tensor src_y,
    torch::Tensor snk_x,
    torch::Tensor snk_y,
    torch::Tensor edge_wt,
    torch::Tensor grad_H,
    torch::Tensor grad_V,
    int R, int C, float cw, float ch
) {
    TORCH_CHECK(src_x.is_cuda(), "lroute_cuda_ext: tensors must be on CUDA");

    int E = (int)src_x.size(0);
    auto opts = torch::TensorOptions().dtype(src_x.dtype()).device(src_x.device());

    auto g_src_y_H = torch::zeros({E}, opts);
    auto g_src_x   = torch::zeros({E}, opts);
    auto g_snk_x_H = torch::zeros({E}, opts);
    auto g_src_y_V = torch::zeros({E}, opts);
    auto g_snk_x_V = torch::zeros({E}, opts);
    auto g_snk_y   = torch::zeros({E}, opts);

    if (E == 0) return {g_src_x, g_src_y_H, g_snk_x_H, g_snk_y};

    const int THREADS = 256;
    const int BLOCKS  = (E + THREADS - 1) / THREADS;

    auto gH = grad_H.contiguous();
    auto gV = grad_V.contiguous();

    h_bwd_kernel<<<BLOCKS, THREADS>>>(
        src_x.data_ptr<float>(), src_y.data_ptr<float>(), snk_x.data_ptr<float>(),
        edge_wt.data_ptr<float>(), gH.data_ptr<float>(),
        g_src_y_H.data_ptr<float>(), g_src_x.data_ptr<float>(), g_snk_x_H.data_ptr<float>(),
        E, R, C, cw, ch
    );
    v_bwd_kernel<<<BLOCKS, THREADS>>>(
        src_y.data_ptr<float>(), snk_x.data_ptr<float>(), snk_y.data_ptr<float>(),
        edge_wt.data_ptr<float>(), gV.data_ptr<float>(),
        g_src_y_V.data_ptr<float>(), g_snk_x_V.data_ptr<float>(), g_snk_y.data_ptr<float>(),
        E, R, C, cw, ch
    );

    // Combine H and V backward contributions for the shared variables
    auto g_src_y = g_src_y_H + g_src_y_V;
    auto g_snk_x = g_snk_x_H + g_snk_x_V;

    return {g_src_x, g_src_y, g_snk_x, g_snk_y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &lroute_forward,  "L-route forward pass  (CUDA)");
    m.def("backward", &lroute_backward, "L-route backward pass (CUDA)");
}
