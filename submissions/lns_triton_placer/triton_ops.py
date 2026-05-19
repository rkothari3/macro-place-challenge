"""
Triton-accelerated L-route congestion kernels.

Replaces the 4 scatter_add calls in lroute_congestion_loss with 2 fused Triton
kernels that eliminate the large [E×C] intermediate tensors (H_lo, H_hi, idx_lo, idx_hi).

For ibm17: E≈184K edges, C=44 columns → saves ~200MB of intermediate allocation per step.

Forward: compute overlap + bilinear weights on-the-fly → atomic scatter into demand grid.
Backward: standard PyTorch gather (correct, no Triton needed in backward).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _h_demand_kernel(
        # per-edge inputs [E]
        edge_wt_ptr, src_y_ptr, x_min_ptr, x_max_ptr,
        # grid geometry [C]
        col_left_ptr,
        # output [R, C]  (pre-zeroed, float32)
        H_demand_ptr,
        E, C, R,
        ch: tl.constexpr,
        cw: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """
        H_demand[row_lo[e], c] += wt[e] * row_w_lo[e] * relu(min(x_max[e], col_r[c]) - max(x_min[e], col_l[c])) / cw
        H_demand[row_hi[e], c] += wt[e] * row_w_hi[e] * ...

        grid = (E // BLOCK_E, C // BLOCK_C)
        """
        e_pid = tl.program_id(0)
        c_pid = tl.program_id(1)

        e_offs = e_pid * BLOCK_E + tl.arange(0, BLOCK_E)   # [BLOCK_E]
        c_offs = c_pid * BLOCK_C + tl.arange(0, BLOCK_C)   # [BLOCK_C]
        e_mask = e_offs < E
        c_mask = c_offs < C

        # Load per-edge scalars
        wt    = tl.load(edge_wt_ptr + e_offs, mask=e_mask, other=0.0)   # [BLOCK_E]
        src_y = tl.load(src_y_ptr   + e_offs, mask=e_mask, other=0.0)
        x_min = tl.load(x_min_ptr   + e_offs, mask=e_mask, other=0.0)
        x_max = tl.load(x_max_ptr   + e_offs, mask=e_mask, other=0.0)

        # Row bilinear weights
        row_f  = src_y / ch
        row_f  = tl.where(row_f < 0.0, 0.0, tl.where(row_f > float(R), float(R), row_f))
        row_lo = row_f.to(tl.int32)
        row_lo = tl.where(row_lo < 0, 0, tl.where(row_lo >= R, R - 1, row_lo))
        row_hi = tl.where(row_lo + 1 >= R, R - 1, row_lo + 1)
        row_w_hi = row_f - row_lo.to(tl.float32)
        row_w_hi = tl.where(row_w_hi < 0.0, 0.0, tl.where(row_w_hi > 1.0, 1.0, row_w_hi))
        row_w_lo = 1.0 - row_w_hi                            # [BLOCK_E]

        # Load per-column geometry
        col_l = tl.load(col_left_ptr + c_offs, mask=c_mask, other=0.0)  # [BLOCK_C]
        col_r = col_l + cw                                               # [BLOCK_C]

        # H column overlap [BLOCK_E, BLOCK_C]
        x_min_2d = tl.expand_dims(x_min,    1)   # [BLOCK_E, 1]
        x_max_2d = tl.expand_dims(x_max,    1)
        col_l_2d = tl.expand_dims(col_l,    0)   # [1, BLOCK_C]
        col_r_2d = tl.expand_dims(col_r,    0)

        ov_raw = tl.minimum(x_max_2d, col_r_2d) - tl.maximum(x_min_2d, col_l_2d)
        h_col_ov = tl.where(ov_raw > 0.0, ov_raw, 0.0) / cw  # [BLOCK_E, BLOCK_C]

        wt_2d      = tl.expand_dims(wt,      1)
        row_w_lo_2d = tl.expand_dims(row_w_lo, 1)
        row_w_hi_2d = tl.expand_dims(row_w_hi, 1)

        h_lo = wt_2d * row_w_lo_2d * h_col_ov   # [BLOCK_E, BLOCK_C]
        h_hi = wt_2d * row_w_hi_2d * h_col_ov

        row_lo_2d = tl.expand_dims(row_lo, 1)   # [BLOCK_E, 1]
        row_hi_2d = tl.expand_dims(row_hi, 1)
        c_offs_2d = tl.expand_dims(c_offs,  0)   # [1, BLOCK_C]

        lo_ptrs = H_demand_ptr + row_lo_2d * C + c_offs_2d   # [BLOCK_E, BLOCK_C]
        hi_ptrs = H_demand_ptr + row_hi_2d * C + c_offs_2d

        mask_2d = tl.expand_dims(e_mask, 1) & tl.expand_dims(c_mask, 0)

        tl.atomic_add(lo_ptrs, h_lo, mask=mask_2d)
        tl.atomic_add(hi_ptrs, h_hi, mask=mask_2d)

    @triton.jit
    def _v_demand_kernel(
        # per-edge inputs [E]
        edge_wt_ptr, snk_x_ptr, y_min_ptr, y_max_ptr,
        # grid geometry [R]
        row_bot_ptr,
        # output [R, C]  (pre-zeroed, float32)
        V_demand_ptr,
        E, C, R,
        ch: tl.constexpr,
        cw: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """
        V_demand[r, col_lo[e]] += wt[e] * col_w_lo[e] * relu(min(y_max[e], row_top[r]) - max(y_min[e], row_bot[r])) / ch
        V_demand[r, col_hi[e]] += ...

        grid = (E // BLOCK_E, R // BLOCK_R)
        """
        e_pid = tl.program_id(0)
        r_pid = tl.program_id(1)

        e_offs = e_pid * BLOCK_E + tl.arange(0, BLOCK_E)
        r_offs = r_pid * BLOCK_R + tl.arange(0, BLOCK_R)
        e_mask = e_offs < E
        r_mask = r_offs < R

        wt    = tl.load(edge_wt_ptr + e_offs, mask=e_mask, other=0.0)
        snk_x = tl.load(snk_x_ptr   + e_offs, mask=e_mask, other=0.0)
        y_min = tl.load(y_min_ptr    + e_offs, mask=e_mask, other=0.0)
        y_max = tl.load(y_max_ptr    + e_offs, mask=e_mask, other=0.0)

        col_f  = snk_x / cw
        col_f  = tl.where(col_f < 0.0, 0.0, tl.where(col_f > float(C), float(C), col_f))
        col_lo = col_f.to(tl.int32)
        col_lo = tl.where(col_lo < 0, 0, tl.where(col_lo >= C, C - 1, col_lo))
        col_hi = tl.where(col_lo + 1 >= C, C - 1, col_lo + 1)
        col_w_hi = col_f - col_lo.to(tl.float32)
        col_w_hi = tl.where(col_w_hi < 0.0, 0.0, tl.where(col_w_hi > 1.0, 1.0, col_w_hi))
        col_w_lo = 1.0 - col_w_hi   # [BLOCK_E]

        row_b = tl.load(row_bot_ptr + r_offs, mask=r_mask, other=0.0)  # [BLOCK_R]
        row_t = row_b + ch

        y_min_2d = tl.expand_dims(y_min, 1)   # [BLOCK_E, 1]
        y_max_2d = tl.expand_dims(y_max, 1)
        row_b_2d = tl.expand_dims(row_b, 0)   # [1, BLOCK_R]
        row_t_2d = tl.expand_dims(row_t, 0)

        ov_raw = tl.minimum(y_max_2d, row_t_2d) - tl.maximum(y_min_2d, row_b_2d)
        v_row_ov = tl.where(ov_raw > 0.0, ov_raw, 0.0) / ch   # [BLOCK_E, BLOCK_R]

        wt_2d      = tl.expand_dims(wt,      1)
        col_w_lo_2d = tl.expand_dims(col_w_lo, 1)
        col_w_hi_2d = tl.expand_dims(col_w_hi, 1)

        v_lo = wt_2d * col_w_lo_2d * v_row_ov   # [BLOCK_E, BLOCK_R]
        v_hi = wt_2d * col_w_hi_2d * v_row_ov

        col_lo_2d = tl.expand_dims(col_lo, 1)   # [BLOCK_E, 1]
        col_hi_2d = tl.expand_dims(col_hi, 1)
        r_offs_2d = tl.expand_dims(r_offs,  0)   # [1, BLOCK_R]

        # V_demand[r, col] → ptr = V_demand_ptr + r*C + col
        lo_ptrs = V_demand_ptr + r_offs_2d * C + col_lo_2d   # [BLOCK_E, BLOCK_R] transposed layout
        hi_ptrs = V_demand_ptr + r_offs_2d * C + col_hi_2d

        mask_2d = tl.expand_dims(e_mask, 1) & tl.expand_dims(r_mask, 0)

        tl.atomic_add(lo_ptrs, v_lo, mask=mask_2d)
        tl.atomic_add(hi_ptrs, v_hi, mask=mask_2d)


# ---------------------------------------------------------------------------
# autograd.Function wrappers
# ---------------------------------------------------------------------------

class _HVDemandFn(torch.autograd.Function):
    """
    Fused H_demand and V_demand computation via Triton.

    Forward: Triton kernels compute overlap + bilinear weights on-the-fly,
             scatter results into H_demand [R, C] and V_demand [R, C].
    Backward: standard PyTorch gathers — no Triton needed.
    """

    @staticmethod
    def forward(
        ctx,
        edge_wt,   # [E]
        src_y,     # [E]  differentiable (from pin_xy)
        snk_x,     # [E]  differentiable
        x_min,     # [E]  differentiable (min of src_x, snk_x — handled by autograd above)
        x_max,     # [E]
        y_min,     # [E]
        y_max,     # [E]
        col_left,  # [C]  constant
        row_bot,   # [R]  constant
        R: int,
        C: int,
        ch: float,
        cw: float,
    ):
        E = edge_wt.shape[0]
        device = edge_wt.device
        dtype = edge_wt.dtype

        H_demand = torch.zeros(R, C, device=device, dtype=dtype)
        V_demand = torch.zeros(R, C, device=device, dtype=dtype)

        if E > 0:
            BLOCK_E = min(64, triton.next_power_of_2(E))
            BLOCK_C = min(64, triton.next_power_of_2(C))
            BLOCK_R = min(64, triton.next_power_of_2(R))

            grid_h = (triton.cdiv(E, BLOCK_E), triton.cdiv(C, BLOCK_C))
            _h_demand_kernel[grid_h](
                edge_wt.contiguous(), src_y.detach().contiguous(),
                x_min.detach().contiguous(), x_max.detach().contiguous(),
                col_left.contiguous(),
                H_demand,
                E, C, R, ch, cw, BLOCK_E, BLOCK_C,
            )

            grid_v = (triton.cdiv(E, BLOCK_E), triton.cdiv(R, BLOCK_R))
            _v_demand_kernel[grid_v](
                edge_wt.contiguous(), snk_x.detach().contiguous(),
                y_min.detach().contiguous(), y_max.detach().contiguous(),
                row_bot.contiguous(),
                V_demand,
                E, C, R, ch, cw, BLOCK_E, BLOCK_R,
            )

        # Save for backward
        ctx.save_for_backward(
            edge_wt, src_y, snk_x, x_min, x_max, y_min, y_max, col_left, row_bot,
        )
        ctx.R, ctx.C, ctx.ch, ctx.cw = R, C, ch, cw

        return H_demand, V_demand

    @staticmethod
    def backward(ctx, grad_H, grad_V):
        (edge_wt, src_y, snk_x, x_min, x_max, y_min, y_max,
         col_left, row_bot) = ctx.saved_tensors
        R, C, ch, cw = ctx.R, ctx.C, ctx.ch, ctx.cw
        device = edge_wt.device
        dtype = edge_wt.dtype
        E = edge_wt.shape[0]

        if E == 0:
            return (None,) * 13

        # ---- H backward ----
        # H_demand[row_lo[e], c] += wt[e] * row_w_lo[e] * h_col_ov[e, c]
        # row_lo, row_w_lo come from src_y; h_col_ov from x_min, x_max

        row_float = (src_y.detach() / ch).clamp(0.0, float(R))
        row_lo = row_float.long().clamp(0, R - 1)
        row_hi = (row_lo + 1).clamp(0, R - 1)
        row_w_hi = (row_float - row_lo.float()).clamp(0.0, 1.0)
        row_w_lo = 1.0 - row_w_hi

        col_right = col_left + cw
        h_col_ov = F.relu(
            torch.minimum(x_max.detach().unsqueeze(1), col_right.unsqueeze(0)) -
            torch.maximum(x_min.detach().unsqueeze(1), col_left.unsqueeze(0))
        ) / cw   # [E, C]

        grad_H_at_lo = grad_H[row_lo]   # [E, C]
        grad_H_at_hi = grad_H[row_hi]   # [E, C]

        # grad w.r.t. src_y: d(H_demand)/d(src_y[e])
        #   = (wt[e]/ch) * Σ_c h_col_ov[e,c] * (grad_H[row_hi, c] - grad_H[row_lo, c])
        grad_src_y = (edge_wt / ch) * (h_col_ov * (grad_H_at_hi - grad_H_at_lo)).sum(dim=1)

        # grad w.r.t. x_min, x_max (via h_col_ov)
        # h_col_ov = relu(min(x_max, col_r) - max(x_min, col_l)) / cw
        # d/d(x_min): -indicator * (-1) = +indicator (where overlap > 0 and x_min is the left edge)
        # Specifically: d(relu(f))/d(x_min) = I[f>0] * d(min(xmax,colr) - max(xmin,coll))/d(xmin)
        #   = -I[f>0 and x_min > col_l]
        H_combined = grad_H_at_lo * row_w_lo.unsqueeze(1) + grad_H_at_hi * row_w_hi.unsqueeze(1)
        # indicator for overlap>0
        ov_raw = (torch.minimum(x_max.detach().unsqueeze(1), col_right.unsqueeze(0)) -
                  torch.maximum(x_min.detach().unsqueeze(1), col_left.unsqueeze(0)))
        ov_active = (ov_raw > 0).float()

        # d/d(x_max): I[f>0 and x_max < col_r] → +ov_active * (x_max_capped)
        x_max_lt_colr = (x_max.detach().unsqueeze(1) < col_right.unsqueeze(0)).float()
        # d/d(x_min): I[f>0 and x_min > col_l] → -ov_active
        x_min_gt_coll = (x_min.detach().unsqueeze(1) > col_left.unsqueeze(0)).float()

        grad_x_max_h = (edge_wt.unsqueeze(1) * H_combined * ov_active * x_max_lt_colr / cw).sum(dim=1)
        grad_x_min_h = -(edge_wt.unsqueeze(1) * H_combined * ov_active * x_min_gt_coll / cw).sum(dim=1)

        # ---- V backward ----
        col_float = (snk_x.detach() / cw).clamp(0.0, float(C))
        col_lo = col_float.long().clamp(0, C - 1)
        col_hi = (col_lo + 1).clamp(0, C - 1)
        col_w_hi = (col_float - col_lo.float()).clamp(0.0, 1.0)
        col_w_lo = 1.0 - col_w_hi

        row_top = row_bot + ch
        v_row_ov = F.relu(
            torch.minimum(y_max.detach().unsqueeze(1), row_top.unsqueeze(0)) -
            torch.maximum(y_min.detach().unsqueeze(1), row_bot.unsqueeze(0))
        ) / ch   # [E, R]

        grad_V_at_lo = grad_V[:, col_lo].t()   # [E, R]: gather columns col_lo per edge
        grad_V_at_hi = grad_V[:, col_hi].t()

        # grad w.r.t. snk_x
        grad_snk_x = (edge_wt / cw) * (v_row_ov * (grad_V_at_hi - grad_V_at_lo)).sum(dim=1)

        # grad w.r.t. y_min, y_max
        V_combined = grad_V_at_lo * col_w_lo.unsqueeze(1) + grad_V_at_hi * col_w_hi.unsqueeze(1)
        v_ov_raw = (torch.minimum(y_max.detach().unsqueeze(1), row_top.unsqueeze(0)) -
                    torch.maximum(y_min.detach().unsqueeze(1), row_bot.unsqueeze(0)))
        v_ov_active = (v_ov_raw > 0).float()
        y_max_lt_rowt = (y_max.detach().unsqueeze(1) < row_top.unsqueeze(0)).float()
        y_min_gt_rowb = (y_min.detach().unsqueeze(1) > row_bot.unsqueeze(0)).float()

        grad_y_max_v = (edge_wt.unsqueeze(1) * V_combined * v_ov_active * y_max_lt_rowt / ch).sum(dim=1)
        grad_y_min_v = -(edge_wt.unsqueeze(1) * V_combined * v_ov_active * y_min_gt_rowb / ch).sum(dim=1)

        return (
            None,           # edge_wt — not differentiable
            grad_src_y,     # src_y
            grad_snk_x,     # snk_x
            grad_x_min_h,   # x_min (H contribution)
            grad_x_max_h,   # x_max (H contribution)
            grad_y_min_v,   # y_min (V contribution)
            grad_y_max_v,   # y_max (V contribution)
            None,           # col_left
            None,           # row_bot
            None, None, None, None,  # R, C, ch, cw
        )


def hv_demand_triton(
    edge_wt, src_y, snk_x, x_min, x_max, y_min, y_max,
    col_left, row_bot, R, C, ch, cw,
):
    """
    Drop-in replacement for the scatter_add section of lroute_congestion_loss.
    Returns H_demand [R, C] and V_demand [R, C].
    Falls back to PyTorch if Triton unavailable or on CPU.
    """
    if not _TRITON_AVAILABLE or not edge_wt.is_cuda:
        return _pytorch_fallback(
            edge_wt, src_y, snk_x, x_min, x_max, y_min, y_max,
            col_left, row_bot, R, C, ch, cw,
        )
    return _HVDemandFn.apply(
        edge_wt, src_y, snk_x, x_min, x_max, y_min, y_max,
        col_left, row_bot, R, C, ch, cw,
    )


def _pytorch_fallback(
    edge_wt, src_y, snk_x, x_min, x_max, y_min, y_max,
    col_left, row_bot, R, C, ch, cw,
):
    """Pure PyTorch fallback (same as original lroute_congestion_loss scatter section)."""
    device = edge_wt.device
    dtype = edge_wt.dtype

    row_float = (src_y / ch).clamp(0.0, float(R))
    row_lo = row_float.detach().long().clamp(0, R - 1)
    row_hi = (row_lo + 1).clamp(0, R - 1)
    row_w_hi = (row_float - row_lo.float()).clamp(0.0, 1.0)
    row_w_lo = 1.0 - row_w_hi

    col_right = col_left + cw
    H_col_ov = F.relu(
        torch.minimum(x_max.unsqueeze(1), col_right.unsqueeze(0)) -
        torch.maximum(x_min.unsqueeze(1), col_left.unsqueeze(0))
    ) / cw
    H_lo = edge_wt.unsqueeze(1) * row_w_lo.unsqueeze(1) * H_col_ov
    H_hi = edge_wt.unsqueeze(1) * row_w_hi.unsqueeze(1) * H_col_ov
    idx_lo = row_lo.unsqueeze(1).expand(-1, C)
    idx_hi = row_hi.unsqueeze(1).expand(-1, C)
    H_demand = torch.zeros(R, C, device=device, dtype=dtype)
    H_demand = H_demand.scatter_add(0, idx_lo, H_lo)
    H_demand = H_demand.scatter_add(0, idx_hi, H_hi)

    col_float = (snk_x / cw).clamp(0.0, float(C))
    col_lo = col_float.detach().long().clamp(0, C - 1)
    col_hi = (col_lo + 1).clamp(0, C - 1)
    col_w_hi = (col_float - col_lo.float()).clamp(0.0, 1.0)
    col_w_lo = 1.0 - col_w_hi
    row_top = row_bot + ch
    V_row_ov = F.relu(
        torch.minimum(y_max.unsqueeze(1), row_top.unsqueeze(0)) -
        torch.maximum(y_min.unsqueeze(1), row_bot.unsqueeze(0))
    ) / ch
    V_lo = edge_wt.unsqueeze(1) * col_w_lo.unsqueeze(1) * V_row_ov
    V_hi = edge_wt.unsqueeze(1) * col_w_hi.unsqueeze(1) * V_row_ov
    V_lo_t = V_lo.t()
    V_hi_t = V_hi.t()
    c_lo_exp = col_lo.unsqueeze(0).expand(R, -1)
    c_hi_exp = col_hi.unsqueeze(0).expand(R, -1)
    V_demand = torch.zeros(R, C, device=device, dtype=dtype)
    V_demand = V_demand.scatter_add(1, c_lo_exp, V_lo_t)
    V_demand = V_demand.scatter_add(1, c_hi_exp, V_hi_t)

    return H_demand, V_demand
