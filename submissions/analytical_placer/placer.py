"""
Analytical global placer: LSE-HPWL + density + L-route congestion surrogate
Optimization: Adam gradient descent → greedy spiral legalization → SA refinement

Pin resolution:
  [0, num_hard)           hard macro → pos[owner] + macro_pin_offsets[owner][slot]
  [num_hard, num_macro)   soft macro → pos[owner] (center)
  [num_macro, ...)        I/O port   → port_positions[owner - num_macro] (fixed)
"""
from __future__ import annotations

import math
import os as _os
import random
import sys as _sys
import time
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

# IBM ICCAD04 macro routing allocation fractions (routes_used_by_macros / routes_per_micron).
# All 17 IBM benchmarks share: hor=30.304/65.957≈0.459, ver=71.304/106.957≈0.667.
# These fractions represent how much routing capacity macros block in cells they occupy.
_MACRO_H_ALLOC_FRAC = 0.459
_MACRO_V_ALLOC_FRAC = 0.667

# ---------------------------------------------------------------------------
# Optional CUDA extension for fast density computation
# ---------------------------------------------------------------------------
# Build once on the eval server:
#   cd submissions/analytical_placer/density_ext && pip install -e .
# Replaces the chunked PyTorch loop with a tiled 2D CUDA kernel using shared memory.
# For ibm17 (N=2604, G=2244): 11 chunk iters × ~9 ops = ~99 kernel launches → 2 launches.
_density_ext_dir = _os.path.join(_os.path.dirname(__file__), 'density_ext')
if _density_ext_dir not in _sys.path:
    _sys.path.insert(0, _density_ext_dir)
try:
    import density_cuda_ext as _DENSITY_CUDA_EXT
    print("[analytical_placer] CUDA density extension loaded")
except ImportError:
    _DENSITY_CUDA_EXT = None


# ---------------------------------------------------------------------------
# Preprocessing: flatten variable-length net_pin_nodes into GPU tensors
# ---------------------------------------------------------------------------

def _preprocess(b: Benchmark, device: torch.device) -> dict:
    """
    Convert all variable-length lists into flat packed tensors for scatter ops.

    Returns dict with:
      pin_net_idx       [total_pins] int64  — which net each pin belongs to
      pin_owner         [total_pins] int64  — owner index (macro or port)
      pin_is_hard       [total_pins] bool   — True if hard macro pin
      pin_is_port       [total_pins] bool   — True if I/O port
      hard_offsets      [total_hard_pins, 2] float32 — stacked macro_pin_offsets
      hard_pin_flat_idx [total_pins] int64  — index into hard_offsets (0 for non-hard)
      num_nets          int
      edge_src_idx      [num_edges] int64   — flat pin index for source pin per edge
      edge_snk_idx      [num_edges] int64   — flat pin index for sink pin per edge
      edge_weights      [num_edges] float32 — net weight for each 2-pin edge
    """
    num_hard = b.num_hard_macros
    num_macro = b.num_macros

    # Stack macro_pin_offsets into a flat [total_hard_pins, 2] tensor
    offset_list = []
    per_hard_offset_start: list[int] = []
    cumulative = 0
    for i in range(num_hard):
        per_hard_offset_start.append(cumulative)
        offs = b.macro_pin_offsets[i]   # [P, 2] or [0, 2]
        if offs.shape[0] > 0:
            offset_list.append(offs)
        cumulative += offs.shape[0]

    hard_offsets = (
        torch.cat(offset_list, dim=0).to(device)
        if offset_list else torch.zeros(0, 2, device=device)
    )

    # Flatten net_pin_nodes into parallel arrays + build 2-pin star edges
    all_net_idx, all_owner, all_slot = [], [], []
    edge_src_flat: list[int] = []
    edge_snk_flat: list[int] = []
    edge_weights_list: list[float] = []
    flat_offset = 0

    for net_i, pins in enumerate(b.net_pin_nodes):
        n = pins.shape[0]
        if n == 0:
            continue
        all_net_idx.append(torch.full((n,), net_i, dtype=torch.long))
        all_owner.append(pins[:, 0])
        all_slot.append(pins[:, 1])

        # Star decomposition: pin[0] = source, pins[1..N-1] = sinks
        # Each (source, sink_j) pair becomes a 2-pin edge for L-route
        if n >= 2:
            w = float(b.net_weights[net_i].item())
            for j in range(1, n):
                edge_src_flat.append(flat_offset)      # index of source pin in pin_xy
                edge_snk_flat.append(flat_offset + j)  # index of sink pin in pin_xy
                edge_weights_list.append(w)

        flat_offset += n

    pin_net_idx = torch.cat(all_net_idx).to(device)   # [total_pins]
    pin_owner   = torch.cat(all_owner).to(device)     # [total_pins]
    pin_slot    = torch.cat(all_slot).to(device)      # [total_pins]

    pin_is_hard = pin_owner < num_hard
    pin_is_port = pin_owner >= num_macro

    # For hard pins: compute absolute index into hard_offsets tensor.
    total_pins = len(pin_net_idx)
    hard_pin_flat_idx = torch.zeros(total_pins, dtype=torch.long)
    pin_owner_cpu = pin_owner.cpu()
    pin_slot_cpu  = pin_slot.cpu()
    pin_is_hard_cpu = pin_is_hard.cpu()
    for k in range(total_pins):
        if pin_is_hard_cpu[k]:
            owner_k = int(pin_owner_cpu[k].item())
            slot_k  = int(pin_slot_cpu[k].item())
            hard_pin_flat_idx[k] = per_hard_offset_start[owner_k] + slot_k
    hard_pin_flat_idx = hard_pin_flat_idx.to(device)

    # Convert edge lists to tensors
    if edge_src_flat:
        edge_src_idx  = torch.tensor(edge_src_flat,     dtype=torch.long,    device=device)
        edge_snk_idx  = torch.tensor(edge_snk_flat,     dtype=torch.long,    device=device)
        edge_weights  = torch.tensor(edge_weights_list, dtype=torch.float32, device=device)
    else:
        edge_src_idx = torch.zeros(0, dtype=torch.long,    device=device)
        edge_snk_idx = torch.zeros(0, dtype=torch.long,    device=device)
        edge_weights = torch.zeros(0, dtype=torch.float32, device=device)

    return dict(
        pin_net_idx=pin_net_idx,
        pin_owner=pin_owner,
        pin_is_hard=pin_is_hard.to(device),
        pin_is_port=pin_is_port.to(device),
        hard_offsets=hard_offsets,
        hard_pin_flat_idx=hard_pin_flat_idx,
        num_nets=b.num_nets,
        edge_src_idx=edge_src_idx,
        edge_snk_idx=edge_snk_idx,
        edge_weights=edge_weights,
    )


# ---------------------------------------------------------------------------
# Differentiable pin position lookup
# ---------------------------------------------------------------------------

def _compute_pin_xy(
    pos: torch.Tensor,        # [num_macros, 2]
    data: dict,
    b: Benchmark,
    port_pos: torch.Tensor,   # [num_ports, 2] on device
) -> torch.Tensor:
    """
    Returns [total_pins, 2] float32 — world coordinates of every pin.
    Differentiable w.r.t. pos.
    """
    num_macro = b.num_macros
    owner = data["pin_owner"]          # [total_pins]
    is_hard = data["pin_is_hard"]      # [total_pins] bool
    is_port = data["pin_is_port"]      # [total_pins] bool

    clamped_owner = owner.clamp(0, num_macro - 1)
    pin_xy = pos[clamped_owner]        # [total_pins, 2]

    if is_hard.any():
        hard_flat = data["hard_pin_flat_idx"]
        offsets   = data["hard_offsets"]
        if offsets.shape[0] > 0:
            offset_xy = offsets[hard_flat]
            pin_xy = pin_xy + offset_xy * is_hard.unsqueeze(1).float()

    if is_port.any() and port_pos.shape[0] > 0:
        port_owner_idx = (owner - num_macro).clamp(min=0)
        port_owner_idx = port_owner_idx.clamp(max=port_pos.shape[0] - 1)
        port_xy = port_pos[port_owner_idx]
        pin_xy = torch.where(is_port.unsqueeze(1), port_xy, pin_xy)

    return pin_xy


# ---------------------------------------------------------------------------
# LSE-HPWL loss (differentiable wirelength surrogate)
# ---------------------------------------------------------------------------

def _scatter_lse(vals: torch.Tensor, idx: torch.Tensor, n: int, alpha: float) -> torch.Tensor:
    """Numerically stable scatter logsumexp: returns [n] tensor."""
    max_v = torch.zeros(n, dtype=vals.dtype, device=vals.device)
    max_v.scatter_reduce_(0, idx, vals, reduce="amax", include_self=True)
    stable = (vals - max_v[idx]) * alpha
    sum_exp = torch.zeros(n, dtype=vals.dtype, device=vals.device)
    sum_exp.scatter_add_(0, idx, stable.exp())
    return max_v + sum_exp.clamp(min=1e-12).log() / alpha


def lse_hpwl_loss(
    pin_xy: torch.Tensor,   # [total_pins, 2]
    data: dict,
    b: Benchmark,
    alpha: float,
) -> torch.Tensor:
    """Differentiable HPWL via log-sum-exp. Returns normalized scalar."""
    net_idx  = data["pin_net_idx"]
    num_nets = data["num_nets"]
    weights  = b.net_weights.to(pin_xy.device)

    x = pin_xy[:, 0]
    y = pin_xy[:, 1]

    lse_x_max =  _scatter_lse( x, net_idx, num_nets, alpha)
    lse_x_min = -_scatter_lse(-x, net_idx, num_nets, alpha)
    lse_y_max =  _scatter_lse( y, net_idx, num_nets, alpha)
    lse_y_min = -_scatter_lse(-y, net_idx, num_nets, alpha)

    hpwl_per_net = (lse_x_max - lse_x_min) + (lse_y_max - lse_y_min)
    norm = (b.canvas_width + b.canvas_height) * num_nets
    return (weights * hpwl_per_net).sum() / norm


# ---------------------------------------------------------------------------
# Density bell-kernel loss (differentiable density surrogate)
# ---------------------------------------------------------------------------

def _make_cell_centers(b: Benchmark, device: torch.device):
    """Returns cell_centers [G, 2] and cell_size [2]."""
    rows, cols = b.grid_rows, b.grid_cols
    cw = b.canvas_width / cols
    ch = b.canvas_height / rows
    col_c = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) * cw
    row_c = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) * ch
    grid_y, grid_x = torch.meshgrid(row_c, col_c, indexing="ij")
    cell_centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [G, 2]
    return cell_centers, torch.tensor([cw, ch], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# CUDA density kernel autograd wrapper
# ---------------------------------------------------------------------------

class _DensityKernel(torch.autograd.Function):
    """
    Wraps density_cuda_ext.forward/backward as a differentiable PyTorch op.

    The autograd.Function pattern lets us:
      - Run a custom CUDA kernel in forward (no Python overhead, no intermediate tensors)
      - Provide an analytically-derived backward kernel (avoids rebuilding the autograd
        graph for 99 sequential chunked operations)

    Usage: cell_density = _DensityKernel.apply(pos, sizes, cell_xy, half_cw, half_ch, inv_cell_area)
    The loss computation (relu, pow, mean) is done in Python with standard autograd.
    """

    @staticmethod
    def forward(ctx, pos, sizes, cell_xy, half_cw, half_ch, inv_cell_area):
        cell_density = _DENSITY_CUDA_EXT.forward(
            pos, sizes, cell_xy, half_cw, half_ch, inv_cell_area
        )
        ctx.save_for_backward(pos, sizes, cell_xy, cell_density)
        ctx.constants = (half_cw, half_ch, inv_cell_area)
        return cell_density

    @staticmethod
    def backward(ctx, grad_cell_density):
        pos, sizes, cell_xy, cell_density = ctx.saved_tensors
        half_cw, half_ch, inv_cell_area = ctx.constants
        grad_pos = _DENSITY_CUDA_EXT.backward(
            grad_cell_density.contiguous(), pos, sizes, cell_xy,
            half_cw, half_ch, inv_cell_area
        )
        # Return None for sizes, cell_xy, half_cw, half_ch, inv_cell_area (not differentiable inputs)
        return grad_pos, None, None, None, None, None


def density_loss(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    cell_centers: torch.Tensor,
    cell_size: torch.Tensor,
    b: Benchmark,
    target_density: float = 1.0,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Differentiable density penalty using exact rectangle overlap.
    cell_density[g] = sum_i (overlap_area(macro_i, cell_g)) / cell_area.

    When density_cuda_ext is available and pos is on CUDA, uses the tiled shared-memory
    kernel (2 kernel launches: forward + loss). Otherwise falls back to the chunked
    PyTorch loop (~9 × num_chunks launches).
    """
    half_cw = cell_size[0] / 2
    half_ch = cell_size[1] / 2
    cell_area = cell_size[0] * cell_size[1]

    if _DENSITY_CUDA_EXT is not None and pos.device.type == 'cuda':
        # Fast path: single CUDA kernel launch with shared-memory tiling
        half_cw_f = half_cw.item()
        half_ch_f = half_ch.item()
        inv_cell_area_f = 1.0 / cell_area.item()
        cell_density = _DensityKernel.apply(
            pos, sizes, cell_centers, half_cw_f, half_ch_f, inv_cell_area_f
        )
        overflow = F.relu(cell_density - target_density)
        return overflow.pow(2).mean()

    # Fallback: chunked PyTorch loop (CPU or no CUDA ext)
    N = pos.shape[0]
    G = cell_centers.shape[0]
    gx = cell_centers[:, 0]
    gy = cell_centers[:, 1]

    cell_density = torch.zeros(G, dtype=pos.dtype, device=pos.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        cx = pos[start:end, 0:1]
        cy = pos[start:end, 1:2]
        hw = sizes[start:end, 0:1] / 2
        hh = sizes[start:end, 1:2] / 2

        lo_x = torch.maximum(cx - hw, gx - half_cw)
        hi_x = torch.minimum(cx + hw, gx + half_cw)
        lo_y = torch.maximum(cy - hh, gy - half_ch)
        hi_y = torch.minimum(cy + hh, gy + half_ch)

        overlap_area = F.relu(hi_x - lo_x) * F.relu(hi_y - lo_y)
        cell_density = cell_density + overlap_area.sum(dim=0) / cell_area

    overflow = F.relu(cell_density - target_density)
    return overflow.pow(2).mean()


# ---------------------------------------------------------------------------
# Direct macro-pair overlap penalty
# ---------------------------------------------------------------------------

def macro_overlap_loss(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    gap: float = 0.02,
) -> torch.Tensor:
    """Penalizes pairwise overlap between hard macros. O(N²) but N≈300 max."""
    x  = pos[:num_hard, 0]
    y  = pos[:num_hard, 1]
    hw = sizes[:num_hard, 0] / 2 + gap / 2
    hh = sizes[:num_hard, 1] / 2 + gap / 2

    dx = (x.unsqueeze(0) - x.unsqueeze(1)).abs()
    dy = (y.unsqueeze(0) - y.unsqueeze(1)).abs()
    px = F.relu(hw.unsqueeze(0) + hw.unsqueeze(1) - dx)
    py = F.relu(hh.unsqueeze(0) + hh.unsqueeze(1) - dy)

    overlap = torch.minimum(px, py) * (px > 0).float() * (py > 0).float()
    mask = torch.triu(torch.ones(num_hard, num_hard, device=pos.device, dtype=torch.bool), diagonal=1)
    return overlap[mask].sum()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# L-route differentiable congestion surrogate (matches plc_client_os.py semantics)
# ---------------------------------------------------------------------------

def lroute_congestion_loss(
    pin_xy: torch.Tensor,    # [total_pins, 2] — differentiable pin world coords
    data: dict,              # from _preprocess(), includes edge_src/snk/weights
    b: Benchmark,
    device: torch.device,
    smooth_range: int = 2,   # IBM benchmarks use smooth_range=2
    pos: torch.Tensor | None = None,    # [num_macros, 2] — for macro blockage term
    sizes: torch.Tensor | None = None,  # [num_macros, 2] — for macro blockage term
) -> torch.Tensor:
    """
    Differentiable L-route congestion surrogate matching plc_client_os.py semantics.

    For each 2-pin connection (source → sink) from star decomposition:
      H segment: horizontal wire at row ≈ source.y, from source.x to sink.x
        → H_demand[r_src, c] += weight × col_overlap(c, x_min, x_max) / cw
      V segment: vertical wire at col ≈ sink.x, from source.y to sink.y
        → V_demand[r, c_snk] += weight × row_overlap(r, y_min, y_max) / ch

    Differentiable via bilinear soft row/col assignment:
      row bilinear: weight to row floor(y/ch) and floor(y/ch)+1, proportional to fraction
      col bilinear: weight to col floor(x/cw) and floor(x/cw)+1, proportional to fraction

    Gradients flow: macro_pos → pin_xy → bilinear weights → demand → congestion.
    The gradient correctly signals "move macros closer to reduce wire length and
    congestion" — opposite of RUDY which pushed macros apart.

    Returns: mean of top-5% routing utilization (H+V concatenated), same as
    plc_client_os.py get_congestion_cost().
    """
    rows = b.grid_rows
    cols = b.grid_cols
    cw = b.canvas_width / cols    # cell width  (μm)
    ch = b.canvas_height / rows   # cell height (μm)

    edge_src = data["edge_src_idx"]   # [E] flat indices into pin_xy
    edge_snk = data["edge_snk_idx"]   # [E]
    edge_wt  = data["edge_weights"]   # [E]

    if edge_src.shape[0] == 0:
        return pin_xy.sum() * 0.0   # differentiable zero

    src_xy = pin_xy[edge_src]  # [E, 2]
    snk_xy = pin_xy[edge_snk]  # [E, 2]

    src_x = src_xy[:, 0]   # [E]
    src_y = src_xy[:, 1]   # [E]
    snk_x = snk_xy[:, 0]   # [E]
    snk_y = snk_xy[:, 1]   # [E]

    # ------------------------------------------------------------------
    # Compute H_demand [R, C] and V_demand [R, C] via scatter_add
    # ------------------------------------------------------------------
    x_min = torch.minimum(src_x, snk_x)
    x_max = torch.maximum(src_x, snk_x)
    y_min = torch.minimum(src_y, snk_y)
    y_max = torch.maximum(src_y, snk_y)

    row_float = (src_y / ch).clamp(0.0, float(rows))
    row_lo = row_float.detach().long().clamp(0, rows - 1)
    row_hi = (row_lo + 1).clamp(0, rows - 1)
    row_w_hi = (row_float - row_lo.float()).clamp(0.0, 1.0)
    row_w_lo = 1.0 - row_w_hi

    col_left  = (torch.arange(cols, device=device, dtype=torch.float32) * cw).unsqueeze(0)
    col_right = col_left + cw
    H_col_ov = F.relu(
        torch.minimum(x_max.unsqueeze(1), col_right) -
        torch.maximum(x_min.unsqueeze(1), col_left)
    ) / cw
    H_lo = edge_wt.unsqueeze(1) * row_w_lo.unsqueeze(1) * H_col_ov
    H_hi = edge_wt.unsqueeze(1) * row_w_hi.unsqueeze(1) * H_col_ov
    idx_lo = row_lo.unsqueeze(1).expand(-1, cols)
    idx_hi = row_hi.unsqueeze(1).expand(-1, cols)
    H_demand = torch.zeros(rows, cols, device=device, dtype=pin_xy.dtype)
    H_demand = H_demand.scatter_add(0, idx_lo, H_lo)
    H_demand = H_demand.scatter_add(0, idx_hi, H_hi)

    col_float = (snk_x / cw).clamp(0.0, float(cols))
    col_lo = col_float.detach().long().clamp(0, cols - 1)
    col_hi = (col_lo + 1).clamp(0, cols - 1)
    col_w_hi = (col_float - col_lo.float()).clamp(0.0, 1.0)
    col_w_lo = 1.0 - col_w_hi
    row_bot = (torch.arange(rows, device=device, dtype=torch.float32) * ch).unsqueeze(0)
    row_top = row_bot + ch
    V_row_ov = F.relu(
        torch.minimum(y_max.unsqueeze(1), row_top) -
        torch.maximum(y_min.unsqueeze(1), row_bot)
    ) / ch
    V_lo = edge_wt.unsqueeze(1) * col_w_lo.unsqueeze(1) * V_row_ov
    V_hi = edge_wt.unsqueeze(1) * col_w_hi.unsqueeze(1) * V_row_ov
    V_lo_t = V_lo.t()
    V_hi_t = V_hi.t()
    c_lo_exp = col_lo.unsqueeze(0).expand(rows, -1)
    c_hi_exp = col_hi.unsqueeze(0).expand(rows, -1)
    V_demand = torch.zeros(rows, cols, device=device, dtype=pin_xy.dtype)
    V_demand = V_demand.scatter_add(1, c_lo_exp, V_lo_t)
    V_demand = V_demand.scatter_add(1, c_hi_exp, V_hi_t)

    # ------------------------------------------------------------------
    # Normalize by routing supply
    #
    # H_supply = cell_height × hroutes_per_micron  (horizontal tracks per cell)
    # V_supply = cell_width  × vroutes_per_micron  (vertical tracks per cell)
    #
    # H_cong[r,c] = H_demand[r,c] / H_supply
    # V_cong[r,c] = V_demand[r,c] / V_supply
    #
    # For ibm01: H_supply ≈ 37.0 tracks/cell, V_supply ≈ 54.4 tracks/cell
    # ------------------------------------------------------------------
    h_supply = float(b.hroutes_per_micron) * ch
    v_supply = float(b.vroutes_per_micron) * cw
    H_cong = H_demand / h_supply   # [R, C]
    V_cong = V_demand / v_supply   # [R, C]

    # ------------------------------------------------------------------
    # Smooth: box filter (matches competition's __smooth_routing_cong)
    #
    # V_cong: smooth horizontally (along column dim) — same row, ±smooth_range cols
    # H_cong: smooth vertically  (along row dim)    — same col, ±smooth_range rows
    #
    # We use F.conv2d with replicate padding. The competition uses a distribution
    # filter (source value / window_size → spreads to neighbors). Both are
    # equivalent to a box-filter average for interior cells; boundary handling
    # differs slightly — acceptable for a surrogate.
    # ------------------------------------------------------------------
    if smooth_range > 0:
        k = 2 * smooth_range + 1
        # V_cong horizontal smoothing: kernel along dim 3 (cols)
        kh = torch.ones(1, 1, 1, k, device=device, dtype=V_cong.dtype) / k
        vc4d = F.pad(V_cong[None, None], (smooth_range, smooth_range, 0, 0), mode='replicate')
        V_cong = F.conv2d(vc4d, kh).squeeze(0).squeeze(0)   # [R, C]

        # H_cong vertical smoothing: kernel along dim 2 (rows)
        kv = torch.ones(1, 1, k, 1, device=device, dtype=H_cong.dtype) / k
        hc4d = F.pad(H_cong[None, None], (0, 0, smooth_range, smooth_range), mode='replicate')
        H_cong = F.conv2d(hc4d, kv).squeeze(0).squeeze(0)   # [R, C]

    # ------------------------------------------------------------------
    # Add macro routing blockage (competition adds this AFTER smoothing net routing).
    #
    # When a hard macro occupies a grid cell, it blocks routing tracks:
    #   V_macro[r,c] = sum_i [macro_i in cell] × (x_overlap_i(c)/cw) × V_ALLOC_FRAC
    #   H_macro[r,c] = sum_i [macro_i in cell] × (y_overlap_i(r)/ch) × H_ALLOC_FRAC
    #
    # IBM alloc fractions: V=0.667 (macros block 67% of vertical tracks),
    #                      H=0.459 (macros block 46% of horizontal tracks).
    # This explains why actual evaluation congestion > our net-only surrogate.
    # ------------------------------------------------------------------
    if pos is not None and sizes is not None and b.num_hard_macros > 0:
        num_h = b.num_hard_macros
        cx = pos[:num_h, 0].unsqueeze(1)  # [H, 1]
        cy = pos[:num_h, 1].unsqueeze(1)
        hw = sizes[:num_h, 0].unsqueeze(1) / 2
        hh = sizes[:num_h, 1].unsqueeze(1) / 2

        col_l = torch.arange(cols, device=device, dtype=pos.dtype) * cw
        col_r = col_l + cw
        row_b = torch.arange(rows, device=device, dtype=pos.dtype) * ch
        row_t = row_b + ch

        # [H, cols]: horizontal overlap in microns
        x_ol = F.relu(torch.minimum(cx + hw, col_r) - torch.maximum(cx - hw, col_l))
        # [H, rows]: vertical overlap in microns
        y_ol = F.relu(torch.minimum(cy + hh, row_t) - torch.maximum(cy - hh, row_b))

        # Soft row-presence indicator: clamp fraction to [0, 1]
        y_ind = (y_ol / ch).clamp(max=1.0)   # [H, rows]
        x_ind = (x_ol / cw).clamp(max=1.0)   # [H, cols]
        x_frac = x_ol / cw                    # [H, cols]
        y_frac = y_ol / ch                    # [H, rows]

        # Detach macro blockage from autograd: blockage improves surrogate calibration
        # (cong_100 at step 99) but its gradient pushes macros to reduce their own
        # cell footprint — this disrupts routing topology. Only net routing gradient
        # should guide macro positions.
        with torch.no_grad():
            # V_macro[r,c] = sum_i y_ind[i,r] × x_frac[i,c] × V_ALLOC_FRAC
            V_macro = y_ind.t() @ x_frac * _MACRO_V_ALLOC_FRAC   # [rows, cols]
            # H_macro[r,c] = sum_i x_ind[i,c] × y_frac[i,r] × H_ALLOC_FRAC
            H_macro = y_frac.t() @ x_ind * _MACRO_H_ALLOC_FRAC   # [rows, cols]

        H_cong = H_cong + H_macro
        V_cong = V_cong + V_macro

    # ------------------------------------------------------------------
    # Top-5% mean of concatenated H+V grids (matches abu(H+V, 0.05))
    #
    # Competition: abu(V_routing_cong + H_routing_cong, 0.05)
    # '+' = Python list concatenation → 2×R×C values total
    # We take top-5% of the combined tensor.
    # torch.topk is differentiable w.r.t. values (not indices).
    # ------------------------------------------------------------------
    combined = torch.cat([H_cong.flatten(), V_cong.flatten()])   # [2*R*C]
    k_top = max(1, int(0.05 * combined.shape[0]))
    return torch.topk(combined, k_top).values.mean()


# ---------------------------------------------------------------------------
# Minimal-perturbation legalization (pairwise separation of hard macros only)
# ---------------------------------------------------------------------------

def _legalize(pos: torch.Tensor, b: Benchmark, time_budget_s: float = 20.0) -> torch.Tensor:
    """
    Hybrid legalization for hard macros:
    1. Iterative pairwise separation (minimal perturbation, O(N²) per iter)
       — capped at time_budget_s to prevent ibm10-style 1188s runtimes
    2. Spiral fallback for macros still overlapping after pairwise phase

    Time budget: ibm10 (537 macros) at 537²×100 passes = 28.8M checks in Python
    would take ~576s worst case. Cap at 20s; spiral handles the rest.
    """
    pos = pos.clone()
    sizes    = b.macro_sizes
    fixed    = b.macro_fixed
    num_hard = b.num_hard_macros
    cw, ch   = b.canvas_width, b.canvas_height
    GAP = 0.02

    movable = [i for i in range(num_hard) if not fixed[i].item()]

    def _clamp(i: int):
        hw, hh = sizes[i, 0].item() / 2, sizes[i, 1].item() / 2
        pos[i, 0] = max(hw, min(pos[i, 0].item(), cw - hw))
        pos[i, 1] = max(hh, min(pos[i, 1].item(), ch - hh))

    # Phase 1: pairwise separation with time cap
    legaliz_start = time.time()
    pairwise_passes = 0
    hit_time_cap = False
    for pass_idx in range(100):
        if time.time() - legaliz_start > time_budget_s:
            hit_time_cap = True
            pairwise_passes = pass_idx
            break
        pairwise_passes = pass_idx + 1
        any_overlap = False
        for a in range(len(movable)):
            for bb in range(a + 1, len(movable)):
                i, j = movable[a], movable[bb]
                xi, yi = pos[i, 0].item(), pos[i, 1].item()
                xj, yj = pos[j, 0].item(), pos[j, 1].item()
                wi, hi = sizes[i, 0].item(), sizes[i, 1].item()
                wj, hj = sizes[j, 0].item(), sizes[j, 1].item()
                px = (wi + wj) / 2 + GAP - abs(xi - xj)
                py = (hi + hj) / 2 + GAP - abs(yi - yj)
                if px <= 0 or py <= 0:
                    continue
                any_overlap = True
                ai, aj = wi * hi, wj * hj
                fi, fj = aj / (ai + aj), ai / (ai + aj)
                if px < py:
                    sx = math.copysign(px, xi - xj)
                    pos[i, 0] = xi + sx * fi
                    pos[j, 0] = xj - sx * fj
                else:
                    sy = math.copysign(py, yi - yj)
                    pos[i, 1] = yi + sy * fi
                    pos[j, 1] = yj - sy * fj
                _clamp(i); _clamp(j)
        if not any_overlap:
            elapsed = time.time() - legaliz_start
            print(f"  [legalize] pairwise converged in {pairwise_passes} passes ({elapsed:.1f}s)")
            return pos

    elapsed_pw = time.time() - legaliz_start
    if hit_time_cap:
        print(f"  [legalize] pairwise TIME CAP after {pairwise_passes} passes ({elapsed_pw:.1f}s) — spiral handles remainder")
    else:
        print(f"  [legalize] pairwise {pairwise_passes} passes ({elapsed_pw:.1f}s), moving to spiral")

    # Phase 2: spiral fallback for remaining violators
    def _overlaps(i: int, others: list) -> bool:
        xi, yi = pos[i, 0].item(), pos[i, 1].item()
        wi, hi = sizes[i, 0].item(), sizes[i, 1].item()
        for j in others:
            if j == i:
                continue
            xj, yj = pos[j, 0].item(), pos[j, 1].item()
            wj, hj = sizes[j, 0].item(), sizes[j, 1].item()
            if (abs(xi - xj) < (wi + wj) / 2 + GAP and
                    abs(yi - yj) < (hi + hj) / 2 + GAP):
                return True
        return False

    def _in_canvas(i: int) -> bool:
        hw, hh = sizes[i, 0].item() / 2, sizes[i, 1].item() / 2
        return (hw <= pos[i, 0].item() <= cw - hw and
                hh <= pos[i, 1].item() <= ch - hh)

    all_hard = list(range(num_hard))
    for i in movable:
        if _in_canvas(i) and not _overlaps(i, all_hard):
            continue
        step = 0.15 * max(sizes[i, 0].item(), sizes[i, 1].item())
        ox, oy = pos[i, 0].item(), pos[i, 1].item()
        ok = False
        for ring in range(1, 300):
            for dx in range(-ring, ring + 1):
                for dy in (-ring, ring):
                    pos[i, 0], pos[i, 1] = ox + dx * step, oy + dy * step
                    if _in_canvas(i) and not _overlaps(i, all_hard):
                        ok = True; break
                if ok: break
            if ok: break
            for dy in range(-ring + 1, ring):
                for dx in (-ring, ring):
                    pos[i, 0], pos[i, 1] = ox + dx * step, oy + dy * step
                    if _in_canvas(i) and not _overlaps(i, all_hard):
                        ok = True; break
                if ok: break
            if ok: break
        if not ok:
            pos[i, 0], pos[i, 1] = ox, oy

    return pos


# ---------------------------------------------------------------------------
# Post-legalization gradient refinement
# ---------------------------------------------------------------------------

def _post_legalize_refine(
    pos_cpu: torch.Tensor,
    b: Benchmark,
    data: dict,
    device: torch.device,
    steps: int = 50,
    cong_w: float = 0.5,
) -> torch.Tensor:
    """
    50-step gradient refinement after legalization.

    Why: legalization resolves overlaps by displacing macros from their
    gradient-optimal positions. This can push congestion up (macros that
    were near their connected components get scattered). A short re-run
    with WL + congestion gradient (no density — macros are already spread)
    recovers some of the lost quality.

    Uses a soft overlap penalty (OVL_W=5) rather than hard projection.
    Macros start legal (ovl=0); the small penalty discourages new overlaps
    without overriding the WL+cong gradient signal.

    Key differences from main gradient:
      DEN_W = 0.3 — light density guard: macros are legal but cong gradient
                    without density pulls them together → density spikes.
                    Observed in results_3: ibm02 +0.18 density from refine alone.
      OVL_W = 5   — light touch; macros start legal so ovl starts at 0
      LR = 0.01   — fine-tuning; smaller than main gradient's 0.05
      alpha = 50  — sharper HPWL for fine-tuning regime
    """
    pos = pos_cpu.clone().to(device)
    sizes    = b.macro_sizes.to(device)
    port_pos = b.port_positions.to(device)
    num_hard = b.num_hard_macros
    cw, ch   = b.canvas_width, b.canvas_height

    movable     = b.get_movable_mask().to(device)
    movable_idx = movable.nonzero(as_tuple=True)[0]
    if len(movable_idx) == 0:
        return pos_cpu

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    cell_centers, cell_size = _make_cell_centers(b, device)

    pos_movable = pos[movable_idx].detach().requires_grad_(True)
    optimizer   = torch.optim.Adam([pos_movable], lr=0.01)

    best_loss    = float("inf")
    best_movable = pos_movable.detach().clone()

    for step in range(steps):
        optimizer.zero_grad()

        p = pos.clone()
        p[movable_idx] = pos_movable
        p_x = p[:, 0].clamp(half_w, cw - half_w)
        p_y = p[:, 1].clamp(half_h, ch - half_h)
        p   = torch.stack([p_x, p_y], dim=1)

        pin_xy = _compute_pin_xy(p, data, b, port_pos)
        wl   = lse_hpwl_loss(pin_xy, data, b, alpha=50.0)
        cong = lroute_congestion_loss(pin_xy, data, b, device, pos=p, sizes=sizes)
        den  = density_loss(p, sizes, cell_centers, cell_size, b, target_density=1.0)
        ovl  = macro_overlap_loss(p, sizes, num_hard)
        loss = wl + cong_w * cong + 0.4 * den + 20.0 * ovl
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            pos_movable[:, 0].clamp_(half_w[movable_idx], cw - half_w[movable_idx])
            pos_movable[:, 1].clamp_(half_h[movable_idx], ch - half_h[movable_idx])

        l = loss.item()
        if l < best_loss:
            best_loss    = l
            best_movable = pos_movable.detach().clone()

    final = pos.clone()
    final[movable_idx] = best_movable
    return final.cpu()


# ---------------------------------------------------------------------------
# SA refinement helpers
# ---------------------------------------------------------------------------

def _has_hard_overlap(i: int, pos: torch.Tensor, sizes: torch.Tensor,
                       num_hard: int, gap: float = 0.02) -> bool:
    """Check if macro i overlaps any other hard macro. O(N)."""
    xi, yi = pos[i, 0].item(), pos[i, 1].item()
    wi, hi = sizes[i, 0].item(), sizes[i, 1].item()
    for j in range(num_hard):
        if j == i:
            continue
        xj, yj = pos[j, 0].item(), pos[j, 1].item()
        wj, hj = sizes[j, 0].item(), sizes[j, 1].item()
        if abs(xi - xj) < (wi + wj) / 2 + gap and abs(yi - yj) < (hi + hj) / 2 + gap:
            return True
    return False


def _nearest_movable(i: int, movable: list, pos: torch.Tensor) -> int:
    """Return index of the nearest movable macro to macro i. O(N)."""
    xi, yi = pos[i, 0].item(), pos[i, 1].item()
    best_j, best_d2 = -1, float("inf")
    for j in movable:
        if j == i:
            continue
        d2 = (pos[j, 0].item() - xi) ** 2 + (pos[j, 1].item() - yi) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_j = j
    return best_j


# ---------------------------------------------------------------------------
# SA refinement (post-legalization)
# ---------------------------------------------------------------------------

def _sa_refinement(
    pos_cpu: torch.Tensor,    # [num_macros, 2] — after legalization, on CPU
    b: Benchmark,
    data: dict,               # from _preprocess(), on device
    device: torch.device,
    max_iters: int = 3000,
    time_budget_s: float = 30.0,
) -> torch.Tensor:
    """SA refinement: L1 WL cost, cold-start, neighbor-biased shifts+swaps."""
    pos  = pos_cpu.clone()
    sizes    = b.macro_sizes
    num_hard = b.num_hard_macros
    cw, ch   = b.canvas_width, b.canvas_height

    movable = [i for i in range(num_hard) if not b.macro_fixed[i].item()]
    if len(movable) < 2:
        return pos_cpu

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    pin_owner_cpu    = data["pin_owner"].cpu()
    edge_src_cpu     = data["edge_src_idx"].cpu()
    edge_snk_cpu     = data["edge_snk_idx"].cpu()
    edge_wt_cpu      = data["edge_weights"].cpu()
    edge_src_owner   = pin_owner_cpu[edge_src_cpu]
    edge_snk_owner   = pin_owner_cpu[edge_snk_cpu]
    port_pos_fixed   = b.port_positions

    @torch.no_grad()
    def _fast_cost(p: torch.Tensor) -> float:
        if port_pos_fixed.shape[0] > 0:
            p_ext = torch.cat([p, port_pos_fixed], dim=0)   # [N+ports, 2]
        else:
            p_ext = p
        max_idx = p_ext.shape[0] - 1
        src_pos = p_ext[edge_src_owner.clamp(0, max_idx)]   # [E, 2]
        snk_pos = p_ext[edge_snk_owner.clamp(0, max_idx)]   # [E, 2]
        return (edge_wt_cpu * (src_pos - snk_pos).abs().sum(dim=1)).sum().item()

    T_canvas = max(cw, ch)
    T       = T_canvas * 0.01
    T_end   = T_canvas * 0.0001
    cooling = (T_end / T) ** (1.0 / max_iters)

    current_cost = _fast_cost(pos)
    best_cost    = current_cost
    best_pos     = pos.clone()

    start_t    = time.time()
    n_accepted = 0
    n_tried    = 0
    last_iter  = 0

    for it in range(max_iters):
        if it % 200 == 0 and time.time() - start_t > time_budget_s:
            break

        last_iter = it
        i = movable[random.randrange(len(movable))]
        old_xi = pos[i, 0].item()
        old_yi = pos[i, 1].item()
        j      = -1
        old_xj = old_yj = 0.0

        if random.random() < 0.6:
            # SHIFT: Gaussian perturbation; sigma proportional to temperature
            sigma  = T
            new_x  = old_xi + random.gauss(0.0, sigma)
            new_y  = old_yi + random.gauss(0.0, sigma)
            new_x  = max(half_w[i].item(), min(new_x, cw - half_w[i].item()))
            new_y  = max(half_h[i].item(), min(new_y, ch - half_h[i].item()))
            pos[i, 0] = new_x
            pos[i, 1] = new_y

            if _has_hard_overlap(i, pos, sizes, num_hard):
                pos[i, 0] = old_xi
                pos[i, 1] = old_yi
                T *= cooling
                continue
        else:
            # SWAP: exchange positions of i and j
            if random.random() < 0.7:
                j = _nearest_movable(i, movable, pos)  # neighbor-biased
            else:
                j = movable[random.randrange(len(movable))]

            if j == i or j < 0:
                T *= cooling
                continue

            old_xj = pos[j, 0].item()
            old_yj = pos[j, 1].item()
            pos[i, 0] = old_xj; pos[i, 1] = old_yj
            pos[j, 0] = old_xi; pos[j, 1] = old_yi

            # Canvas bounds check for both macros at swapped positions
            if (pos[i, 0] < half_w[i] or pos[i, 0] > cw - half_w[i] or
                    pos[i, 1] < half_h[i] or pos[i, 1] > ch - half_h[i] or
                    pos[j, 0] < half_w[j] or pos[j, 0] > cw - half_w[j] or
                    pos[j, 1] < half_h[j] or pos[j, 1] > ch - half_h[j] or
                    _has_hard_overlap(i, pos, sizes, num_hard) or
                    _has_hard_overlap(j, pos, sizes, num_hard)):
                pos[i, 0] = old_xi; pos[i, 1] = old_yi
                pos[j, 0] = old_xj; pos[j, 1] = old_yj
                T *= cooling
                continue

        n_tried += 1
        new_cost = _fast_cost(pos)
        delta    = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / (T + 1e-12)):
            current_cost = new_cost
            n_accepted  += 1
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos  = pos.clone()
        else:
            pos[i, 0] = old_xi; pos[i, 1] = old_yi
            if j >= 0:
                pos[j, 0] = old_xj; pos[j, 1] = old_yj

        T *= cooling

    elapsed = time.time() - start_t
    print(f"  [SA] {last_iter + 1} iters, {n_tried} evals, {n_accepted} accepted  "
          f"fast-WL: {current_cost:.4f} → {best_cost:.4f}  ({elapsed:.1f}s)")
    return best_pos   # already CPU


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class AnalyticalPlacer:
    """
    Analytical global placer.
    Harness calls: placer.place(benchmark) -> [num_macros, 2] Tensor
    """

    def __init__(self):
        pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        b = benchmark
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[analytical_placer] device={device}")

        print("[analytical_placer] Preprocessing benchmark tensors...")
        data = _preprocess(b, device)
        port_pos = b.port_positions.to(device)
        cell_centers, cell_size = _make_cell_centers(b, device)
        sizes = b.macro_sizes.to(device)
        movable = b.get_movable_mask().to(device)
        movable_idx = movable.nonzero(as_tuple=True)[0]

        pos_full = b.macro_positions.clone().to(device)

        TOTAL_STEPS  = 300
        ALPHA_START  = 10.0
        ALPHA_END    = 30.0
        DEN_W_PHASE1 = 2.0
        DEN_W_PHASE2 = 0.5
        OVL_W        = 20.0   # never reduce — causes legalization regression
        PHASE2_START = 100
        TARGET_DEN   = 1.0
        LR           = 0.05
        GRAD_CLIP    = 5.0
        NUM_RESTARTS = 2      # original init + 1 perturbed restart

        cw, ch = b.canvas_width, b.canvas_height
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        # ------------------------------------------------------------------
        # Multi-start phase 1: run NUM_RESTARTS starts for PHASE2_START steps,
        # pick the one with lowest cong_100 at step 99, continue only that one.
        #
        # We preserve the WINNING restart's Adam optimizer (with its moment
        # estimates) so phase 2 continues from a warm-started Adam, not a
        # fresh one. This avoids the tiny-step issue from zero moments.
        # ------------------------------------------------------------------
        best_cong_at_99   = float("inf")
        best_pos_movable  = None   # winning restart's parameter tensor
        best_optimizer    = None   # winning restart's Adam (warm state)
        best_scheduler    = None   # winning restart's LR scheduler
        measured_cong_100 = 0.0
        CONG_W_PHASE2     = 0.10

        PERTURB_SIGMA = max(cw, ch) / 10.0

        for restart_idx in range(NUM_RESTARTS):
            if restart_idx == 0:
                pm_init = pos_full[movable_idx].detach().clone()
            else:
                g = torch.Generator(device=device)
                g.manual_seed(restart_idx * 17)
                noise = torch.randn(pm_init.shape, generator=g, device=device) * PERTURB_SIGMA
                pm_init = (pos_full[movable_idx].detach() + noise)
                pm_init[:, 0].clamp_(half_w[movable_idx], cw - half_w[movable_idx])
                pm_init[:, 1].clamp_(half_h[movable_idx], ch - half_h[movable_idx])

            pos_movable = pm_init.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([pos_movable], lr=LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TOTAL_STEPS, eta_min=0.005
            )

            print(f"[analytical_placer] Restart {restart_idx}: Phase 1 ({PHASE2_START} steps)...")
            for step in range(PHASE2_START):
                optimizer.zero_grad()

                pos = pos_full.clone()
                pos[movable_idx] = pos_movable
                pos_x = pos[:, 0].clamp(half_w, cw - half_w)
                pos_y = pos[:, 1].clamp(half_h, ch - half_h)
                pos   = torch.stack([pos_x, pos_y], dim=1)

                frac  = step / TOTAL_STEPS
                alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * frac

                pin_xy = _compute_pin_xy(pos, data, b, port_pos)
                wl  = lse_hpwl_loss(pin_xy, data, b, alpha)
                den = density_loss(pos, sizes, cell_centers, cell_size, b, target_density=TARGET_DEN)
                ovl = macro_overlap_loss(pos, sizes, b.num_hard_macros)
                loss = wl + DEN_W_PHASE1 * den + OVL_W * ovl  # no cong in phase 1
                loss.backward()

                torch.nn.utils.clip_grad_norm_([pos_movable], GRAD_CLIP)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    pos_movable[:, 0].clamp_(half_w[movable_idx], cw - half_w[movable_idx])
                    pos_movable[:, 1].clamp_(half_h[movable_idx], ch - half_h[movable_idx])

                if step % 50 == 0:
                    print(f"  [r{restart_idx}] step {step:3d}  loss={loss.item():.4f}  "
                          f"wl={wl.item():.4f}  den={den.item():.6f}  alpha={alpha:.1f}")

            # Measure cong_100 at step 99 using the current (end-of-phase-1) positions
            with torch.no_grad():
                p_meas = pos_full.clone()
                p_meas[movable_idx] = pos_movable.detach()
                p_meas[:, 0] = p_meas[:, 0].clamp(half_w, cw - half_w)
                p_meas[:, 1] = p_meas[:, 1].clamp(half_h, ch - half_h)
                pxy_meas = _compute_pin_xy(p_meas, data, b, port_pos)
                cong_100 = lroute_congestion_loss(
                    pxy_meas, data, b, device, pos=p_meas, sizes=sizes
                ).item()

            cong_w_p2 = min(0.30, max(0.10, 0.10 + 0.20 * (cong_100 - 1.2) / 0.6))
            is_best = cong_100 < best_cong_at_99
            print(f"  [restart {restart_idx}] cong_100={cong_100:.4f} → cong_w={cong_w_p2:.3f}"
                  + ("  *** NEW BEST" if is_best else ""))

            if is_best:
                best_cong_at_99  = cong_100
                # Keep the actual parameter tensor + warm Adam — phase 2 continues directly
                best_pos_movable = pos_movable
                best_optimizer   = optimizer
                best_scheduler   = scheduler
                measured_cong_100 = cong_100
                CONG_W_PHASE2    = cong_w_p2

        # ------------------------------------------------------------------
        # Phase 2: continue from the winning restart's warm Adam state
        # ------------------------------------------------------------------
        pos_movable = best_pos_movable
        optimizer   = best_optimizer
        scheduler   = best_scheduler

        best_loss    = float("inf")
        best_movable = pos_movable.detach().clone()

        print(f"[analytical_placer] Phase 2 ({TOTAL_STEPS - PHASE2_START} steps, "
              f"cong_w={CONG_W_PHASE2:.3f}, cong_100={measured_cong_100:.2f})...")
        for step in range(PHASE2_START, TOTAL_STEPS):
            optimizer.zero_grad()

            pos = pos_full.clone()
            pos[movable_idx] = pos_movable
            pos_x = pos[:, 0].clamp(half_w, cw - half_w)
            pos_y = pos[:, 1].clamp(half_h, ch - half_h)
            pos   = torch.stack([pos_x, pos_y], dim=1)

            frac  = step / TOTAL_STEPS
            alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * frac

            pin_xy = _compute_pin_xy(pos, data, b, port_pos)
            wl  = lse_hpwl_loss(pin_xy, data, b, alpha)
            den = density_loss(pos, sizes, cell_centers, cell_size, b, target_density=TARGET_DEN)
            ovl = macro_overlap_loss(pos, sizes, b.num_hard_macros)
            cong = lroute_congestion_loss(pin_xy, data, b, device, pos=pos, sizes=sizes)
            loss = wl + DEN_W_PHASE2 * den + OVL_W * ovl + CONG_W_PHASE2 * cong
            loss.backward()

            torch.nn.utils.clip_grad_norm_([pos_movable], GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pos_movable[:, 0].clamp_(half_w[movable_idx], cw - half_w[movable_idx])
                pos_movable[:, 1].clamp_(half_h[movable_idx], ch - half_h[movable_idx])

            l = loss.item()
            if l < best_loss:
                best_loss    = l
                best_movable = pos_movable.detach().clone()

            if (step - PHASE2_START) % 50 == 0:
                print(f"  step {step:4d}  loss={l:.4f}  wl={wl.item():.4f}  "
                      f"den={den.item():.6f}  cong={cong.item():.4f}  "
                      f"den_w={DEN_W_PHASE2:.2f}  cong_w={CONG_W_PHASE2:.2f}  alpha={alpha:.1f}")

        # Reconstruct and move to CPU
        final_gpu = pos_full.clone()
        final_gpu[movable_idx] = best_movable
        analytical_pos = final_gpu.cpu()

        print("[analytical_placer] Legalizing hard macros...")
        final_pos = _legalize(analytical_pos, b, time_budget_s=120.0)

        cong_w_r1 = 0.5
        cong_w_r2 = min(0.8, 0.4 + 0.4 * (measured_cong_100 - 1.2) / 0.6) if measured_cong_100 > 1.2 else 0.4
        cong_w_r2 = max(0.4, cong_w_r2)

        print(f"[analytical_placer] Post-legalize refine round 1 (40 steps, cong_w={cong_w_r1:.2f})...")
        final_pos = _post_legalize_refine(final_pos, b, data, device, steps=40, cong_w=cong_w_r1)

        print(f"[analytical_placer] Post-legalize refine round 2 (60 steps, cong_w={cong_w_r2:.2f}, "
              f"cong_100={measured_cong_100:.2f})...")
        final_pos = _post_legalize_refine(final_pos, b, data, device, steps=60, cong_w=cong_w_r2)

        return final_pos
