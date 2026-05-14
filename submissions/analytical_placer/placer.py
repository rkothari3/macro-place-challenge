"""
Analytical global placer: LSE-HPWL + density + L-route congestion surrogate
Optimization: Adam gradient descent → greedy spiral legalization → SA refinement

Pin resolution (from net_pin_nodes col0 = owner index):
  [0, num_hard)           hard macro → placement[owner] + macro_pin_offsets[owner][slot]
  [num_hard, num_macro)   soft macro → placement[owner] (center, slot always 0)
  [num_macro, ...)        I/O port   → port_positions[owner - num_macro] (fixed)

Congestion surrogate: L-route (not RUDY).
  RUDY spreads demand over bbox AREA → gradient opposes competition (longer L-routes).
  L-route traces H+V segments → gradient correctly pushes pins closer.
  See findings.md section T for full formula derivation.
"""
from __future__ import annotations

import math
import random
import time
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

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
# RUDY congestion surrogate (kept for reference; NOT used in gradient loop)
# ---------------------------------------------------------------------------

def rudy_congestion_loss(
    pin_xy: torch.Tensor,
    data: dict,
    b: Benchmark,
    device: torch.device,
) -> torch.Tensor:
    """
    RUDY differentiable congestion surrogate (bbox-uniform demand distribution).

    NOT used in the main gradient loop because RUDY's gradient OPPOSES the
    competition's L-route gradient for already-spread placements:
    - RUDY rewards larger bboxes (lower per-cell demand) → pushes macros apart
    - L-route penalizes longer wires (more edge crossings) → pulls macros closer

    Kept here for reference / comparison. Use lroute_congestion_loss() instead.
    """
    rows = b.grid_rows
    cols = b.grid_cols
    cw = b.canvas_width / cols
    ch = b.canvas_height / rows
    num_nets = data["num_nets"]
    net_idx  = data["pin_net_idx"]
    net_weights = b.net_weights.to(device)

    x = pin_xy[:, 0]
    y = pin_xy[:, 1]
    alpha = 50.0

    net_x_max =  _scatter_lse( x, net_idx, num_nets, alpha)
    net_x_min = -_scatter_lse(-x, net_idx, num_nets, alpha)
    net_y_max =  _scatter_lse( y, net_idx, num_nets, alpha)
    net_y_min = -_scatter_lse(-y, net_idx, num_nets, alpha)

    bbox_w = (net_x_max - net_x_min).clamp(min=cw * 0.5)
    bbox_h = (net_y_max - net_y_min).clamp(min=ch * 0.5)
    routing_density = net_weights / (bbox_w * bbox_h)

    col_left  = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(0) * cw
    col_right = col_left + cw
    row_bot   = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(0) * ch
    row_top   = row_bot + ch

    xmin = net_x_min.unsqueeze(1); xmax = net_x_max.unsqueeze(1)
    ymin = net_y_min.unsqueeze(1); ymax = net_y_max.unsqueeze(1)

    overlap_x = F.relu(torch.minimum(xmax, col_right) - torch.maximum(xmin, col_left))
    overlap_y = F.relu(torch.minimum(ymax, row_top)   - torch.maximum(ymin, row_bot))

    scaled_x = routing_density.unsqueeze(1) * overlap_x
    demand   = overlap_y.t() @ scaled_x

    h_supply = float(b.hroutes_per_micron) * ch
    v_supply = float(b.vroutes_per_micron) * cw
    avg_supply = (h_supply + v_supply) / 2.0

    demand_flat = demand.flatten() / avg_supply
    G = rows * cols
    k = max(1, int(0.05 * G))
    return torch.topk(demand_flat, k).values.mean()


# ---------------------------------------------------------------------------
# L-route differentiable congestion surrogate (matches plc_client_os.py semantics)
# ---------------------------------------------------------------------------

def lroute_congestion_loss(
    pin_xy: torch.Tensor,    # [total_pins, 2] — differentiable pin world coords
    data: dict,              # from _preprocess(), includes edge_src/snk/weights
    b: Benchmark,
    device: torch.device,
    smooth_range: int = 2,   # IBM benchmarks use smooth_range=2
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
        cong = lroute_congestion_loss(pin_xy, data, b, device)
        den  = density_loss(p, sizes, cell_centers, cell_size, b, target_density=1.0)
        ovl  = macro_overlap_loss(p, sizes, num_hard)
        loss = wl + cong_w * cong + 0.3 * den + 5.0 * ovl
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
    """
    SA refinement using WL + density surrogate objective.

    Why SA after gradient descent?
    - Gradient descent finds a good basin in a continuous landscape but gets
      stuck in local minima because Adam is a first-order method.
    - SA accepts bad moves with probability exp(-ΔCost/T), allowing escape
      from local minima. At low T (cold start), it mostly accepts improvements.
    - Starting cold (T = 0.01 × canvas_max) preserves the gradient solution's
      quality while allowing micro-corrections.

    Why use WL+density surrogate instead of full plc proxy?
    - plc.get_congestion_cost() is O(nets × grid) in pure Python: ~0.5-2s/call.
      With 30s budget, this gives only ~15-60 SA iterations — nearly nothing.
    - Our torch surrogate evaluates in ~5ms: gives 3000-6000 iterations in 30s.
    - WL+density covers 67% of the proxy cost (0.5×density is the biggest term
      in high-utilization benchmarks). SA on 67% of objective >> SA on 100% with
      15 iterations.

    Why neighbor-biased swaps?
    - Random swaps often propose moves that are far from the current basin
      and get rejected. Neighbor-biased swaps try nearby macros first,
      which have higher acceptance rates and produce more meaningful moves.
      At high T: random swaps dominate (exploration). At low T: neighbor
      swaps dominate (local refinement). We use 70/30 split.

    Temperature schedule:
    - T_start = 0.01 × max(cw, ch): much lower than will_seed's 0.15.
      We start from a good gradient-descent solution, not random.
      High T_start would destroy the gradient's work by accepting random moves.
    - T_end = 0.0001 × max(cw, ch): small perturbations at end.
    - Geometric cooling: T *= (T_end/T_start)^(1/max_iters) each iteration.
    """
    # SA runs entirely on CPU — pos never moves to GPU.
    # Rationale: every `.item()` on a GPU tensor incurs a sync. SA calls overlap
    # checks (which do many .item() calls) and cost on every iteration. Keeping
    # pos on CPU eliminates GPU sync overhead and gives ~10-100x more iterations
    # in the same time budget.
    pos  = pos_cpu.clone()       # CPU throughout
    sizes    = b.macro_sizes     # already CPU
    num_hard = b.num_hard_macros
    cw, ch   = b.canvas_width, b.canvas_height

    movable = [i for i in range(num_hard) if not b.macro_fixed[i].item()]
    if len(movable) < 2:
        return pos_cpu

    half_w = sizes[:, 0] / 2   # [N] CPU
    half_h = sizes[:, 1] / 2   # [N] CPU

    # Precompute macro-owner indices for each edge endpoint (used in fast WL).
    # edge_src/snk_idx are flat pin indices; pin_owner maps pin → macro/port owner.
    # Port owners (>= num_macro) are in b.port_positions; macro owners are in pos.
    # We build a combined [num_macros + num_ports, 2] lookup at eval time.
    pin_owner_cpu    = data["pin_owner"].cpu()
    edge_src_cpu     = data["edge_src_idx"].cpu()
    edge_snk_cpu     = data["edge_snk_idx"].cpu()
    edge_wt_cpu      = data["edge_weights"].cpu()
    edge_src_owner   = pin_owner_cpu[edge_src_cpu]   # [E] macro/port owner of source pin
    edge_snk_owner   = pin_owner_cpu[edge_snk_cpu]   # [E]
    port_pos_fixed   = b.port_positions               # [num_ports, 2] CPU, fixed throughout SA

    @torch.no_grad()
    def _fast_cost(p: torch.Tensor) -> float:
        """
        L1 pairwise wirelength using macro centers (ignores pin offsets).
        Runs in ~0.1ms on CPU vs ~10ms for full LSE+density surrogate.

        Why L1 pairwise WL is a valid SA objective:
        - It tracks the same topology as our gradient surrogate: connected macros
          should be nearby. Minimizing L1 WL finds locally Pareto-improving moves.
        - Pin offsets (typically <1/4 macro size) barely affect the ranking of moves,
          so ignoring them costs almost nothing in solution quality.
        - 100x more iterations in the same budget >> slightly better per-iteration
          accuracy. At cold SA temperatures, move acceptance is near-deterministic
          (accept iff ΔCost < 0), so each iteration is a direct improvement.

        What SA loses by not having density/congestion signal:
        - SA won't explicitly avoid congestion hotspots or dense regions.
        - This is acceptable: the gradient already handled global WL/density/cong.
          SA is a fine-tuner for small local improvements, not a global optimizer.
          Density and congestion change slowly with small macro shifts; WL changes
          immediately. The WL signal alone guides SA to meaningful improvements.
        """
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
# Soft macro force-directed refinement (unused — kept for future experiments)
# ---------------------------------------------------------------------------

def _soft_macro_fd(pos: torch.Tensor, b: Benchmark, steps: int = 300) -> torch.Tensor:
    """Force-directed placement for soft macros. Currently not called."""
    pos = pos.clone()
    num_hard  = b.num_hard_macros
    num_macro = b.num_macros
    cw, ch = b.canvas_width, b.canvas_height

    soft_idx = [i for i in range(num_hard, num_macro) if not b.macro_fixed[i].item()]
    if not soft_idx:
        return pos

    soft_adj: dict[int, list[tuple[int, float]]] = {i: [] for i in soft_idx}
    for net_i, nodes in enumerate(b.net_nodes):
        w = float(b.net_weights[net_i].item())
        node_list = nodes.tolist()
        for ni in node_list:
            if ni >= num_hard and ni < num_macro:
                for nj in node_list:
                    if nj != ni:
                        soft_adj[ni].append((nj, w))

    hard_rep = [(b.macro_sizes[j, 0].item() + b.macro_sizes[j, 1].item()) / 2
                for j in range(num_hard)]

    T = max(cw, ch) * 0.01
    cooling = 0.97

    for _ in range(steps):
        for i in soft_idx:
            fx, fy = 0.0, 0.0
            xi, yi = pos[i, 0].item(), pos[i, 1].item()
            for j, w in soft_adj[i]:
                if j < num_macro:
                    xj, yj = pos[j, 0].item(), pos[j, 1].item()
                else:
                    port_j = j - num_macro
                    if port_j < b.port_positions.shape[0]:
                        xj, yj = b.port_positions[port_j, 0].item(), b.port_positions[port_j, 1].item()
                    else:
                        continue
                dx, dy = xj - xi, yj - yi
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                fx += w * dx / dist; fy += w * dy / dist
            for j in range(num_hard):
                xj, yj = pos[j, 0].item(), pos[j, 1].item()
                dx, dy = xi - xj, yi - yj
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                rep = hard_rep[j]
                if dist < rep * 2.5:
                    s = rep * rep / (dist * dist + 1e-6)
                    fx += s * dx / dist; fy += s * dy / dist
            norm = math.sqrt(fx * fx + fy * fy) + 1e-6
            scale = min(T, norm) / norm
            xi = max(b.macro_sizes[i, 0].item() / 2, min(xi + fx * scale, cw - b.macro_sizes[i, 0].item() / 2))
            yi = max(b.macro_sizes[i, 1].item() / 2, min(yi + fy * scale, ch - b.macro_sizes[i, 1].item() / 2))
            pos[i, 0] = xi; pos[i, 1] = yi
        T *= cooling

    return pos


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

        # Preprocess into GPU tensors (runs once)
        print("[analytical_placer] Preprocessing benchmark tensors...")
        data = _preprocess(b, device)
        port_pos = b.port_positions.to(device)
        cell_centers, cell_size = _make_cell_centers(b, device)
        sizes = b.macro_sizes.to(device)
        movable = b.get_movable_mask().to(device)

        movable_idx = movable.nonzero(as_tuple=True)[0]  # [num_movable]

        # Init from current benchmark positions
        pos_full    = b.macro_positions.clone().to(device)
        pos_movable = pos_full[movable_idx].detach().requires_grad_(True)

        # ------------------------------------------------------------------
        # Hyperparameters
        #
        # CONG_W phases:
        #   Phase 1 (steps 0-99): CONG_W = 0 — let density+overlap resolve first.
        #     Adding congestion before overlaps are gone causes the L-route gradient
        #     to fight the overlap penalty → unstable optimization.
        #   Phase 2 (steps 100-299): CONG_W = 0.3 — L-route gradient pulls macros
        #     into low-congestion configurations. Start AFTER overlaps ≈ 0.
        #
        # OVL_W = 20 throughout: RUDY's regression (Session 2) showed that
        #   reducing OVL_W to 5 in phase 2 allowed congestion gradient to recreate
        #   small overlaps → 20-51x slower legalization. Keep OVL_W = 20 always.
        # ------------------------------------------------------------------
        # 300 steps (fixed): 1000-step runs diverged on hard benchmarks (ibm02 cong 2.4→7.6)
        # More SA time (60s) compensates for fewer gradient steps.
        TOTAL_STEPS    = 300
        ALPHA_START    = 10.0
        ALPHA_END      = 30.0
        DEN_W_PHASE1   = 2.0    # strong cell density penalty
        DEN_W_PHASE2   = 0.5    # keep at 0.5: needed to resist L-route compression → fewer overlaps → faster legalization
        OVL_W_PHASE1   = 20.0   # direct macro-pair overlap penalty
        OVL_W_PHASE2   = 20.0   # NEVER reduce — prevents legalization regression
        CONG_W_PHASE1  = 0.0    # no congestion until overlaps resolve
        CONG_W_PHASE2  = 0.3    # L-route surrogate; overridden adaptively at step PHASE2_START-1
        PHASE2_START   = 100
        TARGET_DEN     = 1.0
        LR             = 0.05
        GRAD_CLIP      = 5.0

        optimizer = torch.optim.Adam([pos_movable], lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_STEPS, eta_min=0.005
        )

        best_loss    = float("inf")
        best_movable = pos_movable.detach().clone()
        measured_cong_100 = 0.0   # filled at step PHASE2_START-1, used for SA restart decision

        cw, ch = b.canvas_width, b.canvas_height
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        print(f"[analytical_placer] Gradient descent ({TOTAL_STEPS} steps)...")
        for step in range(TOTAL_STEPS):
            optimizer.zero_grad()

            pos = pos_full.clone()
            pos[movable_idx] = pos_movable

            pos_x = pos[:, 0].clamp(half_w, cw - half_w)
            pos_y = pos[:, 1].clamp(half_h, ch - half_h)
            pos   = torch.stack([pos_x, pos_y], dim=1)

            frac  = step / TOTAL_STEPS
            alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * frac

            den_w  = DEN_W_PHASE1  if step < PHASE2_START else DEN_W_PHASE2
            ovl_w  = OVL_W_PHASE1  if step < PHASE2_START else OVL_W_PHASE2
            cong_w = CONG_W_PHASE1 if step < PHASE2_START else CONG_W_PHASE2

            pin_xy = _compute_pin_xy(pos, data, b, port_pos)
            wl  = lse_hpwl_loss(pin_xy, data, b, alpha)
            den = density_loss(pos, sizes, cell_centers, cell_size, b, target_density=TARGET_DEN)
            ovl = macro_overlap_loss(pos, sizes, b.num_hard_macros)

            if cong_w > 0:
                cong = lroute_congestion_loss(pin_xy, data, b, device)
            else:
                cong = torch.zeros(1, device=device).squeeze()

            loss = wl + den_w * den + ovl_w * ovl + cong_w * cong
            loss.backward()

            torch.nn.utils.clip_grad_norm_([pos_movable], GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pos_movable[:, 0].clamp_(half_w[movable_idx], cw - half_w[movable_idx])
                pos_movable[:, 1].clamp_(half_h[movable_idx], ch - half_h[movable_idx])

            # Adaptive CONG_W: measure L-route surrogate just before phase 2 starts.
            # Low-congestion benchmarks (ibm09/ibm11) regressed with CONG_W=0.3 because
            # L-route compressed macros that were already well-spread → density spike.
            # Formula: scale from 0.10 (cong_100≤1.2) to 0.30 (cong_100≥1.8).
            if step == PHASE2_START - 1:
                with torch.no_grad():
                    p_meas = pos_full.clone()
                    p_meas[movable_idx] = pos_movable
                    p_meas[:, 0] = p_meas[:, 0].clamp(half_w, cw - half_w)
                    p_meas[:, 1] = p_meas[:, 1].clamp(half_h, ch - half_h)
                    pxy_meas = _compute_pin_xy(p_meas, data, b, port_pos)
                    cong_100 = lroute_congestion_loss(pxy_meas, data, b, device).item()
                CONG_W_PHASE2 = min(0.30, max(0.10, 0.10 + 0.20 * (cong_100 - 1.2) / 0.6))
                measured_cong_100 = cong_100
                print(f"  [adaptive] cong_100={cong_100:.4f} → CONG_W_PHASE2={CONG_W_PHASE2:.3f}")

            l = loss.item()
            if l < best_loss:
                best_loss    = l
                best_movable = pos_movable.detach().clone()

            if step % 50 == 0:
                print(f"  step {step:4d}  loss={l:.4f}  wl={wl.item():.4f}  "
                      f"den={den.item():.6f}  cong={cong.item():.4f}  "
                      f"den_w={den_w:.2f}  cong_w={cong_w:.2f}  alpha={alpha:.1f}")

        # Reconstruct and move to CPU
        final_gpu = pos_full.clone()
        final_gpu[movable_idx] = best_movable
        analytical_pos = final_gpu.cpu()

        # Phase 3: legalize hard macros (120s cap — most benchmarks converge well
        # within this; only ibm10's ~1100s extreme case gets truncated)
        print("[analytical_placer] Legalizing hard macros...")
        final_pos = _legalize(analytical_pos, b, time_budget_s=120.0)

        # Phase 4: post-legalization gradient refinement — DISABLED
        # cong gradient (cong_w=0.5) caused density to spike (ibm01: 0.576→0.934)
        # by clustering macros after legalization had spread them cleanly.
        # print("[analytical_placer] Post-legalization gradient refinement (50 steps)...")
        # final_pos = _post_legalize_refine(final_pos, b, data, device, steps=50, cong_w=0.5)

        # Phase 5: SA refinement — multiple restarts for high-congestion benchmarks.
        # ibm02/ibm06/ibm15/ibm18 have cong_100 > 2.3 and score above 1.4; a single
        # SA run can get stuck. 3 × 30s = 90s extra per benchmark, worth the trade.
        n_sa_trials = 3 if measured_cong_100 > 2.3 else 1
        if n_sa_trials > 1:
            print(f"[analytical_placer] SA refinement ({n_sa_trials} restarts × 30s, cong_100={measured_cong_100:.2f})...")
        else:
            print("[analytical_placer] SA refinement (fast WL, 30s budget)...")

        # Fast L1 WL cost for comparing SA trial results
        _powner  = data["pin_owner"].cpu()
        _esrc    = data["edge_src_idx"].cpu()
        _esnk    = data["edge_snk_idx"].cpu()
        _ewt     = data["edge_weights"].cpu()
        _portpos = b.port_positions  # [P, 2] CPU

        def _eval_wl(pos_cpu: torch.Tensor) -> float:
            p_ext = torch.cat([pos_cpu, _portpos], dim=0) if _portpos.shape[0] > 0 else pos_cpu
            mi = p_ext.shape[0] - 1
            s = p_ext[_powner[_esrc].clamp(0, mi)]
            k = p_ext[_powner[_esnk].clamp(0, mi)]
            return (_ewt * (s - k).abs().sum(1)).sum().item()

        best_sa_pos  = None
        best_sa_cost = float('inf')
        for trial in range(n_sa_trials):
            random.seed(trial * 37)
            torch.manual_seed(trial * 37)
            trial_pos = _sa_refinement(
                final_pos, b, data, device,
                max_iters=3000, time_budget_s=60.0
            )

            trial_cost = _eval_wl(trial_pos)
            if n_sa_trials > 1:
                print(f"  [SA trial {trial+1}/{n_sa_trials}] wl_cost={trial_cost:.1f}")
            if trial_cost < best_sa_cost:
                best_sa_cost = trial_cost
                best_sa_pos = trial_pos
        final_pos = best_sa_pos

        return final_pos
