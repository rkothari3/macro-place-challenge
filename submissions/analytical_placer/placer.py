"""
Analytical global placer: LSE-HPWL + bell-kernel density surrogate
Optimization: Adam gradient descent → greedy spiral legalization → soft-macro FD

Pin resolution (from net_pin_nodes col0 = owner index):
  [0, num_hard)           hard macro → placement[owner] + macro_pin_offsets[owner][slot]
  [num_hard, num_macro)   soft macro → placement[owner] (center, slot always 0)
  [num_macro, ...)        I/O port   → port_positions[owner - num_macro] (fixed)
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Preprocessing: flatten variable-length net_pin_nodes into GPU tensors
# ---------------------------------------------------------------------------

def _preprocess(b: Benchmark, device: torch.device) -> dict:
    """
    Convert all variable-length lists into flat packed tensors for scatter ops.

    Returns dict with:
      pin_net_idx      [total_pins] int64  — which net each pin belongs to
      pin_owner        [total_pins] int64  — owner index (macro or port)
      pin_is_hard      [total_pins] bool   — True if hard macro pin
      pin_is_port      [total_pins] bool   — True if I/O port
      hard_offsets     [total_hard_pins, 2] float32 — stacked macro_pin_offsets
      hard_pin_flat_idx [total_pins] int64 — index into hard_offsets (0 for non-hard)
      num_nets         int
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

    # Flatten net_pin_nodes into parallel arrays
    all_net_idx, all_owner, all_slot = [], [], []
    for net_i, pins in enumerate(b.net_pin_nodes):
        n = pins.shape[0]
        if n == 0:
            continue
        all_net_idx.append(torch.full((n,), net_i, dtype=torch.long))
        all_owner.append(pins[:, 0])
        all_slot.append(pins[:, 1])

    pin_net_idx = torch.cat(all_net_idx).to(device)   # [total_pins]
    pin_owner   = torch.cat(all_owner).to(device)     # [total_pins]
    pin_slot    = torch.cat(all_slot).to(device)      # [total_pins]

    pin_is_hard = pin_owner < num_hard
    pin_is_port = pin_owner >= num_macro

    # For hard pins: compute absolute index into hard_offsets tensor.
    # This Python loop runs once at startup; ~1s for ibm17's ~200k pins.
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

    return dict(
        pin_net_idx=pin_net_idx,
        pin_owner=pin_owner,
        pin_is_hard=pin_is_hard.to(device),
        pin_is_port=pin_is_port.to(device),
        hard_offsets=hard_offsets,
        hard_pin_flat_idx=hard_pin_flat_idx,
        num_nets=b.num_nets,
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

    # Start with owner macro center (valid for soft macros; overridden for hard/port)
    clamped_owner = owner.clamp(0, num_macro - 1)
    pin_xy = pos[clamped_owner]        # [total_pins, 2]

    # Hard macro pins: add offset (offset is 0 for non-hard, masked by is_hard)
    if is_hard.any():
        hard_flat = data["hard_pin_flat_idx"]    # [total_pins]
        offsets   = data["hard_offsets"]         # [total_hard_pins, 2]
        # For pins where hard_offsets is empty, hard_flat will be 0 but
        # is_hard ensures the offset only applies to real hard pins.
        if offsets.shape[0] > 0:
            offset_xy = offsets[hard_flat]       # [total_pins, 2]
            pin_xy = pin_xy + offset_xy * is_hard.unsqueeze(1).float()

    # I/O ports: replace with fixed port position (no gradient from ports)
    if is_port.any() and port_pos.shape[0] > 0:
        port_owner_idx = (owner - num_macro).clamp(min=0)   # [total_pins]
        port_owner_idx = port_owner_idx.clamp(max=port_pos.shape[0] - 1)
        port_xy = port_pos[port_owner_idx]                   # [total_pins, 2]
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
    """
    Differentiable HPWL via log-sum-exp. Returns normalized scalar.
    As alpha → ∞, this converges to true HPWL.
    """
    net_idx  = data["pin_net_idx"]    # [total_pins]
    num_nets = data["num_nets"]
    weights  = b.net_weights.to(pin_xy.device)   # [num_nets]

    x = pin_xy[:, 0]
    y = pin_xy[:, 1]

    lse_x_max =  _scatter_lse( x, net_idx, num_nets, alpha)
    lse_x_min = -_scatter_lse(-x, net_idx, num_nets, alpha)
    lse_y_max =  _scatter_lse( y, net_idx, num_nets, alpha)
    lse_y_min = -_scatter_lse(-y, net_idx, num_nets, alpha)

    hpwl_per_net = (lse_x_max - lse_x_min) + (lse_y_max - lse_y_min)  # [num_nets]

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
    grid_y, grid_x = torch.meshgrid(row_c, col_c, indexing="ij")   # [rows, cols]
    cell_centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [G, 2]
    return cell_centers, torch.tensor([cw, ch], dtype=torch.float32, device=device)


def density_loss(
    pos: torch.Tensor,           # [N, 2] macro centers (grad-tracked)
    sizes: torch.Tensor,         # [N, 2] macro (w, h) — NOT grad-tracked
    cell_centers: torch.Tensor,  # [G, 2]
    cell_size: torch.Tensor,     # [2] = (cw, ch)
    b: Benchmark,
    target_density: float = 1.0,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Differentiable density penalty using exact rectangle overlap.
    cell_density[g] = sum_i (overlap_area(macro_i, cell_g)) / cell_area.
    Processes macros in chunks to bound GPU memory.
    """
    N = pos.shape[0]
    G = cell_centers.shape[0]
    half_cw = cell_size[0] / 2
    half_ch = cell_size[1] / 2
    cell_area = cell_size[0] * cell_size[1]

    gx = cell_centers[:, 0]  # [G]
    gy = cell_centers[:, 1]  # [G]

    cell_density = torch.zeros(G, dtype=pos.dtype, device=pos.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        cx = pos[start:end, 0:1]            # [C, 1]
        cy = pos[start:end, 1:2]            # [C, 1]
        hw = sizes[start:end, 0:1] / 2     # [C, 1]
        hh = sizes[start:end, 1:2] / 2     # [C, 1]

        # True axis-aligned rectangle overlap: [C, G]
        lo_x = torch.maximum(cx - hw, gx - half_cw)   # [C, G]
        hi_x = torch.minimum(cx + hw, gx + half_cw)
        lo_y = torch.maximum(cy - hh, gy - half_ch)
        hi_y = torch.minimum(cy + hh, gy + half_ch)

        overlap_area = F.relu(hi_x - lo_x) * F.relu(hi_y - lo_y)  # [C, G]
        cell_density = cell_density + overlap_area.sum(dim=0) / cell_area

    overflow = F.relu(cell_density - target_density)
    return overflow.pow(2).mean()


# ---------------------------------------------------------------------------
# Direct macro-pair overlap penalty (catches small overlaps missed by cell density)
# ---------------------------------------------------------------------------

def macro_overlap_loss(
    pos: torch.Tensor,      # [num_macros, 2]
    sizes: torch.Tensor,    # [num_macros, 2]
    num_hard: int,
    gap: float = 0.02,
) -> torch.Tensor:
    """
    Penalizes pairwise overlap between hard macros. O(N²) but N≈300 max.
    Returns total overlap area (scalar).
    """
    x  = pos[:num_hard, 0]   # [N]
    y  = pos[:num_hard, 1]
    hw = sizes[:num_hard, 0] / 2 + gap / 2  # [N] half-width + gap/2
    hh = sizes[:num_hard, 1] / 2 + gap / 2

    # Pairwise penetration [N, N]
    dx = (x.unsqueeze(0) - x.unsqueeze(1)).abs()   # [N, N]
    dy = (y.unsqueeze(0) - y.unsqueeze(1)).abs()
    px = F.relu(hw.unsqueeze(0) + hw.unsqueeze(1) - dx)
    py = F.relu(hh.unsqueeze(0) + hh.unsqueeze(1) - dy)

    # Overlap area per pair: min(px, py) selects the smaller axis
    overlap = torch.minimum(px, py) * (px > 0).float() * (py > 0).float()  # [N, N]

    # Upper triangle only (avoid double-counting), zero diagonal
    mask = torch.triu(torch.ones(num_hard, num_hard, device=pos.device, dtype=torch.bool), diagonal=1)
    return overlap[mask].sum()


# ---------------------------------------------------------------------------
# Minimal-perturbation legalization (pairwise separation of hard macros only)
# ---------------------------------------------------------------------------

def _legalize(pos: torch.Tensor, b: Benchmark) -> torch.Tensor:
    """
    Hybrid legalization for hard macros:
    1. Iterative pairwise separation (minimal perturbation, O(N²) per iter)
    2. Spiral fallback only for macros that still overlap after pairwise phase
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

    # ---------- Phase 1: pairwise separation (100 passes) ----------
    for _ in range(100):
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
            return pos

    # ---------- Phase 2: spiral fallback for remaining violators ----------
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
# Soft macro force-directed refinement
# ---------------------------------------------------------------------------

def _soft_macro_fd(pos: torch.Tensor, b: Benchmark, steps: int = 300) -> torch.Tensor:
    """
    Force-directed placement for soft macros, treating hard macros as fixed.
    Uses net_nodes (not net_pin_nodes) for connectivity since soft macros pin at center.
    """
    pos = pos.clone()
    num_hard  = b.num_hard_macros
    num_macro = b.num_macros
    cw, ch = b.canvas_width, b.canvas_height

    soft_idx = [i for i in range(num_hard, num_macro) if not b.macro_fixed[i].item()]
    if not soft_idx:
        return pos

    # Build adjacency: soft_macro_idx → list of (other_idx, weight)
    soft_adj: dict[int, list[tuple[int, float]]] = {i: [] for i in soft_idx}
    for net_i, nodes in enumerate(b.net_nodes):
        w = float(b.net_weights[net_i].item())
        node_list = nodes.tolist()
        for ni in node_list:
            if ni >= num_hard and ni < num_macro:  # soft macro
                for nj in node_list:
                    if nj != ni:
                        soft_adj[ni].append((nj, w))

    # Precompute hard macro repulsion radii
    hard_rep = [
        (b.macro_sizes[j, 0].item() + b.macro_sizes[j, 1].item()) / 2
        for j in range(num_hard)
    ]

    T = max(cw, ch) * 0.01
    cooling = 0.97

    for _ in range(steps):
        for i in soft_idx:
            fx, fy = 0.0, 0.0
            xi, yi = pos[i, 0].item(), pos[i, 1].item()

            # Attractive spring to connected nodes
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
                fx += w * dx / dist
                fy += w * dy / dist

            # Repulsive from hard macros
            for j in range(num_hard):
                xj, yj = pos[j, 0].item(), pos[j, 1].item()
                dx, dy = xi - xj, yi - yj
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                rep = hard_rep[j]
                if dist < rep * 2.5:
                    strength = rep * rep / (dist * dist + 1e-6)
                    fx += strength * dx / dist
                    fy += strength * dy / dist

            norm = math.sqrt(fx * fx + fy * fy) + 1e-6
            scale = min(T, norm) / norm
            xi += fx * scale
            yi += fy * scale

            hw = b.macro_sizes[i, 0].item() / 2
            hh = b.macro_sizes[i, 1].item() / 2
            xi = max(hw, min(xi, cw - hw))
            yi = max(hh, min(yi, ch - hh))
            pos[i, 0] = xi
            pos[i, 1] = yi

        T *= cooling

    return pos


# ---------------------------------------------------------------------------
# Main placer class (API: place(self, benchmark) -> Tensor)
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
        pos_full = b.macro_positions.clone().to(device)
        pos_movable = pos_full[movable_idx].detach().requires_grad_(True)

        TOTAL_STEPS    = 300
        ALPHA_START    = 10.0
        ALPHA_END      = 30.0
        DEN_W_PHASE1   = 2.0    # strong cell+macro overlap penalty
        DEN_W_PHASE2   = 0.4    # gentle spreading in phase 2
        OVL_W_PHASE1   = 20.0   # direct macro-pair overlap penalty (phase 1)
        OVL_W_PHASE2   = 5.0    # keep macro-pair penalty active in phase 2
        PHASE2_START   = 100    # switch after overlaps resolved
        TARGET_DEN     = 1.0    # penalize cell overflow
        LR             = 0.05
        GRAD_CLIP      = 5.0

        optimizer = torch.optim.Adam([pos_movable], lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_STEPS, eta_min=0.005
        )

        best_loss    = float("inf")
        best_movable = pos_movable.detach().clone()

        cw, ch = b.canvas_width, b.canvas_height
        half_w = sizes[:, 0] / 2   # [N]
        half_h = sizes[:, 1] / 2   # [N]

        print(f"[analytical_placer] Gradient descent ({TOTAL_STEPS} steps)...")
        for step in range(TOTAL_STEPS):
            optimizer.zero_grad()

            # Reconstruct full pos tensor; only movable portion is differentiable
            pos = pos_full.clone()
            pos[movable_idx] = pos_movable

            # Clamp to canvas (differentiable, passes grad through)
            pos_x = pos[:, 0].clamp(half_w, cw - half_w)
            pos_y = pos[:, 1].clamp(half_h, ch - half_h)
            pos = torch.stack([pos_x, pos_y], dim=1)

            frac  = step / TOTAL_STEPS
            alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * frac

            pin_xy = _compute_pin_xy(pos, data, b, port_pos)
            wl  = lse_hpwl_loss(pin_xy, data, b, alpha)
            den = density_loss(pos, sizes, cell_centers, cell_size, b,
                               target_density=TARGET_DEN)
            ovl = macro_overlap_loss(pos, sizes, b.num_hard_macros)

            den_w = DEN_W_PHASE1 if step < PHASE2_START else DEN_W_PHASE2
            ovl_w = OVL_W_PHASE1 if step < PHASE2_START else OVL_W_PHASE2
            loss  = wl + den_w * den + ovl_w * ovl
            loss.backward()

            torch.nn.utils.clip_grad_norm_([pos_movable], GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            # Hard-project back to canvas after optimizer step
            with torch.no_grad():
                pos_movable[:, 0].clamp_(half_w[movable_idx], cw - half_w[movable_idx])
                pos_movable[:, 1].clamp_(half_h[movable_idx], ch - half_h[movable_idx])

            l = loss.item()
            if l < best_loss:
                best_loss    = l
                best_movable = pos_movable.detach().clone()

            if step % 50 == 0:
                print(f"  step {step:4d}  loss={l:.4f}  wl={wl.item():.4f}  "
                      f"den={den.item():.6f}  den_w={den_w:.2f}  alpha={alpha:.1f}")

        # Reconstruct and move to CPU
        final_gpu = pos_full.clone()
        final_gpu[movable_idx] = best_movable
        analytical_pos = final_gpu.cpu()

        # Phase 3: legalize hard macros (soft macros kept at initial positions
        # from the .plc file, which are already placement-tool optimized)
        print("[analytical_placer] Legalizing hard macros...")
        final_pos = _legalize(analytical_pos, b)

        return final_pos
