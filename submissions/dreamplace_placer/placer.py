"""
DREAMPlace-style macro placer fallback.

The real DREAMPlace CUDA ops are not available in this environment: PyPI
provides only a placeholder package with no dreamplace.ops modules, and the
MacroPlacement submodule does not vendor DREAMPlace. This placer follows the
Step 1B fallback architecture instead:

  - weighted-average wirelength (WAW) instead of LSE HPWL
  - pure PyTorch Nesterov updates instead of Adam
  - existing differentiable density and L-route congestion surrogates
  - explicit hard-macro clearance support for NG45 submissions
  - joint hard+soft optimization plus soft-only repositioning after legalization
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark
from submissions.analytical_placer.placer import (
    _compute_pin_xy,
    _make_cell_centers,
    _preprocess,
    density_loss,
    lroute_congestion_loss,
    macro_overlap_loss,
)


def _scatter_weighted_average(
    vals: torch.Tensor,
    net_idx: torch.Tensor,
    num_nets: int,
    gamma: float,
    maximize: bool,
) -> torch.Tensor:
    """Per-net weighted average max/min coordinate used by WAW wirelength."""
    signed = vals / gamma if maximize else -vals / gamma
    max_signed = torch.full((num_nets,), -float("inf"), dtype=vals.dtype, device=vals.device)
    max_signed.scatter_reduce_(0, net_idx, signed, reduce="amax", include_self=True)
    max_signed = torch.where(torch.isfinite(max_signed), max_signed, torch.zeros_like(max_signed))
    exp_v = (signed - max_signed[net_idx]).exp()

    denom = torch.zeros(num_nets, dtype=vals.dtype, device=vals.device)
    numer = torch.zeros(num_nets, dtype=vals.dtype, device=vals.device)
    denom.scatter_add_(0, net_idx, exp_v)
    numer.scatter_add_(0, net_idx, vals * exp_v)
    return numer / denom.clamp(min=1e-12)


def weighted_average_wirelength_loss(
    pin_xy: torch.Tensor,
    data: dict,
    net_weights: torch.Tensor,
    norm: float,
    gamma: float = 0.5,
) -> torch.Tensor:
    """
    Weighted-average wirelength surrogate.

    WAW estimates max(x), min(x), max(y), and min(y) by softmax-weighted
    averages. It is smoother than exact HPWL but sharper and less explosive
    than LSE on large nets when gamma is kept modest.
    """
    net_idx = data["pin_net_idx"]
    num_nets = data["num_nets"]
    x = pin_xy[:, 0]
    y = pin_xy[:, 1]

    x_max = _scatter_weighted_average(x, net_idx, num_nets, gamma, maximize=True)
    x_min = _scatter_weighted_average(x, net_idx, num_nets, gamma, maximize=False)
    y_max = _scatter_weighted_average(y, net_idx, num_nets, gamma, maximize=True)
    y_min = _scatter_weighted_average(y, net_idx, num_nets, gamma, maximize=False)

    hpwl = (x_max - x_min) + (y_max - y_min)
    return (net_weights * hpwl).sum() / norm


def nesterov_step(
    x: torch.Tensor,
    x_prev: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    t_prev: float,
) -> tuple[torch.Tensor, float]:
    """Apply the session-plan Nesterov update to an already computed gradient."""
    t_cur = (1.0 + math.sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0
    y = x + (t_prev / t_cur) * (x - x_prev)
    return y - step_size * grad, t_cur


def _nearest_rect_distance(gap_x: torch.Tensor, gap_y: torch.Tensor) -> torch.Tensor:
    """
    Distance between two axis-aligned rectangles from signed x/y gaps.

    Positive gap means separated on that axis; negative means overlap on that
    axis. This gives the nearest edge distance for separated rectangles and 0
    when rectangles overlap.
    """
    gx = F.relu(gap_x)
    gy = F.relu(gap_y)
    diagonal = torch.sqrt(gx * gx + gy * gy + 1e-12)
    axis_aligned = torch.maximum(gx, gy)
    both_separated = (gx > 0) & (gy > 0)
    return torch.where(both_separated, diagonal, axis_aligned)


def macro_clearance_loss(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    min_gap: float = 12.0,
) -> torch.Tensor:
    """Soft barrier for hard macro pairs whose nearest-edge gap is too small."""
    if num_hard <= 1 or min_gap <= 0:
        return pos.sum() * 0.0

    hard_pos = pos[:num_hard]
    hard_sizes = sizes[:num_hard]
    x = hard_pos[:, 0]
    y = hard_pos[:, 1]
    hw = hard_sizes[:, 0] / 2
    hh = hard_sizes[:, 1] / 2

    gap_x = (x.unsqueeze(0) - x.unsqueeze(1)).abs() - (hw.unsqueeze(0) + hw.unsqueeze(1))
    gap_y = (y.unsqueeze(0) - y.unsqueeze(1)).abs() - (hh.unsqueeze(0) + hh.unsqueeze(1))
    distance = _nearest_rect_distance(gap_x, gap_y)

    mask = torch.triu(torch.ones(num_hard, num_hard, device=pos.device, dtype=torch.bool), diagonal=1)
    violation = F.relu(float(min_gap) - distance[mask])
    return violation.pow(2).sum()


def _hard_soft_overlap_loss(pos: torch.Tensor, sizes: torch.Tensor, num_hard: int) -> torch.Tensor:
    """Penalize soft macros that overlap hard macros; soft-soft overlap is allowed."""
    if num_hard == 0 or num_hard >= pos.shape[0]:
        return pos.sum() * 0.0

    hard_pos = pos[:num_hard]
    soft_pos = pos[num_hard:]
    hard_sizes = sizes[:num_hard]
    soft_sizes = sizes[num_hard:]

    dx = (hard_pos[:, 0:1] - soft_pos[:, 0].unsqueeze(0)).abs()
    dy = (hard_pos[:, 1:2] - soft_pos[:, 1].unsqueeze(0)).abs()
    min_x = hard_sizes[:, 0:1] / 2 + soft_sizes[:, 0].unsqueeze(0) / 2
    min_y = hard_sizes[:, 1:2] / 2 + soft_sizes[:, 1].unsqueeze(0) / 2
    px = F.relu(min_x - dx)
    py = F.relu(min_y - dy)
    return (px * py).sum()


def _effective_clearance_gap(b: Benchmark) -> float:
    """
    Use 12 um where it is physically plausible, but avoid destroying ICCAD04.

    IBM canvases are only tens of microns wide at roughly 80% utilization, so a
    12 um all-pairs hard-macro clearance is geometrically impossible. The 12 um
    rule is a Grand Prize NG45 requirement, so the IBM proxy track keeps the
    normal legality gap while NG45-sized canvases receive the requested gap.
    """
    if b.canvas_width < 100.0 or b.canvas_height < 100.0:
        return 0.02
    return 12.0


def _has_pin_connectivity(b: Benchmark) -> bool:
    """Return True when Benchmark has enough pin data for WAW/congestion."""
    return (
        len(b.net_pin_nodes) == b.num_nets
        and len(b.net_pin_nodes) > 0
        and len(b.macro_pin_offsets) >= b.num_hard_macros
    )


def _clamp_to_canvas(pos: torch.Tensor, sizes: torch.Tensor, b: Benchmark) -> torch.Tensor:
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    x = pos[:, 0].clamp(half_w, float(b.canvas_width) - half_w)
    y = pos[:, 1].clamp(half_h, float(b.canvas_height) - half_h)
    return torch.stack([x, y], dim=1)


def _has_overlap_with_placed(
    idx: int,
    cand_x: float,
    cand_y: float,
    placed: list[int],
    pos: torch.Tensor,
    sizes: torch.Tensor,
    gap: float,
) -> bool:
    wi = sizes[idx, 0].item()
    hi = sizes[idx, 1].item()
    for j in placed:
        xj = pos[j, 0].item()
        yj = pos[j, 1].item()
        wj = sizes[j, 0].item()
        hj = sizes[j, 1].item()
        if abs(cand_x - xj) < (wi + wj) / 2 + gap and abs(cand_y - yj) < (hi + hj) / 2 + gap:
            return True
    return False


def _tetris_legalize(pos: torch.Tensor, b: Benchmark, gap: float) -> torch.Tensor:
    """
    Greedy spatial legalization for hard macros.

    This avoids the old all-pairs iterative separation loop. Hard macros are
    processed largest-first and placed at the nearest legal spiral candidate
    against already placed macros, which behaves like Tetris compaction while
    keeping the search local.
    """
    out = pos.clone()
    sizes = b.macro_sizes
    num_hard = b.num_hard_macros
    fixed = b.macro_fixed
    cw = float(b.canvas_width)
    ch = float(b.canvas_height)

    fixed_indices = [i for i in range(num_hard) if fixed[i].item()]
    movable_indices = [i for i in range(num_hard) if not fixed[i].item()]
    order = sorted(movable_indices, key=lambda i: float(sizes[i, 0] * sizes[i, 1]), reverse=True)
    placed: list[int] = []
    failed = 0

    for i in fixed_indices:
        hw = sizes[i, 0].item() / 2
        hh = sizes[i, 1].item() / 2
        out[i, 0] = min(max(out[i, 0].item(), hw), cw - hw)
        out[i, 1] = min(max(out[i, 1].item(), hh), ch - hh)
        placed.append(i)

    for i in order:
        hw = sizes[i, 0].item() / 2
        hh = sizes[i, 1].item() / 2
        ox = min(max(out[i, 0].item(), hw), cw - hw)
        oy = min(max(out[i, 1].item(), hh), ch - hh)
        step = max(0.02, 0.10 * max(sizes[i, 0].item(), sizes[i, 1].item()), gap)
        best = None
        best_d2 = float("inf")

        for ring in range(0, 600):
            found_ring = False
            offsets: list[tuple[int, int]] = []
            if ring == 0:
                offsets.append((0, 0))
            else:
                for dx in range(-ring, ring + 1):
                    offsets.append((dx, -ring))
                    offsets.append((dx, ring))
                for dy in range(-ring + 1, ring):
                    offsets.append((-ring, dy))
                    offsets.append((ring, dy))

            for dx, dy in offsets:
                cx = min(max(ox + dx * step, hw), cw - hw)
                cy = min(max(oy + dy * step, hh), ch - hh)
                if _has_overlap_with_placed(i, cx, cy, placed, out, sizes, gap):
                    continue
                d2 = (cx - ox) * (cx - ox) + (cy - oy) * (cy - oy)
                if d2 < best_d2:
                    best = (cx, cy)
                    best_d2 = d2
                    found_ring = True
            if found_ring:
                break

        if best is None:
            failed += 1
            best = (ox, oy)
        out[i, 0], out[i, 1] = best
        placed.append(i)

    if failed:
        print(f"  [legalize] warning: {failed} hard macros kept at nearest clamped positions")
    return out


def _clearance_stats(pos: torch.Tensor, sizes: torch.Tensor, num_hard: int, min_gap: float) -> tuple[int, float]:
    if num_hard <= 1 or min_gap <= 0:
        return 0, 0.0
    with torch.no_grad():
        hard_pos = pos[:num_hard]
        hard_sizes = sizes[:num_hard]
        x = hard_pos[:, 0]
        y = hard_pos[:, 1]
        hw = hard_sizes[:, 0] / 2
        hh = hard_sizes[:, 1] / 2
        gap_x = (x.unsqueeze(0) - x.unsqueeze(1)).abs() - (hw.unsqueeze(0) + hw.unsqueeze(1))
        gap_y = (y.unsqueeze(0) - y.unsqueeze(1)).abs() - (hh.unsqueeze(0) + hh.unsqueeze(1))
        distance = _nearest_rect_distance(gap_x, gap_y)
        mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
        violations = F.relu(float(min_gap) - distance[mask])
        count = int((violations > 1e-4).sum().item())
        max_push = float(violations.max().item()) if count else 0.0
        return count, max_push


class DreamPlaceMacroPlacer:
    """
    Phase 1: global placement with WAW + density + L-route congestion using Nesterov.
    Phase 2: spatial Tetris-style hard macro legalization.
    Phase 3: soft macro repositioning after hard macros are fixed.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        b = benchmark
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[dreamplace_placer] device={device}")

        has_pins = _has_pin_connectivity(b)
        data = _preprocess(b, device) if has_pins else None
        port_pos = b.port_positions.to(device)
        cell_centers, cell_size = _make_cell_centers(b, device)
        sizes = b.macro_sizes.to(device)
        net_weights = b.net_weights.to(device)
        movable_idx = b.get_movable_mask().to(device).nonzero(as_tuple=True)[0]

        if movable_idx.numel() == 0:
            return b.macro_positions.clone()

        pos_full = b.macro_positions.clone().to(device)
        x = pos_full[movable_idx].detach().clone()
        x_prev = x.clone()
        t_prev = 1.0

        norm = float((b.canvas_width + b.canvas_height) * max(1, b.num_nets))
        clearance_gap = _effective_clearance_gap(b)
        print(f"[dreamplace_placer] clearance_gap={clearance_gap:.2f}um")
        if not has_pins:
            print("[dreamplace_placer] no pin connectivity; using geometry-only fallback")

        total_steps = 500 if device.type == "cuda" else 220
        lambda_d = 0.0
        lambda_cap = 2.0
        base_step = max(b.canvas_width, b.canvas_height) * (0.0025 if device.type == "cuda" else 0.0015)
        best_loss = float("inf")
        best_x = x.clone()

        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        print(f"[dreamplace_placer] Nesterov global placement ({total_steps} steps)...")
        for step in range(total_steps):
            t_cur = (1.0 + math.sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0
            y = (x + (t_prev / t_cur) * (x - x_prev)).detach().requires_grad_(True)

            p = pos_full.clone()
            p[movable_idx] = y
            p = _clamp_to_canvas(p, sizes, b)

            gamma = max(0.10, 1.0 * (0.995 ** step))
            if has_pins:
                pin_xy = _compute_pin_xy(p, data, b, port_pos)
                wl = weighted_average_wirelength_loss(pin_xy, data, net_weights, norm, gamma=gamma)
                cong = lroute_congestion_loss(pin_xy, data, b, device) if step >= 50 else p.sum() * 0.0
            else:
                wl = p.sum() * 0.0
                cong = p.sum() * 0.0
            den = density_loss(p, sizes, cell_centers, cell_size, b, target_density=1.0)
            ovl = macro_overlap_loss(p, sizes, b.num_hard_macros, gap=0.02)
            clear = macro_clearance_loss(p, sizes, b.num_hard_macros, min_gap=clearance_gap)
            hard_soft = _hard_soft_overlap_loss(p, sizes, b.num_hard_macros)

            loss = wl + lambda_d * den + 0.5 * cong + 20.0 * ovl + 0.1 * clear + 0.05 * hard_soft
            grad = torch.autograd.grad(loss, y)[0]
            grad_norm = grad.norm().clamp(min=1e-12)
            if grad_norm > 10.0:
                grad = grad * (10.0 / grad_norm)

            step_size = base_step * (0.10 + 0.90 * (1.0 - step / max(1, total_steps)))
            x_next, _ = nesterov_step(x, x_prev, grad, step_size=step_size, t_prev=t_prev)

            with torch.no_grad():
                x_prev = x
                x = x_next.detach()
                x[:, 0].clamp_(half_w[movable_idx], float(b.canvas_width) - half_w[movable_idx])
                x[:, 1].clamp_(half_h[movable_idx], float(b.canvas_height) - half_h[movable_idx])
                t_prev = t_cur

                lambda_d = 0.01 if lambda_d == 0.0 else min(lambda_cap, lambda_d * 1.05)
                if den.item() < 1e-4:
                    lambda_d *= 0.98

                selection = wl + 0.5 * den + 0.5 * cong + 0.25 * ovl + 0.1 * clear + 0.05 * hard_soft
                l = float(selection.detach().item())
                if step >= 25 and l < best_loss:
                    best_loss = l
                    best_x = x.clone()

            if step % 50 == 0:
                print(
                    f"  step {step:4d} loss={loss.item():.4f} wl={wl.item():.4f} "
                    f"den={den.item():.5f} cong={cong.item():.4f} "
                    f"ovl={ovl.item():.4f} clear={clear.item():.2f} lambda_d={lambda_d:.3f}"
                )

        final = pos_full.clone()
        final[movable_idx] = best_x
        final = _clamp_to_canvas(final, sizes, b).cpu()

        print("[dreamplace_placer] Tetris-style hard macro legalization...")
        start = time.time()
        final = _tetris_legalize(final, b, gap=clearance_gap)
        count, max_push = _clearance_stats(final, b.macro_sizes, b.num_hard_macros, clearance_gap)
        print(f"  [clearance] violations={count} max_needed_push={max_push:.3f}um ({time.time() - start:.1f}s)")

        print("[dreamplace_placer] Soft macro repositioning...")
        final = self._reposition_soft_macros(final, b, data, device)
        return final

    def _reposition_soft_macros(
        self,
        pos_cpu: torch.Tensor,
        b: Benchmark,
        data: dict | None,
        device: torch.device,
    ) -> torch.Tensor:
        soft_mask = b.get_soft_macro_mask() & b.get_movable_mask()
        soft_idx = soft_mask.nonzero(as_tuple=True)[0].to(device)
        if soft_idx.numel() == 0:
            return pos_cpu

        sizes = b.macro_sizes.to(device)
        has_pins = data is not None
        port_pos = b.port_positions.to(device)
        net_weights = b.net_weights.to(device)
        cell_centers, cell_size = _make_cell_centers(b, device)
        pos_full = pos_cpu.clone().to(device)
        x = pos_full[soft_idx].detach().clone()
        x_prev = x.clone()
        t_prev = 1.0
        norm = float((b.canvas_width + b.canvas_height) * max(1, b.num_nets))
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        steps = 80 if device.type == "cuda" else 40
        step_size = max(b.canvas_width, b.canvas_height) * 0.001

        for _ in range(steps):
            t_cur = (1.0 + math.sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0
            y = (x + (t_prev / t_cur) * (x - x_prev)).detach().requires_grad_(True)
            p = pos_full.clone()
            p[soft_idx] = y
            p = _clamp_to_canvas(p, sizes, b)

            if has_pins:
                pin_xy = _compute_pin_xy(p, data, b, port_pos)
                wl = weighted_average_wirelength_loss(pin_xy, data, net_weights, norm, gamma=0.5)
            else:
                wl = p.sum() * 0.0
            den = density_loss(p, sizes, cell_centers, cell_size, b, target_density=1.0)
            hard_soft = _hard_soft_overlap_loss(p, sizes, b.num_hard_macros)
            loss = wl + 0.3 * den + 0.1 * hard_soft
            grad = torch.autograd.grad(loss, y)[0]
            grad_norm = grad.norm().clamp(min=1e-12)
            if grad_norm > 10.0:
                grad = grad * (10.0 / grad_norm)

            x_next, _ = nesterov_step(x, x_prev, grad, step_size=step_size, t_prev=t_prev)
            with torch.no_grad():
                x_prev = x
                x = x_next.detach()
                x[:, 0].clamp_(half_w[soft_idx], float(b.canvas_width) - half_w[soft_idx])
                x[:, 1].clamp_(half_h[soft_idx], float(b.canvas_height) - half_h[soft_idx])
                t_prev = t_cur

        out = pos_full.clone()
        out[soft_idx] = x
        return _clamp_to_canvas(out, sizes, b).cpu()
