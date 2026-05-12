"""
test_density_kernel.py — correctness + timing for density_cuda_ext

Run after building:
  cd submissions/analytical_placer/density_ext && pip install -e .
  python test_density_kernel.py

Tests:
  1. Hardcoded 10-macro, 9-cell correctness vs PyTorch reference (< 1e-4 error)
  2. Backward gradient correctness via torch.autograd.gradcheck
  3. ibm17-scale timing: N=2604, G=2244 (100 iterations, CUDA events)
"""

import torch
import torch.nn.functional as F
import time

# ── PyTorch reference implementation (matches placer.py density_loss) ──────
def density_forward_pytorch(pos, sizes, cell_xy, half_cw, half_ch, inv_cell_area,
                             chunk_size=256):
    N = pos.shape[0]
    G = cell_xy.shape[0]
    gx = cell_xy[:, 0]
    gy = cell_xy[:, 1]
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
        overlap = F.relu(hi_x - lo_x) * F.relu(hi_y - lo_y)
        cell_density = cell_density + overlap.sum(dim=0) * inv_cell_area
    return cell_density


def run_tests():
    import density_cuda_ext

    print("=" * 60)
    print("density_cuda_ext — correctness + timing")
    print("=" * 60)

    device = torch.device('cuda')

    # ── Test 1: 10-macro, 9-cell hardcoded correctness ──────────────────────
    print("\n[Test 1] 10 macros, 9 cells (3×3 grid on 3×3 μm canvas)")

    canvas = 3.0
    rows, cols = 3, 3
    cw = canvas / cols
    ch = canvas / rows
    half_cw = cw / 2
    half_ch = ch / 2
    inv_cell_area = 1.0 / (cw * ch)

    # Cell centers: 3×3 grid
    xs = torch.tensor([cw/2, cw*1.5, cw*2.5])
    ys = torch.tensor([ch/2, ch*1.5, ch*2.5])
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    cell_xy = torch.stack([gx.flatten(), gy.flatten()], dim=1).cuda()  # [9, 2]

    # 10 macros: some inside one cell, some spanning multiple
    pos = torch.tensor([
        [0.5, 0.5],   # macro 0: center of cell (0,0)
        [1.5, 0.5],   # macro 1: center of cell (0,1)
        [0.5, 1.5],   # macro 2: center of cell (1,0)
        [1.5, 1.5],   # macro 3: center of cell (1,1) — 4 macros overlapping same cell
        [0.5, 0.5],   # macro 4: same as macro 0
        [2.5, 2.5],   # macro 5: corner cell (2,2)
        [1.0, 1.0],   # macro 6: corner between 4 cells
        [0.2, 0.2],   # macro 7: small, cell (0,0)
        [2.8, 2.8],   # macro 8: small, cell (2,2)
        [1.5, 0.8],   # macro 9: cell (0,1)
    ], dtype=torch.float32).cuda()

    sizes = torch.tensor([
        [0.4, 0.4],  # macro 0
        [0.4, 0.4],  # macro 1
        [0.4, 0.4],  # macro 2
        [0.4, 0.4],  # macro 3
        [0.4, 0.4],  # macro 4 (duplicate of 0)
        [0.4, 0.4],  # macro 5
        [0.8, 0.8],  # macro 6: spans 4 cells
        [0.1, 0.1],  # macro 7
        [0.1, 0.1],  # macro 8
        [0.3, 0.3],  # macro 9
    ], dtype=torch.float32).cuda()

    ref = density_forward_pytorch(pos, sizes, cell_xy, half_cw, half_ch, inv_cell_area)
    out = density_cuda_ext.forward(pos, sizes, cell_xy, half_cw, half_ch, inv_cell_area)

    max_err = (ref - out.cpu()).abs().max().item()
    print(f"  PyTorch ref: {ref.tolist()}")
    print(f"  CUDA kernel: {out.cpu().tolist()}")
    print(f"  Max abs error: {max_err:.2e}", "✓ PASS" if max_err < 1e-4 else "✗ FAIL")

    # ── Test 2: Gradient correctness via finite differences ─────────────────
    print("\n[Test 2] Gradient correctness (finite differences, N=10, G=9)")

    from placer_autograd_wrapper import _DensityKernel

    pos_fd = pos.clone().requires_grad_(True)
    cell_density = _DensityKernel.apply(pos_fd, sizes, cell_xy, half_cw, half_ch, inv_cell_area)
    overflow = F.relu(cell_density - 0.5)
    loss = overflow.pow(2).mean()
    loss.backward()
    analytic_grad = pos_fd.grad.clone()

    # Finite differences
    eps = 1e-3
    fd_grad = torch.zeros_like(analytic_grad)
    for d in range(2):
        for i in range(pos.shape[0]):
            pos_p = pos.clone(); pos_p[i, d] += eps
            pos_m = pos.clone(); pos_m[i, d] -= eps
            f_p = F.relu(density_forward_pytorch(pos_p, sizes, cell_xy, half_cw, half_ch, inv_cell_area) - 0.5).pow(2).mean()
            f_m = F.relu(density_forward_pytorch(pos_m, sizes, cell_xy, half_cw, half_ch, inv_cell_area) - 0.5).pow(2).mean()
            fd_grad[i, d] = (f_p - f_m) / (2 * eps)

    fd_err = (analytic_grad.cpu() - fd_grad).abs().max().item()
    print(f"  Max gradient error vs finite diff: {fd_err:.2e}",
          "✓ PASS" if fd_err < 1e-2 else "✗ FAIL")

    # ── Test 3: ibm17-scale timing ───────────────────────────────────────────
    print("\n[Test 3] ibm17-scale timing: N=2604, G=2244")

    N_big, G_big = 2604, 2244
    pos_big   = torch.rand(N_big, 2, device=device) * 72.6
    sizes_big = torch.rand(N_big, 2, device=device) * 5.0 + 0.5
    cell_xy_big = torch.rand(G_big, 2, device=device) * 72.6
    cw_b, ch_b = 72.6 / 51, 72.6 / 44
    half_cw_b, half_ch_b = cw_b / 2, ch_b / 2
    inv_cell_area_b = 1.0 / (cw_b * ch_b)

    # Warmup
    for _ in range(10):
        _ = density_cuda_ext.forward(pos_big, sizes_big, cell_xy_big,
                                      half_cw_b, half_ch_b, inv_cell_area_b)
    torch.cuda.synchronize()

    # Time forward: CUDA events (not CPU clock) for accurate GPU timing
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(100):
        _ = density_cuda_ext.forward(pos_big, sizes_big, cell_xy_big,
                                      half_cw_b, half_ch_b, inv_cell_area_b)
    end_ev.record()
    torch.cuda.synchronize()
    fwd_ms = start_ev.elapsed_time(end_ev) / 100
    print(f"  CUDA forward avg:  {fwd_ms:.3f} ms/call")

    # Time backward
    grad_density = torch.rand(G_big, device=device)
    for _ in range(10):
        _ = density_cuda_ext.backward(grad_density, pos_big, sizes_big, cell_xy_big,
                                       half_cw_b, half_ch_b, inv_cell_area_b)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(100):
        _ = density_cuda_ext.backward(grad_density, pos_big, sizes_big, cell_xy_big,
                                       half_cw_b, half_ch_b, inv_cell_area_b)
    end_ev.record()
    torch.cuda.synchronize()
    bwd_ms = start_ev.elapsed_time(end_ev) / 100
    print(f"  CUDA backward avg: {bwd_ms:.3f} ms/call")
    print(f"  Combined fwd+bwd:  {fwd_ms+bwd_ms:.3f} ms/call")

    # Compare PyTorch reference timing
    pos_big_cpu   = pos_big.cpu()
    sizes_big_cpu = sizes_big.cpu()
    cell_xy_big_cpu = cell_xy_big.cpu()
    t0 = time.time()
    for _ in range(20):
        _ = density_forward_pytorch(pos_big_cpu, sizes_big_cpu, cell_xy_big_cpu,
                                    half_cw_b, half_ch_b, inv_cell_area_b)
    pytorch_ms = (time.time() - t0) / 20 * 1000
    print(f"  PyTorch CPU forward avg: {pytorch_ms:.3f} ms/call")
    print(f"  Speedup (CUDA vs PyTorch CPU): {pytorch_ms/fwd_ms:.1f}x")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available — skipping tests")
    else:
        run_tests()
