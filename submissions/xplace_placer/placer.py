"""
Xplace-based macro placer.

Pipeline:
  1. Convert Benchmark → bookshelf (.nodes/.nets/.pl/.scl/.wts/.aux)
  2. Run Xplace GP (GPU-accelerated ePlace density + Nesterov)
  3. Read back GP placement (bottom-left → center coords)
  4. Legalize: pairwise separation (from analytical_placer)
  5. Post-legalize L-route congestion + WL gradient refinement
  6. Return final [num_macros, 2] tensor

Xplace location: detected from XPLACE_HOME env var or /opt/xplace.
Falls back to the analytical placer if Xplace is unavailable.

Xplace ref: https://github.com/cuhk-eda/Xplace (CUHK, DAC'22/TCAD'23/ICCAD'24)
"""

from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

import torch

from macro_place.benchmark import Benchmark

# ---------------------------------------------------------------------------
# Import proven loss + legalization functions from the analytical placer.
# This avoids re-implementing and avoids subtle bugs.
# ---------------------------------------------------------------------------
_analytical_dir = os.path.join(os.path.dirname(__file__), '..', 'analytical_placer')
_analytical_path = os.path.join(_analytical_dir, 'placer.py')

for _ext_subdir in ('lroute_ext', 'density_ext'):
    _d = os.path.join(_analytical_dir, _ext_subdir)
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

_spec = importlib.util.spec_from_file_location("_aplacer", _analytical_path)
_amod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_amod)

_preprocess         = _amod._preprocess
_compute_pin_xy     = _amod._compute_pin_xy
_lse_hpwl_loss      = _amod.lse_hpwl_loss
_density_loss       = _amod.density_loss
_make_cell_centers  = _amod._make_cell_centers
_overlap_loss       = _amod.macro_overlap_loss
_lroute_loss        = _amod.lroute_congestion_loss
_legalize           = _amod._legalize

# Per-analytical placer constant
_TARGET_DEN = _amod.TARGET_DEN if hasattr(_amod, "TARGET_DEN") else 1.0


# ---------------------------------------------------------------------------
# Xplace location detection
# ---------------------------------------------------------------------------

def _find_xplace() -> str | None:
    candidates = [
        os.environ.get("XPLACE_HOME", ""),
        "/opt/xplace",
        os.path.expanduser("~/.xplace"),
        os.path.join(os.path.dirname(__file__), "../../research_repos/Xplace"),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "main.py")):
            return os.path.abspath(c)
    return None


# ---------------------------------------------------------------------------
# Post-legalize gradient refinement (WL + congestion, Adam)
# ---------------------------------------------------------------------------

def _post_legalize_refine(
    pos_cpu: torch.Tensor,
    b: Benchmark,
    data: dict,
    device: torch.device,
    steps: int = 60,
    cong_w: float = 0.5,
    wl_w: float = 1.0,
    den_w: float = 0.1,
    ovl_w: float = 20.0,
    lr: float = 0.005,
) -> torch.Tensor:
    sizes = b.macro_sizes.to(device)
    port_pos = b.port_positions.to(device)
    fixed = b.macro_fixed
    movable_idx = (~fixed).nonzero(as_tuple=True)[0]
    cw, ch = b.canvas_width, b.canvas_height

    cell_centers, cell_size = _make_cell_centers(b, device)

    pos_full = pos_cpu.to(device)
    pos_movable = pos_full[movable_idx].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pos_movable], lr=lr)

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    best_loss = float("inf")
    best_movable = pos_movable.detach().clone()

    for _ in range(steps):
        optimizer.zero_grad()
        pos = pos_full.clone()
        pos[movable_idx] = pos_movable
        pos_x = pos[:, 0].clamp(half_w, cw - half_w)
        pos_y = pos[:, 1].clamp(half_h, ch - half_h)
        pos = torch.stack([pos_x, pos_y], dim=1)

        pin_xy = _compute_pin_xy(pos, data, b, port_pos)
        wl   = _lse_hpwl_loss(pin_xy, data, b, alpha=50.0)
        cong = _lroute_loss(pin_xy, data, b, device, pos=pos, sizes=sizes)
        den  = _density_loss(pos, sizes, cell_centers, cell_size, b,
                             target_density=_TARGET_DEN)
        ovl  = _overlap_loss(pos, sizes, b.num_hard_macros)
        loss = wl_w * wl + cong_w * cong + den_w * den + ovl_w * ovl
        loss.backward()

        torch.nn.utils.clip_grad_norm_([pos_movable], 1.0)
        optimizer.step()

        with torch.no_grad():
            pos_movable[:, 0].clamp_(half_w[movable_idx], cw - half_w[movable_idx])
            pos_movable[:, 1].clamp_(half_h[movable_idx], ch - half_h[movable_idx])

        l = loss.item()
        if l < best_loss:
            best_loss = l
            best_movable = pos_movable.detach().clone()

    final = pos_full.clone()
    final[movable_idx] = best_movable
    return final.cpu()


# ---------------------------------------------------------------------------
# Surrogate proxy: 1.0*WL + 0.5*Congestion + 0.5*Density (matches the real
# proxy-cost weights). Used to rank Xplace vs the analytical fallback without
# needing a PlacementCost object inside place().
# ---------------------------------------------------------------------------

def _surrogate_proxy(
    pos_cpu: torch.Tensor,
    b: Benchmark,
    data: dict,
    device: torch.device,
) -> float:
    sizes = b.macro_sizes.to(device)
    port_pos = b.port_positions.to(device)
    cell_centers, cell_size = _make_cell_centers(b, device)
    with torch.no_grad():
        pos = pos_cpu.to(device)
        pin_xy = _compute_pin_xy(pos, data, b, port_pos)
        wl   = _lse_hpwl_loss(pin_xy, data, b, alpha=50.0)
        cong = _lroute_loss(pin_xy, data, b, device, pos=pos, sizes=sizes)
        den  = _density_loss(pos, sizes, cell_centers, cell_size, b,
                             target_density=_TARGET_DEN)
        proxy = 1.0 * wl + 0.5 * cong + 0.5 * den
    return float(proxy.item())


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class XplacePlacer:
    """
    GPU macro placer using Xplace as the global placement engine.
    Falls back to analytical placer if Xplace is not installed.
    """

    def place(self, b: Benchmark) -> torch.Tensor:
        xplace_home = _find_xplace()

        if xplace_home is None:
            print("[xplace_placer] WARNING: Xplace not found. "
                  "Set XPLACE_HOME env var or install at /opt/xplace. "
                  "Falling back to analytical placer.")
            return self._fallback(b)

        print(f"[xplace_placer] Using Xplace at: {xplace_home}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[xplace_placer] Device: {device}")

        with tempfile.TemporaryDirectory(prefix="xplace_") as tmpdir:
            return self._run_xplace(b, xplace_home, tmpdir, device)

    def _run_xplace(
        self, b: Benchmark, xplace_home: str, tmpdir: str, device: torch.device
    ) -> torch.Tensor:
        from submissions.xplace_placer.to_bookshelf import write_bookshelf, read_bookshelf_pl

        design = b.name.replace("/", "_").replace(" ", "_")
        bk_dir = os.path.join(tmpdir, "bookshelf")
        aux_path = write_bookshelf(b, bk_dir, design)
        print(f"[xplace_placer] Bookshelf written: {bk_dir}")

        result_dir = os.path.join(tmpdir, "result")
        exp_id = "exp0"
        output_dir = "output"
        output_prefix = "placement"

        # bookshelf_variety is NOT a CLI arg in Xplace (it's a params dict key set via
        # custom_path). Xplace defaults to "ispd2005" for bookshelf format automatically.
        # --design_name is also set from custom_path by get_custom_design_params, but
        # args.exp_id is built before that override, so exp_id dir uses the CLI default.
        # We therefore don't try to predict the exact output path — use glob by mtime.
        cmd = [
            sys.executable,
            os.path.join(xplace_home, "main.py"),
            "--custom_path",
            f"aux:{aux_path},design_name:{design},benchmark:iccad04",
            "--load_from_raw", "True",
            "--mixed_size", "False",
            "--legalization", "False",
            "--detail_placement", "False",
            "--write_global_placement", "True",
            "--write_placement", "True",
            "--result_dir", result_dir,
            "--exp_id", exp_id,
            "--output_dir", output_dir,
            "--output_prefix", output_prefix,
            "--draw_placement", "False",
            "--verbose_cpp_log", "False",
            "--deterministic", "True",
            # IBM macro benchmarks have no real standard cells, so GP overflow
            # plateaus around ~0.09 and never reaches the 0.07 default. Without
            # a reachable target, no best solution is ever recorded and density
            # weight spirals to NaN. Stop at 0.15 so a clean placement is saved.
            "--stop_overflow", "0.15",
            "--target_density", "0.8",
            "--num_threads", "2",
            "--gpu", "0" if device.type == "cuda" else "-1",
        ]

        print("[xplace_placer] Running Xplace GP...")
        t0 = time.time()
        env = os.environ.copy()
        env["PYTHONPATH"] = xplace_home + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(cmd, cwd=xplace_home, env=env)
        elapsed = time.time() - t0
        print(f"[xplace_placer] Xplace done in {elapsed:.1f}s (rc={result.returncode})")

        # Locate the GP output .pl file.  Xplace modifies args.exp_id with a datetime
        # prefix before find_design_params sets args.design_name, so the actual output
        # directory name is unpredictable.  Find it by modification time.
        gp_candidates = glob.glob(
            os.path.join(result_dir, "**", "*_gp.pl"), recursive=True
        )
        if not gp_candidates:
            # fallback: any .pl in result_dir (may include a dp result)
            gp_candidates = glob.glob(
                os.path.join(result_dir, "**", "*.pl"), recursive=True
            )
        if not gp_candidates:
            print("[xplace_placer] No output .pl found — falling back to analytical")
            return self._fallback(b)
        gp_pl = max(gp_candidates, key=os.path.getmtime)
        print(f"[xplace_placer] Using output: {gp_pl}")

        pos = read_bookshelf_pl(gp_pl, b)

        # Bug 1: Xplace's GP can diverge to NaN/Inf (density weight spiral) yet
        # still write a .pl with non-finite coords. read_bookshelf_pl copies
        # them through silently. Detect and fall back before they poison
        # legalization.
        if not torch.isfinite(pos).all():
            print("[xplace_placer] WARNING: Xplace GP produced non-finite "
                  "coordinates (diverged) — falling back to analytical.")
            return self._fallback(b)
        print("[xplace_placer] GP placement loaded.")

        # Legalize
        print("[xplace_placer] Legalizing...")
        t1 = time.time()
        pos = _legalize(pos, b, time_budget_s=120.0)
        print(f"[xplace_placer] Legalization: {time.time()-t1:.1f}s")

        # Post-legalize congestion refinement
        data = _preprocess(b, device)

        print("[xplace_placer] Post-legalize refine round 1 (cong_w=0.5)...")
        pos = _post_legalize_refine(pos, b, data, device, steps=60, cong_w=0.5)
        pos = _legalize(pos, b, time_budget_s=30.0)

        print("[xplace_placer] Post-legalize refine round 2 (cong_w=0.7)...")
        pos = _post_legalize_refine(pos, b, data, device, steps=60, cong_w=0.7)
        pos = _legalize(pos, b, time_budget_s=30.0)

        if not torch.isfinite(pos).all():
            print("[xplace_placer] WARNING: non-finite positions after "
                  "legalize/refine — falling back to analytical.")
            return self._fallback(b)

        # Bug 2: A plateaued GP can leave nodes in poor positions so that the
        # Xplace pipeline loses to the plain analytical placer (observed
        # ibm01: 0.9459 vs 0.8940). Compute both and keep the better one under
        # a surrogate that matches the real proxy weights. This makes the
        # Xplace placer never regress below the analytical baseline.
        print("[xplace_placer] Computing analytical fallback for comparison...")
        pos_fb = self._fallback(b)

        proxy_x = _surrogate_proxy(pos, b, data, device)
        proxy_f = _surrogate_proxy(pos_fb, b, data, device)
        print(f"[xplace_placer] surrogate proxy — xplace={proxy_x:.4f} "
              f"analytical={proxy_f:.4f}")

        if torch.isfinite(pos_fb).all() and proxy_f < proxy_x:
            print("[xplace_placer] Analytical wins — returning fallback.")
            return pos_fb
        print("[xplace_placer] Xplace wins — returning Xplace placement.")
        return pos

    def _fallback(self, b: Benchmark) -> torch.Tensor:
        fallback_path = os.path.join(
            os.path.dirname(__file__), "..", "analytical_placer", "placer.py"
        )
        spec = importlib.util.spec_from_file_location("_fallback_placer", fallback_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.AnalyticalPlacer().place(b)
