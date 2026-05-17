"""
Converts a macro_place.benchmark.Benchmark object to bookshelf format files
that Xplace can read (bookshelf_variety=ispd2005).

Output files: {outdir}/{design}.{nodes,nets,pl,scl,wts,aux}

Coordinate convention:
  Benchmark: center (x, y) in microns
  Bookshelf .pl: bottom-left (x, y) in site units (site_w = 1 micron here)
"""

import math
import os
import re

import torch

from macro_place.benchmark import Benchmark


def _safe_name(name: str) -> str:
    """Replace whitespace and special chars with underscores."""
    return re.sub(r"[^\w\.\-]", "_", name)


_SCALE = 1000  # microns → nanometers; all bookshelf coords are integers in nm


def write_bookshelf(b: Benchmark, outdir: str, design: str) -> str:
    """
    Write bookshelf files for benchmark b.

    All coordinates written as integers in nanometers (1 nm = 1 site unit).
    Xplace reads widths/heights as atoi() — they MUST be integers.

    Returns path to the .aux file (the Xplace entry point).
    """
    os.makedirs(outdir, exist_ok=True)

    # --- Build node lists -----------------------------------------------
    num_hard = b.num_hard_macros
    num_macros = b.num_macros
    num_ports = b.port_positions.shape[0]

    macro_names = [_safe_name(n) for n in b.macro_names]
    port_names = [f"io_port_{i}" for i in range(num_ports)]

    def _nm(v: float) -> int:
        return int(round(v * _SCALE))

    # --- .nodes ---------------------------------------------------------
    nodes_path = os.path.join(outdir, f"{design}.nodes")
    num_terminals = int(b.macro_fixed.sum().item()) + num_ports

    with open(nodes_path, "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {num_macros + num_ports}\n")
        f.write(f"NumTerminals : {num_terminals}\n\n")
        for i in range(num_macros):
            w = _nm(b.macro_sizes[i, 0].item())
            h = _nm(b.macro_sizes[i, 1].item())
            name = macro_names[i]
            suffix = "\tterminal" if b.macro_fixed[i].item() else ""
            f.write(f"\t{name}\t{w}\t{h}{suffix}\n")
        for i in range(num_ports):
            f.write(f"\t{port_names[i]}\t1\t1\tterminal\n")

    # --- .pl (bottom-left corner = center - size/2, in nm) --------------
    pl_path = os.path.join(outdir, f"{design}.pl")
    with open(pl_path, "w") as f:
        f.write("UCLA pl 1.0\n\n")
        for i in range(num_macros):
            cx = b.macro_positions[i, 0].item()
            cy = b.macro_positions[i, 1].item()
            w = b.macro_sizes[i, 0].item()
            h = b.macro_sizes[i, 1].item()
            bl_x = _nm(cx - w / 2.0)
            bl_y = _nm(cy - h / 2.0)
            name = macro_names[i]
            if b.macro_fixed[i].item():
                f.write(f"\t{name}\t{bl_x}\t{bl_y}\t:\tN\t/FIXED\n")
            else:
                f.write(f"\t{name}\t{bl_x}\t{bl_y}\t:\tN\n")
        for i in range(num_ports):
            px = _nm(b.port_positions[i, 0].item())
            py = _nm(b.port_positions[i, 1].item())
            f.write(f"\t{port_names[i]}\t{px}\t{py}\t:\tN\t/FIXED_NI\n")

    # --- .nets ----------------------------------------------------------
    nets_path = os.path.join(outdir, f"{design}.nets")
    # Build name lookup: bench index -> name
    node_name_map = macro_names + port_names

    use_pin_level = len(b.net_pin_nodes) == b.num_nets

    total_pins = 0
    net_lines = []
    kept_net_indices = []  # original net indices of non-empty nets (for .wts matching)

    for net_i in range(b.num_nets):
        if use_pin_level:
            pins = b.net_pin_nodes[net_i]  # [num_pins, 2]
            pin_rows = []
            for j in range(pins.shape[0]):
                owner = int(pins[j, 0].item())
                slot = int(pins[j, 1].item())
                if owner >= len(node_name_map):
                    continue
                nname = node_name_map[owner]
                if owner < num_hard:
                    offsets = b.macro_pin_offsets[owner]
                    if offsets.shape[0] > slot:
                        ox = offsets[slot, 0].item() * _SCALE
                        oy = offsets[slot, 1].item() * _SCALE
                    else:
                        ox, oy = 0.0, 0.0
                else:
                    ox, oy = 0.0, 0.0
                pin_rows.append((nname, ox, oy))
        else:
            nodes = b.net_nodes[net_i]  # [num_nodes]
            pin_rows = []
            for j in range(nodes.shape[0]):
                owner = int(nodes[j].item())
                if owner >= len(node_name_map):
                    continue
                nname = node_name_map[owner]
                pin_rows.append((nname, 0.0, 0.0))

        if not pin_rows:
            continue

        total_pins += len(pin_rows)
        block = [f"NetDegree : {len(pin_rows)}  net_{net_i}"]
        for nname, ox, oy in pin_rows:
            block.append(f"\t{nname}\tB\t:\t{ox:.6f}\t{oy:.6f}")
        net_lines.append("\n".join(block))
        kept_net_indices.append(net_i)

    with open(nets_path, "w") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {len(net_lines)}\n")
        f.write(f"NumPins : {total_pins}\n\n")
        for block in net_lines:
            f.write(block + "\n")

    # --- .wts -----------------------------------------------------------
    wts_path = os.path.join(outdir, f"{design}.wts")
    with open(wts_path, "w") as f:
        f.write("UCLA wts 1.0\n\n")
        for orig_i in kept_net_indices:
            w_val = b.net_weights[orig_i].item()
            f.write(f"net_{orig_i}\t{w_val:.4f}\n")

    # --- .scl (synthetic rows covering entire canvas, in nm) ------------
    scl_path = os.path.join(outdir, f"{design}.scl")
    W_nm = _nm(b.canvas_width)
    H_nm = _nm(b.canvas_height)

    # Use evaluation grid for row count; rows in nm, sitewidth = 1 nm
    n_rows = max(b.grid_rows, 10)
    row_h_nm = max(1, H_nm // n_rows)
    n_sites = max(1, W_nm // 1)   # 1 site = 1 nm

    with open(scl_path, "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {n_rows}\n\n")
        for r in range(n_rows):
            coord_nm = r * row_h_nm
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    : {coord_nm}\n")
            f.write(f"  Height        : {row_h_nm}\n")
            f.write(f"  Sitewidth     : 1\n")
            f.write(f"  Sitespacing   : 1\n")
            f.write(f"  SiteOrient    : 1\n")
            f.write(f"  SiteSymmetry  : 1\n")
            f.write(f"  SubrowOrigin  : 0  NumSites : {n_sites}\n")
            f.write("End\n")

    # --- .aux -----------------------------------------------------------
    aux_path = os.path.join(outdir, f"{design}.aux")
    with open(aux_path, "w") as f:
        f.write(
            f"RowBasedPlacement : {design}.nodes {design}.nets {design}.wts "
            f"{design}.pl {design}.scl\n"
        )

    return aux_path


def read_bookshelf_pl(pl_path: str, b: Benchmark) -> torch.Tensor:
    """
    Read Xplace output .pl file and return [num_macros, 2] tensor of
    CENTER coordinates in microns.

    Xplace writes bottom-left in nm (integers); we convert back:
      center_um = (bl_nm + size_nm/2) / SCALE
    """
    macro_names = [_safe_name(n) for n in b.macro_names]
    name_to_idx = {n: i for i, n in enumerate(macro_names)}

    pos = b.macro_positions.clone()  # fallback = original positions

    with open(pl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            if name not in name_to_idx:
                continue
            try:
                bl_x_nm = float(parts[1])
                bl_y_nm = float(parts[2])
            except ValueError:
                continue
            idx = name_to_idx[name]
            w_nm = b.macro_sizes[idx, 0].item() * _SCALE
            h_nm = b.macro_sizes[idx, 1].item() * _SCALE
            # convert nm bottom-left → micron center
            pos[idx, 0] = (bl_x_nm + w_nm / 2.0) / _SCALE
            pos[idx, 1] = (bl_y_nm + h_nm / 2.0) / _SCALE

    return pos
