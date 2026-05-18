"""
XplacePlacer — delegates to the analytical placer.

History: this submission previously ran Xplace (GP + GPU legalization +
detailed placement) as the global placement engine. Four Colab runs across
ibm01/ibm02 proved stock Xplace is structurally wrong for this challenge:

  - The proxy (1.0*WL + 0.5*Den + 0.5*Cong) is overlap-tolerant: density is
    a soft cost, not a legality constraint. Xplace's value-add is producing
    *legal* placements, which inflates the dominant WL term ~3.5x. It lost
    to the analytical placer on every benchmark (ibm01 surrogate 1.118 vs
    0.958; ibm02 1.423 vs 1.232).
  - Xplace's macro legalizer is infeasible with the ~250 fixed "macros" the
    classifier produces and aborts with a C++ assertion (SIGABRT) on
    benchmarks with >1 movable macro (ibm02 wasted 132s then crashed).

Since the analytical placer directly optimizes the real proxy and wins
every time, this placer now simply delegates to it. The Xplace integration
(to_bookshelf.py) is kept in the directory for reference but unused.
"""

from __future__ import annotations

import importlib.util
import os

import torch

from macro_place.benchmark import Benchmark

_ANALYTICAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "analytical_placer", "placer.py"
)


def _load_analytical():
    spec = importlib.util.spec_from_file_location(
        "_analytical_placer", _ANALYTICAL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AnalyticalPlacer()


class XplacePlacer:
    """Thin wrapper that runs the analytical placer."""

    def place(self, b: Benchmark) -> torch.Tensor:
        return _load_analytical().place(b)
