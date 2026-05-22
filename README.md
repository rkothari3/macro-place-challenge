# Partcl/HRT Macro Placement Challenge

<img src="assets/HRT.png" alt="Hudson River Trading" height="80"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="assets/partcl.png" alt="Partcl" height="80">

**Win $20,000 by developing better macro placement algorithms!**

Partcl and Hudson River Trading are excited to co-host a competition to solve the macro placement problem. 

## About Macro Placement

Macro placement is the problem of positioning large fixed-size blocks (SRAMs, IPs, analog macros, etc.) on a chip floorplan so that routing congestion, timing, power delivery, and area constraints are balanced. Unlike standard-cell placement, macros have strong geometric and connectivity constraints, so the challenge is to explore a highly discrete design space while minimizing wirelength, avoiding blockages, and preserving downstream routability and timing quality.

For example, the **ibm01** benchmark has:
- **246 hard macros** of varying sizes (ranging from 0.8 to 27 μm², with 33× size variation)
- **7,269 nets** connecting macros to each other and to 894 pre-placed standard cell clusters
- **A 22.9 × 23.0 μm canvas** with 42.8% area utilization

<p align="center">
  <img src="assets/sa_ibm01.gif" alt="Simulated annealing on ibm01" width="600"><br>
  <img src="assets/fd_ibm01.gif" alt="Force-directed placement on ibm01" width="600">
</p>

## About HRT Hardware

Hudson River Trading (HRT) is a leading quantitative trading firm at the forefront of technical innovation in global financial markets.

HRT’s Hardware team builds the high-performance compute systems at the core of our trading infrastructure. We use FPGAs and ASICs to drive low-latency decision-making and power custom solutions across the trading stack, from bespoke circuits to machine learning accelerators.

We’re proud to sponsor this competition because advancing macro placement and low-level hardware optimization directly aligns with the kinds of hard, performance-critical engineering challenges our teams tackle every day.

Joining Hudson River Trading’s hardware team means working alongside leading engineers in one of the most advanced computing environments in global finance. Learn more about open roles at [hudsonrivertrading.com](https://www.hudsonrivertrading.com/).

## About Partcl

Partcl is rebuilding chip design infrastructure from the ground up for the GPU era.

Modern chip design is slow, fragmented, and fundamentally constrained by tools built decades ago. Critical workflows like timing analysis and placement still take hours to days - limiting how much engineers can explore and optimize.

We’re changing that.

Partcl develops GPU-accelerated systems for physical design that run orders of magnitude faster than legacy tools. Our goal is simple: make iteration cheap enough that design space exploration becomes the default, not the exception.

## Background Papers
[1] [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)

[2] [Assessment of Reinforcement Learning for Macro Placement](https://vlsicad.ucsd.edu/Publications/Conferences/396/c396.pdf)

[3] [Reevaluating Google's Reinforcement Learning for IC Macro Placement](https://cacm.acm.org/research/reevaluating-googles-reinforcement-learning-for-ic-macro-placement/)

[4] [A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)

## 🏆 Prizes

- **$20,000 — Grand Prize:** The top 7 submissions by proxy score are evaluated through the OpenROAD flow on NG45 designs (including hidden designs). Among those 7, the submission that beats the SA and RePlAce baselines (reported in [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)) by the largest margin on WNS, TNS, and Area wins the Grand Prize. 
- **$20,000 — First Place (Proxy):** Awarded to the #1 submission by proxy score. Only awarded if no submission qualifies for the Grand Prize.
- **$5,000 — Second Place:** Awarded to the runner-up of the Grand Prize. If no submission qualifies for the Grand Prize, awarded to the #2 submission by proxy score.
- **$4,000 — Innovation Award:** Granted to the most creative or technically innovative approach among the top entries, as determined by the judging panel.
- **Swag:** Every valid submission gets HRT swag!
- **Note:** An additional score adjustment will be applied based on human-expert analysis of the resulting placement.

For full Grand Prize scoring rules, feasibility gate, tie-breaking, and ORFS-failure handling, see [`SCORING.md`](SCORING.md).

## Submission Format

- All submissions will be via google form. Submissions may be made public or private before the end of judging.
- Private submissions will be required to share repository with judges so they may clone/evaluate the method.
- Teams may be up to 5 individuals.
- The deadline for submissions is May 21, 2026, 11:59 pacific.
- All teams may only submit one algorithm.
- **All winning implementations must be made open-source under Apache 2.0 or GPL**
- All submissions must be registered via this [Submission Link](https://forms.gle/YDRtYV5Vq68SZgKW9).
- All submissions must be under 1 hour end-to-end runtime (per benchmark) for the macro placement algorithm.
- All submissions will be evaluated on a AMD EPYC 9655P with 16 cores + 100GB of memory and an NVIDIA RTX 6000 Ada 48GB.
- Submissions may include a `Dockerfile` to define their own runtime environment. If present, the judges will build the image and run the eval against it (with `--network none` enforced at run time, so any `pip install` / `apt-get install` steps must happen at build time). Otherwise, the submission's `placer.py` is mounted into the judges' standard image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, Python 3.11).

## Additional Rules

### Allowed

- **Any algorithmic approach**: SA, RL, GNN, analytical methods, hybrid approaches, learning-based, etc.
- **Any framework**: PyTorch, TensorFlow, JAX, or pure Python/C++
- **Any optimization technique**: Gradient descent, evolutionary algorithms, local search, etc.
- **Training on public benchmarks**: You can learn from the IBM benchmark data
- **Hard-macro orientation flips** (Klein-4 only: `N`, `FN`, `FS`, `S`) — carried to Tier 2 via an optional `orientations.pt` sidecar

### Not Allowed

- Modifying the evaluation functions (must use TILOS MacroPlacement evaluator as-is)
- Hardcoding solutions for specific benchmarks (must be general algorithm)
- Using external/proprietary placement tools (must be open-source submission)
- Exceeding runtime limits (1 hour per benchmark hard timeout)
- Overlaps in resulting placement (strictly zero overlap between hard macros — no tolerance. Participants should add small gaps in their legalization to avoid float-precision edge cases.)
- 90° rotations of hard macros (`R90`, `R270`, `FE`, `FW`) — the fakeram45 SRAMs in our benchmarks aren't designed for rotation (pin access and internal metal direction assume a fixed orientation)
- Resizing soft macros — soft-macro size is a proxy-only concept for density/congestion that doesn't translate to Tier 2; sizes are locked to the initial `.plc` values on every `compute_proxy_cost` call


## 🚀 Quick Start

### Installation 

```bash
# Clone the repository
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Install the package and all dependencies
uv sync

# Verify the setup
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
```

### Run Your First Example

```bash
# Run the greedy row placer on ibm01
uv run evaluate submissions/examples/greedy_row_placer.py

# Run on all 17 IBM benchmarks
uv run evaluate submissions/examples/greedy_row_placer.py --all

# Run on NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla)
uv run evaluate submissions/examples/greedy_row_placer.py --ng45

# Visualize the result
uv run evaluate submissions/examples/greedy_row_placer.py --vis
uv run evaluate submissions/examples/greedy_row_placer.py --all --vis
```

Running on all benchmarks produces a summary like:
```
Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
   ibm01    2.0463    1.3166    0.9976    -55.4%     -105.1%         0
   ibm02    2.0431    1.9072    1.8370     -7.1%      -11.2%         0
   ...
     AVG    2.2109    2.1251    1.4578     -4.0%      -51.7%         0
```

The greedy placer achieves zero overlaps but makes no attempt to optimize wirelength or connectivity — your job is to do better! See [`SETUP.md`](SETUP.md) for the full API reference and [`submissions/examples/`](submissions/examples/) for working examples.

## 🎯 IBM Benchmark Suite (ICCAD04)

We evaluate on the complete ICCAD04 IBM benchmark suite:

| Benchmark | Macros | Nets | Canvas (μm) | Area Util. | SA Baseline | RePlAce Baseline |
|-----------|--------|------|-------------|------------|-------------|------------------|
| **ibm01** | 246 | 7,269 | 22.9×23.0 | 42.8% | 1.3166 | **0.9976** ⭐ |
| **ibm02** | 254 | 7,538 | 23.2×23.5 | 43.1% | 1.9072 | **1.8370** ⭐ |
| **ibm03** | 269 | 8,045 | 24.1×24.3 | 44.2% | 1.7401 | **1.3222** ⭐ |
| **ibm04** | 285 | 8,654 | 24.8×25.1 | 44.8% | 1.5037 | **1.3024** ⭐ |
| **ibm06** | 318 | 9,745 | 26.1×26.5 | 46.1% | 2.5057 | **1.6187** ⭐ |
| **ibm07** | 335 | 10,328 | 26.8×27.2 | 46.8% | 2.0229 | **1.4633** ⭐ |
| **ibm08** | 352 | 10,901 | 27.5×27.9 | 47.4% | 1.9239 | **1.4285** ⭐ |
| **ibm09** | 369 | 11,463 | 28.1×28.5 | 48.0% | 1.3875 | **1.1194** ⭐ |
| **ibm10** | 387 | 12,018 | 28.8×29.2 | 48.6% | 2.1108 | **1.5009** ⭐ |
| **ibm11** | 405 | 12,568 | 29.4×29.8 | 49.2% | 1.7111 | **1.1774** ⭐ |
| **ibm12** | 423 | 13,111 | 30.1×30.5 | 49.8% | 2.8261 | **1.7261** ⭐ |
| **ibm13** | 441 | 13,647 | 30.7×31.1 | 50.4% | 1.9141 | **1.3355** ⭐ |
| **ibm14** | 460 | 14,178 | 31.4×31.8 | 51.0% | 2.2750 | **1.5436** ⭐ |
| **ibm15** | 479 | 14,704 | 32.0×32.4 | 51.6% | 2.3000 | **1.5159** ⭐ |
| **ibm16** | 498 | 15,225 | 32.7×33.1 | 52.2% | 2.2337 | **1.4780** ⭐ |
| **ibm17** | 517 | 15,741 | 33.3×33.7 | 52.8% | 3.6726 | **1.6446** ⭐ |
| **ibm18** | 537 | 16,253 | 34.0×34.4 | 53.4% | 2.7755 | **1.7722** ⭐ |

Each benchmark includes:
- Hard macros (you place these)
- Soft macros (you can also place these)
- Nets connecting all components
- Initial placement (hand-crafted, serves as reference)

**Baseline Analysis:**
- RePlAce (⭐) consistently outperforms SA across all benchmarks
- RePlAce achieves 15-55% lower proxy cost than SA
- Both baselines achieve zero overlaps (enforced as hard constraint)

## 💡 Why This Is Hard

Despite "only" 246-537 macros, this problem is extremely challenging:

1. **Massive search space**: ~10^800 possible placements (even with constraints)
2. **Conflicting objectives**: Wirelength wants clustering, density wants spreading, congestion wants routing space
3. **Non-convex landscape**: Millions of local minima, discontinuities, plateaus
4. **Long-range dependencies**: Moving one macro affects costs globally through thousands of nets
5. **Hard constraints**: No overlaps between heterogeneous sizes (33× size variation)
6. **Tight packing**: 43-53% area utilization leaves little slack
7. **Runtime matters**: Must be fast enough to be practical (< 5 minutes ideal)

Classical methods (SA, RePlAce) have been refined for decades but still have room for improvement!

## 📖 Documentation

- **Setup & API Reference**: [`SETUP.md`](SETUP.md) - Infrastructure details, benchmark format, cost computation, testing

## 📚 References

- **TILOS MacroPlacement**: [GitHub Repository](https://github.com/TILOS-AI-Institute/MacroPlacement)
  - Source of evaluation infrastructure
  - ICCAD04 benchmarks
  - SA and RePlAce baseline implementations

- **ICCAD04 Benchmarks**: Classic macro placement benchmark suite used in academic research

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE.md](LICENSE.md) for details.
