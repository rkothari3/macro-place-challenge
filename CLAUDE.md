# Macro Placement Challenge 2026

## Environment
- Conda env: macro (already exists)
- Run: conda run -n macro python ...
- No uv — use pip

## Key Files
- Core package: macro_place/
- Proxy cost: macro_place/objective.py
- Best submission: submissions/will_seed/placer.py
- Benchmarks: external/MacroPlacement/Testcases/ICCAD04/
- Greedy baseline: submissions/examples/greedy_row_placer.py

## Project Goal
Beat RePlAce proxy cost baseline of 1.4578 (avg across 17 IBM benchmarks)
Proxy cost = 1.0×WL + 0.5×Density + 0.5×Congestion (lower is better)
Current #1 (will_seed): 1.5338 avg

## Evaluation
Entry point: find in scripts/ or src/ — use conda run -n macro python <script> -b ibm01

## Session Notes
See .claude/findings.md and .claude/progress.md for full context.
