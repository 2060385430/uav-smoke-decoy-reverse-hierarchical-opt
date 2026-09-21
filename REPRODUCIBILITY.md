# Reproducibility

This document lists everything needed to reproduce the results in the paper: environment, random seeds, hyperparameters, evaluation fidelity, and wall-clock times.

## Hardware and Software Environment

All experiments were conducted on a single workstation: AMD Ryzen 7 7840HS CPU (8 cores), Windows 11, Python 3.12.14. Dependencies: NumPy 2.4.4, SciPy 1.18.1, pandas 3.0.2, Matplotlib 3.10.9, openpyxl 3.1.5 (see `requirements.txt`). No GPU or parallel computing was used.

## Random Seeds

| Component | Seed setting |
|---|---|
| Problems 2–4 (DE main runs) | `seed = 42` |
| Problem 2 multi-seed stability test | 8 seeds, `seed = 7k + 13`, k = 0,…,7 |
| Problem 4 per-drone sub-problems | `seed = 42 + hash(drone) mod 1000`; refinement `seed = 42` |
| Problem 5 Stage 1 (candidate pool) | 4 seeds per unit sub-problem |
| Problem 5 Stages 2–5 (RHO pipeline) | `seed = 999 / 1000 / 1999 / 2000 + iteration index`, stage-dependent |
| Problem 5 global 40-D refinement | `seed = 42` |
| Perturbation robustness tests | `seed = 20240500 + 100·p` for p ∈ {1%, 3%, 5%} |

## Hyperparameters

| Problem | Encoding / Dim. | Optimizer | Budget | Key settings |
|---|---|---|---|---|
| P1 | Deterministic | Grid sampling + bisection | N = 300 points/circle, dt = 0.01 s | Bisection tol = 1e-6 |
| P2 | 4-D | DE + L-BFGS-B polish | maxiter = 18, popsize = 12 | F ∈ (0.5, 1.0), CR = 0.7 |
| P3 | 8-D, time-mapping | DE | maxiter = 40, popsize = 15 | Gap encoding g ∈ [1, 5] s |
| P4 | 3 × 4-D + 12-D | DE → enumeration (3375) → DE | maxiter = 25 / 15, popsize = 12 / 8 | K = 15 candidates per drone |
| P5 Stage 1 | 15 × 8-D units | DE/rand/1 + warm-start rounds | G = 35, NP = 14, 4 seeds | F ∈ [0.5, 1.5], CR = 0.75; pool of 39 candidates |
| P5 Stages 2–5 | Assignment + 40-D | Joint screening (Top-8) + DE | maxiter = 40, popsize = 10 | F ∈ (0.5, 1.0), CR = 0.7 |

## Evaluation Accuracy

Two fidelities: **medium** (120 points/circle, dt = 0.02 s, 35 bisection iterations) for search, **fine** (200 points, dt = 0.005 s, 40 iterations) for final verification. On the Problem 5 champion solution the two differ by 0.000082 s.

## Computation Time

| Experiment | Wall-clock time |
|---|---|
| Problem 1 (incl. convergence checks) | < 1 min |
| Problem 2 (incl. 8-seed test) | ≈ 2 min |
| Problem 3 | ≈ 3 min |
| Problem 4 | ≈ 5 min |
| Problem 5 Stage 1 (candidate pool regeneration) | ≈ 2 h (resumable checkpoints) |
| Problem 5 RHO pipeline | ≈ 60 s per ablation configuration |
| Problem 5 perturbation robustness (per level, 30 runs) | ≈ 66 s |

## Candidate Pool

`问题5_RHO正式流水线/问题5_输入_候选库39组.pkl` is the exact pool used in the paper (13 non-empty assignment keys × 3 candidates = 39). `问题5_阶段一_候选库生成脚本.py` regenerates a pool from scratch (with resumable checkpoints); `问题5_阶段一_重生成验证.json` records the regeneration check, in which the champion assignment reappears in the new pool's Top-8.
