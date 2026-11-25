# Reproducibility

This document lists everything needed to reproduce the results in the paper: environment, random seeds, hyperparameters, evaluation fidelity, and wall-clock times.

## Hardware and Software Environment

All experiments were conducted on a single workstation: AMD Ryzen 7 7840HS CPU (8 cores), Windows 11, Python 3.12.14. Dependencies: NumPy 2.4.4, SciPy 1.18.1, pandas 3.0.2, Matplotlib 3.10.9, openpyxl 3.1.5, cma 4.5.0, scikit-learn 1.9.0 (see `requirements.txt`). No GPU or parallel computing was used.

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
| 实验1 wind re-optimization (constrained) | seeds `[42, 431, 977, 2024]` per Top-8 assignment |
| 实验3 Monte-Carlo robustness | `seed = seed_base + global index` (per-level `seed_base`, script default 310000), N = 1000 per level |
| 实验4 baseline comparison | `seed = 20240000 + 977k`, k = 0,…,19 (BO: 5 runs) |

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

## RHO pipeline NFE accounting (实验4 budget note)

The 16,000-NFE budget in the baseline comparison applies to the **baseline optimizers only**. RHO's own pipeline cost is reported separately and breaks down as:

- Stage 1 base channel: 15 keys × 4 seeds × (35 + 1) generations × NP = 14 → **30,240 NFE**
- Stage 1 rescue channel: 20 rescue rounds in total (0–4 per key, see `问题5_阶段一_重生成验证.json`) × 4 seeds × 61 × 14 → **68,320 NFE**
- Stage 1 intensify channel: 15 keys × 4 rounds × (1 warm-started + 2 fresh runs) × 81 × 14 → **204,120 NFE**
- Subtotal ≈ **3.03×10⁵ NFE**; plus medium-fidelity re-evaluation and deduplication of candidates → **≈ 3.16×10⁵ NFE**, the figure quoted in Section 4.4 of the paper.

Stages 2–5 (screening + Top-8 40-D refinements) add a small further amount on top of this.

## Statistical tests (实验4, updated 2026-09-23)

- Wilcoxon signed-rank tests against the RHO result are **two-sided, exact** (`scipy.stats.wilcoxon(diffs, alternative="two-sided", method="exact")`, SciPy 1.18.1). With all 20 runs below RHO, the exact two-sided p for DE / PSO / GA / CMA-ES is 1.91×10⁻⁶; for BO (n = 5) it is 6.25×10⁻² (not significant).
- Table 3 reports mean ± std with the **population std (ddof = 0)** over independent runs.
- No multiple-comparison correction is applied; the four population-based comparisons remain below 0.001 regardless.

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
| 实验1 extended-model sensitivity sweeps (100 fine evaluations) | ≈ 3 min |
| 实验1 wind re-optimization demonstration (8 refinements) | ≈ 3 min |
| 实验1 coverage-constrained wind re-optimization | ≈ 3 min |
| 实验4 fair comparison (85 runs, resumable) | mean 18 s (DE) – 762 s (BO) per run |

## Supplementary experiments — reproduction commands

Run each command inside the corresponding folder (keep the folder siblings of `问题5_RHO正式流水线/`, as shipped).
Figure numbering below follows the final paper layout (2026-09-24): the 实验1 wind heatmap is paper Fig. 7, the K/λ sensitivity figure is paper Fig. 8, the extended-cloud-model figure is Appendix B Fig. B.2, the 实验3 merged Monte-Carlo figure is paper Fig. 5, the 实验4 baseline figure is paper Fig. 6, and the Problem 5 joint-Gaussian robustness figure is Appendix B Fig. B.1.

```bash
# 实验1 — extended model + sensitivity + wind re-optimization
python -c "import extended_objective as e; e.parity_check()"   # nominal-point parity (≤ 1e-12 s)
python run_sensitivity.py                 # 100-case sweep → sensitivity_results.json
python run_wind_reopt.py                  # unconstrained re-optimization → wind_reopt_results.json
python run_wind_reopt_constrained.py      # all-three-covered re-optimization (16.425 s)
python make_figs_comment1.py              # regenerate paper Figs. 7–9 (fig9/10/11_*.png/svg/tiff, macaron palette)

# 实验2 — global-optimality verification
python gv_p2.py                           # Problem 2: 100k anchored samples + 24 polishes (4.587445 s)
python gv_p3.py                           # Problem 3: 60 combinations + refinements (6.463488 s)

# 实验3 — Monte-Carlo robustness
python mc_robustness_q5.py                # 3 levels × N = 1000 → mc_results_*.json
python analyze_mc.py                      # summary → mc_summary.json; figures incl. merged paper Fig. 5
                                          # (图_合并_分布与成功率.png) plus the three standalone plots

# 实验4 — fair baseline comparison
python run_fair_comparison.py             # 85 runs, resumable → fair_comparison_results.json
python bo_finish.py                       # resume BO with per-step checkpoints if interrupted
python summarize_results.py               # → 对比表.md/.csv (two-sided exact Wilcoxon)
python make_fig_comment4.py               # regenerate paper Fig. 6 (fig_baseline_comparison.*, macaron palette)
```

## Candidate Pool

`问题5_RHO正式流水线/问题5_输入_候选库39组.pkl` is the exact pool used in the paper (13 non-empty assignment keys × 3 candidates = 39). `问题5_阶段一_候选库生成脚本.py` regenerates a pool from scratch (with resumable checkpoints); `问题5_阶段一_重生成验证.json` records the regeneration check, in which the champion assignment reappears in the new pool's Top-8.
