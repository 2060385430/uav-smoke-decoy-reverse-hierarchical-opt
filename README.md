# RHO: Cooperative Deployment of UAV Smoke Decoy Bombs via Reverse Hierarchical Optimization

Official code and data for the paper:

> **Cooperative Deployment Strategy of UAV Smoke Decoy Bombs Based on Reverse Hierarchical Optimization**

This repository contains the complete, runnable pipeline for all five problems in the paper, including the candidate pool, fixed random seeds, hyperparameters, and all numerical results (masked durations, ablation data, robustness tests).

## Quick Start

```bash
pip install -r requirements.txt
```

Then enter any problem folder and run its main script (filenames are self-descriptive; see the manifest below and `README_zh.md` for the full annotated manifest in Chinese):

| Folder | Content | Headline result |
|---|---|---|
| `问题1_定参数基准计算/` | Problem 1: fixed-parameter baseline (sampling + bisection) | 1.391642 s |
| `问题2_单机单弹优化/` | Problem 2: single UAV, single bomb, 4-D DE | 4.581712 s |
| `问题3_单机三弹时间映射/` | Problem 3: single UAV, three bombs, 8-D time-mapping DE | 6.697052 s |
| `问题4_三机三弹分治/` | Problem 4: three UAVs, divide-and-conquer + 12-D refinement | 11.593395 s |
| `问题5_RHO正式流水线/` | Problem 5: **full RHO pipeline** (Stage 1–5 + ablation) | 22.717447 s |
| `问题5_早期探索与对比实验_非论文流水线/` | Early explorations & baselines — **not** part of the paper pipeline | — |

## Supplementary experiments (added 2026-09-22, updated 2026-09-24)

| Folder | Content | Headline result |
|---|---|---|
| `扩展模型与风场敏感性分析/` | Comment 1: extended cloud model (wind advection, Gaussian diffusion, time-varying settling) + 100-case sensitivity sweep + wind re-optimization | Worst-case retention 10.7 %; coverage-constrained re-optimization recovers 16.425 s (+196 %) |
| `MINLP与全局最优性验证/` | Comment 2: MINLP view + answer-free global-optimality verification on Problems 2–3 | P2 within 0.13 % of independent reference; P3 solution remains best-known |
| `蒙特卡洛鲁棒性实验/` | Comment 3: Monte-Carlo robustness under three error levels (N = 1000 each) | L1 expected duration 13.78 s; 95 % assured 8.26 s |
| `与基准算法的公平对比实验/` | Comment 4: fair comparison vs DE / PSO / GA / CMA-ES / BO under a common protocol | None of 85 baseline runs reaches 22.717447 s; two-sided exact Wilcoxon p < 0.001 (BO: p = 0.0625, n.s.) |

Naming convention: `主脚本_*` = runnable solver, `结果_*` = data output, `图_*` = figures, `补跑/重跑/验证/阶段一_*` = consistency & regeneration scripts.

## Revision history

- **2026-09-24 (paper v7, final figure layout)** — All paper figures unified to one macaron palette (Figs. 8–11 recolored via `make_figs_comment1.py` / `make_fig_comment4.py`; Figs. 1–7 already used it). Monte-Carlo Figs. 6+7 merged into a single four-panel Fig. 5 (`analyze_mc.py` additionally writes `图_合并_分布与成功率.png`); the joint-Gaussian robustness figure and the extended-cloud-model figure moved to Appendix B (Figs. B.1, B.2); remaining figures renumbered (baseline old Fig. 8 → Fig. 6, wind heatmap old Fig. 10 → Fig. 7, K/λ sensitivity old Fig. 11 → Fig. 8). TIFF outputs are now LZW-compressed. No experiment data was changed.
- **2026-09-23 (paper v7)** — 实验4: Wilcoxon tests switched to **two-sided exact** (`scipy.stats.wilcoxon(..., alternative="two-sided", method="exact")`), summary table regenerated; 实验1: **coverage-constrained wind re-optimization** added (`run_wind_reopt_constrained.py`, best all-three-covered plan 16.425 s; the unconstrained 13.261 s plan leaves M3 uncovered); Figs. 8–11 regenerated, publication formats (SVG/TIFF) added. No raw experiment data was changed.
- **2026-09-22** — Supplementary experiment folders 实验1–实验4 added.

## Reproducibility

All stochastic components use fixed seeds (main seed `42`; stage-specific seed rules documented in `REPRODUCIBILITY.md`). Every number in the paper can be reproduced by re-running the corresponding script. See **`REPRODUCIBILITY.md`** for the full seed table, hyperparameter table, evaluation fidelity settings, and wall-clock times.

Verified environment: Windows 11, Python 3.12.14, AMD Ryzen 7 7840HS (CPU only).

## Citation

If you use this code, please cite the paper (citation entry will be added upon publication).

## License

MIT (see `LICENSE`).
