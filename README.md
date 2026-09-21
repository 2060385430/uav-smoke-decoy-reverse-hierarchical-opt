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

Naming convention: `主脚本_*` = runnable solver, `结果_*` = data output, `图_*` = figures, `补跑/重跑/验证/阶段一_*` = consistency & regeneration scripts.

## Reproducibility

All stochastic components use fixed seeds (main seed `42`; stage-specific seed rules documented in `REPRODUCIBILITY.md`). Every number in the paper can be reproduced by re-running the corresponding script. See **`REPRODUCIBILITY.md`** for the full seed table, hyperparameter table, evaluation fidelity settings, and wall-clock times.

Verified environment: Windows 11, Python 3.12.14, AMD Ryzen 7 7840HS (CPU only).

## Citation

If you use this code, please cite the paper (citation entry will be added upon publication).

## License

MIT (see `LICENSE`).
