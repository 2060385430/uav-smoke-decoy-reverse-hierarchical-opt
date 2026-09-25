# 实验1 · 风场/扩散/时变沉降/时变半径扩展模型 + 敏感性分析（2026-09-22 新增）

> 补充实验 1 的全部代码与数据，对应论文 **v6（实验1-5全部合并）** 中的红色高亮内容：§2.1 扩展模型段 + Fig. 9、§2.2 Remark（降维定理对平移球心/时变半径仍成立）、新增 **4.5 Sensitivity Analysis** 节（4 段正文 + Fig. 10 + Fig. 11）、假设 (1) 改写、结论局限句改写、参考文献 [14][15]。
> 采用**路线 B**：扩展模型 + 标称参数点（w = 0、K = 0、λ = 0）逐位退化为原模型，**论文全部旧数字一律不改**。

## 扩展模型形式（与论文 §2.1 红字段一致）

| 扩展项 | 模型 | 文献依据 |
|---|---|---|
| 风场平流 | 云团球心叠加常值水平风速 w（m/s）× 风向 φ（°）的平移 | [14] Hua C 等, Laser & Infrared, 2019, 49(2): 217-221 |
| 扩散致半径增长 | 瞬时体源高斯扩散：R(τ) = √(R0² + 4Kτ)，K 为等效扩散系数 | [14] 同上 |
| 浓度衰减阈值 | 浓度稀释比 (R0/R(τ))³，低于阈值 0.35 视为失效（遮蔽时长截断） | [15] Bao S, Zhang Z H, Electro-Optic Technology Application, 2015, 30(4): 20-23 |
| 时变沉降速度 | v_s(τ) = v0·e^(−λτ)，λ 为衰减系数 | [15] 同上 |

**标称点退化校验（parity）**：w = 0、K = 0、λ = 0 时扩展评估器与原 RHO 评估器逐位一致——medium 精度最大偏差 9.66e-13 s，fine 精度冠军解偏差 3.91e-14 s（`extended_objective.py` 内置 `parity_check`）。因此论文 §3–§4 的全部已有结果在新模型下**保持有效**，无需改动。

## 实验设置

| 项目 | 设置 |
|---|---|
| 评估对象 | 论文冠军方案（分配 (M1,M2,M1,M1,M3) 的 40 维最优解，从消融 JSON 的 `refines["M1,M2,M1,M1,M3"].x` 读取） |
| 组 A：风场敏感性 | 风速 w ∈ {0,1,…,10} m/s × 风向 φ ∈ {0°,45°,…,315°}，共 88 组 |
| 组 B：扩散系数 | K ∈ {0, 0.25, 0.5, 1, 2, 3, 5} m²/s，共 7 组 |
| 组 C：沉降衰减 | λ ∈ {0, 0.02, 0.05, 0.1, 0.2} s⁻¹，共 5 组 |
| 评估口径 | fine 精度（400 采样点、dt = 0.005 s、二分 40 次），与论文 22.717447 s 同一度量 |
| 重优化演示 | 在代表性不利风况（w = 5 m/s, φ = 270°）下，对标称 Top-8 分配逐一重跑 RHO 精修 |

## 核心结果（与论文 §4.5 一致）

- **风场**：最有利 +6.6%（w = 1 m/s, φ = 225°，总时长 24.226 s）；最恶劣保持率 10.7%（w = 10 m/s, φ = 270°，总时长 2.434 s）；88 组全部 success_all3 = True（三枚来袭弹均仍有遮蔽）。
- **扩散系数 K**：非单调——K = 1 时峰值 24.842 s（保持率 109.4%，半径增长使覆盖窗口扩大），K = 5 时降至 20.703 s（91.1%，浓度阈值截断主导）。
- **沉降衰减 λ**：保持率 76.4%–101.0%（λ = 0.02 时 101.0%，λ = 0.1 时 76.4%）。
- **风况重优化**：w = 5 m/s、φ = 270° 下，标称冠军仅余 5.558 s。无约束总时长重优化的最优方案达 13.261 s，但该方案 M3 未被遮蔽；施加“三枚导弹必须全部保留正向遮蔽”约束后，最优重优化方案（分配 M1,M2,M3,M1,M1）达 **16.425 s（+196%）**，且 M1/M2/M3 分别为 8.03/5.48/2.91 s。说明 RHO 可直接在扩展模型下进行覆盖约束的风况重规划。

## 文件清单

| 文件 | 说明 |
|---|---|
| `extended_objective.py` | 扩展物理评估器：`Phys` 类（wx/wy/K/lam/c_ratio = 0.35）、`shielded_ext`、`eval_global_med_ext`、`eval_plan_med_ext`、`eval_plan_fine_ext`、`_champion_x()`（读消融 JSON 冠军解）、`NOMINAL`、`parity_check`（标称点退化校验）。注意：`shielded_ext` 按 axis = 1 逐采样点求投影，与原 `rho.shielded_scalar` 严格一致 |
| `run_sensitivity.py` | 敏感性跑批主脚本：组 A/B/C 共 100 组，结果写入 `sensitivity_results.json` |
| `run_wind_reopt.py` | 风况重优化脚本：5 m/s、270° 风下对标称 Top-8 分配重跑 RHO 精修（候选库沿用标称库），结果写入 `wind_reopt_results.json` |
| `run_wind_reopt_constrained.py` | 三弹全覆盖约束的重优化脚本：从无约束精修解热启动，软惩罚约束下重跑 Top-8，结果写入 `wind_reopt_constrained_results.json` |
| `make_figs_comment1.py` | 论文 Fig. 9 / Fig. 10 / Fig. 11 生成脚本（中文字形用 mathtext 处理下标） |
| `sensitivity_results.json` | 100 组敏感性原始数据（逐组的 total / retention / per_missile / success_all3） |
| `wind_reopt_results.json` | 重优化原始数据（场景、标称解该风况下表现、8 个精修的 40 维解向量与 fine 时长、最优结果） |
| `wind_reopt_constrained_results.json` | 三弹全覆盖约束重优化原始数据（标称解、8 个精修与 best_all3 最优方案） |
| `fig9_model_extension.png` | 论文 Fig. 9：R(t)/浓度稀释比/v_s(t) 三联示意（§2.1） |
| `fig10_wind_sensitivity.png` | 论文 Fig. 10：风速×风向保持率热力图（标注最恶劣点 10.7%） |
| `fig11_K_lambda_reopt.png` | 论文 Fig. 11：K 曲线 / λ 曲线 / 重优化柱状三联 |

## 复现命令

```bash
# 环境：Python 3.12 + numpy / scipy / cma==4.5.0 / matplotlib
# 注意：extended_objective.py 通过相对路径引用 问题5_RHO正式流水线 的 RHO 主脚本与消融 JSON，
#       请保持本文件夹与 问题5_RHO正式流水线 同级（本归档已满足）。

python -c "import extended_objective as e; e.parity_check()"   # 标称点退化校验（med ≤1e-12 / fine ≤1e-12）
python run_sensitivity.py        # 敏感性跑批（100 组，约 3 min）
python run_wind_reopt.py         # 风况重优化（8 个精修，约 3 min）
python run_wind_reopt_constrained.py  # 三弹全覆盖约束重优化（约 3 min）
python make_figs_comment1.py     # 重新生成 Fig. 9 / 10 / 11
```

## 与论文的对应关系

| 论文内容 | 数据来源 |
|---|---|
| §2.1 扩展模型段（R(t)、浓度阈值、v_s(t)、风场平流） | `extended_objective.py` 的 `Phys` 类；Fig. 9 ← `make_figs_comment1.py` |
| "标称参数点退化为原模型，已有结果保持有效" | `extended_objective.py` 的 `parity_check`（med 9.66e-13 / fine 3.91e-14） |
| Fig. 10 热力图；"最有利 +6.6%、最恶劣 10.7%（10 m/s, 270°）" | `sensitivity_results.json` 组 A（88 组） |
| Fig. 11 左/中图；"K = 1 峰值 +9.4%、K = 5 降至 91.1%；λ 保持率 76.4%–101.0%" | `sensitivity_results.json` 组 B / C |
| Fig. 11 右图；"5 m/s、270° 风下标称解 5.56 s → 全覆盖约束重优化 16.43 s（+196%）；无约束重优化 13.26 s 但 M3 未覆盖" | `wind_reopt_constrained_results.json` + `wind_reopt_results.json` |
| 参考文献 [14][15] | 见上表文献依据 |
