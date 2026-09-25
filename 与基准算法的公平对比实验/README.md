# 实验4 · 基线公平对比实验（2026-09-22 新增）

> 补充实验 4（基线公平对比）的全部代码与数据，对应论文 v4 中新增的 **4.4 Comparison with Baseline Optimizers** 节（红色高亮：一段正文 + Table 3 + Fig. 8）。
 > 结论：85 个基线 run 无一达到 RHO 的 22.717447 s；最强基线均值 GA 7.885 s（比 RHO 低 65.3%，即 RHO 领先 188%）；全部基线单次最好成绩 PSO 12.046 s，仍差 10.67 s；双侧 Wilcoxon 检验：DE/PSO/GA/CMA-ES p<0.001，BO p=0.0625 不显著（BO 仅 5 次运行）。RHO 完整流程另含离线候选库生成（约 3.16×10^5 NFE），论文中已与基线 16,000 NFE 口径分开报告。

## 统一实验口径

| 项目 | 设置 |
|---|---|
| 搜索目标 | `eval_global_med`（medium：240 采样点、dt=0.02 s、二分 35 次），向量化 + 多进程 |
| 终评口径 | `eval_global_fine`（fine：400 采样点、dt=0.005 s、二分 40 次），与 RHO 的 22.717447 s 同一度量 |
| 预算 | DE / PSO / GA / CMA-ES：16,000 NFE/run（种群×代数相应设置，CMA-ES 关闭全部早停跑满预算）；BO：1,000 NFE/run（GP 代理立方复杂度限制） |
| 搜索空间 | 问题 5 完整 40 维盒约束 |
| 种子 | 20240000 + 977k，k = 0,…,19；每算法 20 次独立运行（BO 5 次，受 wall-clock 限制） |
| 基线 | DE（scipy，直接 40 维 = 任务要求的 direct 40-D DE）、PSO、GA、CMA-ES（cma 4.5.0）、BO（sklearn GP Matern-2.5 + EI） |

## 文件清单

| 文件 | 说明 |
|---|---|
| `fast_objective.py` | 加速版目标函数：与 RHO 原 `eval_global_med` 逐位一致（21 样本 parity 校验最大偏差 9.66e-13 s），向量化 + 多进程，随机解加速约 1262 倍 |
| `run_fair_comparison.py` | 跑批主脚本：五算法统一预算、固定种子、断点续跑（已完成 run 自动跳过），结果写入 `fair_comparison_results.json` |
| `bo_finish.py` | BO 补跑脚本：主脚本的 BO 无步级断点，本脚本每步保存 npz 断点，被杀后可续跑 |
| `fix_tainted_de.py` | 一次性修复脚本：把 DE 被 `--quick` 残留的 4 个 800-NFE run 按同种子重跑为 16,000 NFE（已执行完毕，留存备查） |
 | `summarize_results.py` | 汇总脚本：读取结果 JSON，输出 `对比表.md/.csv`（均值±std、中位数、最好/最差、>=RHO 占比、双侧 Wilcoxon p） |
| `make_fig_comment4.py` | 论文 Fig. 8 生成脚本 |
| `fair_comparison_results.json` | **全部 85 个 run 的原始数据**（配置 + 逐 run 的 seed / NFE / fine 时长 / wall-clock） |
| `对比表.md` / `.csv` | 汇总对比表（= 论文 Table 3 的数据来源） |
| `fig_baseline_comparison.png` | 论文 Fig. 8（300 dpi） |

## 复现命令

```bash
# 环境：Python 3.12 + numpy / scipy / scikit-learn / cma==4.5.0 / matplotlib
# 注意：fast_objective.py 通过相对路径引用 问题5_RHO正式流水线 的 RHO 主脚本与候选库，
#       请保持本文件夹与 问题5_RHO正式流水线 同级（本归档已满足）。

python run_fair_comparison.py            # 全量跑批（断点续跑；约 85 run）
python bo_finish.py                      # 若 BO 被中断，用本脚本带断点补跑
python summarize_results.py              # 重新生成对比表
python make_fig_comment4.py              # 重新生成 Fig. 8
```

## 与论文的对应关系

| 论文内容 | 数据来源 |
|---|---|
 | Table 3 各行均值±std / 最好值 / 双侧 Wilcoxon p | `fair_comparison_results.json` → `summarize_results.py` → `对比表.md` |
| "85 runs 无一达到 22.717 s"、"最好单次 12.046 s" | `fair_comparison_results.json` 逐 run 记录 |
 | "RHO 领先最强基线均值 188%" | (22.717447 − 7.885) / 7.885，GA 均值见对比表；论文同时写明“GA 比 RHO 低 65.3%”，避免方向误读 |
| Fig. 8 | `make_fig_comment4.py` → `fig_baseline_comparison.png` |

备注：BO 仅 5 次运行，原因是其单次 wall-clock 均值 762 s（GP 在 40 维、样本数趋近 1000 时立方复杂度），远高于其他算法 16,000 NFE 的耗时；已在论文中说明。
