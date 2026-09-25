# 实验 3 蒙特卡洛鲁棒性实验（2026-09-22 新增，对应论文 v3 红字部分）

> 本文件夹是**补充实验 3（鲁棒/机会约束优化）路线一**的全部代码与数据，
> 对应论文 v3 中 Fig. 5 之后新增的红色高亮内容（两段正文 + Fig. 6 + Fig. 7）。

## 与论文的对应关系

| 论文位置 | 来源文件 |
|---|---|
| Fig. 6 遮蔽时长分布直方图 | `图_遮蔽时长分布直方图.png`（由 `analyze_mc.py` 生成） |
| Fig. 7 成功率—阈值曲线 | `图_成功率_阈值曲线.png`（由 `analyze_mc.py` 生成） |
| 正文数据（13.78/8.41/4.23 s、置信保证时长 9.39/8.26/5.93 s、有效率 100%/99.2%/88.7%、分导弹 72.9%/25.3% 等） | `mc_summary.json`（由 `analyze_mc.py` 汇总） |
| 分导弹箱线图（正文未用，备查） | `图_分导弹箱线图.png` |

## 文件清单

| 文件 | 说明 |
|---|---|
| `mc_robustness_q5.py` | 蒙特卡洛主实验脚本（自包含）：四类物理随机变量扰动 + 三档误差 + N=1000/档 + fine 子集交叉验证；逐次独立种子，断点续跑 |
| `mc_results_medium.json` | 主实验原始记录：3 档 × 1000 次，逐次总时长 + 分导弹时长 |
| `mc_results_fine.json` | fine 精度交叉验证原始记录：3 档 × 100 次（独立种子 910000+） |
| `analyze_mc.py` | 统计分析 + 出图脚本（马卡龙配色，与论文图件一致） |
| `mc_summary.json` | 统计汇总：均值/标准差/分位数/置信保证时长/分导弹指标/墙钟时间 |
| `结果报告_蒙特卡洛鲁棒性评估.md` | 完整实验报告（设计、结果、发现、可复现性说明） |

## 复现方法

```bash
# 依赖：仅需 numpy + matplotlib；基准解读取相对路径 ../问题5/ablation/ablation_results.json
# （即把本文件夹放在与 问题5/ 同级的位置运行；或将脚本中 ABL_JSON 指向
#   问题5_RHO正式流水线/问题5_结果_消融A0至A4_最优22.717447s.json，两者结构一致）

# 主实验（每档 1000 次可分多批跑，自动断点续跑累计）
python mc_robustness_q5.py --level L1 --reps 1000 --precision medium
python mc_robustness_q5.py --level L2 --reps 1000 --precision medium
python mc_robustness_q5.py --level L3 --reps 1000 --precision medium

# fine 精度交叉验证（每档 100 次）
python mc_robustness_q5.py --level L1 --reps 100 --precision fine --seed-base 910000
python mc_robustness_q5.py --level L2 --reps 100 --precision fine --seed-base 910000
python mc_robustness_q5.py --level L3 --reps 100 --precision fine --seed-base 910000

# 统计汇总 + 三张图
python analyze_mc.py
```

- 随机种子：medium 主实验 `310000 + 全局序号`；fine 子集 `910000 + 全局序号`（逐次独立）。
- 墙钟时间：medium 每档约 360 s（1000 次），fine 每档约 220 s（100 次），AMD Ryzen 7 7840HS。
- 误差档定义与硬约束处理见 `结果报告_蒙特卡洛鲁棒性评估.md` 第 1 节。
