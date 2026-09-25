# 实验 2 全局最优性验证（2026-09-22 新增，对应论文 §3.5）

> 本文件夹是**补充实验 2（MINLP 联合建模 + 全局最优性 gap）实验部分**的全部代码与数据，
> 对应论文 §3.5 "A Mixed-Integer Nonlinear Programming View"（建模小节，纯文字）。
> 与 §4.3 末段（全局最优性独立验证结果）。

## 方法概述（与论文 §4.3 末段对应）

独立全局搜索利用论文降维定理的直接推论：导弹—目标视线束的最大横截面半径 = 目标半径 7 m < 云团半径 10 m，
因此**云团中心落在视线束轴线上即完全遮蔽**。采样 (遮蔽时刻 t_obs, 轴上相对位置 λ, 无人机速度 v) 并反演
(heading, speed, drop, det)，即可遍历全部"可达 ∩ 完全遮蔽"方案，实现无答案导向的全局候选生成。

- **问题 2（4 维单弹）**：100,000 个视线束锚定样本（粗扫 dt=0.1 s 仅用于筛选）→ Top-24 候选全界 DE 抛光 → fine 复核。
  全局参考 **4.587445 s** vs 论文解 4.581712 s → **论文解在最优参考的 0.125% 以内**。
- **问题 3（8 维三弹并集）**：从 P2 单弹池按 (航向, 速度) 分桶组成 60 个三弹组合 → 12 次全界 DE 抛光 + 2 次长程精修。
  全局参考 **6.463488 s** vs 论文解 6.697052 s → **独立搜索未超越论文解（低 3.6%），论文解保持 best-known**。

## 与论文的对应关系

| 论文表述 | 来源文件 |
|---|---|
| "4.5874 s, only 0.13% above the RHO solution" | `gv_p2_result.json` → `global_reference_fine_s` / `rel_gap_pct_of_reference` |
| "100,000 anchored samples, 24 candidates refined" | `gv_p2_scan.json`（n_done、top）、`gv_p2_result.json`（polished_all，24 条） |
| "at best 6.4635 s, 3.6% below the RHO solution" | `gv_p3_result.json` → `global_reference_fine_s` / `rel_gap_pct_of_reference` |
| "60 three-bomb combinations, 12 DE runs + two long-budget refinements" | `gv_p3_scan.json`（n_done=60）、`gv_p3_result.json`（polished_all 14 条 = 12 抛光 + 2 精修） |

## 文件清单

| 文件 | 说明 |
|---|---|
| `gv_p2.py` | P2 全局验证脚本：视线束锚定反演采样（`scan`，断点续跑）+ Top-K 全界 DE 抛光与 fine 复核（`polish`）；物理模型逐行复制自 `问题2/problem2_solver.py` |
| `gv_p3.py` | P3 全局验证脚本：从 P2 单弹池组三弹（`triples`）+ 时间映射编码 8 维全界 DE 抛光（`polish`）；复用 `gv_p2.py` 的物理模型 |
| `gv_p2_scan.json` | P2 粗扫池：10 万样本、232 个可行单弹、Top-200 候选 |
| `gv_p2_result.json` | P2 抛光结果：24 条 fine 复核记录 + gap 汇总 |
| `gv_p3_scan.json` | P3 三弹候选池：60 个组合及粗扫并集时长 |
| `gv_p3_result.json` | P3 抛光/精修结果：14 条 fine 复核记录 + gap 汇总 |

## 复现方法

```bash
# 依赖：numpy + scipy；两个脚本需放在同一目录（gv_p3 引用 gv_p2）
# 阶段一：P2 锚定粗扫（可分批，自动断点续跑；单批 5 万点约 2.7 min，Ryzen 7 7840HS）
python gv_p2.py scan 50000
python gv_p2.py scan 50000
# 阶段二：P2 抛光（每批 8 个候选约 4 min，累计至 24）
python gv_p2.py polish 8
python gv_p2.py polish 16
python gv_p2.py polish 24
# 阶段三：P3 从 P2 池组三弹
python gv_p3.py triples
# 阶段四：P3 抛光（每批 4 个候选约 4.5 min，累计至 12）
python gv_p3.py polish 4
python gv_p3.py polish 8
python gv_p3.py polish 12
# 长程精修（maxiter=50 两轮）见 gv_p3_result.json 中 seed_rank=888/999 记录的脚本备注
```

- 随机种子：P2 粗扫 `770000+序号`、抛光 `880000+k`；P3 抛光 `550000+k`、长程精修 `888000/999000`。
- 评估精度：粗扫 40 点/圆周 dt=0.1 s（仅筛选）；抛光 medium 120 点 dt=0.02 s 二分 35 次；复核 fine 200 点 dt=0.005 s 二分 40 次（与论文口径一致）。
- 注意：论文问题 3 原求解使用了围绕已知答案的窄化边界（航向 170°–190° 等），本验证全部使用**未窄化的全物理边界**。
