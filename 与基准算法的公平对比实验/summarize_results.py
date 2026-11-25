# -*- coding: utf-8 -*-
"""汇总公平对比结果：统计表（均值±std、最好/最差、>=RHO 比例、Wilcoxon p）
输出 对比表.md 与 对比表.csv。"""
import json, os
import numpy as np
from scipy import stats as sstats

HERE = os.path.dirname(os.path.abspath(__file__))
RHO_FINE = 22.717447

state = json.load(open(os.path.join(HERE, "fair_comparison_results.json"), encoding="utf-8"))
rows = []
for name in ["DE", "PSO", "GA", "CMA-ES", "BO"]:
    runs = state["results"].get(name) or []
    vals = np.array([d["fine_duration_s"] for d in runs])
    if len(vals) == 0:
        continue
    walls = np.array([d["wall_time_s"] for d in runs])
    nfe = runs[0]["nfe"]
    frac = float(np.mean(vals >= RHO_FINE))
    try:
        pval = float(sstats.wilcoxon(vals - RHO_FINE, alternative="two-sided", method="exact").pvalue)
    except Exception:
        pval = float("nan")
    rows.append({
        "algo": name, "n_runs": len(vals), "nfe": nfe,
        "mean": vals.mean(), "std": vals.std(),
        "best": vals.max(), "worst": vals.min(),
        "median": float(np.median(vals)),
        "frac_ge_rho": frac, "wilcoxon_p": pval,
        "mean_wall_s": walls.mean(),
    })

lines = []
lines.append("# 实验 4 公平对比结果汇总（fine 精度系统总遮蔽时长，秒）")
lines.append("")
lines.append(f"RHO（本文方法，消融 A0）：**{RHO_FINE}** s")
lines.append("")
lines.append("统一口径：搜索目标 = eval_global_med（medium：240 采样点, dt=0.02, 二分35）；"
             "终评 = eval_global_fine（fine：400 采样点, dt=0.005, 二分40）；"
             "预算 = 16000 NFE（BO 为 1000 NFE，GP 代理成本限制）；"
             "种子 = 20240000 + 977k。")
lines.append("")
lines.append("| 算法 | runs | NFE/run | 均值±标准差 | 中位数 | 最好 | 最差 | >=RHO 占比 | 双侧 Wilcoxon p (vs RHO) | 平均耗时/run |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    p = r["wilcoxon_p"]
    ps = f"{p:.2e}" if p == p else "—"
    lines.append(
        f"| {r['algo']} | {r['n_runs']} | {r['nfe']} | "
        f"{r['mean']:.3f} ± {r['std']:.3f} | {r['median']:.3f} | "
        f"{r['best']:.3f} | {r['worst']:.3f} | {r['frac_ge_rho']*100:.0f}% | {ps} | "
        f"{r['mean_wall_s']:.0f} s |")
lines.append("")
lines.append(f"注：RHO 相对最强基线的提升 = (RHO - 最强基线均值) / 最强基线均值。")
if rows:
    best_base = max(rows, key=lambda r: r["mean"])
    imp = (RHO_FINE - best_base["mean"]) / best_base["mean"] * 100
    lines.append(f"最强基线 = {best_base['algo']}（均值 {best_base['mean']:.3f} s），"
                 f"RHO 提升 **{imp:.1f}%**；相对最强基线的单次最好成绩 "
                 f"{max(r['best'] for r in rows):.3f} s，RHO 仍领先 "
                 f"{(RHO_FINE - max(r['best'] for r in rows)):.2f} s。")
lines.append("检验口径为双侧 Wilcoxon 符号秩检验（exact）：DE/PSO/GA/CMA-ES 均为 p<0.001；"
             "BO 的 p=0.0625，在 0.05 水平不显著（BO 仅 5 次独立运行）。")
lines.append("RHO 完整流程的总 NFE 不采用与基线相同的 16,000 口径，另含离线候选库生成"
             "（约 3.16×10^5 NFE）与 Top-8 精修，论文中以 RHO 流水线成本单独报告。")

md = "\n".join(lines)
open(os.path.join(HERE, "对比表.md"), "w", encoding="utf-8").write(md)

import csv
with open(os.path.join(HERE, "对比表.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["algo", "n_runs", "nfe_per_run", "mean", "std", "median",
                "best", "worst", "frac_ge_rho", "wilcoxon_p", "mean_wall_s"])
    for r in rows:
        w.writerow([r["algo"], r["n_runs"], r["nfe"], round(r["mean"], 4),
                    round(r["std"], 4), round(r["median"], 4), round(r["best"], 4),
                    round(r["worst"], 4), r["frac_ge_rho"], r["wilcoxon_p"],
                    round(r["mean_wall_s"], 1)])
print(md)
