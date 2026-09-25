# -*- coding: utf-8 -*-
"""实验3 蒙特卡洛结果统计分析与出图"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def load(prec):
    with open(os.path.join(HERE, f"mc_results_{prec}.json"), encoding="utf-8") as f:
        return json.load(f)

med, fine = load("medium"), load("fine")
T0 = med["T0_baseline_this_precision_s"]

LEVEL_NAMES = {"L1": "nominal", "L2": "moderate", "L3": "severe"}
summary = {}

print(f"基准总遮蔽 T0 = {T0:.4f} s (medium) / {fine['T0_baseline_this_precision_s']:.4f} s (fine 复算)")
print()
for lv in ["L1", "L2", "L3"]:
    s = med["levels"][lv]
    arr = np.array([r[0] for r in s["records"]])
    per_m = np.array([r[1:] for r in s["records"]])
    n = len(arr)
    qs = np.percentile(arr, [1, 5, 10, 25, 50, 75, 95, 99])
    entry = {
        "n": n,
        "valid_rate_P_T_gt_0.01s": round(float(np.mean(arr > 0.01)), 4),
        "mean_s": round(float(arr.mean()), 4),
        "std_s": round(float(arr.std()), 4),
        "cv": round(float(arr.std() / arr.mean()), 4),
        "min_s": round(float(arr.min()), 4),
        "max_s": round(float(arr.max()), 4),
        "keep_ratio_vs_T0": round(float(arr.mean() / T0), 4),
        "quantiles_s": {f"P{q}": round(float(v), 4) for q, v in
                        zip([1, 5, 10, 25, 50, 75, 95, 99], qs)},
        # 置信水平口径：以 90%/95%/99% 置信度能保证的最低遮蔽时长 = P10/P5/P1 分位数
        "guaranteed_duration_at_confidence": {
            "90%": round(float(qs[2]), 4),
            "95%": round(float(qs[1]), 4),
            "99%": round(float(qs[0]), 4),
        },
        "per_missile_mean_s": {m: round(float(per_m[:, i].mean()), 4)
                               for i, m in enumerate(["M1", "M2", "M3"])},
        "per_missile_valid_rate": {m: round(float(np.mean(per_m[:, i] > 0.01)), 4)
                                   for i, m in enumerate(["M1", "M2", "M3"])},
        "gap_adjust_fraction": round(s["n_gap_adjust"] / (n * 10), 4),  # 每rep最多10个相邻对(5机x2)
        "wall_time_s": s["wall_time_s"],
    }
    # fine 子集对照
    farr = np.array([r[0] for r in fine["levels"][lv]["records"]])
    entry["fine_subset_check"] = {
        "n": len(farr),
        "mean_s": round(float(farr.mean()), 4),
        "std_s": round(float(farr.std()), 4),
        "valid_rate": round(float(np.mean(farr > 0.01)), 4),
        "mean_diff_vs_medium_s": round(float(farr.mean() - arr.mean()), 4),
    }
    summary[lv] = entry
    print(f"[{lv} {LEVEL_NAMES[lv]}] n={n} 有效率={entry['valid_rate_P_T_gt_0.01s']*100:.1f}% "
          f"mean={entry['mean_s']}s std={entry['std_s']}s "
          f"保证时长@90/95/99%={entry['guaranteed_duration_at_confidence']} "
          f"fine对照mean={entry['fine_subset_check']['mean_s']}s")

with open(os.path.join(HERE, "mc_summary.json"), "w", encoding="utf-8") as f:
    json.dump({"T0_medium_s": T0, "T0_fine_s": fine["T0_baseline_this_precision_s"],
               "levels": summary}, f, ensure_ascii=False, indent=1)
print("\nsaved mc_summary.json")

# ================= 出图 =================
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__import__('sys').executable), '..', '..')))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
# 与论文现有图件一致的马卡龙配色与白底风格（取自 make_figs_en.py）
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": True,
                     "grid.color": "#DDDDDD", "grid.linewidth": 0.7,
                     "axes.edgecolor": "#9A9A9A", "axes.linewidth": 0.9})

colors = {"L1": "#6FB3E0", "L2": "#F0B27A", "L3": "#F1948A"}   # 马卡龙蓝/橙/朱红
C_GRAY = "#555555"
labels = {"L1": "L1 nominal (σ_h=0.5°, σ_v=0.5, σ_t=σ_f=0.05s)",
          "L2": "L2 moderate (σ_h=1.0°, σ_v=1.0, σ_t=σ_f=0.10s)",
          "L3": "L3 severe (σ_h=2.0°, σ_v=2.0, σ_t=σ_f=0.20s)"}

# 图1：遮蔽时长分布直方图（3档并排）
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
bins = np.linspace(0, 24, 49)
for ax, lv in zip(axes, ["L1", "L2", "L3"]):
    arr = np.array([r[0] for r in med["levels"][lv]["records"]])
    ax.hist(arr, bins=bins, color=colors[lv], alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.axvline(T0, color=C_GRAY, ls="--", lw=1.4, label=f"deterministic T0 = {T0:.2f} s")
    ax.axvline(arr.mean(), color=C_GRAY, ls=":", lw=1.6,
               label=f"MC mean = {arr.mean():.2f} s")
    ax.set_title(labels[lv].split(" (")[0], fontsize=11)
    ax.set_xlabel("total masking duration (s)")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("count (N = 1000)")
fig.suptitle("Distribution of total masking duration under Monte Carlo perturbation (Problem 5 champion plan)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(HERE, "图_遮蔽时长分布直方图.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# 图2：成功率—任务阈值曲线 P(T >= tau)
fig, ax = plt.subplots(figsize=(7.6, 4.8))
taus = np.linspace(0, 24, 241)
for lv in ["L1", "L2", "L3"]:
    arr = np.array([r[0] for r in med["levels"][lv]["records"]])
    ax.plot(taus, [np.mean(arr >= t) * 100 for t in taus], color=colors[lv],
            lw=1.8, label=labels[lv])
for conf, sty in [(90, "--"), (95, ":"), (99, "-.")]:
    ax.axhline(conf, color="gray", lw=0.8, ls=sty, alpha=0.6)
ax.text(23.8, 90, "90%", ha="right", va="bottom", fontsize=8, color="gray")
ax.text(23.8, 95, "95%", ha="right", va="bottom", fontsize=8, color="gray")
ax.text(23.8, 99, "99%", ha="right", va="bottom", fontsize=8, color="gray")
ax.set_xlabel("required masking duration threshold τ (s)")
ax.set_ylabel("success rate  P(T ≥ τ)  (%)")
ax.set_title("Success rate vs. mission threshold under three perturbation levels (N = 1000)")
ax.set_xlim(0, 24); ax.set_ylim(0, 102)
ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "图_成功率_阈值曲线.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# 图3：分导弹箱线图
fig, ax = plt.subplots(figsize=(7.6, 4.6))
data, ticklabels, c = [], [], []
for lv in ["L1", "L2", "L3"]:
    per_m = np.array([r[1:] for r in med["levels"][lv]["records"]])
    for i, m in enumerate(["M1", "M2", "M3"]):
        data.append(per_m[:, i]); ticklabels.append(f"{lv}\n{m}"); c.append(colors[lv])
bp = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color=C_GRAY))
for patch, col in zip(bp["boxes"], c):
    patch.set_facecolor(col); patch.set_alpha(0.85)
ax.set_xticklabels(ticklabels, fontsize=8)
ax.set_ylabel("masking duration (s)")
ax.set_title("Per-missile masking duration under perturbation (N = 1000 per level)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "图_分导弹箱线图.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print("saved 3 figures")

# 图4（论文精简版）：分布直方图 + 成功率曲线 合并双联图（马卡龙配色不变，数据不变）
fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.25)
bins = np.linspace(0, 24, 49)
for j, lv in enumerate(["L1", "L2", "L3"]):
    ax = fig.add_subplot(gs[0, j])
    arr = np.array([r[0] for r in med["levels"][lv]["records"]])
    ax.hist(arr, bins=bins, color=colors[lv], alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.axvline(T0, color=C_GRAY, ls="--", lw=1.3, label=f"deterministic T0 = {T0:.2f} s")
    ax.axvline(arr.mean(), color=C_GRAY, ls=":", lw=1.4,
               label=f"MC mean = {arr.mean():.2f} s")
    ax.set_title(f"({chr(97 + j)}) {labels[lv].split(' (')[0]}", fontsize=10.5)
    ax.set_xlabel("total masking duration (s)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    if j == 0:
        ax.set_ylabel("count (N = 1000)", fontsize=9)
ax = fig.add_subplot(gs[1, :])
taus = np.linspace(0, 24, 241)
for lv in ["L1", "L2", "L3"]:
    arr = np.array([r[0] for r in med["levels"][lv]["records"]])
    ax.plot(taus, [np.mean(arr >= t) * 100 for t in taus], color=colors[lv],
            lw=1.8, label=labels[lv])
for conf, sty in [(90, "--"), (95, ":"), (99, "-.")]:
    ax.axhline(conf, color="gray", lw=0.8, ls=sty, alpha=0.6)
ax.text(23.8, 90, "90%", ha="right", va="bottom", fontsize=8, color="gray")
ax.text(23.8, 95, "95%", ha="right", va="bottom", fontsize=8, color="gray")
ax.text(23.8, 99, "99%", ha="right", va="bottom", fontsize=8, color="gray")
ax.set_xlabel("required masking duration threshold τ (s)", fontsize=9)
ax.set_ylabel("success rate  P(T ≥ τ)  (%)", fontsize=9)
ax.set_title("(d) Success rate vs. mission threshold", fontsize=10.5)
ax.set_xlim(0, 24); ax.set_ylim(0, 102)
ax.tick_params(labelsize=8.5)
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
fig.savefig(os.path.join(HERE, "图_合并_分布与成功率.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved combined figure (merged Fig.6+Fig.7)")
