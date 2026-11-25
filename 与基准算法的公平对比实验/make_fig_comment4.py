# -*- coding: utf-8 -*-
"""实验4：基线公平对比图（横向条形 + 误差棒 + RHO 参考线），供论文插入。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

HERE = Path(__file__).parent
data = json.load(open(HERE / "fair_comparison_results.json", encoding="utf-8"))
RHO = data["config"]["rho_reference_fine"]

order = ["BO", "CMA-ES", "DE", "PSO", "GA"]  # 按均值升序，GA 最强在最上
labels = {"DE": "DE (direct 40-D)", "PSO": "PSO", "GA": "GA",
          "CMA-ES": "CMA-ES", "BO": "BO"}
stats = {}
for a, runs in data["results"].items():
    v = np.array([r["fine_duration_s"] for r in runs])
    stats[a] = (v.mean(), v.std(), v.max())

fig, ax = plt.subplots(figsize=(6.4, 3.4))
y = np.arange(len(order))
means = [stats[a][0] for a in order]
stds = [stats[a][1] for a in order]
bests = [stats[a][2] for a in order]

# 马卡龙色板（与论文 Figs 1-7 一致）：浅蓝填充 #C4E0F2 / 柔蓝 #6FB3E0 / 陶土 #D9755E / 柔珊瑚 #F1948A
bars = ax.barh(y, means, xerr=stds, height=0.58, color="#C4E0F2",
               edgecolor="#6FB3E0", linewidth=0.8,
               error_kw=dict(elinewidth=1.0, ecolor="#6FB3E0", capsize=3),
               label="Baseline mean ± std (fine fidelity)", zorder=2)
# 最好单次标记
ax.scatter(bests, y, marker="D", s=28, color="#D9755E", zorder=3,
           label="Best single run")
ax.axvline(RHO, color="#F1948A", linewidth=1.8, zorder=4,
           label=f"RHO (ours) = {RHO:.2f} s")
ax.text(RHO - 0.4, len(order) - 0.32, f"RHO = {RHO:.2f} s",
        color="#D9755E", fontsize=9, ha="right", va="bottom", fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels([labels[a] for a in order], fontsize=9)
ax.set_xlabel("Total masking duration (s, fine fidelity, larger is better)", fontsize=9)
ax.tick_params(axis="x", labelsize=8.5)
ax.set_xlim(0, 25.5)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24),
          ncol=3, frameon=False)
ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "fig_baseline_comparison.svg", bbox_inches="tight")
fig.savefig(HERE / "fig_baseline_comparison.pdf", bbox_inches="tight")
fig.savefig(HERE / "fig_baseline_comparison.tiff", dpi=600, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
out = HERE / "fig_baseline_comparison.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved:", out)
