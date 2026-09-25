# -*- coding: utf-8 -*-
"""实验1：三张论文图（马卡龙色系，与论文 Figs 1-8 配色一致）。
fig9  模型扩展示意：R(t) 膨胀 / 浓度比衰减 / v_s(t) 时变沉降
fig10 风况敏感性热力图：保持率(%) over 风速×风向
fig11 扩散系数与沉降衰减敏感性 + 风况重优化对比"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except Exception:
    require_matplotlib_panel_alignment = None

# ---- 马卡龙色板（与 make_figs_en.py / analyze_mc.py / make_fig_comment4.py 一致）----
MC_BLUE = "#6FB3E0"    # 柔蓝
MC_ORANGE = "#F0B27A"  # 柔橙
MC_MINT = "#8FD0B8"    # 薄荷绿
MC_VERM = "#F1948A"    # 柔珊瑚红
MC_PURPLE = "#B4A7D6"  # 薰衣草紫
MC_SERIES = [MC_BLUE, MC_ORANGE, MC_MINT, MC_VERM, MC_PURPLE]
MC_DEEP_RED = "#D9755E"   # 陶土（标注/强调，同 Figs 1-5）
MC_DEEP_MINT = "#4FA383"  # 深薄荷（细线可见性）
# 热力图用 pastel 连续色带：柔珊瑚红 -> 淡黄 -> 薄荷绿
MC_CMAP = LinearSegmentedColormap.from_list(
    "macaron_r2g", ["#F1948A", "#F7DC9B", "#8FD0B8"])

HERE = Path(__file__).parent
S = json.load(open(HERE / "sensitivity_results.json", encoding="utf-8"))
R = json.load(open(HERE / "wind_reopt_results.json", encoding="utf-8"))
RC = json.load(open(HERE / "wind_reopt_constrained_results.json", encoding="utf-8"))
R0, C_TH = 10.0, 0.35


def save_fig(fig, stem, multipanel=False):
    fig.tight_layout()
    if multipanel and require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(HERE / (stem + ".alignment.json")),
            overlay_svg=str(HERE / (stem + ".alignment.svg")),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True)
    fig.savefig(HERE / (stem + ".svg"), bbox_inches="tight")
    fig.savefig(HERE / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(HERE / (stem + ".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(HERE / (stem + ".png"), dpi=300, bbox_inches="tight")

# ---------------- Fig 9: 模型扩展示意 ----------------
t = np.linspace(0, 20, 400)
Ks = [0, 0.5, 1, 2, 5]
lams = [0, 0.02, 0.05, 0.1, 0.2]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
ax = axes[0]
for i, K in enumerate(Ks):
    ax.plot(t, np.sqrt(R0**2 + 4*K*t), color=MC_SERIES[i], lw=1.8, label=f"K = {K}")
ax.set_xlabel("time after detonation (s)"); ax.set_ylabel("effective radius R(t) (m)")
ax.set_title("(a) Diffusion-driven radius growth", fontsize=10)
ax.legend(fontsize=8, title="K (m²/s)", title_fontsize=8,
          loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5, frameon=False)
ax.grid(alpha=0.3)
ax = axes[1]
for i, K in enumerate(Ks[1:]):
    ax.plot(t, (R0/np.sqrt(R0**2 + 4*K*t))**3, color=MC_SERIES[i], lw=1.8, label=f"K = {K}")
ax.axhline(C_TH, color=MC_DEEP_RED, ls="--", lw=1.2, label=f"threshold = {C_TH}")
ax.set_xlabel("time after detonation (s)"); ax.set_ylabel("concentration ratio $(R_0/R)^3$")
ax.set_title("(b) Concentration attenuation", fontsize=10)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24),
          ncol=5, frameon=False); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
ax = axes[2]
for i, lam in enumerate(lams):
    vs = 3.0*np.exp(-lam*t) if lam > 0 else np.full_like(t, 3.0)
    ax.plot(t, vs, color=MC_SERIES[i], lw=1.8, label=f"λ = {lam}")
ax.set_xlabel("time after detonation (s)"); ax.set_ylabel("settling velocity $v_s(t)$ (m/s)")
ax.set_title("(c) Time-varying settling velocity", fontsize=10)
ax.legend(fontsize=8, title="λ (1/s)", title_fontsize=8,
          loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5, frameon=False)
ax.grid(alpha=0.3)
save_fig(fig, "fig9_model_extension", multipanel=True)
plt.close(fig)

# ---------------- Fig 10: 风况热力图 ----------------
speeds = list(range(0, 11))
phis = list(range(0, 360, 45))
Z = np.zeros((len(phis), len(speeds)))
for i, phi in enumerate(phis):
    for j, w in enumerate(speeds):
        Z[i, j] = S[f"A_w{w}_phi{phi}"]["retention"] * 100
fig, ax = plt.subplots(figsize=(7.6, 3.6))
im = ax.imshow(Z, aspect="auto", origin="lower", cmap=MC_CMAP,
               extent=[-0.5, 10.5, -22.5, 382.5], vmin=0, vmax=110)
ax.contour(speeds, phis, Z, levels=[25, 50, 75, 100], colors="#555555",
                linewidths=0.6, alpha=0.55)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("retention of total obscuration duration (%)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
ax.set_xlabel("wind speed (m/s)", fontsize=9)
ax.set_ylabel("wind direction (deg)", fontsize=9)
ax.set_yticks(phis)
ax.tick_params(labelsize=8.5)
ax.grid(False)  # 热力图不叠加网格线
# 标注最差点
wi, pi = np.unravel_index(np.argmin(Z), Z.shape)
ax.plot(speeds[pi], phis[wi], "kx", ms=9, mew=2)
ax.annotate(f"worst: {Z[wi, pi]:.1f}%", (speeds[pi], phis[wi]),
            textcoords="offset points", xytext=(-64, 8), fontsize=8)
save_fig(fig, "fig10_wind_sensitivity", multipanel=False)
plt.close(fig)

# ---------------- Fig 11: K / λ 曲线 + 重优化 ----------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
ax = axes[0]
kb = sorted((r["K"], r["retention"]*100) for r in S.values() if r["group"] == "B_diffusion")
ax.plot([k for k, _ in kb], [v for _, v in kb], "o-", color=MC_BLUE, lw=1.8)
ax.axhline(100, color="gray", ls=":", lw=1)
ax.set_xlabel("diffusion coefficient K (m²/s)", fontsize=9)
ax.set_ylabel("retention (%)", fontsize=9)
ax.set_title("(a) Sensitivity to diffusion coefficient", fontsize=10)
ax.grid(alpha=0.3); ax.tick_params(labelsize=8.5)
ax = axes[1]
lc = sorted((r["lam"], r["retention"]*100) for r in S.values() if r["group"] == "C_settling")
ax.plot([k for k, _ in lc], [v for _, v in lc], "s-", color=MC_DEEP_MINT, lw=1.8)
ax.axhline(100, color="gray", ls=":", lw=1)
ax.set_xlabel("settling decay rate λ (1/s)", fontsize=9)
ax.set_ylabel("retention (%)", fontsize=9)
ax.set_title("(b) Sensitivity to settling decay", fontsize=10)
ax.grid(alpha=0.3); ax.tick_params(labelsize=8.5)
ax = axes[2]
nom_wind = R["nominal_under_wind"]["total"]
reopt_all3 = RC["best_all3"]["fine_total"]
reopt_unconstrained = R["best"]["fine_total"]
labels = ["nominal plan\nunder wind",
          "re-optimized\n(all 3 missiles)",
          "re-optimized\n(unconstrained)"]
bars = ax.bar(labels, [nom_wind, reopt_all3, reopt_unconstrained],
              width=0.55, color=[MC_VERM, MC_MINT, MC_ORANGE],
              edgecolor="#555555", linewidth=0.6)
ax.axhline(22.717, color="gray", ls="--", lw=1.2)
ax.text(1.55, 23.4, "nominal, no wind = 22.72 s",
        fontsize=8, ha="center", va="center")
for b, v in zip(bars, [nom_wind, reopt_all3, reopt_unconstrained]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.35, f"{v:.2f} s",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("total duration (s)", fontsize=9)
ax.set_title("(c) Re-optimization under wind 5 m/s, 270°", fontsize=10)
ax.set_ylim(0, 26)
ax.set_xlim(-0.55, 2.55)
ax.tick_params(labelsize=8.5)
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "fig11_K_lambda_reopt", multipanel=True)
plt.close(fig)
print("saved fig9/fig10/fig11 (macaron palette)")
print(f"worst wind point: w={speeds[pi]} phi={phis[wi]} retention={Z[wi,pi]:.1f}%")
print(f"re-opt all3: {nom_wind:.3f} -> {reopt_all3:.3f} s "
      f"(+{(reopt_all3/nom_wind-1)*100:.0f}%); "
      f"unconstrained: {reopt_unconstrained:.3f} s")
