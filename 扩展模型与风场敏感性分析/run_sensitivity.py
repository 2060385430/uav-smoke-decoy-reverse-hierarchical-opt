# -*- coding: utf-8 -*-
"""实验1：敏感性分析跑批（冠军方案，fine 精度终评）。
A: 风速 0-10 m/s × 8 风向（K=0, λ=0）  88 组
B: 扩散系数 K ∈ {0,0.25,0.5,1,2,3,5}（无风, λ=0）  7 组
C: 沉降衰减 λ ∈ {0,0.02,0.05,0.1,0.2}（无风, K=0）  5 组
带 JSON 断点：已完成的组自动跳过，可分块多次运行。"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import extended_objective as exo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sensitivity_results.json")

CHAMP_X, CHAMP_ASN = exo._champion_x()
NOMINAL_TOTAL = 22.71744735505382

def build_jobs():
    jobs = []
    for w in range(0, 11):
        for phi in range(0, 360, 45):
            rad = np.deg2rad(phi)
            jobs.append({"group": "A_wind", "w": float(w), "phi": float(phi),
                         "phys": exo.Phys(wx=w * np.cos(rad), wy=w * np.sin(rad))})
    for K in (0, 0.25, 0.5, 1, 2, 3, 5):
        jobs.append({"group": "B_diffusion", "K": K, "phys": exo.Phys(K=K)})
    for lam in (0, 0.02, 0.05, 0.1, 0.2):
        jobs.append({"group": "C_settling", "lam": lam, "phys": exo.Phys(lam=lam)})
    return jobs

def job_key(j):
    if j["group"] == "A_wind":
        return f"A_w{j['w']:.0f}_phi{j['phi']:.0f}"
    if j["group"] == "B_diffusion":
        return f"B_K{j['K']}"
    return f"C_lam{j['lam']}"

def main():
    state = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    jobs = build_jobs()
    todo = [j for j in jobs if job_key(j) not in state]
    print(f"总 {len(jobs)} 组，已完成 {len(jobs)-len(todo)}，本次待跑 {len(todo)}", flush=True)
    for j in todo:
        t0 = time.time()
        total, per_m = exo.eval_plan_fine_ext(CHAMP_X, list(CHAMP_ASN), j["phys"])
        rec = {"group": j["group"], "total": total,
               "retention": total / NOMINAL_TOTAL,
               "per_missile": per_m,
               "success_all3": all(v > 0 for v in per_m.values())}
        for k in ("w", "phi", "K", "lam"):
            if k in j:
                rec[k] = j[k]
        state[job_key(j)] = rec
        json.dump(state, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[{job_key(j)}] total={total:.4f}s retention={rec['retention']:.3f} "
              f"wall={time.time()-t0:.1f}s", flush=True)
    print("敏感性跑批完成" if len(state) == len(jobs) else "部分完成，可再次运行续跑")

if __name__ == "__main__":
    main()
