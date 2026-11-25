# -*- coding: utf-8 -*-
"""实验1：风况重优化演示。
场景：|w| = 5 m/s、风向 270°（敏感性扫描中 5 m/s 最恶劣方向，标称冠军方案
在该风况下 fine 总遮蔽仅 5.558 s，保持率 24.5%）。
流程：对筛选 Top-8 分配逐一在扩展物理下重跑 RHO 精修（阶段五，DE 同参数），
取 fine 最优；与"标称冠军方案直接用于该风况"对比。
候选库沿用标称库（阶段一不重生成，文中作为限制说明）。
带 JSON 断点，可分块续跑。"""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import differential_evolution
import extended_objective as exo
rho = exo.rho

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wind_reopt_results.json")

W_SPEED, W_PHI = 5.0, 270.0
PHYS = exo.Phys(wx=W_SPEED * np.cos(np.deg2rad(W_PHI)),
                wy=W_SPEED * np.sin(np.deg2rad(W_PHI)))

def load_top8():
    d = json.load(open(os.path.join(
        HERE, "..", "问题5_RHO正式流水线",
        "问题5_结果_消融A0至A4_最优22.717447s.json"), encoding="utf-8"))
    return d["screen_joint"]

def refine_ext(assignment, cand_by_drone, seed=42, maxiter=15):
    """rho.refine 的扩展物理版：同初始化、同 DE 参数，仅目标函数换扩展物理。"""
    x0 = rho.encode_from_candidate(assignment, cand_by_drone)
    rng = np.random.RandomState(seed)
    init_pop = [x0.copy()]
    while len(init_pop) < 10:
        p = x0.copy()
        for j, (lo, hi) in enumerate(rho.BOUNDS_40):
            p[j] = np.clip(p[j] + rng.normal(0, (hi - lo) * 0.03), lo, hi)
        init_pop.append(p)
    res = differential_evolution(
        lambda x: exo.eval_plan_med_ext(x, assignment, PHYS), rho.BOUNDS_40,
        maxiter=maxiter, popsize=10, seed=seed,
        mutation=(0.5, 1.0), recombination=0.8, polish=False,
        init=np.array(init_pop), workers=1, tol=1e-9)
    return res.x

def main():
    state = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    state.setdefault("scenario", {"w": W_SPEED, "phi": W_PHI})
    # 基准：标称冠军方案在该风况下的表现
    if "nominal_under_wind" not in state:
        xc, asn = exo._champion_x()
        total, per_m = exo.eval_plan_fine_ext(xc, list(asn), PHYS)
        state["nominal_under_wind"] = {"total": total, "per_missile": per_m}
        print(f"[基准] 标称冠军在 w=5 φ=270° 下 fine = {total:.4f} s", flush=True)
        json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    lib = pickle.load(open(os.path.join(
        HERE, "..", "问题5_RHO正式流水线", "问题5_输入_候选库39组.pkl"), "rb"))
    top8 = load_top8()
    done = state.setdefault("refines", {})
    for rank, item in enumerate(top8):
        key = ",".join(item["assignment"])
        if key in done:
            continue
        t0 = time.time()
        cand = {d: lib[(d, a)][idx] for (d, a), idx in
                zip(((dr, a) for dr, a in zip(rho.DRONE_LIST, item["assignment"])),
                    (item["cand_map"][dr] for dr in rho.DRONE_LIST))}
        x = refine_ext(tuple(item["assignment"]), cand)
        total, per_m = exo.eval_plan_fine_ext(x, item["assignment"], PHYS)
        done[key] = {"rank_nominal": rank, "x": x.tolist(),
                     "fine_total": total, "per_missile": per_m,
                     "time": round(time.time() - t0, 1)}
        json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[重优化 {rank+1}/8] {key} fine = {total:.4f} s "
              f"耗时 {time.time()-t0:.0f}s", flush=True)
    if len(done) == len(top8):
        best = max(done.items(), key=lambda kv: kv[1]["fine_total"])
        state["best"] = {"assignment": best[0], "fine_total": best[1]["fine_total"],
                         "per_missile": best[1]["per_missile"]}
        json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[完成] 最优重优化分配 {best[0]} fine = {best[1]['fine_total']:.4f} s "
              f"vs 基准 {state['nominal_under_wind']['total']:.4f} s")

if __name__ == "__main__":
    main()
