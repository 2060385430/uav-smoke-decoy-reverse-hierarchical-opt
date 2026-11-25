# -*- coding: utf-8 -*-
"""Constrained wind replanning: every Top-8 assignment must keep all three
missiles positively obscured. Complements run_wind_reopt.py, whose unconstrained
best total sacrifices M3.
"""
import os
import sys
import json
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import differential_evolution

import extended_objective as exo

rho = exo.rho
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wind_reopt_constrained_results.json")

W_SPEED, W_PHI = 5.0, 270.0
SEEDS = [42, 431, 977, 2024]
MAXITER = 20
PHYS = exo.Phys(wx=W_SPEED * np.cos(np.deg2rad(W_PHI)),
                wy=W_SPEED * np.sin(np.deg2rad(W_PHI)))


def load_top8():
    d = json.load(open(os.path.join(
        HERE, "..", "问题5_RHO正式流水线",
        "问题5_结果_消融A0至A4_最优22.717447s.json"), encoding="utf-8"))
    return d["screen_joint"]


def refine_ext(assignment, x0, seed=42, maxiter=20):
    rng = np.random.RandomState(seed)
    init_pop = [x0.copy()]
    while len(init_pop) < 10:
        p = x0.copy()
        for j, (lo, hi) in enumerate(rho.BOUNDS_40):
            p[j] = np.clip(p[j] + rng.normal(0, (hi - lo) * 0.03), lo, hi)
        init_pop.append(p)
    res = differential_evolution(
        lambda x: exo.eval_plan_med_ext_cover_soft(x, assignment, PHYS),
        rho.BOUNDS_40,
        maxiter=maxiter, popsize=10, seed=seed,
        mutation=(0.5, 1.0), recombination=0.8, polish=False,
        init=np.array(init_pop), workers=1, tol=1e-9)
    return res.x


def main():
    state = {"scenario": {"w": W_SPEED, "phi": W_PHI}}

    xc, asn = exo._champion_x()
    total, per_m = exo.eval_plan_fine_ext(xc, list(asn), PHYS)
    state["nominal_under_wind"] = {"total": total, "per_missile": per_m}

    unconstrained = json.load(open(
        os.path.join(HERE, "wind_reopt_results.json"), encoding="utf-8"))

    lib = pickle.load(open(os.path.join(
        HERE, "..", "问题5_RHO正式流水线", "问题5_输入_候选库39组.pkl"), "rb"))
    top8 = load_top8()
    done = state["refines"] = {}

    for rank, item in enumerate(top8):
        key = ",".join(item["assignment"])
        t0 = time.time()
        cand = {d: lib[(d, a)][idx] for (d, a), idx in
                zip(((dr, a) for dr, a in zip(rho.DRONE_LIST, item["assignment"])),
                    (item["cand_map"][dr] for dr in rho.DRONE_LIST))}
        warm = np.array(unconstrained["refines"][key]["x"])
        best = None
        for seed in SEEDS:
            x = refine_ext(tuple(item["assignment"]), warm, seed=seed, maxiter=MAXITER)
            total, per_m = exo.eval_plan_fine_ext(x, item["assignment"], PHYS)
            covered = bool(all(v > 0 for v in per_m.values()))
            if best is None or (covered and total > best["fine_total"]):
                best = {"rank_nominal": rank, "x": x.tolist(),
                        "fine_total": total, "per_missile": per_m,
                        "all3_covered": covered,
                        "seed": seed, "time": round(time.time() - t0, 1)}
        done[key] = best
        json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[constrained {rank+1}/8] {key} fine={best['fine_total']:.4f}s "
              f"all3={best['all3_covered']} seed={best['seed']}", flush=True)

    feasible = {k: v for k, v in done.items() if v.get("all3_covered")}
    if feasible:
        best_key = max(feasible, key=lambda k: feasible[k]["fine_total"])
        state["best_all3"] = {"assignment": best_key,
                              "fine_total": feasible[best_key]["fine_total"],
                              "per_missile": feasible[best_key]["per_missile"]}
    else:
        state["best_all3"] = None
    json.dump(state, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done] best_all3 =", state["best_all3"])


if __name__ == "__main__":
    main()
