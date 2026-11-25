import os
import json
import time
import importlib.util
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ablation_rho.py")
ABL_JSON = os.path.join(HERE, "ablation_results.json")
OUT_JSON = os.path.join(HERE, "medium_fine_check.json")

CHAMPION_KEY = "M1,M2,M1,M1,M3"
PAPER_CLAIM_DIFF = 0.000191

# ---- 导入 ablation_rho.py（不执行其 __main__） ----
spec = importlib.util.spec_from_file_location("ablation_rho", SRC)
rho = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rho)


def eval_plan_at(x, assignment, E, dt, bisect):
    """与 ablation_rho.eval_plan_fine 同结构，但精度参数外置。
    返回 (总时长, 各导弹时长, 各导弹区间)。"""
    plan = rho.decode(x, assignment)
    per_m = {m: [] for m in rho.MISSILE_LIST}
    for drone, (theta, v, drops, dets, m) in plan.items():
        for d, t in zip(drops, dets):
            iv = rho.interval_for(drone, m, theta, v, d, t, E, dt, bisect=bisect)
            if iv is not None:
                per_m[m].append(iv)
    per_m_dur = {m: rho.merge(ivs) for m, ivs in per_m.items()}
    return sum(per_m_dur.values()), per_m_dur, per_m


def main():
    with open(ABL_JSON, "r", encoding="utf-8") as f:
        st = json.load(f)
    champ = st["refines"][CHAMPION_KEY]
    x = np.array(champ["x"], dtype=float)
    assignment = tuple(CHAMPION_KEY.split(","))
    fine_stored = champ["fine_total"]

    # ---- medium：120 点/圆周、dt=0.02、二分 35 次（同 eval_plan_med 精度） ----
    t0 = time.time()
    med_total, med_per_m, med_ivs = eval_plan_at(
        x, assignment, rho.E_MED, 0.02, 35)
    t_med = time.time() - t0

    # ---- fine：200 点/圆周、dt=0.005、二分 40 次（同 eval_plan_fine 精度） ----
    t0 = time.time()
    fine_total, fine_per_m, fine_ivs = eval_plan_at(
        x, assignment, rho.E_FINE, 0.005, 40)
    t_fine = time.time() - t0

    diff = abs(fine_total - med_total)
    out = {
        "meta": {
            "script": os.path.basename(__file__),
            "purpose": "验证论文 medium/fine 差异 0.000191 s 的说法",
            "champion_key": CHAMPION_KEY,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "medium": {
            "params": {"num_per_circle": 120, "dt": 0.02, "bisect_iters": 35},
            "total_s": float(med_total),
            "per_missile_s": {m: float(v) for m, v in med_per_m.items()},
            "per_missile_intervals_s": {
                m: [[float(s), float(e)] for s, e in ivs]
                for m, ivs in med_ivs.items()},
            "elapsed_s": round(t_med, 2),
        },
        "fine": {
            "params": {"num_per_circle": 200, "dt": 0.005, "bisect_iters": 40},
            "total_s": float(fine_total),
            "per_missile_s": {m: float(v) for m, v in fine_per_m.items()},
            "per_missile_intervals_s": {
                m: [[float(s), float(e)] for s, e in ivs]
                for m, ivs in fine_ivs.items()},
            "elapsed_s": round(t_fine, 2),
        },
        "comparison": {
            "fine_total_recomputed_s": float(fine_total),
            "fine_total_stored_in_ablation_json_s": float(fine_stored),
            "fine_recompute_matches_stored": bool(abs(fine_total - fine_stored) < 1e-9),
            "abs_diff_medium_vs_fine_s": float(diff),
            "paper_claimed_diff_s": PAPER_CLAIM_DIFF,
            "diff_matches_paper_claim_1e6": bool(abs(diff - PAPER_CLAIM_DIFF) < 5e-7),
            "diff_same_order_as_paper_claim": bool(abs(diff - PAPER_CLAIM_DIFF) < 5e-5),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("=" * 66)
    print(f"冠军向量 {CHAMPION_KEY}（40 维，来自 ablation_results.json）")
    print(f"  medium（120点, dt=0.02, 二分35）: {med_total:.6f} s  {med_per_m}")
    print(f"  fine  （200点, dt=0.005, 二分40）: {fine_total:.6f} s  {fine_per_m}")
    print(f"  json 存档 fine_total            : {fine_stored:.6f} s"
          f"  （重算一致: {out['comparison']['fine_recompute_matches_stored']}）")
    print(f"  |fine - medium| = {diff:.6f} s")
    print(f"  论文声称差异    = {PAPER_CLAIM_DIFF:.6f} s")
    print(f"  与论文声称之差  = {diff - PAPER_CLAIM_DIFF:+.6f} s")
    print(f"[落盘] {OUT_JSON}")


if __name__ == "__main__":
    main()
