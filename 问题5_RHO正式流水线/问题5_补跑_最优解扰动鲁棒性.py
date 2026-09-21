# -*- coding: utf-8 -*-
"""
问题5 最优解（A0 完整 RHO 冠军方案）联合高斯扰动鲁棒性补跑
=================================================================
基准解：ablation_results.json -> refines["M1,M2,M1,M1,M3"].x（40 维，fine 总遮蔽 22.717447 s）
扰动方式：对 40 维每一维加 N(0, (p*|x_i|)^2) 的相对高斯扰动，随后截断到 BOUNDS_40
扰动档：p ∈ {0.01, 0.03, 0.05}，每档 30 次重复（每档独立固定种子，可复现）
评估精度：fine（每圆周 200 点、dt=0.005 s、二分 40 次），与 ablation_rho.eval_plan_fine 完全一致
指标：
  (a) 有效率     = 30 次中总遮蔽时长 > 0.01 s 的比例
  (b) 平均保持率 = mean(T_扰动) / T0
用法：python robustness_q5.py 0.01   （分档运行；结果增量写入 robustness_q5_results.json）

说明：本脚本自包含——物理模型与评估函数逐行复制自同目录 ablation_rho.py
（samples / shielded_scalar / interval_for / merge / decode / BOUNDS_40 / 物理常量），
未改动原文件任何内容。
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ABL_JSON = os.path.join(HERE, "ablation_results.json")
OUT_JSON = os.path.join(HERE, "robustness_q5_results.json")

# ================= 物理常量（复制自 ablation_rho.py） =================
G = 9.8
R = 10.0
SINK = 3.0
VALID = 20.0
MIN_Z = 2.0
EPS = 1e-12
TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TARGET_R = 7.0
TARGET_H = 10.0
MISSILE_SPEED = 300.0
MISSILES = {
    "M1": np.array([20000.0, 0.0, 2000.0]),
    "M2": np.array([19000.0, 600.0, 2100.0]),
    "M3": np.array([18000.0, -600.0, 1900.0]),
}
DRONES = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}
DRONE_LIST = ["FY1", "FY2", "FY3", "FY4", "FY5"]
MISSILE_LIST = ["M1", "M2", "M3"]
MDIR = {m: -p / np.linalg.norm(p) for m, p in MISSILES.items()}
ARRIVAL = {m: np.linalg.norm(p) / MISSILE_SPEED for m, p in MISSILES.items()}

# ================= 采样与判定（复制自 ablation_rho.py） =================
def samples(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bottom = np.column_stack([
        TARGET_CENTER[0] + TARGET_R * np.cos(theta),
        TARGET_CENTER[1] + TARGET_R * np.sin(theta),
        np.full(n, TARGET_CENTER[2])])
    top = np.column_stack([
        TARGET_CENTER[0] + TARGET_R * np.cos(theta),
        TARGET_CENTER[1] + TARGET_R * np.sin(theta),
        np.full(n, TARGET_CENTER[2] + TARGET_H)])
    return np.vstack([bottom, top])

E_FINE = samples(200)

def shielded_scalar(t, drone, m, heading, speed, drop, det, E):
    init = DRONES[drone]
    uav = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = init + speed * drop * uav
    det_xy = drop_pos[:2] + speed * det * uav[:2]
    det_z = drop_pos[2] - 0.5 * G * det * det
    t_start = drop + det
    if t < t_start or t > t_start + VALID or det_z < MIN_Z:
        return False
    smoke = np.array([det_xy[0], det_xy[1], det_z - SINK * (t - t_start)])
    if smoke[2] < MIN_Z:
        return False
    S = MISSILES[m] + MISSILE_SPEED * t * MDIR[m]
    MP = E - S
    MC = smoke - S
    a = np.maximum(np.sum(MP * MP, axis=1), EPS)
    proj = np.clip(np.sum(MP * MC, axis=1) / a, 0.0, 1.0)
    nearest = S + proj[:, None] * MP
    dist = np.linalg.norm(nearest - smoke, axis=1)
    return bool(np.all(dist <= R + EPS))

def interval_for(drone, m, heading, speed, drop, det, E, dt, bisect=40):
    init = DRONES[drone]
    uav = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = init + speed * drop * uav
    det_xy = drop_pos[:2] + speed * det * uav[:2]
    det_z = drop_pos[2] - 0.5 * G * det * det
    t_start = drop + det
    t_end = min(t_start + VALID, ARRIVAL[m])
    if det_z < MIN_Z or t_start >= t_end - EPS:
        return None
    t_coarse = np.arange(t_start, t_end + dt, dt)
    T = len(t_coarse)
    S = MISSILES[m] + MISSILE_SPEED * t_coarse[:, None] * MDIR[m]
    smoke_z = det_z - SINK * (t_coarse - t_start)
    valid_z = smoke_z >= MIN_Z - EPS
    C = np.empty((T, 3))
    C[:, 0] = det_xy[0]; C[:, 1] = det_xy[1]; C[:, 2] = smoke_z
    MP = E[None, :, :] - S[:, None, :]
    MC = C[:, None, :] - S[:, None, :]
    a = np.maximum(np.sum(MP * MP, axis=2), EPS)
    proj = np.clip(np.sum(MP * MC, axis=2) / a, 0.0, 1.0)
    nearest = S[:, None, :] + proj[:, :, None] * MP
    dist = np.linalg.norm(nearest - C[:, None, :], axis=2)
    covered = np.all(dist <= R + EPS, axis=1) & valid_z
    if not np.any(covered):
        return None
    first = int(np.argmax(covered))
    last = int(len(covered) - 1 - np.argmax(covered[::-1]))
    la = t_coarse[first]
    lo, hi = max(t_start, la - dt), la
    if not shielded_scalar(lo, drone, m, heading, speed, drop, det, E):
        for _ in range(bisect):
            mid = (lo + hi) / 2
            if shielded_scalar(mid, drone, m, heading, speed, drop, det, E):
                hi = mid
            else:
                lo = mid
    else:
        hi = lo
    left = hi
    ra = t_coarse[last]
    lo, hi = ra, min(t_end, ra + dt)
    if not shielded_scalar(hi, drone, m, heading, speed, drop, det, E):
        for _ in range(bisect):
            mid = (lo + hi) / 2
            if shielded_scalar(mid, drone, m, heading, speed, drop, det, E):
                lo = mid
            else:
                hi = mid
    else:
        lo = hi
    return float(left), float(lo)

def merge(ivs):
    valid = [iv for iv in ivs if iv is not None]
    if not valid:
        return 0.0
    valid.sort()
    total, cs, ce = 0.0, valid[0][0], valid[0][1]
    for s, e in valid[1:]:
        if s <= ce + EPS:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + (ce - cs)

# ================= 方案编解码（复制自 ablation_rho.py） =================
def decode(x, assignment):
    """40维向量 -> 每架无人机 (theta, v, drops[3], dets[3])；时间映射编码 d2=d1+g1, d3=d2+g2"""
    plan = {}
    for i, drone in enumerate(DRONE_LIST):
        seg = x[8 * i: 8 * i + 8]
        theta, v, d1, g1, g2 = seg[0], seg[1], seg[2], seg[3], seg[4]
        t1, t2, t3 = seg[5], seg[6], seg[7]
        drops = [d1, d1 + g1, d1 + g1 + g2]
        plan[drone] = (theta, v, drops, [t1, t2, t3], assignment[i])
    return plan

BOUNDS_40 = []
for i, drone in enumerate(DRONE_LIST):
    z0 = DRONES[drone][2]
    det_max = min(12.0, np.sqrt(2 * max(z0 - MIN_Z, 0.1) / G))
    BOUNDS_40 += [(0.0, 2 * np.pi), (70.0, 140.0), (0.0, 55.0),
                  (1.0, 10.0), (1.0, 10.0),
                  (0.1, det_max), (0.1, det_max), (0.1, det_max)]

def eval_plan_fine(x, assignment):
    """fine 精度总遮蔽时长（每圆周 200 点、dt=0.005 s、二分 40 次）"""
    plan = decode(x, assignment)
    per_m = {m: [] for m in MISSILE_LIST}
    for drone, (theta, v, drops, dets, m) in plan.items():
        for d, t in zip(drops, dets):
            iv = interval_for(drone, m, theta, v, d, t, E_FINE, 0.005, bisect=40)
            if iv is not None:
                per_m[m].append(iv)
    return sum(merge(ivs) for ivs in per_m.values())

# ================= 扰动鲁棒性主流程 =================
ASSIGNMENT = ("M1", "M2", "M1", "M1", "M3")   # FY1/FY3/FY4→M1, FY2→M2, FY5→M3
N_REP = 30
PRECISION = "fine (200 点/圆周, dt=0.005 s, 二分 40 次)"

def load_baseline():
    with open(ABL_JSON, "r", encoding="utf-8") as f:
        st = json.load(f)
    key = ",".join(ASSIGNMENT)
    return np.array(st["refines"][key]["x"]), float(st["refines"][key]["fine_total"])

def run_level(p, x0, T0):
    """单档扰动：30 次相对高斯扰动 + fine 评估。种子按档固定，可复现。"""
    rng = np.random.RandomState(20240500 + int(round(p * 100)))
    durs = []
    t_start = time.time()
    for k in range(N_REP):
        xp = x0.copy()
        for j, (lo, hi) in enumerate(BOUNDS_40):
            xp[j] = np.clip(xp[j] + rng.normal(0.0, p * abs(x0[j])), lo, hi)
        durs.append(float(eval_plan_fine(xp, ASSIGNMENT)))
        print(f"  [p={p:.2f}] rep {k+1:2d}/{N_REP}: T={durs[-1]:.4f}s", flush=True)
    arr = np.array(durs)
    valid_rate = float(np.mean(arr > 0.01))
    keep_rate = float(arr.mean() / T0)
    return {
        "p": p,
        "seed": 20240500 + int(round(p * 100)),
        "n_rep": N_REP,
        "durations_s": [round(v, 6) for v in durs],
        "valid_rate_T_gt_0.01s": valid_rate,
        "mean_keep_ratio_vs_T0": keep_rate,
        "mean_s": float(arr.mean()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "std_s": float(arr.std()),
        "wall_time_s": round(time.time() - t_start, 1),
    }

def main():
    x0, T0_json = load_baseline()
    # 复算基准（防 json 与代码不一致）
    T0 = float(eval_plan_fine(x0, ASSIGNMENT))
    print(f"基准解 fine 复算 T0 = {T0:.6f} s（json 记录 {T0_json:.6f} s）", flush=True)

    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            out = json.load(f)
    else:
        out = {
            "baseline_assignment": list(ASSIGNMENT),
            "baseline_x": [float(v) for v in x0],
            "T0_baseline_fine_s": T0,
            "T0_from_ablation_json_s": T0_json,
            "precision": PRECISION,
            "perturbation": "x_i += N(0, (p*|x_i|)^2)，逐维截断到 BOUNDS_40",
            "levels": {},
        }
    out["T0_baseline_fine_s"] = T0

    targets = [float(a) for a in sys.argv[1:]] or [0.01, 0.03, 0.05]
    for p in targets:
        key = f"{p:.2f}"
        if key in out["levels"]:
            print(f"档 p={key} 已完成，跳过", flush=True)
            continue
        print(f"=== 扰动档 p = ±{int(p*100)}% ===", flush=True)
        out["levels"][key] = run_level(p, x0, T0)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        lv = out["levels"][key]
        print(f"  -> 有效率={lv['valid_rate_T_gt_0.01s']*100:.1f}%  "
              f"平均保持率={lv['mean_keep_ratio_vs_T0']*100:.1f}%  "
              f"mean={lv['mean_s']:.4f}s min={lv['min_s']:.4f}s max={lv['max_s']:.4f}s",
              flush=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
