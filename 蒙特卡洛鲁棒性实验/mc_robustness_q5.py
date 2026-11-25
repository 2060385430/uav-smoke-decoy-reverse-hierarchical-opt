# -*- coding: utf-8 -*-
"""
实验3 · 问题5 冠军方案蒙特卡洛鲁棒性评估（物理建模随机扰动）
=================================================================
基准解：问题5/ablation/ablation_results.json -> refines["M1,M2,M1,M1,M3"].x
        （40 维，fine 总遮蔽 22.717447 s）

与既有"±1%/±3%/±5% 统一相对扰动"补跑的区别：
  本实验按补充实验 3 的要求，把四类物理量分别建模为独立随机变量：
    (1) 投放时刻 release time   ~ N(0, sigma_t^2)   秒
    (2) 引信延时 fuse delay     ~ N(0, sigma_f^2)   秒
    (3) 无人机速度 speed        ~ N(0, sigma_v^2)   m/s
    (4) 无人机航向 heading      ~ N(0, sigma_h^2)   rad
  扰动直接作用于解码后的物理方案（不经过时间映射再编码）。

硬约束处理：
  - 同一无人机相邻投放间隔 >= 1 s 为发射装置硬约束：扰动后顺序修正
    d_{k+1} = max(d_{k+1}, d_k + 1.0)，并统计触发比例；
  - 引信延时截断到 [0.1, 12] s（与优化搜索界一致）；
  - 速度截断到 [70, 140] m/s；航向截断到 [0, 2*pi)。

误差量级三档（工程容差假设，正态分布）：
  L1  nominal : sigma_h=0.5 deg, sigma_v=0.5 m/s, sigma_t=0.05 s, sigma_f=0.05 s
  L2  moderate: sigma_h=1.0 deg, sigma_v=1.0 m/s, sigma_t=0.10 s, sigma_f=0.10 s
  L3  severe  : sigma_h=2.0 deg, sigma_v=2.0 m/s, sigma_t=0.20 s, sigma_f=0.20 s

用法（断点续跑，结果增量写入 mc_results_<precision>.json）：
  python mc_robustness_q5.py --level L1 --reps 200 --precision medium
  python mc_robustness_q5.py --level L1 --reps 100 --precision fine --seed-base 910000

每次重复使用独立种子 seed = seed_base + 全局序号，完全可复现。
物理模型与评估函数逐行复制自 问题5/ablation/ablation_rho.py，未改动原文件。
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ABL_JSON = os.path.join(HERE, "..", "问题5", "ablation", "ablation_results.json")

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
ASSIGNMENT = ("M1", "M2", "M1", "M1", "M3")   # FY1/FY3/FY4→M1, FY2→M2, FY5→M3

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

E_MED = samples(120)     # medium：120 点/圆周、dt=0.02 s、二分 35 次
E_FINE = samples(200)    # fine  ：200 点/圆周、dt=0.005 s、二分 40 次

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

def interval_for(drone, m, heading, speed, drop, det, E, dt, bisect=35):
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

# ================= 方案解码 + 物理扰动 =================
def decode_physical(x):
    """40维向量 -> {drone: (m, theta, v, drops[3], dets[3])}（物理量，不再编码）"""
    plan = {}
    for i, drone in enumerate(DRONE_LIST):
        seg = x[8 * i: 8 * i + 8]
        theta, v, d1, g1, g2 = seg[0], seg[1], seg[2], seg[3], seg[4]
        drops = [d1, d1 + g1, d1 + g1 + g2]
        dets = [seg[5], seg[6], seg[7]]
        plan[drone] = [ASSIGNMENT[i], theta, v, drops, dets]
    return plan

def eval_physical(plan, E, dt, bisect):
    per_m = {m: [] for m in MISSILE_LIST}
    for drone, (m, theta, v, drops, dets) in plan.items():
        for d, t in zip(drops, dets):
            iv = interval_for(drone, m, theta, v, d, t, E, dt, bisect=bisect)
            if iv is not None:
                per_m[m].append(iv)
    per = {m: merge(ivs) for m, ivs in per_m.items()}
    return sum(per.values()), per

LEVELS = {
    "L1": dict(name="nominal",  sigma_h=np.radians(0.5), sigma_v=0.5, sigma_t=0.05, sigma_f=0.05),
    "L2": dict(name="moderate", sigma_h=np.radians(1.0), sigma_v=1.0, sigma_t=0.10, sigma_f=0.10),
    "L3": dict(name="severe",   sigma_h=np.radians(2.0), sigma_v=2.0, sigma_t=0.20, sigma_f=0.20),
}
MIN_GAP = 1.0          # 发射装置最小投放间隔（硬约束）
DET_LO, DET_HI = 0.1, 12.0
V_LO, V_HI = 70.0, 140.0

def perturb_plan(base_plan, rng, lv):
    """对物理方案加一次扰动，返回 (新plan, gap修正次数)"""
    plan = {}
    n_adj = 0
    for drone, (m, theta, v, drops, dets) in base_plan.items():
        th = (theta + rng.normal(0.0, lv["sigma_h"])) % (2 * np.pi)
        vv = float(np.clip(v + rng.normal(0.0, lv["sigma_v"]), V_LO, V_HI))
        dp = [d + rng.normal(0.0, lv["sigma_t"]) for d in drops]
        dp[0] = max(dp[0], 0.0)
        for k in (1, 2):                      # 顺序修正最小投放间隔
            if dp[k] < dp[k - 1] + MIN_GAP:
                dp[k] = dp[k - 1] + MIN_GAP
                n_adj += 1
        dt_ = [float(np.clip(t + rng.normal(0.0, lv["sigma_f"]), DET_LO, DET_HI)) for t in dets]
        plan[drone] = [m, th, vv, dp, dt_]
    return plan, n_adj

# ================= 主流程（断点续跑） =================
def load_baseline():
    with open(ABL_JSON, "r", encoding="utf-8") as f:
        st = json.load(f)
    key = ",".join(ASSIGNMENT)
    return np.array(st["refines"][key]["x"]), float(st["refines"][key]["fine_total"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--reps", type=int, required=True, help="本批新增重复次数")
    ap.add_argument("--precision", default="medium", choices=["medium", "fine"])
    ap.add_argument("--seed-base", type=int, default=310000)
    args = ap.parse_args()

    E, dt, bis = (E_MED, 0.02, 35) if args.precision == "medium" else (E_FINE, 0.005, 40)
    out_json = os.path.join(HERE, f"mc_results_{args.precision}.json")

    x0, T0_fine_json = load_baseline()
    base_plan = decode_physical(x0)

    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            out = json.load(f)
    else:
        T0_med, per0 = eval_physical(base_plan, E, dt, bis)
        out = {
            "experiment": "实验3 蒙特卡洛鲁棒性：投放时刻/引信延时/速度/航向 独立高斯扰动",
            "baseline_assignment": list(ASSIGNMENT),
            "T0_baseline_this_precision_s": T0_med,
            "T0_baseline_per_missile_s": per0,
            "T0_from_ablation_json_fine_s": T0_fine_json,
            "precision": args.precision,
            "hard_constraints": "投放间隔>=1s 顺序修正；引信延时[0.1,12]s；速度[70,140]m/s；航向[0,2pi)",
            "levels": {},
        }
        print(f"基准解 {args.precision} 复算 T0 = {T0_med:.6f} s  per-missile={ {k: round(v,4) for k,v in per0.items()} }", flush=True)

    lv = LEVELS[args.level]
    slot = out["levels"].setdefault(args.level, {
        "sigma": {"heading_deg": float(np.degrees(lv["sigma_h"])), "speed_mps": lv["sigma_v"],
                  "release_s": lv["sigma_t"], "fuse_s": lv["sigma_f"]},
        "seed_base": args.seed_base,
        "records": [],          # 每条: [total, M1, M2, M3]
        "n_gap_adjust": 0,
        "wall_time_s": 0.0,
    })

    n0 = len(slot["records"])
    t0 = time.time()
    for k in range(args.reps):
        rng = np.random.RandomState(args.seed_base + n0 + k)
        plan, n_adj = perturb_plan(base_plan, rng, lv)
        tot, per = eval_physical(plan, E, dt, bis)
        slot["records"].append([round(tot, 6), round(per["M1"], 6),
                                round(per["M2"], 6), round(per["M3"], 6)])
        slot["n_gap_adjust"] += n_adj
        if (k + 1) % 20 == 0 or k == args.reps - 1:
            print(f"  [{args.level}/{args.precision}] rep {n0+k+1}: T={tot:.4f}s", flush=True)
    slot["wall_time_s"] = round(slot["wall_time_s"] + time.time() - t0, 1)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    arr = np.array([r[0] for r in slot["records"]])
    print(f"=== {args.level} 累计 {len(arr)} 次 | 有效率(>0.01s)={np.mean(arr>0.01)*100:.1f}% "
          f"mean={arr.mean():.4f}s std={arr.std():.4f}s "
          f"P5={np.percentile(arr,5):.4f}s P50={np.percentile(arr,50):.4f}s ===", flush=True)
    print(f"saved {out_json}", flush=True)

if __name__ == "__main__":
    main()
