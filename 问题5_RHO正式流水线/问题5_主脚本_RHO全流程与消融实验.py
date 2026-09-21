# -*- coding: utf-8 -*-
"""
问题5 RHO 消融实验
变体：
  A0 完整 RHO（筛选Top-8 + 40维精修）
  A1 去局部精修（仅 阶段一+二+三）
  A2 去任务分配枚举（贪心分配 + 精修）
  A3 去联合筛选（独立时长求和排序 + 精修Top-8）
  A4 去分层（直接40维全局DE）
  S  筛选宽度敏感性 Top-N ∈ {1,2,4,8}（复用A0精修结果）
锚点：论文报告系统总遮蔽时长 20.46s（fine 精度）
断点续跑：结果写入 ablation_results.json，已完成项自动跳过
"""
import os, sys, json, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from itertools import product
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "..", "candidate_lib.pkl")
OUT_JSON = os.path.join(HERE, "ablation_results.json")

# ================= 物理常量 =================
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

# ================= 采样与判定 =================
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

E_MED = samples(120)
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

# ================= 方案编解码 =================
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

def encode_from_candidate(assignment, cand_by_drone):
    """由候选库候选参数构造40维初始向量"""
    x = np.zeros(40)
    for i, drone in enumerate(DRONE_LIST):
        c = cand_by_drone[drone]
        drops = sorted(c["drop_times"])
        g1 = max(drops[1] - drops[0], 1.0)
        g2 = max(drops[2] - drops[1], 1.0)
        x[8 * i: 8 * i + 8] = [c["theta"], c["v"], drops[0], g1, g2,
                               c["det_delays"][0], c["det_delays"][1], c["det_delays"][2]]
    return x

BOUNDS_40 = []
for i, drone in enumerate(DRONE_LIST):
    z0 = DRONES[drone][2]
    det_max = min(12.0, np.sqrt(2 * max(z0 - MIN_Z, 0.1) / G))
    BOUNDS_40 += [(0.0, 2 * np.pi), (70.0, 140.0), (0.0, 55.0),
                  (1.0, 10.0), (1.0, 10.0),
                  (0.1, det_max), (0.1, det_max), (0.1, det_max)]

def eval_plan_med(x, assignment):
    plan = decode(x, assignment)
    per_m = {m: [] for m in MISSILE_LIST}
    for drone, (theta, v, drops, dets, m) in plan.items():
        if not (70.0 <= v <= 140.0):
            return 1000.0
        for d, t in zip(drops, dets):
            iv = interval_for(drone, m, theta, v, d, t, E_MED, 0.02)
            if iv is not None:
                per_m[m].append(iv)
    total = sum(merge(ivs) for ivs in per_m.values())
    return -total if total > 0 else 1000.0

def eval_plan_fine(x, assignment):
    plan = decode(x, assignment)
    per_m = {m: [] for m in MISSILE_LIST}
    detail = {}
    for drone, (theta, v, drops, dets, m) in plan.items():
        for d, t in zip(drops, dets):
            iv = interval_for(drone, m, theta, v, d, t, E_FINE, 0.005, bisect=40)
            if iv is not None:
                per_m[m].append(iv)
                detail.setdefault(m, []).append(iv)
    per_m_dur = {m: merge(ivs) for m, ivs in per_m.items()}
    return sum(per_m_dur.values()), per_m_dur

# ================= 阶段二+三：枚举分配 + 联合筛选 =================
def screen_assignments(lib, joint=True):
    """返回 [(score, assignment, cand_idx_map), ...] 按分数降序。
    joint=True: 每分配内枚举候选组合，按三导弹区间并集总和评分
    joint=False: 各无人机取最优候选，按独立时长简单求和评分（忽略重叠）"""
    results = []
    for choices in product(range(3), repeat=5):
        covered = {MISSILE_LIST[c] for c in choices}
        if len(covered) < 3:
            continue
        assignment = tuple(MISSILE_LIST[c] for c in choices)
        cand_lists = []
        ok = True
        for i, drone in enumerate(DRONE_LIST):
            cands = lib.get((drone, assignment[i]), [])
            if not cands:
                ok = False
                break
            cand_lists.append(cands)
        if not ok:
            continue
        if not joint:
            score = sum(max(c["duration"] for c in cl) for cl in cand_lists)
            best_map = {DRONE_LIST[i]: max(range(len(cand_lists[i])),
                                           key=lambda k: cand_lists[i][k]["duration"])
                        for i in range(5)}
            results.append((score, assignment, best_map))
            continue
        best_score, best_map = -1.0, None
        for combo in product(*[range(len(cl)) for cl in cand_lists]):
            per_m = {m: [] for m in MISSILE_LIST}
            for i, drone in enumerate(DRONE_LIST):
                c = cand_lists[i][combo[i]]
                per_m[assignment[i]].extend(
                    [(float(s), float(e)) for s, e in c["intervals"]])
            score = sum(merge(ivs) for ivs in per_m.values())
            if score > best_score:
                best_score = score
                best_map = {DRONE_LIST[i]: combo[i] for i in range(5)}
        results.append((best_score, assignment, best_map))
    results.sort(key=lambda r: -r[0])
    return results

def greedy_assignment(lib):
    """每机选最优导弹，再修补覆盖"""
    best_pair = {}
    for drone in DRONE_LIST:
        best_m, best_d = None, -1
        for m in MISSILE_LIST:
            cands = lib.get((drone, m), [])
            if cands and cands[0]["duration"] > best_d:
                best_d, best_m = cands[0]["duration"], m
        best_pair[drone] = best_m
    assign = {d: best_pair[d] for d in DRONE_LIST}
    covered = set(assign.values())
    for m in MISSILE_LIST:
        if m not in covered:
            # 找改派损失最小的无人机
            best_drone, best_loss = None, float("inf")
            for d in DRONE_LIST:
                cur = lib[(d, assign[d])][0]["duration"]
                alt = lib.get((d, m), [])
                alt_d = alt[0]["duration"] if alt else 0.0
                loss = cur - alt_d
                if list(assign.values()).count(assign[d]) > 1 and loss < best_loss:
                    best_loss, best_drone = loss, d
            if best_drone is not None:
                assign[best_drone] = m
            covered = set(assign.values())
    assignment = tuple(assign[d] for d in DRONE_LIST)
    best_map = {}
    for i, drone in enumerate(DRONE_LIST):
        cands = lib[(drone, assignment[i])]
        best_map[drone] = max(range(len(cands)), key=lambda k: cands[k]["duration"])
    return assignment, best_map

# ================= 阶段四：40维局部精修 =================
def refine(assignment, cand_by_drone, seed=42, maxiter=15):
    x0 = encode_from_candidate(assignment, cand_by_drone)
    rng = np.random.RandomState(seed)
    init_pop = [x0.copy()]
    while len(init_pop) < 10:
        p = x0.copy()
        for j, (lo, hi) in enumerate(BOUNDS_40):
            p[j] = np.clip(p[j] + rng.normal(0, (hi - lo) * 0.03), lo, hi)
        init_pop.append(p)
    res = differential_evolution(
        eval_plan_med, BOUNDS_40, args=(assignment,),
        maxiter=maxiter, popsize=10, seed=seed,
        mutation=(0.5, 1.0), recombination=0.8, polish=False,
        init=np.array(init_pop), workers=1, tol=1e-9)
    return res.x

def interval_or_guidance(drone, m, heading, speed, drop, det, E, dt):
    """返回 (iv, guidance)：有遮蔽时 iv=(s,e)、guidance=None；
    无遮蔽时 iv=None、guidance=有效时间窗内云团到各视线的最大距离的最小值"""
    init = DRONES[drone]
    uav = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = init + speed * drop * uav
    det_xy = drop_pos[:2] + speed * det * uav[:2]
    det_z = drop_pos[2] - 0.5 * G * det * det
    t_start = drop + det
    t_end = min(t_start + VALID, ARRIVAL[m])
    if det_z < MIN_Z or t_start >= t_end - EPS:
        return None, 500.0
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
    dist = np.where(valid_z[:, None], dist, 1e5)
    max_per_t = np.max(dist, axis=1)
    guidance = float(np.min(max_per_t))
    covered = np.all(dist <= R + EPS, axis=1)
    if not np.any(covered):
        return None, guidance
    iv = interval_for(drone, m, heading, speed, drop, det, E, dt)
    return iv, None

def eval_global_med_guided(x):
    """A4'目标：有遮蔽时最大化并集；无遮蔽时以距离引导项替代"""
    per_m = {m: [] for m in MISSILE_LIST}
    guid_sum = 0.0
    for i, drone in enumerate(DRONE_LIST):
        seg = x[8 * i: 8 * i + 8]
        theta, v, d1, g1, g2 = seg[0], seg[1], seg[2], seg[3], seg[4]
        drops = [d1, d1 + g1, d1 + g1 + g2]
        if not (70.0 <= v <= 140.0):
            return 1000.0
        for d, t in zip(drops, seg[5:8]):
            for m in MISSILE_LIST:
                iv, g = interval_or_guidance(drone, m, theta, v, d, t, E_MED, 0.05)
                if iv is not None:
                    per_m[m].append(iv)
                else:
                    guid_sum += g
    total = sum(merge(ivs) for ivs in per_m.values())
    if total > 0:
        return -total
    return 100.0 + guid_sum / 45.0  # 无遮蔽：平均引导距离越小越好

def eval_global_med(x):
    """A4目标：40维全局DE，每枚弹对三枚导弹分别计算遮蔽区间，计入各导弹并集"""
    per_m = {m: [] for m in MISSILE_LIST}
    for i, drone in enumerate(DRONE_LIST):
        seg = x[8 * i: 8 * i + 8]
        theta, v, d1, g1, g2 = seg[0], seg[1], seg[2], seg[3], seg[4]
        drops = [d1, d1 + g1, d1 + g1 + g2]
        if not (70.0 <= v <= 140.0):
            return 1000.0
        for d, t in zip(drops, seg[5:8]):
            for m in MISSILE_LIST:
                iv = interval_for(drone, m, theta, v, d, t, E_MED, 0.02)
                if iv is not None:
                    per_m[m].append(iv)
    total = sum(merge(ivs) for ivs in per_m.values())
    return -total if total > 0 else 1000.0

def refine_global(seed=42, maxiter=40, popsize=10):
    """A4：直接40维全局DE（无分层结构）"""
    res = differential_evolution(
        eval_global_med, BOUNDS_40, maxiter=maxiter, popsize=popsize, seed=seed,
        mutation=(0.5, 1.0), recombination=0.7, polish=False,
        init='random', workers=1, tol=1e-9)
    return res.x

# ================= 断点存储 =================
def load_state():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(st):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)

# ================= 主流程 =================
def main():
    t0 = time.time()
    with open(PKL, "rb") as f:
        lib = pickle.load(f)
    st = load_state()

    # --- 锚点校验：xlsx 最终方案的 fine 评估 ---
    if "anchor" not in st:
        anchor_assign = ("M1", "M2", "M1", "M1", "M3")
        anchor_cand = {d: lib[(d, anchor_assign[i])][0] for i, d in enumerate(DRONE_LIST)}
        x0 = encode_from_candidate(anchor_assign, anchor_cand)
        fine, per_m = eval_plan_fine(x0, anchor_assign)
        st["anchor"] = {"fine_total": fine, "per_missile": per_m}
        save_state(st)
        print(f"[锚点] 候选库最优组合 fine 总遮蔽 = {fine:.4f}s {per_m}", flush=True)

    # --- 阶段二+三：联合筛选 ---
    if "screen_joint" not in st:
        ranked = screen_assignments(lib, joint=True)
        st["screen_joint"] = [
            {"score": s, "assignment": list(a), "cand_map": cm}
            for s, a, cm in ranked[:8]]
        st["n_valid_assignments"] = len(ranked)
        save_state(st)
        print(f"[筛选] 有效分配 {len(ranked)} 种，Top1={ranked[0][0]:.4f}s "
              f"Top8={ranked[7][0]:.4f}s", flush=True)

    # --- 阶段二+三（消融）：独立求和排序 ---
    if "screen_indep" not in st:
        ranked = screen_assignments(lib, joint=False)
        st["screen_indep"] = [
            {"score": s, "assignment": list(a), "cand_map": cm}
            for s, a, cm in ranked[:8]]
        save_state(st)

    # --- A2 贪心分配 ---
    if "greedy" not in st:
        assignment, best_map = greedy_assignment(lib)
        st["greedy"] = {"assignment": list(assignment), "cand_map": best_map}
        save_state(st)
        print(f"[贪心] {assignment}", flush=True)

    # --- 阶段四精修（A0 Top-8 / A2 / A3 Top-8，按 assignment 缓存） ---
    jobs = []
    for item in st["screen_joint"]:
        jobs.append(("A0", item["assignment"], item["cand_map"]))
    jobs.append(("A2", st["greedy"]["assignment"], st["greedy"]["cand_map"]))
    for item in st["screen_indep"]:
        jobs.append(("A3", item["assignment"], item["cand_map"]))

    refines = st.setdefault("refines", {})
    for tag, assignment_list, cand_map in jobs:
        key = ",".join(assignment_list)
        if key in refines:
            continue
        assignment = tuple(assignment_list)
        cand_by_drone = {d: lib[(d, assignment[i])][cand_map[d]]
                         for i, d in enumerate(DRONE_LIST)}
        print(f"[精修] {assignment} ...", flush=True)
        ts = time.time()
        x = refine(assignment, cand_by_drone)
        fine, per_m = eval_plan_fine(x, assignment)
        refines[key] = {"x": x.tolist(), "fine_total": fine,
                        "per_missile": per_m, "time": time.time() - ts}
        save_state(st)
        print(f"[精修] {assignment} fine={fine:.4f}s 耗时{time.time()-ts:.0f}s", flush=True)

    # --- A4 直接全局DE（分块热启动，断点续跑） ---
    if "A4" not in st:
        part = st.get("A4_partial", {"iters": 0, "x": None, "fun": None, "time": 0.0})
        ts = time.time()
        while part["iters"] < 30:
            init = 'random'
            if part["x"] is not None:
                x0 = np.array(part["x"])
                rng = np.random.RandomState(999 + part["iters"])
                rows = [x0.copy()]
                while len(rows) < 10:
                    p = x0.copy()
                    for j, (lo, hi) in enumerate(BOUNDS_40):
                        p[j] = np.clip(p[j] + rng.normal(0, (hi - lo) * 0.05), lo, hi)
                    rows.append(p)
                init = np.array(rows)
            res = differential_evolution(
                eval_global_med, BOUNDS_40, maxiter=6, popsize=5,
                seed=1000 + part["iters"], mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init=init, workers=1, tol=1e-9)
            if part["fun"] is None or res.fun < part["fun"]:
                part["fun"], part["x"] = float(res.fun), res.x.tolist()
            part["iters"] += 6
            part["time"] += time.time() - ts
            st["A4_partial"] = part
            save_state(st)
            print(f"[A4] 迭代{part['iters']}/30 当前最优={-part['fun']:.4f}s", flush=True)
            break  # 每次调用最多跑一块，防超时
        if part["iters"] >= 30:
            x = np.array(part["x"])
            per_m = {m: [] for m in MISSILE_LIST}
            for i, drone in enumerate(DRONE_LIST):
                seg = x[8 * i: 8 * i + 8]
                drops = [seg[2], seg[2] + seg[3], seg[2] + seg[3] + seg[4]]
                for d, t in zip(drops, seg[5:8]):
                    for m in MISSILE_LIST:
                        iv = interval_for(drone, m, seg[0], seg[1], d, t, E_FINE, 0.005, bisect=40)
                        if iv is not None:
                            per_m[m].append(iv)
            per_m_dur = {m: merge(ivs) for m, ivs in per_m.items()}
            st["A4"] = {"x": x.tolist(), "fine_total": sum(per_m_dur.values()),
                        "per_missile": per_m_dur, "time": part["time"]}
            save_state(st)
            print(f"[A4] fine={st['A4']['fine_total']:.4f}s 耗时{st['A4']['time']:.0f}s", flush=True)
        else:
            return  # 未完成，下次继续

    # --- A4' 带距离引导的直接全局DE（分块热启动） ---
    if "A4g" not in st:
        part = st.get("A4g_partial", {"iters": 0, "x": None, "fun": None, "time": 0.0})
        ts = time.time()
        while part["iters"] < 30:
            init = 'random'
            if part["x"] is not None:
                x0 = np.array(part["x"])
                rng = np.random.RandomState(1999 + part["iters"])
                rows = [x0.copy()]
                while len(rows) < 10:
                    p = x0.copy()
                    for j, (lo, hi) in enumerate(BOUNDS_40):
                        p[j] = np.clip(p[j] + rng.normal(0, (hi - lo) * 0.05), lo, hi)
                    rows.append(p)
                init = np.array(rows)
            res = differential_evolution(
                eval_global_med_guided, BOUNDS_40, maxiter=3, popsize=5,
                seed=2000 + part["iters"], mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init=init, workers=1, tol=1e-9)
            if part["fun"] is None or res.fun < part["fun"]:
                part["fun"], part["x"] = float(res.fun), res.x.tolist()
            part["iters"] += 3
            part["time"] += time.time() - ts
            st["A4g_partial"] = part
            save_state(st)
            print(f"[A4'] 迭代{part['iters']}/30 当前目标值={part['fun']:.4f}", flush=True)
            break
        if part["iters"] >= 30:
            x = np.array(part["x"])
            per_m = {m: [] for m in MISSILE_LIST}
            for i, drone in enumerate(DRONE_LIST):
                seg = x[8 * i: 8 * i + 8]
                drops = [seg[2], seg[2] + seg[3], seg[2] + seg[3] + seg[4]]
                for d, t in zip(drops, seg[5:8]):
                    for m in MISSILE_LIST:
                        iv = interval_for(drone, m, seg[0], seg[1], d, t, E_FINE, 0.005, bisect=40)
                        if iv is not None:
                            per_m[m].append(iv)
            per_m_dur = {m: merge(ivs) for m, ivs in per_m.items()}
            st["A4g"] = {"x": x.tolist(), "fine_total": sum(per_m_dur.values()),
                         "per_missile": per_m_dur, "time": part["time"]}
            save_state(st)
            print(f"[A4'] fine={st['A4g']['fine_total']:.4f}s 耗时{st['A4g']['time']:.0f}s", flush=True)
        else:
            return

    # --- 汇总 ---
    summary = {}
    a0_keys = [",".join(it["assignment"]) for it in st["screen_joint"]]
    a0_scores = [st["refines"][k]["fine_total"] for k in a0_keys]
    summary["A0_full_RHO"] = max(a0_scores)
    summary["S_topN"] = {str(n): max(a0_scores[:n]) for n in (1, 2, 4, 8)}
    summary["A1_no_refine"] = st["anchor"]["fine_total"]
    gk = ",".join(st["greedy"]["assignment"])
    summary["A2_no_enum"] = st["refines"][gk]["fine_total"]
    a3_keys = [",".join(it["assignment"]) for it in st["screen_indep"]]
    summary["A3_no_joint_screen"] = max(st["refines"][k]["fine_total"] for k in a3_keys)
    summary["A4_no_hierarchy"] = st["A4"]["fine_total"]
    summary["A4g_guided_DE"] = st["A4g"]["fine_total"]
    st["summary"] = summary
    save_state(st)

    print("\n================ 消融实验汇总（fine 精度，系统总遮蔽时长 s）================")
    print(f"A0 完整 RHO            : {summary['A0_full_RHO']:.4f}")
    print(f"A1 去局部精修          : {summary['A1_no_refine']:.4f}")
    print(f"A2 去任务分配枚举(贪心): {summary['A2_no_enum']:.4f}")
    print(f"A3 去联合筛选(独立和)  : {summary['A3_no_joint_screen']:.4f}")
    print(f"A4 去分层(直接40维DE)  : {summary['A4_no_hierarchy']:.4f}")
    print(f"A4' 带距离引导的40维DE : {summary['A4g_guided_DE']:.4f}")
    print("Top-N 敏感性:", {k: round(v, 4) for k, v in summary["S_topN"].items()})
    print(f"总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
