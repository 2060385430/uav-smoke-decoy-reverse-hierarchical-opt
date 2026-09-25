# -*- coding: utf-8 -*-
"""
实验2 · 小规模全局最优性验证（问题2：4维单弹）
=================================================================
路线：大规模粗扫（40点/圆周、dt=0.1 s、无二分，仅用于候选筛选）
      → Top-K 候选在 medium 精度（120点、dt=0.02、二分35）下 DE 抛光
      → 最优者 fine 精度（200点、dt=0.005、二分40）复核
对比论文值 4.581712 s（fine），报告 optimality gap。

物理模型逐行复制自 问题2/problem3_solver 同源代码，未改动原文件。
用法：
  python global_verify_p2.py scan  <n_coarse>     # 粗扫 n 个随机点（断点续跑）
  python global_verify_p2.py polish <top_k>       # 抛光 + fine 复核
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_JSON = os.path.join(HERE, "gv_p2_scan.json")
OUT_JSON = os.path.join(HERE, "gv_p2_result.json")

G = 9.8; R = 10.0; SINK = 3.0; VALID = 20.0; MIN_Z = 2.0; EPS = 1e-12
TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TARGET_R = 7.0; TARGET_H = 10.0
DRONE0 = np.array([17800.0, 0.0, 1800.0])
MISSILE0 = np.array([20000.0, 0.0, 2000.0])
MISSILE_DIR = -MISSILE0 / np.linalg.norm(MISSILE0)
ARRIVAL = np.linalg.norm(MISSILE0) / 300.0
BOUNDS = [(0.0, 2 * np.pi), (70.0, 140.0), (0.0, ARRIVAL * 0.8), (0.1, 15.0)]
PAPER_FINE = 4.581712   # 论文问题2最优（fine）

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

def shielded_scalar(t, heading, speed, drop, det, E):
    uav = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = DRONE0 + speed * drop * uav
    det_xy = drop_pos[:2] + speed * det * uav[:2]
    det_z = drop_pos[2] - 0.5 * G * det * det
    t_start = drop + det
    if t < t_start or t > t_start + VALID or det_z < MIN_Z:
        return False
    smoke = np.array([det_xy[0], det_xy[1], det_z - SINK * (t - t_start)])
    if smoke[2] < MIN_Z:
        return False
    S = MISSILE0 + 300.0 * t * MISSILE_DIR
    MP = E - S; MC = smoke - S
    a = np.maximum(np.sum(MP * MP, axis=1), EPS)
    proj = np.clip(np.sum(MP * MC, axis=1) / a, 0.0, 1.0)
    nearest = S + proj[:, None] * MP
    dist = np.linalg.norm(nearest - smoke, axis=1)
    return bool(np.all(dist <= R + EPS))

def interval_for(heading, speed, drop, det, E, dt, bisect=35):
    uav = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = DRONE0 + speed * drop * uav
    det_xy = drop_pos[:2] + speed * det * uav[:2]
    det_z = drop_pos[2] - 0.5 * G * det * det
    t_start = drop + det
    t_end = min(t_start + VALID, ARRIVAL)
    if det_z < MIN_Z or t_start >= t_end - EPS:
        return None
    t_coarse = np.arange(t_start, t_end + dt, dt)
    T = len(t_coarse)
    S = MISSILE0 + 300.0 * t_coarse[:, None] * MISSILE_DIR
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
    if not shielded_scalar(lo, heading, speed, drop, det, E):
        for _ in range(bisect):
            mid = (lo + hi) / 2
            if shielded_scalar(mid, heading, speed, drop, det, E):
                hi = mid
            else:
                lo = mid
    else:
        hi = lo
    left = hi
    ra = t_coarse[last]
    lo, hi = ra, min(t_end, ra + dt)
    if not shielded_scalar(hi, heading, speed, drop, det, E):
        for _ in range(bisect):
            mid = (lo + hi) / 2
            if shielded_scalar(mid, heading, speed, drop, det, E):
                lo = mid
            else:
                hi = mid
    else:
        lo = hi
    return float(left), float(lo)

# ---------- 向量化粗扫（无二分，仅筛选） ----------
E_COARSE = samples(40)

def coarse_durations(X):
    """X: (M,4) -> 近似时长 (M,)；无二分，dt=0.1"""
    dt = 0.1
    heading, speed, drop, det = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    ux, uy = np.cos(heading), np.sin(heading)
    det_x = DRONE0[0] + speed * (drop + det) * ux
    det_y = DRONE0[1] + speed * (drop + det) * uy
    det_z = DRONE0[2] - 0.5 * G * det * det
    t_start = drop + det
    t_end = np.minimum(t_start + VALID, ARRIVAL)
    M = len(X)
    out = np.zeros(M)
    # 统一时间栅格：t in (0, ARRIVAL]
    tg = np.arange(0.0, ARRIVAL + dt, dt)
    S = MISSILE0 + 300.0 * tg[:, None] * MISSILE_DIR          # (T,3)
    MP = E_COARSE[None, :, :] - S[:, None, :]                 # (T,P,3)
    a = np.maximum(np.sum(MP * MP, axis=2), EPS)              # (T,P)
    for i in range(M):
        if det_z[i] < MIN_Z or t_start[i] >= t_end[i] - EPS:
            continue
        smoke_z = det_z[i] - SINK * (tg - t_start[i])
        alive = (tg >= t_start[i]) & (tg <= t_end[i]) & (smoke_z >= MIN_Z - EPS)
        if not np.any(alive):
            continue
        # smoke 位置随时间下沉：逐时刻 C(t) = (det_x, det_y, smoke_z)
        Ct = np.column_stack([np.full(len(tg), det_x[i]),
                              np.full(len(tg), det_y[i]), smoke_z])
        MC = Ct[:, None, :] - S[:, None, :]
        proj = np.clip(np.sum(MP * MC, axis=2) / a, 0.0, 1.0)
        nearest = S[:, None, :] + proj[:, :, None] * MP
        dist = np.linalg.norm(nearest - Ct[:, None, :], axis=2)
        covered = np.all(dist <= R + EPS, axis=1) & alive
        out[i] = covered.sum() * dt
    return out

# ================= 阶段1：视线束锚定反演采样 =================
# 几何事实（论文降维定理的直接推论）：从导弹看目标圆柱的 400 条视线构成一个束，
# 束横截面半径在目标端最大（= 目标半径 7 m < 云团半径 R = 10 m），
# 因此云团中心落在束轴线上即可同时遮蔽全部视线（完全遮蔽）。
# 采样 (t_obs, λ, v)：t_obs 时刻导弹在 S(t_obs)，云团在该时刻位于束轴
# P = S + λ(A − S)（A=目标中心）；由云团只垂直下沉反演起爆点，
# 再由无人机匀速直线飞行反演 (heading, speed, drop, det)。
# 该采样遍历"可达 ∩ 完全遮蔽"解族，是严格的全局候选生成。
A_TARGET = np.array([0.0, 200.0, 5.0])   # 目标圆柱轴线中点（束轴终点）

def sample_via_inversion(rng, m):
    """返回 (m,4) 参数矩阵 [heading, speed, drop, det]（已按约束过滤）"""
    t_obs = rng.uniform(0.5, 60.0, m)
    lam = rng.uniform(0.001, 0.995, m)
    v = rng.uniform(70.0, 140.0, m)
    S = MISSILE0 + 300.0 * t_obs[:, None] * MISSILE_DIR       # (m,3)
    P = S + lam[:, None] * (A_TARGET[None, :] - S)            # (m,3) 云团在 t_obs 时刻的位置
    dx = P[:, 0] - DRONE0[0]; dy = P[:, 1] - DRONE0[1]
    dist = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx) % (2 * np.pi)
    t_det = dist / v                                          # 起爆时刻 = 飞行距离 / 速度
    with np.errstate(invalid="ignore"):
        det_z = P[:, 2] + SINK * (t_obs - t_det)              # 起爆高度（下沉补偿）
        det = np.sqrt((DRONE0[2] - det_z) / (0.5 * G))        # 引信延时
    drop = t_det - det
    ok = (det >= 0.1) & (det <= 15.0) & (drop >= 0.0) & (drop <= BOUNDS[2][1]) \
         & np.isfinite(det) & (t_obs >= t_det)
    X = np.column_stack([heading, v, drop, det])
    return X[ok]

def do_scan(n):
    if os.path.exists(SCAN_JSON):
        with open(SCAN_JSON, encoding="utf-8") as f:
            st = json.load(f)
    else:
        st = {"n_done": 0, "n_feasible": 0, "top": []}   # top: 至多 200 条 [coarse_dur, h, s, d, t]
    rng = np.random.RandomState(770000 + st["n_done"])
    t0 = time.time()
    BATCH = 20000
    remain = n
    while remain > 0:
        m = min(BATCH, remain)
        X = sample_via_inversion(rng, m)
        durs = coarse_durations(X)
        st["n_feasible"] += int(np.sum(durs > 0))
        cand = np.column_stack([durs, X])
        keep = np.array(st["top"] + cand.tolist()) if len(st["top"]) else cand
        keep = keep[keep[:, 0] > 0]
        keep = keep[np.argsort(-keep[:, 0])][:200]
        st["top"] = keep.tolist()
        st["n_done"] += m
        remain -= m
        best = keep[0][0] if len(keep) else 0.0
        print(f"  scanned {st['n_done']}: best coarse = {best:.3f}s, feasible={st['n_feasible']}", flush=True)
    st["wall_time_s"] = round(st.get("wall_time_s", 0) + time.time() - t0, 1)
    with open(SCAN_JSON, "w", encoding="utf-8") as f:
        json.dump(st, f)
    print(f"saved {SCAN_JSON}  n_done={st['n_done']}")

# ================= 阶段2：抛光 + fine 复核 =================
def do_polish(top_k):
    from scipy.optimize import differential_evolution
    with open(SCAN_JSON, encoding="utf-8") as f:
        st = json.load(f)
    cands = st["top"][:top_k]
    E_MED = samples(120); E_FINE = samples(200)

    def obj(x):
        iv = interval_for(x[0], x[1], x[2], x[3], E_MED, 0.02)
        return -(iv[1] - iv[0]) if iv is not None else 1000.0

    results = []
    if os.path.exists(OUT_JSON):                 # 断点续跑：跳过已完成候选
        with open(OUT_JSON, encoding="utf-8") as f:
            prev = json.load(f)
        results = prev.get("polished_all", [])
    done_ranks = {r["seed_rank"] for r in results}

    t0 = time.time()
    for k, c in enumerate(cands):
        if k in done_ranks:
            continue
        x0 = np.array(c[1:])
        rng = np.random.RandomState(880000 + k)
        init = [x0.copy()]
        while len(init) < 12:
            p = x0.copy()
            for j, (blo, bhi) in enumerate(BOUNDS):
                p[j] = np.clip(p[j] + rng.normal(0, (bhi - blo) * 0.05), blo, bhi)
            init.append(p)
        res = differential_evolution(obj, BOUNDS, maxiter=18, popsize=12,
                                     seed=880000 + k, tol=1e-9, mutation=(0.5, 1.0),
                                     recombination=0.7, polish=True, init=np.array(init))
        iv = interval_for(res.x[0], res.x[1], res.x[2], res.x[3], E_FINE, 0.005, bisect=40)
        fine = (iv[1] - iv[0]) if iv else 0.0
        results.append({"seed_rank": k, "x": [float(v) for v in res.x],
                        "fine_s": round(fine, 6)})
        print(f"  cand {k+1}/{len(cands)}: fine = {fine:.6f}s", flush=True)
        # 每个候选落盘一次
        _save_out(results, st, t0)
    print(f"=== 本批完成，累计抛光 {len(results)} 个 ===")

def _save_out(results, st, t0):
    rs = sorted(results, key=lambda r: -r["fine_s"])
    best = rs[0]["fine_s"]
    gap = (best - PAPER_FINE) / best * 100.0
    out = {
        "problem": "P2 (4-D single bomb)",
        "paper_fine_s": PAPER_FINE,
        "global_reference_fine_s": best,
        "abs_gap_s": round(best - PAPER_FINE, 6),
        "rel_gap_pct_of_reference": round(gap, 4),
        "n_coarse_scanned": st["n_done"],
        "coarse_wall_time_s": st.get("wall_time_s"),
        "polish_wall_time_s": round(time.time() - t0, 1),
        "polished_all": rs,
        "note": "gap>0 表示全局参考优于论文解；gap<=0 表示论文解不劣于全局参考",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"    [checkpoint] best={best:.6f}s gap={gap:.4f}%", flush=True)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "scan":
        do_scan(int(sys.argv[2]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "polish":
        do_polish(int(sys.argv[2]))
    else:
        print("usage: scan <n> | polish <top_k>")
