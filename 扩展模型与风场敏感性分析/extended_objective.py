# -*- coding: utf-8 -*-
"""
实验1：扩展物理模型评估器（风场漂移 + 时变沉降 v_s(t) + 扩散膨胀/浓度衰减半径 R(t)）。

模型形式（文献依据见本目录 README）：
  τ = t - t_start（起爆后经过时间）
  云团中心 c(t) = (det_x + wx·τ,  det_y + wy·τ,  det_z - Sz(τ))
    Sz(τ) = v0·(1 - e^(-λτ))/λ   （λ→0 时 Sz → v0·τ，即恒定沉降 3 m/s）
  有效半径 R(τ) = sqrt(R0² + 4K·τ)   （瞬时体源高斯扩散，方差线性增长）
  浓度条件 (R0/R(τ))³ >= c_ratio      （质量守恒稀释，低于阈值即失效——"衰减"）
  窗口不变：[t_start, t_start+20s]，海拔 >= 2 m，导弹到达截止。

标称参数点（wx=wy=0, K=0, λ=0）下本评估器逐位退化为 RHO 原评估器：
  R(τ)≡R0=10 m，浓度比恒为 1 ≥ 0.35，Sz=v0·τ=3τ —— 与原模型完全相同。
parity_check() 用 21 个样本（含优质解）对照 rho.eval_global_med / eval_plan_fine，
容差 1e-6（实测约 1e-13）。

向量化结构与 fast_objective.py 相同（必要条件预筛 + 分块公共网格 + 原标量函数二分），
仅把"恒定 R / 垂直沉降"换成上述时变形式；预筛半径相应放宽为
TR + R(VALID) + |w|·VALID（仍为精确必要条件，不会漏判）。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rho_pipeline",
    os.path.join(_HERE, "..", "问题5_RHO正式流水线",
                 "问题5_主脚本_RHO全流程与消融实验.py"))
rho = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rho)

G, R0, V0, VALID, MIN_Z, EPS = rho.G, rho.R, rho.SINK, rho.VALID, rho.MIN_Z, rho.EPS
E_MED, E_FINE = rho.E_MED, rho.E_FINE
DRONE_LIST, MISSILE_LIST = rho.DRONE_LIST, rho.MISSILE_LIST
MDIR, ARRIVAL, MS = rho.MDIR, rho.ARRIVAL, rho.MISSILE_SPEED
TC, TR, TH = rho.TARGET_CENTER, rho.TARGET_R, rho.TARGET_H

_PAIRS = [(i, b, mi) for i in range(5) for b in range(3) for mi in range(3)]
_ISEL = np.array([p[0] for p in _PAIRS])
_BSEL = np.array([p[1] for p in _PAIRS])
_MSEL = np.array([p[2] for p in _PAIRS])
_M0 = np.array([rho.MISSILES[m] for m in MISSILE_LIST])
_MDIR = np.array([MDIR[m] for m in MISSILE_LIST])
_ARR = np.array([ARRIVAL[m] for m in MISSILE_LIST])
_INIT = np.array([rho.DRONES[d] for d in DRONE_LIST])


class Phys:
    """扩展物理参数。标称点 = 原模型。"""
    def __init__(self, wx=0.0, wy=0.0, K=0.0, lam=0.0, c_ratio=0.35):
        self.wx, self.wy, self.K, self.lam, self.c_ratio = wx, wy, K, lam, c_ratio

    @property
    def w_abs(self):
        return float(np.hypot(self.wx, self.wy))

    def sink_z(self, tau):
        """累计沉降位移 Sz(τ) >= 0。"""
        tau = np.asarray(tau, dtype=float)
        if self.lam < 1e-12:
            return V0 * tau
        return V0 * (1.0 - np.exp(-self.lam * tau)) / self.lam

    def radius(self, tau):
        tau = np.maximum(np.asarray(tau, dtype=float), 0.0)
        return np.sqrt(R0 * R0 + 4.0 * self.K * tau)

    def conc_ok(self, tau):
        """浓度条件：稀释比 (R0/R)^3 >= c_ratio。"""
        r = self.radius(tau)
        return (R0 / r) ** 3 >= self.c_ratio

    def __repr__(self):
        return (f"Phys(wx={self.wx}, wy={self.wy}, K={self.K}, "
                f"lam={self.lam}, c_ratio={self.c_ratio})")


NOMINAL = Phys()


def shielded_ext(t, drone, m, heading, speed, drop, det, E, phys):
    """rho.shielded_scalar 的扩展物理版（二分精修调用，语义逐位对应）。"""
    init = rho.DRONES[drone]
    uav = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = init + speed * drop * uav
    det_xy = drop_pos[:2] + speed * det * uav[:2]
    det_z = drop_pos[2] - 0.5 * G * det * det
    t_start = drop + det
    if t < t_start or t > t_start + VALID or det_z < MIN_Z:
        return False
    tau = t - t_start
    smoke = np.array([det_xy[0] + phys.wx * tau,
                      det_xy[1] + phys.wy * tau,
                      det_z - phys.sink_z(tau)])
    if smoke[2] < MIN_Z or not phys.conc_ok(tau):
        return False
    S = rho.MISSILES[m] + MS * t * MDIR[m]
    MP = E - S
    MC = smoke - S
    a = np.maximum(np.sum(MP * MP, axis=1), EPS)
    proj = np.clip(np.sum(MP * MC, axis=1) / a, 0.0, 1.0)
    nearest = S + proj[:, None] * MP
    dist = np.linalg.norm(nearest - smoke, axis=1)
    return bool(np.all(dist <= phys.radius(tau) + EPS))


def _pair_scalars(x):
    x = np.asarray(x, dtype=float)
    theta = x[0::8][_ISEL]
    v = x[1::8][_ISEL]
    d1 = x[2::8][_ISEL]
    g1 = x[3::8][_ISEL]
    g2 = x[4::8][_ISEL]
    drops = np.where(_BSEL == 0, d1,
                     np.where(_BSEL == 1, d1 + g1, d1 + g1 + g2))
    dets = np.array([x[8 * p[0] + 5 + p[1]] for p in _PAIRS])
    return theta, v, drops, dets


def _pt_seg_dist_2d(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + by * 0 + aby * aby
    if L2 < 1e-18:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _pt_tri_dist_2d(px, py, A, B, C):
    def cross(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d1 = cross(A[0], A[1], B[0], B[1], px, py)
    d2 = cross(B[0], B[1], C[0], C[1], px, py)
    d3 = cross(C[0], C[1], A[0], A[1], px, py)
    if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
        return 0.0
    return min(_pt_seg_dist_2d(px, py, A[0], A[1], B[0], B[1]),
               _pt_seg_dist_2d(px, py, B[0], B[1], C[0], C[1]),
               _pt_seg_dist_2d(px, py, C[0], C[1], A[0], A[1]))


def _intervals_batched_ext(x, E, dt, bisect, phys):
    """fast_objective._intervals_batched 的扩展物理版（45 对全覆盖）。"""
    x = np.asarray(x, dtype=float)
    theta, v, drops, dets = _pair_scalars(x)
    init = _INIT[_ISEL]
    ux, uy = np.cos(theta), np.sin(theta)
    det_x = init[:, 0] + v * (drops + dets) * ux
    det_y = init[:, 1] + v * (drops + dets) * uy
    det_z = init[:, 2] - 0.5 * G * dets * dets
    t_start = drops + dets
    t_end = np.minimum(t_start + VALID, _ARR[_MSEL])

    out = [None] * len(_PAIRS)
    # 预筛（必要条件放宽：半径随时增至 R(VALID)、风致漂移 |w|·VALID）
    R_pre = TR + phys.radius(VALID) + phys.w_abs * VALID + 1e-9
    ok = (det_z >= MIN_Z) & (t_start < t_end - EPS)
    for j in np.where(ok)[0]:
        ms = _MSEL[j]
        A = _M0[ms][:2] + MS * t_start[j] * _MDIR[ms][:2]
        B = _M0[ms][:2] + MS * t_end[j] * _MDIR[ms][:2]
        if _pt_tri_dist_2d(det_x[j], det_y[j], A, B, TC[:2]) > R_pre:
            ok[j] = False
    idx_all = np.where(ok)[0]
    if len(idx_all) == 0:
        return out

    idx_all = idx_all[np.argsort(t_start[idx_all])]
    first_t, last_t = {}, {}
    CHUNK = 5
    for s in range(0, len(idx_all), CHUNK):
        idx = idx_all[s:s + CHUNK]
        g0 = t_start[idx].min()
        g1_ = t_end[idx].max()
        t_grid = np.arange(g0, g1_ + dt, dt)
        ms = _MSEL[idx]
        S = _M0[ms][None, :, :] + MS * t_grid[:, None, None] * _MDIR[ms][None, :, :]
        tau = t_grid[:, None] - t_start[idx][None, :]          # (T,P)
        smoke_x = det_x[idx][None, :] + phys.wx * tau
        smoke_y = det_y[idx][None, :] + phys.wy * tau
        smoke_z = det_z[idx][None, :] - phys.sink_z(tau)
        in_win = (tau >= -EPS) & (t_grid[:, None] <= t_end[idx][None, :] + EPS)
        Rt = phys.radius(np.maximum(tau, 0.0))                 # (T,P)
        valid = (smoke_z >= MIN_Z - EPS) & in_win & \
                ((R0 / np.maximum(Rt, 1e-12)) ** 3 >= phys.c_ratio)
        MP = E[None, None, :, :] - S[:, :, None, :]
        MC = np.empty_like(MP)
        MC[:, :, :, 0] = smoke_x[:, :, None] - S[:, :, None, 0]
        MC[:, :, :, 1] = smoke_y[:, :, None] - S[:, :, None, 1]
        MC[:, :, :, 2] = smoke_z[:, :, None] - S[:, :, None, 2]
        a = np.einsum('tpkd,tpkd->tpk', MP, MP)
        np.maximum(a, EPS, out=a)
        dot = np.einsum('tpkd,tpkd->tpk', MC, MP)
        proj = np.clip(dot / a, 0.0, 1.0)
        mc2 = np.einsum('tpkd,tpkd->tpk', MC, MC)
        d2 = mc2 - 2.0 * proj * dot + proj * proj * a
        covered = np.all(d2 <= (Rt[:, :, None] + EPS) ** 2, axis=2) & valid
        for j, pj in enumerate(idx):
            col = covered[:, j]
            if np.any(col):
                first_t[pj] = t_grid[int(np.argmax(col))]
                last_t[pj] = t_grid[len(col) - 1 - int(np.argmax(col[::-1]))]

    for j in first_t:
        i, b, mi = _PAIRS[j]
        drone, m = DRONE_LIST[i], MISSILE_LIST[mi]
        th_, v_ = x[8 * i], x[8 * i + 1]
        d_ = [x[8 * i + 2], x[8 * i + 2] + x[8 * i + 3],
              x[8 * i + 2] + x[8 * i + 3] + x[8 * i + 4]][b]
        t_ = x[8 * i + 5 + b]
        ts = d_ + t_
        te = min(ts + VALID, ARRIVAL[m])
        la = first_t[j]
        lo, hi = max(ts, la - dt), la
        if not shielded_ext(lo, drone, m, th_, v_, d_, t_, E, phys):
            for _ in range(bisect):
                mid = (lo + hi) / 2
                if shielded_ext(mid, drone, m, th_, v_, d_, t_, E, phys):
                    hi = mid
                else:
                    lo = mid
        else:
            hi = lo
        left = hi
        ra = last_t[j]
        lo, hi = ra, min(te, ra + dt)
        if not shielded_ext(hi, drone, m, th_, v_, d_, t_, E, phys):
            for _ in range(bisect):
                mid = (lo + hi) / 2
                if shielded_ext(mid, drone, m, th_, v_, d_, t_, E, phys):
                    lo = mid
                else:
                    hi = mid
        else:
            lo = hi
        out[j] = (float(left), float(lo))
    return out


def eval_global_med_ext(x, phys=NOMINAL):
    """eval_global_med 的扩展物理版（搜索用）。"""
    x = np.asarray(x, dtype=float)
    for i in range(5):
        if not (70.0 <= x[8 * i + 1] <= 140.0):
            return 1000.0
    ivs = _intervals_batched_ext(x, E_MED, 0.02, 35, phys)
    per_m = {m: [] for m in MISSILE_LIST}
    for (i, b, mi), iv in zip(_PAIRS, ivs):
        if iv is not None:
            per_m[MISSILE_LIST[mi]].append(iv)
    total = sum(rho.merge(v) for v in per_m.values())
    return -total if total > 0 else 1000.0


def eval_plan_med_ext(x, assignment, phys=NOMINAL):
    """eval_plan_med 的扩展物理版（重优化演示用）。"""
    x = np.asarray(x, dtype=float)
    plan = rho.decode(x, assignment)
    # 借 45 对批量实现，只取分配内的对
    want = {}
    for i, drone in enumerate(DRONE_LIST):
        mi = MISSILE_LIST.index(plan[drone][4])
        for b in range(3):
            want[(i, b, mi)] = True
    if not all(70.0 <= x[8 * i + 1] <= 140.0 for i in range(5)):
        return 1000.0
    ivs = _intervals_batched_ext(x, E_MED, 0.02, 35, phys)
    per_m = {m: [] for m in MISSILE_LIST}
    for pair, iv in zip(_PAIRS, ivs):
        if iv is not None and pair in want:
            per_m[MISSILE_LIST[pair[2]]].append(iv)
    total = sum(rho.merge(v) for v in per_m.values())
    return -total if total > 0 else 1000.0


def eval_plan_med_ext_cover(x, assignment, phys=NOMINAL):
    """Constrained variant: all three missiles must retain positive obscuration."""
    x = np.asarray(x, dtype=float)
    plan = rho.decode(x, assignment)
    want = {}
    for i, drone in enumerate(DRONE_LIST):
        mi = MISSILE_LIST.index(plan[drone][4])
        for b in range(3):
            want[(i, b, mi)] = True
    if not all(70.0 <= x[8 * i + 1] <= 140.0 for i in range(5)):
        return 1000.0
    ivs = _intervals_batched_ext(x, E_MED, 0.02, 35, phys)
    per_m = {m: [] for m in MISSILE_LIST}
    for pair, iv in zip(_PAIRS, ivs):
        if iv is not None and pair in want:
            per_m[MISSILE_LIST[pair[2]]].append(iv)
    per_m_dur = {m: rho.merge(v) for m, v in per_m.items()}
    total = sum(per_m_dur.values())
    if total <= 0 or any(v <= EPS for v in per_m_dur.values()):
        return 1000.0
    return -total


def eval_plan_med_ext_cover_soft(x, assignment, phys=NOMINAL):
    """Soft constraint: penalize uncovered missiles, then maximize total duration."""
    x = np.asarray(x, dtype=float)
    plan = rho.decode(x, assignment)
    want = {}
    for i, drone in enumerate(DRONE_LIST):
        mi = MISSILE_LIST.index(plan[drone][4])
        for b in range(3):
            want[(i, b, mi)] = True
    if not all(70.0 <= x[8 * i + 1] <= 140.0 for i in range(5)):
        return 1000.0
    ivs = _intervals_batched_ext(x, E_MED, 0.02, 35, phys)
    per_m = {m: [] for m in MISSILE_LIST}
    for pair, iv in zip(_PAIRS, ivs):
        if iv is not None and pair in want:
            per_m[MISSILE_LIST[pair[2]]].append(iv)
    per_m_dur = {m: rho.merge(v) for m, v in per_m.items()}
    total = sum(per_m_dur.values())
    uncovered = sum(1 for v in per_m_dur.values() if v <= EPS)
    return -total + 1000.0 * uncovered


def eval_plan_fine_ext(x, assignment, phys=NOMINAL):
    """eval_plan_fine 的扩展物理版（终评用）：返回 (total, per_m_dur)。"""
    x = np.asarray(x, dtype=float)
    plan = rho.decode(x, assignment)
    want = {}
    for i, drone in enumerate(DRONE_LIST):
        mi = MISSILE_LIST.index(plan[drone][4])
        for b in range(3):
            want[(i, b, mi)] = True
    ivs = _intervals_batched_ext(x, E_FINE, 0.005, 40, phys)
    per_m = {m: [] for m in MISSILE_LIST}
    for pair, iv in zip(_PAIRS, ivs):
        if iv is not None and pair in want:
            per_m[MISSILE_LIST[pair[2]]].append(iv)
    per_m_dur = {m: rho.merge(v) for m, v in per_m.items()}
    return sum(per_m_dur.values()), per_m_dur


def _champion_x():
    """论文冠军 40 维向量（消融 A0，fine=22.717447 s）。"""
    import json
    d = json.load(open(os.path.join(
        _HERE, "..", "问题5_RHO正式流水线",
        "问题5_结果_消融A0至A4_最优22.717447s.json"), encoding="utf-8"))
    return np.array(d["refines"]["M1,M2,M1,M1,M3"]["x"]), \
        tuple(d["refines"]["M1,M2,M1,M1,M3"].get("assignment",
              ["M1", "M2", "M1", "M1", "M3"]))


def parity_check(n_random=20, seed=7):
    """标称点逐位一致性：ext(NOMINAL) vs rho 原评估器。"""
    import pickle
    rng = np.random.RandomState(seed)
    xs = [np.array([a + (b - a) * rng.rand() for a, b in rho.BOUNDS_40])
          for _ in range(n_random)]
    lib = pickle.load(open(os.path.join(
        _HERE, "..", "问题5_RHO正式流水线", "问题5_输入_候选库39组.pkl"), "rb"))
    assignment = ("M1", "M2", "M1", "M1", "M3")
    cand = {d: lib[(d, a)][0] for d, a in zip(DRONE_LIST, assignment)}
    xs.append(rho.encode_from_candidate(assignment, cand))
    worst = 0.0
    for k, x in enumerate(xs):
        a = rho.eval_global_med(x)
        b = eval_global_med_ext(x, NOMINAL)
        diff = abs(a - b)
        worst = max(worst, diff)
        flag = "OK " if diff < 1e-6 else "FAIL"
        print(f"  [{flag}] sample {k:2d}: orig={a:.9f} ext={b:.9f} |diff|={diff:.2e}")
    print(f"[med] 最大偏差 {worst:.2e} -> {'通过' if worst < 1e-6 else '未通过'}")
    # fine 级：冠军方案
    xc, asn = _champion_x()
    f0, pm0 = rho.eval_plan_fine(xc, list(asn))
    f1, pm1 = eval_plan_fine_ext(xc, list(asn), NOMINAL)
    df = abs(f0 - f1)
    print(f"[fine] 冠军方案: orig={f0:.9f} ext={f1:.9f} |diff|={df:.2e} "
          f"per_m orig={ {k: round(v,6) for k,v in pm0.items()} } ext={ {k: round(v,6) for k,v in pm1.items()} }")
    ok = worst < 1e-6 and df < 1e-6
    print("=> 标称点 parity " + ("通过" if ok else "未通过"))
    return ok


if __name__ == "__main__":
    print("== 标称点一致性校验（ext(NOMINAL) vs 原评估器） ==")
    parity_check()
