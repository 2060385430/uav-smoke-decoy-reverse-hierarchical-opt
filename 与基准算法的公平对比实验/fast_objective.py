# -*- coding: utf-8 -*-
"""
medium 精度目标函数的加速版（数学语义与 RHO 主脚本 eval_global_med 完全一致）。
两级加速，均为"必要条件预筛 + 等价向量化"，不改变任何输出：
  1. 预筛（精确必要条件，绝无漏判）：
     - 水平：|det_xy - (0,200)| <= TARGET_R + R 才可能覆盖（遮蔽要求云团
       到某视线段距离 <= R，而视线段端点 E_k 必在目标圆周上）；
     - 竖直：云团 z 区间须与 [-R, TARGET_H + R] 有交。
  2. 通过预筛的 (弹,导弹) 对按 t_start 排序后分块，在块内用公共时间网格
     + 窗口掩码批量计算覆盖（块内窗口跨度小，计算量≈实际活跃窗口）。
二分精修仍调用原始 shielded_scalar，保证区间端点逐位一致。
运行本脚本即执行一致性校验（容差 1e-6 s）与耗时对比。
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

G, R_, SINK, VALID, MIN_Z, EPS = rho.G, rho.R, rho.SINK, rho.VALID, rho.MIN_Z, rho.EPS
E_MED = rho.E_MED                      # (240, 3)
DRONE_LIST, MISSILE_LIST = rho.DRONE_LIST, rho.MISSILE_LIST
MDIR, ARRIVAL = rho.MDIR, rho.ARRIVAL
MS = rho.MISSILE_SPEED
TC = rho.TARGET_CENTER                 # (0, 200, 0)
TR, TH = rho.TARGET_R, rho.TARGET_H
shielded_scalar = rho.shielded_scalar

_PAIRS = [(i, b, mi) for i in range(5) for b in range(3) for mi in range(3)]
_ISEL = np.array([p[0] for p in _PAIRS])
_BSEL = np.array([p[1] for p in _PAIRS])
_MSEL = np.array([p[2] for p in _PAIRS])
_M0 = np.array([rho.MISSILES[m] for m in MISSILE_LIST])      # (3,3)
_MDIR = np.array([MDIR[m] for m in MISSILE_LIST])            # (3,3)
_ARR = np.array([ARRIVAL[m] for m in MISSILE_LIST])          # (3,)
_INIT = np.array([rho.DRONES[d] for d in DRONE_LIST])        # (5,3)

# 预筛半径：TARGET_R + R（必要条件，见模块 docstring）
R_PRE = TR + R_ + 1e-9


def _pair_scalars(x):
    """40 维向量 -> 45 对的标量参数。"""
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
    """点到 2D 线段的距离（标量）。"""
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    if L2 < 1e-18:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _pt_tri_dist_2d(px, py, A, B, C):
    """点到 2D 三角形 ABC 的距离（内部为 0，标量）。"""
    # 重心符号法判断是否在三角形内
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


def _intervals_batched(x, dt=0.02):
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
    # ---- 精确必要条件预筛（绝不漏判） ----
    # 3D 距离 <= R => 水平距离 <= R；云团活跃窗内导弹视线段的水平投影落在
    # 三角形(M_xy(t_start), M_xy(t_end), 目标圆心) 外扩 TARGET_R 的区域内，
    # 故 det_xy 到该三角形的距离 > TARGET_R + R 时不可能覆盖。
    ok = (det_z >= MIN_Z) & (t_start < t_end - EPS)
    for j in np.where(ok)[0]:
        ms = _MSEL[j]
        A = _M0[ms][:2] + MS * t_start[j] * _MDIR[ms][:2]
        B = _M0[ms][:2] + MS * t_end[j] * _MDIR[ms][:2]
        if _pt_tri_dist_2d(det_x[j], det_y[j], A, B, TC[:2]) > R_PRE:
            ok[j] = False
    idx_all = np.where(ok)[0]
    if len(idx_all) == 0:
        return out

    # ---- 按 t_start 排序分块，块内公共网格 ----
    idx_all = idx_all[np.argsort(t_start[idx_all])]
    first_t = {}
    last_t = {}
    CHUNK = 5
    for s in range(0, len(idx_all), CHUNK):
        idx = idx_all[s:s + CHUNK]
        g0 = t_start[idx].min()
        g1_ = t_end[idx].max()
        t_grid = np.arange(g0, g1_ + dt, dt)
        T = len(t_grid)
        ms = _MSEL[idx]
        S = _M0[ms][None, :, :] + MS * t_grid[:, None, None] * _MDIR[ms][None, :, :]
        smoke_z = det_z[idx][None, :] - SINK * (t_grid[:, None] - t_start[idx][None, :])
        in_win = (t_grid[:, None] >= t_start[idx][None, :] - EPS) & \
                 (t_grid[:, None] <= t_end[idx][None, :] + EPS)
        valid_z = (smoke_z >= MIN_Z - EPS) & in_win
        # dist^2 = |MC|^2 - 2*proj*(MC.MP) + proj^2*|MP|^2, proj=clip((MC.MP)/|MP|^2,0,1)
        MP = E_MED[None, None, :, :] - S[:, :, None, :]       # (T,P,240,3)
        MC = np.empty_like(MP)
        MC[:, :, :, 0] = det_x[idx][None, :, None] - S[:, :, None, 0]
        MC[:, :, :, 1] = det_y[idx][None, :, None] - S[:, :, None, 1]
        MC[:, :, :, 2] = smoke_z[:, :, None] - S[:, :, None, 2]
        a = np.einsum('tpkd,tpkd->tpk', MP, MP)
        np.maximum(a, EPS, out=a)
        dot = np.einsum('tpkd,tpkd->tpk', MC, MP)
        proj = np.clip(dot / a, 0.0, 1.0)
        mc2 = np.einsum('tpkd,tpkd->tpk', MC, MC)
        d2 = mc2 - 2.0 * proj * dot + proj * proj * a
        covered = np.all(d2 <= (R_ + EPS) ** 2, axis=2) & valid_z
        for j, pj in enumerate(idx):
            col = covered[:, j]
            if np.any(col):
                first_t[pj] = t_grid[int(np.argmax(col))]
                last_t[pj] = t_grid[len(col) - 1 - int(np.argmax(col[::-1]))]

    # ---- 二分精修（仅覆盖对；调用原始标量函数） ----
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
        if not shielded_scalar(lo, drone, m, th_, v_, d_, t_, E_MED):
            for _ in range(35):
                mid = (lo + hi) / 2
                if shielded_scalar(mid, drone, m, th_, v_, d_, t_, E_MED):
                    hi = mid
                else:
                    lo = mid
        else:
            hi = lo
        left = hi
        ra = last_t[j]
        lo, hi = ra, min(te, ra + dt)
        if not shielded_scalar(hi, drone, m, th_, v_, d_, t_, E_MED):
            for _ in range(35):
                mid = (lo + hi) / 2
                if shielded_scalar(mid, drone, m, th_, v_, d_, t_, E_MED):
                    lo = mid
                else:
                    hi = mid
        else:
            lo = hi
        out[j] = (float(left), float(lo))
    return out


def eval_med_fast(x):
    """eval_global_med 的加速版，返回相同标量。"""
    x = np.asarray(x, dtype=float)
    for i in range(5):
        if not (70.0 <= x[8 * i + 1] <= 140.0):
            return 1000.0
    ivs = _intervals_batched(x, dt=0.02)
    per_m = {m: [] for m in MISSILE_LIST}
    for (i, b, mi), iv in zip(_PAIRS, ivs):
        if iv is not None:
            per_m[MISSILE_LIST[mi]].append(iv)
    total = sum(rho.merge(v) for v in per_m.values())
    return -total if total > 0 else 1000.0


def _champion_like_x():
    """由候选库构造一个优质 40 维向量（触发大量覆盖/二分路径）。"""
    import pickle
    lib = pickle.load(open(os.path.join(
        _HERE, "..", "问题5_RHO正式流水线",
        "问题5_输入_候选库39组.pkl"), "rb"))
    assignment = ("M1", "M2", "M1", "M1", "M3")
    cand = {d: lib[(d, a)][0] for d, a in zip(DRONE_LIST, assignment)}
    return rho.encode_from_candidate(assignment, cand)


def parity_check(n_random=20, seed=7):
    rng = np.random.RandomState(seed)
    xs = [np.array([a + (b - a) * rng.rand() for a, b in rho.BOUNDS_40])
          for _ in range(n_random)]
    xs.append(_champion_like_x())
    worst = 0.0
    for k, x in enumerate(xs):
        a = rho.eval_global_med(x)
        b = eval_med_fast(x)
        diff = abs(a - b)
        worst = max(worst, diff)
        flag = "OK " if diff < 1e-6 else "FAIL"
        print(f"  [{flag}] sample {k:2d}: orig={a:.9f} fast={b:.9f} |diff|={diff:.2e}")
    print(f"最大偏差 {worst:.2e} -> {'通过' if worst < 1e-6 else '未通过'}")
    return worst < 1e-6


if __name__ == "__main__":
    import time
    print("== 一致性校验 ==")
    parity_check()
    print("== 耗时对比（随机解 / 优质解） ==")
    rng = np.random.RandomState(1)
    xr = np.array([a + (b - a) * rng.rand() for a, b in rho.BOUNDS_40])
    xc = _champion_like_x()
    for name, x in [("随机解", xr), ("优质解", xc)]:
        t0 = time.time(); rho.eval_global_med(x); t1 = time.time()
        eval_med_fast(x); t2 = time.time()
        print(f"  {name}: orig {t1-t0:.3f}s / fast {t2-t1:.3f}s "
              f"(加速 {(t1-t0)/max(t2-t1,1e-9):.1f}x)")
