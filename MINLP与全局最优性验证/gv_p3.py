# -*- coding: utf-8 -*-
"""
实验2 · 小规模全局最优性验证（问题3：8维三弹并集）
=================================================================
采样：弹1用视线束锚定反演（与 P2 相同，决定航向/速度/首弹）；
      弹2/3 在同一射线轨迹上投影锚定（共享航向与速度），拒绝间隔<1s 样本。
抛光：论文同款时间映射编码 8 维 DE（全界，无答案导向窄化）。
对比论文值 6.697052 s（fine），报告 optimality gap。

用法：
  python gv_p3.py scan  <n_coarse>
  python gv_p3.py polish <top_k>
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gv_p2 import (samples, shielded_scalar, interval_for, sample_via_inversion,
                   MISSILE0, MISSILE_DIR, DRONE0, ARRIVAL, A_TARGET,
                   G, R, SINK, VALID, MIN_Z, EPS)

SCAN_JSON = os.path.join(HERE, "gv_p3_scan.json")
OUT_JSON = os.path.join(HERE, "gv_p3_result.json")
PAPER_FINE = 6.697052   # 论文问题3最优（fine，并集）
BOUNDS_DET = (0.1, 15.0)
DROP_MAX = ARRIVAL * 0.8

# ---------- 三弹反演采样 ----------
def sample_p3(rng, m, K=40):
    X1 = sample_via_inversion(rng, m)          # 弹1锚定 (m1,4)，决定射线(航向)与速度
    out = []
    for row in X1:
        h, v, d1, t1 = row
        u = np.array([np.cos(h), np.sin(h)])
        for _ in range(K):                      # 同一射线多次尝试补齐弹2/3
            bombs = [(d1, t1)]
            ok = True
            for _ in range(2):                  # 弹2/3 投影锚定
                t_obs = d1 + t1 + 3.0 + rng.uniform(-20.0, 20.0)
                lam = rng.uniform(0.001, 0.995)
                S = MISSILE0 + 300.0 * t_obs * MISSILE_DIR
                P = S + lam * (A_TARGET - S)
                d_k = float(np.dot(P[:2] - DRONE0[:2], u))   # 射线投影距离
                if d_k <= 0:
                    ok = False; break
                t_det = d_k / v
                det_z = P[2] + SINK * (t_obs - t_det)
                q = (DRONE0[2] - det_z) / (0.5 * G)
                if q <= 0:
                    ok = False; break
                det = float(np.sqrt(q))
                drop = t_det - det
                if not (BOUNDS_DET[0] <= det <= BOUNDS_DET[1] and 0.0 <= drop <= DROP_MAX
                        and t_obs >= t_det):
                    ok = False; break
                bombs.append((drop, det))
            if not ok:
                continue
            ds = sorted(bombs, key=lambda b: b[0])
            if ds[1][0] - ds[0][0] < 1.0 or ds[2][0] - ds[1][0] < 1.0:
                continue                        # 投放间隔硬约束 ≥1 s
            out.append([h, v, ds[0][0], ds[0][1], ds[1][0], ds[1][1], ds[2][0], ds[2][1]])
            break                               # 每条射线保留一个可行三弹组合
    return np.array(out) if out else np.empty((0, 8))

# ---------- 向量化三弹并集粗扫（dt=0.1，无二分） ----------
E_COARSE = samples(40)
_DT = 0.1
_TG = np.arange(0.0, ARRIVAL + _DT, _DT)
_S = MISSILE0 + 300.0 * _TG[:, None] * MISSILE_DIR
_MP = E_COARSE[None, :, :] - _S[:, None, :]
_A2 = np.maximum(np.sum(_MP * _MP, axis=2), EPS)

def _covered_one(h, v, drop, det):
    u = np.array([np.cos(h), np.sin(h)])
    det_xy = DRONE0[:2] + v * (drop + det) * u
    det_z = DRONE0[2] - 0.5 * G * det * det
    t_start = drop + det
    t_end = min(t_start + VALID, ARRIVAL)
    if det_z < MIN_Z or t_start >= t_end - EPS:
        return None
    smoke_z = det_z - SINK * (_TG - t_start)
    alive = (_TG >= t_start) & (_TG <= t_end) & (smoke_z >= MIN_Z - EPS)
    if not np.any(alive):
        return None
    Ct = np.column_stack([np.full(len(_TG), det_xy[0]),
                          np.full(len(_TG), det_xy[1]), smoke_z])
    MC = Ct[:, None, :] - _S[:, None, :]
    proj = np.clip(np.sum(_MP * MC, axis=2) / _A2, 0.0, 1.0)
    nearest = _S[:, None, :] + proj[:, :, None] * _MP
    dist = np.linalg.norm(nearest - Ct[:, None, :], axis=2)
    return np.all(dist <= R + EPS, axis=1) & alive

def coarse_union(X):
    out = np.zeros(len(X))
    for i, row in enumerate(X):
        cov = None
        for k in range(3):
            c = _covered_one(row[0], row[1], row[2 + 2 * k], row[3 + 2 * k])
            if c is not None:
                cov = c if cov is None else (cov | c)
        if cov is not None:
            out[i] = cov.sum() * _DT
    return out

# ================= 阶段1：从 P2 单弹池组三弹候选 =================
def do_triples():
    """读取 P2 视线束锚定粗扫池（gv_p2_scan.json），按 (航向,速度) 分桶组三弹组合，
    粗扫并集时长，保留 Top-200 写入 gv_p3_scan.json"""
    from collections import defaultdict
    with open(os.path.join(HERE, "gv_p2_scan.json"), encoding="utf-8") as f:
        st2 = json.load(f)
    singles = [r for r in st2["top"] if r[0] > 0.0]
    print(f"P2 池单弹候选: {len(singles)}")
    groups = defaultdict(list)
    for r in singles:
        h, v, d, t = r[1], r[2], r[3], r[4]
        groups[(int(np.degrees(h)) // 5, int(v) // 10)].append((h, v, d, t, r[0]))
    cands = []
    for key, g in groups.items():
        if len(g) < 3:
            continue
        g.sort(key=lambda z: -z[4])
        h0, v0 = g[0][0], g[0][1]
        adj = []
        for (h, v, d, t, dur) in g[:12]:          # 组内最优12枚参与组合
            dist = v * (d + t)                     # 保持起爆点，重算共享速度下的 drop
            d_new = dist / v0 - t
            if d_new >= 0.0:
                adj.append((d_new, t))
        from itertools import combinations
        for combo in combinations(adj, 3):
            ds = sorted(combo, key=lambda b: b[0])
            if ds[1][0] - ds[0][0] < 1.0 or ds[2][0] - ds[1][0] < 1.0:
                continue
            cands.append([h0, v0, ds[0][0], ds[0][1], ds[1][0], ds[1][1], ds[2][0], ds[2][1]])
    print(f"三弹候选: {len(cands)}")
    t0 = time.time()
    durs = coarse_union(np.array(cands)) if cands else np.array([])
    order = np.argsort(-durs)[:200]
    top = [[float(durs[i])] + [float(v) for v in cands[i]] for i in order if durs[i] > 0]
    st = {"n_done": len(cands), "n_feasible": int(np.sum(durs > 0)),
          "source": "P2 scan pool triples", "top": top,
          "wall_time_s": round(time.time() - t0, 1)}
    with open(SCAN_JSON, "w", encoding="utf-8") as f:
        json.dump(st, f)
    best = top[0][0] if top else 0.0
    print(f"saved {SCAN_JSON}  best coarse union = {best:.3f}s  feasible={st['n_feasible']}")

# ================= 阶段2：8维DE抛光 + fine 复核 =================
BOUNDS_8 = [(0.0, 2 * np.pi), (70.0, 140.0), (0.0, DROP_MAX),
            (1.0, 5.0), (1.0, 5.0), BOUNDS_DET, BOUNDS_DET, BOUNDS_DET]

def decode8(x):
    """时间映射编码 -> (h, v, drops[3], dets[3])"""
    h, v, d1, g1, g2 = x[0], x[1], x[2], x[3], x[4]
    return h, v, [d1, d1 + g1, d1 + g1 + g2], [x[5], x[6], x[7]]

def encode8(row):
    """粗扫原始行 [h,v,d1,t1,d2,t2,d3,t3]（drops 已升序）-> 编码"""
    h, v = row[0], row[1]
    d1, d2, d3 = row[2], row[4], row[6]
    return np.array([h, v, d1, d2 - d1, d3 - d2, row[3], row[5], row[7]])

def eval_union(x, E, dt, bisect):
    h, v, drops, dets = decode8(x)
    ivs = []
    for d, t in zip(drops, dets):
        iv = interval_for(h, v, d, t, E, dt, bisect=bisect)
        if iv is not None:
            ivs.append(iv)
    if not ivs:
        return 0.0
    ivs.sort()
    total, cs, ce = 0.0, ivs[0][0], ivs[0][1]
    for s, e in ivs[1:]:
        if s <= ce + EPS:
            ce = max(ce, e)
        else:
            total += ce - cs; cs, ce = s, e
    return total + (ce - cs)

def _save_out(results, st, t0):
    rs = sorted(results, key=lambda r: -r["fine_s"])
    best = rs[0]["fine_s"]
    gap = (best - PAPER_FINE) / best * 100.0
    out = {
        "problem": "P3 (8-D single UAV, 3 bombs, union)",
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

def do_polish(top_k):
    from scipy.optimize import differential_evolution
    with open(SCAN_JSON, encoding="utf-8") as f:
        st = json.load(f)
    cands = st["top"][:top_k]
    E_MED = samples(120); E_FINE = samples(200)

    def obj(x):
        return -eval_union(x, E_MED, 0.02, 35)

    results = []
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            results = json.load(f).get("polished_all", [])
    done_ranks = {r["seed_rank"] for r in results}

    t0 = time.time()
    for k, c in enumerate(cands):
        if k in done_ranks:
            continue
        x0 = encode8(np.array(c[1:]))
        rng = np.random.RandomState(550000 + k)
        init = [x0.copy()]
        while len(init) < 16:
            p = x0.copy()
            for j, (blo, bhi) in enumerate(BOUNDS_8):
                p[j] = np.clip(p[j] + rng.normal(0, (bhi - blo) * 0.05), blo, bhi)
            init.append(p)
        res = differential_evolution(obj, BOUNDS_8, maxiter=12, popsize=8,
                                     seed=550000 + k, tol=1e-9, mutation=(0.5, 1.0),
                                     recombination=0.7, polish=False, init=np.array(init))
        fine = eval_union(res.x, E_FINE, 0.005, 40)
        results.append({"seed_rank": k, "x": [float(v) for v in res.x],
                        "fine_s": round(fine, 6)})
        print(f"  cand {k+1}/{len(cands)}: fine = {fine:.6f}s", flush=True)
        _save_out(results, st, t0)
    print(f"=== 本批完成，累计抛光 {len(results)} 个 ===")

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "triples":
        do_triples()
    elif len(sys.argv) >= 2 and sys.argv[1] == "polish":
        do_polish(int(sys.argv[2]))
    else:
        print("usage: triples | polish <top_k>")
