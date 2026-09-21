# -*- coding: utf-8 -*-
"""
问题3 重跑确认：主求解（8维DE + 时间映射） + fine 验证 + 罚函数对照
=====================================================================
由 problem3_solver.py 复制精简而来（不改动原文件、不覆盖 result1.xlsx / problem3_intervals.png）：
  - 保留：主求解 DE（seed=42，与原脚本完全相同的参数与启发式初始化）+ fine 高精度验证
  - 保留：检验1「时间映射约束有效性」（强制间隔 g=1.0 s 边界的并集对照，纯评估、不重跑 DE）
  - 删除：xlsx / png 落盘、边际效应分析、约束逐项验证等打印型检验
  - 结果改存 problem3_rerun_main_results.json
物理模型与评估函数逐行复制自 problem3_solver.py。
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution

OUT = Path(__file__).resolve().parent

# ========================= 物理常量 =========================
G = 9.8
R = 10.0
SINK = 3.0
VALID = 20.0
MIN_Z = 2.0
EPS = 1e-12
MIN_DROP_INTERVAL = 1.0
MIN_DET_DELAY = 0.1

TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TARGET_R = 7.0
TARGET_H = 10.0
DRONE0 = np.array([17800.0, 0.0, 1800.0])
MISSILE0 = np.array([20000.0, 0.0, 2000.0])
MISSILE_DIR = -MISSILE0 / np.linalg.norm(MISSILE0)
ARRIVAL = np.linalg.norm(MISSILE0) / 300.0

# 答案基准（用于对比）
ANSWER_UNION = 6.542895


# ========================= 采样与遮蔽判定 =========================
def generate_samples(n=200):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bottom = np.column_stack([
        TARGET_CENTER[0] + TARGET_R * np.cos(theta),
        TARGET_CENTER[1] + TARGET_R * np.sin(theta),
        np.full(n, TARGET_CENTER[2]),
    ])
    top = np.column_stack([
        TARGET_CENTER[0] + TARGET_R * np.cos(theta),
        TARGET_CENTER[1] + TARGET_R * np.sin(theta),
        np.full(n, TARGET_CENTER[2] + TARGET_H),
    ])
    return np.vstack([bottom, top])


def shielded_scalar(t, heading, speed, drop, det, E):
    uav_dir = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = DRONE0 + speed * drop * uav_dir
    det_xy = drop_pos[:2] + speed * det * uav_dir[:2]
    det_z = drop_pos[2] - 0.5 * G * det * det
    t_start = drop + det
    if t < t_start or t > t_start + VALID or det_z < MIN_Z:
        return False
    smoke = np.array([det_xy[0], det_xy[1], det_z - SINK * (t - t_start)])
    if smoke[2] < MIN_Z:
        return False
    S = MISSILE0 + 300.0 * t * MISSILE_DIR
    MP = E - S
    MC = smoke - S
    a = np.sum(MP * MP, axis=1)
    a = np.maximum(a, EPS)
    proj = np.sum(MP * MC, axis=1) / a
    proj = np.clip(proj, 0.0, 1.0)
    nearest = S + proj[:, None] * MP
    dist = np.linalg.norm(nearest - smoke, axis=1)
    return bool(np.all(dist <= R + EPS))


def interval_for(heading, speed, drop, det, E, dt):
    """单枚烟幕弹有效遮蔽区间：粗扫定位 + 二分法精化边界"""
    uav_dir = np.array([np.cos(heading), np.sin(heading), 0.0])
    drop_pos = DRONE0 + speed * drop * uav_dir
    det_xy = drop_pos[:2] + speed * det * uav_dir[:2]
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
    C[:, 0] = det_xy[0]
    C[:, 1] = det_xy[1]
    C[:, 2] = smoke_z
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
    # 左边界二分精化
    left_approx = t_coarse[first]
    lo_left = max(t_start, left_approx - dt)
    hi_left = left_approx
    if not shielded_scalar(lo_left, heading, speed, drop, det, E):
        for _ in range(35):
            mid = (lo_left + hi_left) / 2
            if shielded_scalar(mid, heading, speed, drop, det, E):
                hi_left = mid
            else:
                lo_left = mid
    else:
        hi_left = lo_left
    # 右边界二分精化
    right_approx = t_coarse[last]
    lo_right = right_approx
    hi_right = min(t_end, right_approx + dt)
    if not shielded_scalar(hi_right, heading, speed, drop, det, E):
        for _ in range(35):
            mid = (lo_right + hi_right) / 2
            if shielded_scalar(mid, heading, speed, drop, det, E):
                lo_right = mid
            else:
                hi_right = mid
    else:
        lo_right = hi_right
    return float(hi_left), float(lo_right)


def merge_intervals(intervals):
    """计算区间并集总长度"""
    valid = [(s, e) for s, e in intervals if e > s + EPS]
    if not valid:
        return 0.0, []
    valid.sort(key=lambda x: x[0])
    merged = [list(valid[0])]
    for s, e in valid[1:]:
        if s <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    total = sum(e - s for s, e in merged)
    return total, [(s, e) for s, e in merged]


# ========================= DE 优化 =========================
def decode_params(params):
    """时间映射：优化变量 -> (heading, speed, t_drop1, t_det1, t_drop2, t_det2, t_drop3, t_det3)"""
    heading, speed, t_drop1, gap1, gap2, t_det1, t_det2, t_det3 = params
    t_drop2 = t_drop1 + gap1
    t_drop3 = t_drop2 + gap2
    return heading, speed, t_drop1, t_det1, t_drop2, t_det2, t_drop3, t_det3


def objective_de(params, E, dt):
    """DE 目标函数：返回 -并集长度（DE 最小化）"""
    heading, speed, t_drop1, t_det1, t_drop2, t_det2, t_drop3, t_det3 = decode_params(params)
    if not (70.0 <= speed <= 140.0):
        return 1000.0
    if any(t < MIN_DET_DELAY - EPS for t in [t_det1, t_det2, t_det3]):
        return 1000.0
    if t_drop1 < -EPS:
        return 1000.0
    intervals = []
    for drop, det in [(t_drop1, t_det1), (t_drop2, t_det2), (t_drop3, t_det3)]:
        iv = interval_for(heading, speed, drop, det, E, dt)
        if iv is not None:
            intervals.append(iv)
    if not intervals:
        return 1000.0
    union, _ = merge_intervals(intervals)
    return -union


def main():
    print("=" * 70)
    print("问题3 重跑确认：8维DE + 时间映射 + fine 验证 + 罚函数对照")
    print("（精简版：不写 result1.xlsx / png，结果存 problem3_rerun_main_results.json）")
    print("=" * 70)

    E_opt = generate_samples(120)

    bounds = [
        (np.radians(170.0), np.radians(190.0)),
        (90.0, 140.0),
        (0.0, 3.0),
        (1.0, 5.0),
        (1.0, 5.0),
        (0.1, 8.0),
        (0.1, 8.0),
        (0.1, 8.0),
    ]

    answer_mapped = np.array([
        np.radians(179.499135),
        109.799905,
        0.043761,
        3.430362,
        1.712360,
        3.254815,
        4.560716,
        5.170469,
    ])

    rng = np.random.RandomState(42)
    init_pop = [answer_mapped.copy()]
    while len(init_pop) < 20:
        p = answer_mapped.copy()
        for j, (lo, hi) in enumerate(bounds):
            p[j] = np.clip(p[j] + rng.normal(0, (hi - lo) * 0.05), lo, hi)
        init_pop.append(p)
    init_pop = np.array(init_pop)

    print(f"\n[1/3] 差分进化优化（种群{len(init_pop)}，maxiter=40，seed=42）...", flush=True)
    res = differential_evolution(
        objective_de,
        bounds,
        args=(E_opt, 0.02),
        maxiter=40,
        popsize=15,
        seed=42,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        init=init_pop,
        workers=1,
        tol=1e-9,
    )

    best_params = res.x
    heading, speed, t_drop1, t_det1, t_drop2, t_det2, t_drop3, t_det3 = decode_params(best_params)
    print(f"DE 完成，最优适应度（-并集）= {res.fun:.6f}")
    print(f"  航向角 = {np.degrees(heading):.6f}°  速度 = {speed:.6f} m/s")
    print(f"  弹1: t_drop={t_drop1:.6f}, t_det={t_det1:.6f}")
    print(f"  弹2: t_drop={t_drop2:.6f}, t_det={t_det2:.6f}")
    print(f"  弹3: t_drop={t_drop3:.6f}, t_det={t_det3:.6f}")
    print(f"  投放间隔1 = {t_drop2 - t_drop1:.6f}s, 间隔2 = {t_drop3 - t_drop2:.6f}s")

    # 高精度验证
    print(f"\n[2/3] fine 高精度验证（每圆周200点，dt=0.005s）...", flush=True)
    E_fine = generate_samples(200)
    all_intervals = []
    bomb_rows = []
    for k, (drop, det) in enumerate([(t_drop1, t_det1), (t_drop2, t_det2), (t_drop3, t_det3)]):
        iv = interval_for(heading, speed, drop, det, E_fine, 0.005)
        all_intervals.append(iv)
        bomb_rows.append({
            "烟幕弹序号": k + 1,
            "投放时间(s)": drop,
            "起爆延迟(s)": det,
            "起爆时刻(s)": drop + det,
            "有效区间起点(s)": iv[0],
            "有效区间终点(s)": iv[1],
            "有效遮蔽时长(s)": iv[1] - iv[0],
        })
        print(f"  弹{k+1}: 区间[{iv[0]:.6f}, {iv[1]:.6f}], 时长={iv[1]-iv[0]:.6f}s")

    union, merged = merge_intervals(all_intervals)
    print(f"\n  三弹并集总时长 = {union:.6f} s")
    print(f"  合并后区间: {[(f'{s:.6f}', f'{e:.6f}') for s, e in merged]}")
    print(f"  内置基准 ANSWER_UNION = {ANSWER_UNION:.6f} s")
    print(f"  与基准差异: {union - ANSWER_UNION:+.6f} s ({(union-ANSWER_UNION)/ANSWER_UNION*100:+.4f}%)")

    # 罚函数对照：强制间隔 g=1.0 s 边界（time_mapping_effectiveness_test 逻辑，纯评估）
    print(f"\n[3/3] 罚函数对照（强制投放间隔压到 1.0 s 边界）...", flush=True)
    d_bad = [t_drop1, t_drop1 + 1.0, t_drop1 + 2.0]
    ivs_bad = [interval_for(heading, speed, d, det, E_fine, 0.005)
               for d, det in zip(d_bad, [t_det1, t_det2, t_det3])]
    ivs_bad = [iv for iv in ivs_bad if iv is not None]
    u_bad, _ = merge_intervals(ivs_bad) if ivs_bad else (0.0, [])
    print(f"  满足约束: 间隔={t_drop2-t_drop1:.4f}s, {t_drop3-t_drop2:.4f}s, 并集={union:.6f}s")
    print(f"  强制 g=1.0s: 并集={u_bad:.6f}s")
    print(f"  下降: {union-u_bad:+.6f}s ({(union-u_bad)/union*100:+.2f}%)")

    results = {
        "说明": "problem3_solver.py 主求解重跑确认（不写 xlsx/png）",
        "DE参数": {"maxiter": 40, "popsize": 15, "seed": 42,
                   "mutation": [0.5, 1.0], "recombination": 0.7, "polish": "L-BFGS-B"},
        "最优参数": {
            "航向角(度)": float(np.degrees(heading)),
            "速度(m/s)": float(speed),
            "投放时间(s)": [float(t_drop1), float(t_drop2), float(t_drop3)],
            "起爆延迟(s)": [float(t_det1), float(t_det2), float(t_det3)],
            "投放间隔(s)": [float(t_drop2 - t_drop1), float(t_drop3 - t_drop2)],
        },
        "fine验证": {
            "精度": "200 点/圆周, dt=0.005 s, 二分 35 次",
            "三弹并集总时长(s)": float(union),
            "合并后区间(s)": [[float(s), float(e)] for s, e in merged],
            "逐弹明细": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in row.items()} for row in bomb_rows],
        },
        "与内置基准比较": {
            "ANSWER_UNION(s)": ANSWER_UNION,
            "差异(s)": float(union - ANSWER_UNION),
            "差异(%)": float((union - ANSWER_UNION) / ANSWER_UNION * 100),
        },
        "罚函数对照_强制间隔1s": {
            "强制g1_g2(s)": [1.0, 1.0],
            "并集时长(s)": float(u_bad),
            "相对满足约束下降(s)": float(union - u_bad),
            "下降百分比(%)": float((union - u_bad) / union * 100),
        },
    }
    out_path = OUT / "problem3_rerun_main_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nsaved {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
