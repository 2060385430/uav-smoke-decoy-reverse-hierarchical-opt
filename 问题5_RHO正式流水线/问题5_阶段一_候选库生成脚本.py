# -*- coding: utf-8 -*-
"""
问题5 RHO 阶段一：候选库生成（论文 3.2 节）
=============================================
对 15 个独立 8 维单元子问题 (drone, missile) ∈ FY1..FY5 × M1..M3，
固定该机仅干扰该弹，优化 3 枚烟幕弹的完整投放方案。

决策向量 x = [θ, v, d1, g1, g2, t1, t2, t3]（8 维）
  θ      航向角        [0, 2π]
  v      速度          [70, 140]
  d1     首次投放时刻  [0, 55]
  g1,g2  间隔增量      [1, 10]   时间映射：投放时刻 = [d1, d1+g1, d1+g1+g2]，间隔≥1s 天然满足
  t1..t3 起爆延迟      [0.1, det_max]  det_max = min(12, sqrt(2*(z0-2)/9.8))，由起爆高度决定
（边界与消费端 ablation_rho.BOUNDS_40 对应分量逐一致）

分段目标函数：
  有遮蔽时  最大化 3 弹遮蔽区间并集长度（式14）→ 目标 = -并集长度
  无遮蔽时  以距离引导项 D(x)（式15，有效时间窗内云团到各采样点视线的最大距离的最小值）
            替代，D 越小越好 → 目标 = 100 + mean(D_i)
  （口径复用 ablation_rho.interval_or_guidance / eval_global_med_guided 的既有实现）

求解器：DE/rand/1/bin，SEEDS 个随机种子，每种子 G=35 代、NP=14，
  缩放因子 F∈[0.5,1.5] 逐代抖动（mutation=(0.5,1.5) dither），CR=0.75。
  scipy 的 popsize 是种群倍数，为精确实现 NP=14，初始种群以 14×8 数组显式传入
  （init 为数组时 popsize 被忽略，种群规模恰为 14）。
  规格偏差：基础 4 种子 × G=35 后候选不足 Top-3 的组合，以额外种子
  （每轮 4 个、G=60，至多 RESCUE_ROUNDS 轮）补投，保证候选库覆盖。

精度：优化阶段 coarse（每圆周 60 点、dt=0.25 s，仅作区间定位）；
      候选以 medium（每圆周 120 点、dt=0.02 s、二分 35 次）复评。
去重后每个组合保留 Top-3 候选。

断点续跑：每完成一个 (drone, missile) 组合即追加写入 stage1_checkpoint.json，
          重跑时跳过已完成组合；最终输出 candidate_lib_regen.pkl。
"""
import os, sys, json, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablation_rho as A  # 复用消费端物理模型与评估函数，保证口径一致

CHECKPOINT = os.path.join(HERE, "stage1_checkpoint.json")
OUT_PKL = os.path.join(HERE, "candidate_lib_regen.pkl")

# ================= 可调配置 =================
SEEDS = [2024, 7, 42, 1234]   # 论文 3.2 节：4 个随机种子
GENS = 35                     # 每种子 G=35 代
NP = 14                       # 种群规模 14
F_RANGE = (0.5, 1.5)          # 缩放因子逐代抖动
CR = 0.75
TOP_K = 3                     # 去重后每组合保留 Top-3
# 补投配置（规格偏差，见报告）：基础 4 种子 × G=35 后，候选不足 Top-3 的组合
# 以额外种子与更长代数补投，直至凑满 Top-3 或补投轮次耗尽。
RESCUE_ROUNDS = 4             # 最多补投轮数
RESCUE_SEEDS = 4              # 每轮额外种子数
RESCUE_GENS = 60              # 补投每种子代数
# 强化通道（规格偏差，见报告）：以当前最优候选热启动 + 新鲜随机种子、更长代数，
# 继续提升各组合候选质量（仍 DE/rand/1、F∈[0.5,1.5] 抖动、CR=0.75、NP=14）。
INTENSIFY_ROUNDS = 4          # 强化轮数
INTENSIFY_FRESH_SEEDS = 2     # 每轮新鲜随机种子数
INTENSIFY_GENS = 80           # 强化每种子代数

E_COARSE = A.samples(60)      # 优化：每圆周 60 点
DT_COARSE = 0.25
# medium 复评直接复用消费端 A.E_MED（每圆周 120 点）+ interval_for(dt=0.02, bisect=35)


def unit_bounds(drone):
    """该无人机 8 维单元子问题边界，与 BOUNDS_40 对应分量一致"""
    i = A.DRONE_LIST.index(drone)
    return A.BOUNDS_40[8 * i: 8 * i + 8]


def unit_objective(x, drone, m):
    """分段目标：有遮蔽最大化并集（式14）；无遮蔽以距离引导 D(x)（式15）替代"""
    theta, v, d1, g1, g2, t1, t2, t3 = x
    drops = [d1, d1 + g1, d1 + g1 + g2]
    ivs, guid = [], []
    for d, t in zip(drops, (t1, t2, t3)):
        iv, g = A.interval_or_guidance(drone, m, theta, v, d, t, E_COARSE, DT_COARSE)
        if iv is not None:
            ivs.append(iv)
        else:
            guid.append(g)
    total = A.merge(ivs)
    if total > 0:
        return -total
    return 100.0 + float(np.mean(guid)) if guid else 100.0


def medium_eval(drone, m, x):
    """medium 精度复评一个解，返回候选 dict；完全无遮蔽返回 None"""
    theta, v, d1, g1, g2, t1, t2, t3 = x
    drops = [d1, d1 + g1, d1 + g1 + g2]
    ivs = []
    for d, t in zip(drops, (t1, t2, t3)):
        iv = A.interval_for(drone, m, theta, v, d, t, A.E_MED, 0.02)  # bisect=35
        if iv is not None:
            ivs.append(iv)
    if not ivs:
        return None
    return {
        "v": float(v),
        "theta": float(theta),
        "drop_times": [float(d) for d in drops],
        "det_delays": [float(t1), float(t2), float(t3)],
        "intervals": [(float(s), float(e)) for s, e in ivs],
        "duration": float(A.merge(ivs)),
    }


def dedup_key(c):
    """去重键：参数取整到网格，同一网格视为重复"""
    return (round(c["theta"], 3), round(c["v"], 2),
            tuple(round(d, 2) for d in c["drop_times"]),
            tuple(round(t, 2) for t in c["det_delays"]))


def solve_unit(drone, m, seeds, gens):
    """求解一个单元子问题：给定种子集 DE，medium 复评，去重取 Top-3"""
    bounds = unit_bounds(drone)
    cands = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        init = lo + (hi - lo) * rng.rand(NP, len(bounds))  # NP=14 初始种群
        res = differential_evolution(
            unit_objective, bounds, args=(drone, m),
            strategy="rand1bin", maxiter=gens, popsize=NP,
            mutation=F_RANGE, recombination=CR, polish=False,
            init=init, seed=seed, workers=1, tol=1e-9)
        c = medium_eval(drone, m, res.x)
        if c is not None:
            cands.append(c)
    return cands


def select_topk(cands):
    """去重 + Top-K"""
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda c: -c["duration"]):
        k = dedup_key(c)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
        if len(uniq) >= TOP_K:
            break
    return uniq


def encode_unit(c):
    """候选 dict -> 8 维决策向量（时间映射逆变换）"""
    drops = sorted(c["drop_times"])
    g1 = max(drops[1] - drops[0], 1.0)
    g2 = max(drops[2] - drops[1], 1.0)
    return np.array([c["theta"], c["v"], drops[0], g1, g2,
                     c["det_delays"][0], c["det_delays"][1], c["det_delays"][2]])


def de_run(drone, m, bounds, init, seed, gens):
    """单次 DE/rand/1/bin（F 抖动、CR=0.75、NP=14 显式种群）"""
    res = differential_evolution(
        unit_objective, bounds, args=(drone, m),
        strategy="rand1bin", maxiter=gens, popsize=NP,
        mutation=F_RANGE, recombination=CR, polish=False,
        init=init, seed=seed, workers=1, tol=1e-9)
    return medium_eval(drone, m, res.x)


def intensify_unit(drone, m, cands, rounds, start_round=0):
    """强化通道：当前最优候选热启动（±5% 高斯扰动种群）+ 新鲜随机种子"""
    bounds = unit_bounds(drone)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pool = list(cands)
    for i in range(rounds):
        r = start_round + i  # 全局轮次编号，保证续跑时种子不重复
        x_best = encode_unit(select_topk(pool)[0])
        rng = np.random.RandomState(77000 + r)
        rows = [x_best]
        while len(rows) < NP:
            p = x_best + rng.normal(0.0, 1.0, len(bounds)) * (hi - lo) * 0.05
            rows.append(np.clip(p, lo, hi))
        c = de_run(drone, m, bounds, np.array(rows), 77000 + r, INTENSIFY_GENS)
        if c is not None:
            pool.append(c)
        for s in range(INTENSIFY_FRESH_SEEDS):
            seed = 78000 + 100 * r + s
            rng = np.random.RandomState(seed)
            init = lo + (hi - lo) * rng.rand(NP, len(bounds))
            c = de_run(drone, m, bounds, init, seed, INTENSIFY_GENS)
            if c is not None:
                pool.append(c)
    return select_topk(pool)


def load_ckpt():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ckpt(st):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def dump_lib(st):
    """由断点状态重建完整 15 键候选库并写 pkl"""
    lib = {}
    for drone in A.DRONE_LIST:
        for m in A.MISSILE_LIST:
            key = f"{drone}_{m}"
            cands = st.get(key, {}).get("candidates", [])
            lib[(drone, m)] = [
                {**c, "intervals": [tuple(iv) for iv in c["intervals"]]}
                for c in cands]
    with open(OUT_PKL, "wb") as f:
        pickle.dump(lib, f)
    return lib


def main():
    t0 = time.time()
    st = load_ckpt()
    for drone in A.DRONE_LIST:
        for m in A.MISSILE_LIST:
            key = f"{drone}_{m}"
            if key in st:  # 断点续跑：跳过已完成组合
                print(f"[跳过] {key} 已完成", flush=True)
                continue
            ts = time.time()
            # --- 基础通道：论文 3.2 节规格的 4 种子 × G=35 ---
            pool = solve_unit(drone, m, SEEDS, GENS)
            n_seeds, extra_rounds = len(SEEDS), 0
            # --- 补投通道（规格偏差）：候选不足 Top-3 时追加种子与代数 ---
            while len(select_topk(pool)) < TOP_K and extra_rounds < RESCUE_ROUNDS:
                extra_rounds += 1
                seeds = [90000 + 1000 * extra_rounds + s for s in range(RESCUE_SEEDS)]
                pool += solve_unit(drone, m, seeds, RESCUE_GENS)
                n_seeds += len(seeds)
            cands = select_topk(pool)
            st[key] = {
                "candidates": [
                    {**c, "intervals": [list(iv) for iv in c["intervals"]]}
                    for c in cands],
                "seeds": n_seeds, "base_gens": GENS,
                "rescue_rounds": extra_rounds, "rescue_gens": RESCUE_GENS,
                "np": NP,
                "time_s": round(time.time() - ts, 2),
            }
            save_ckpt(st)
            dump_lib(st)  # 每组合完成后同步更新 pkl，部分结果亦可用
            best = cands[0]["duration"] if cands else 0.0
            print(f"[完成] {key}: {len(cands)} 候选, 最优(medium)={best:.4f}s, "
                  f"种子 {n_seeds} 个(补投{extra_rounds}轮), "
                  f"耗时 {time.time()-ts:.1f}s", flush=True)
    lib = dump_lib(st)
    n_nonempty = sum(1 for v in lib.values() if v)
    print(f"[汇总] 15 组合全部完成，非空键 {n_nonempty} 个，"
          f"总耗时 {time.time()-t0:.0f}s -> {OUT_PKL}", flush=True)


def main_intensify():
    """强化模式（--intensify）：对已完成的非空组合逐组合强化，断点续跑"""
    t0 = time.time()
    st = load_ckpt()
    for drone in A.DRONE_LIST:
        for m in A.MISSILE_LIST:
            key = f"{drone}_{m}"
            entry = st.get(key)
            if not entry or not entry["candidates"]:
                continue  # 空组合（FY1_M2/FY1_M3）无可强化
            done = entry.get("intensify_rounds_done", 0)
            if done >= INTENSIFY_ROUNDS:
                print(f"[跳过] {key} 已强化 {done} 轮", flush=True)
                continue
            ts = time.time()
            cands = [
                {**c, "intervals": [tuple(iv) for iv in c["intervals"]]}
                for c in entry["candidates"]]
            cands = intensify_unit(drone, m, cands, INTENSIFY_ROUNDS - done,
                                   start_round=done)
            entry["candidates"] = [
                {**c, "intervals": [list(iv) for iv in c["intervals"]]}
                for c in cands]
            entry["intensify_rounds_done"] = INTENSIFY_ROUNDS
            entry["intensify_gens"] = INTENSIFY_GENS
            entry["time_s"] = round(entry.get("time_s", 0) + time.time() - ts, 2)
            save_ckpt(st)
            dump_lib(st)
            best = cands[0]["duration"] if cands else 0.0
            print(f"[强化] {key}: {len(cands)} 候选, 最优(medium)={best:.4f}s, "
                  f"耗时 {time.time()-ts:.1f}s", flush=True)
    print(f"[强化汇总] 耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--intensify" in sys.argv:
        main_intensify()
    else:
        main()
