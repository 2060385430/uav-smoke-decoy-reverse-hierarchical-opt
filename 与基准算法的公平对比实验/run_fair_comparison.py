# -*- coding: utf-8 -*-
"""
补充实验 4：基线公平对比（fair comparison）跑批脚本
====================================================
公平性口径（与论文/REPRODUCIBILITY.md 一致）：
  1. 同一目标函数：所有基线直接优化 40 维问题（"直接 40 维"设定），
     搜索目标 = RHO 阶段五精修所用的 eval_global_med（medium 精度：
     240 采样点，dt=0.02 s，二分 35 次），经 fast_objective.py 批量化加速，
     数学上与原函数逐位一致（parity_check 通过，最大偏差 < 1e-12 s）；
  2. 统一计算预算：以目标函数评估次数（NFE）为主口径，默认 BUDGET=16000，
     恰等于 RHO 阶段五精修预算（maxiter=40, popsize=10 -> 16400，取整 16000）；
     RHO 另有阶段一候选库生成开销未计入，即预算口径实际上对基线更有利；
  3. 统一精度评估：每个 run 的最优解统一用 eval_global_fine
     （fine 精度：400 采样点，dt=0.005 s，二分 40 次）复算，
     与 RHO 的 22.717447 s 同口径比较；
  4. 独立重复：每种算法 N_RUNS 个不同种子独立运行，报告均值±标准差、
     最好/最差、达到 RHO 水平的比例、双侧 Wilcoxon 符号秩检验 p 值。
算法：DE(scipy, 即"直接 40 维 DE")、PSO、GA(实数编码)、CMA-ES(cma 库)、
      贝叶斯优化(sklearn GP+EI)。
并行：多进程（默认 7 进程）评估种群，Windows/Linux 均可。
输出：fair_comparison_results.json（含全部设置、种子、逐 run 结果与耗时），断点续跑。
用法：
  python run_fair_comparison.py            # 正式跑批（数小时，建议夜间）
  python run_fair_comparison.py --quick    # 快速自检（约 2-3 分钟）
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import differential_evolution
from scipy import stats as sstats
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT_JSON = os.path.join(HERE, "fair_comparison_results.json")

import fast_objective as fo          # 加速目标 + RHO 模块引用
rho = fo.rho
BOUNDS_40 = rho.BOUNDS_40
RHO_FINE = 22.717447                  # RHO 论文结果（fine 精度，消融 A0）

LO = np.array([b[0] for b in BOUNDS_40])
HI = np.array([b[1] for b in BOUNDS_40])
DIM = 40


def _eval_one(x):
    """多进程 worker：medium 精度单点评估（返回最小化目标值）。"""
    return fo.eval_med_fast(x)


def eval_global_fine(x):
    """fine 精度终评（与 eval_global_med 同语义，E_FINE/dt=0.005/bisect=40）。"""
    per_m = {m: [] for m in rho.MISSILE_LIST}
    for i in range(5):
        seg = x[8 * i: 8 * i + 8]
        theta, v, d1, g1, g2 = seg[0], seg[1], seg[2], seg[3], seg[4]
        if not (70.0 <= v <= 140.0):
            return 0.0
        drops = [d1, d1 + g1, d1 + g1 + g2]
        for d, t in zip(drops, seg[5:8]):
            for m in rho.MISSILE_LIST:
                iv = rho.interval_for(rho.DRONE_LIST[i], m, theta, v, d, t,
                                      rho.E_FINE, 0.005, bisect=40)
                if iv is not None:
                    per_m[m].append(iv)
    return sum(rho.merge(ivs) for ivs in per_m.values())


# ================= 算法实现（均按 NFE 预算精确控制） =================
def run_de(seed, budget, eval_pop):
    """DE：scipy differential_evolution，popsize=10 -> 400 个体/代。
    maxiter = budget//400 - 1，实际 NFE 由 res.nfev 精确记录。"""
    popsize = 10
    maxiter = max(1, budget // (popsize * DIM) - 1)
    res = differential_evolution(
        _eval_one, BOUNDS_40, maxiter=maxiter, popsize=popsize, seed=seed,
        mutation=(0.5, 1.0), recombination=0.7, polish=False,
        init='random', tol=0.0, atol=-1e18, workers=eval_pop,
        updating='deferred')
    return res.x, int(res.nfev)


def run_pso(seed, budget, eval_pop, n_particles=40):
    """PSO：40 粒子，w:0.9->0.4，c1=c2=1.5（与归档基线一致）。"""
    rng = np.random.RandomState(seed)
    iters = max(1, budget // n_particles)
    X = LO + (HI - LO) * rng.rand(n_particles, DIM)
    V = 0.1 * (HI - LO) * rng.randn(n_particles, DIM)
    pbest, pval = X.copy(), np.full(n_particles, -np.inf)
    gbest, gval = None, -np.inf
    for it in range(iters):
        vals = np.array([-f for f in eval_pop(_eval_one, X)])
        better = vals > pval
        pbest[better], pval[better] = X[better], vals[better]
        if vals.max() > gval:
            gval, gbest = vals.max(), X[vals.argmax()].copy()
        w = 0.9 - 0.5 * it / max(iters - 1, 1)
        r1, r2 = rng.rand(n_particles, DIM), rng.rand(n_particles, DIM)
        V = w * V + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest - X)
        X = np.clip(X + V, LO, HI)
    return gbest, n_particles * iters


def run_ga(seed, budget, eval_pop, pop_size=100, elite=2, alpha=0.5, mut_rate=0.1):
    """GA：实数编码，锦标赛选择(k=3) + BLX-0.5 交叉 + 高斯变异 + 精英保留。"""
    rng = np.random.RandomState(seed)
    gens = max(1, budget // pop_size)
    P = LO + (HI - LO) * rng.rand(pop_size, DIM)
    gbest, gval = None, -np.inf
    span = HI - LO
    for g in range(gens):
        vals = np.array([-f for f in eval_pop(_eval_one, P)])
        order = np.argsort(-vals)
        if vals[order[0]] > gval:
            gval, gbest = vals[order[0]], P[order[0]].copy()
        newP = [P[i].copy() for i in order[:elite]]
        while len(newP) < pop_size:
            def tourney():
                idx = rng.randint(0, pop_size, 3)
                return P[idx[np.argmax(vals[idx])]]
            p1, p2 = tourney(), tourney()
            u = rng.rand(DIM)
            lo_c = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
            hi_c = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
            child = lo_c + (hi_c - lo_c) * u
            mut = rng.rand(DIM) < mut_rate
            child[mut] += 0.1 * span[mut] * rng.randn(int(mut.sum()))
            newP.append(np.clip(child, LO, HI))
        P = np.array(newP)
    return gbest, pop_size * gens


def run_cmaes(seed, budget, eval_pop):
    """CMA-ES：cma 库，sigma0=0.3（归一化意义下），active CMA。"""
    import cma
    x0 = 0.5 * (LO + HI)
    es = cma.CMAEvolutionStrategy(
        x0, 0.3, {'bounds': [LO.tolist(), HI.tolist()],
                  'maxfevals': budget, 'maxiter': 10 ** 9,
                  'seed': seed, 'verbose': -9, 'CMA_active': True,
                  'tolfun': 0, 'tolfunhist': 0, 'tolx': 0,
                  'tolstagnation': 10 ** 9, 'tolflatfitness': 10 ** 9,
                  'tolupsigma': 1e30, 'tolfacupx': 1e30})
    best_x, best_v = None, np.inf
    nfe = 0
    while not es.stop():
        X = es.ask()
        F = list(eval_pop(_eval_one, [np.asarray(x) for x in X]))
        nfe += len(X)
        es.tell(X, F)
        for x, f in zip(X, F):
            if f < best_v:
                best_v, best_x = f, np.asarray(x).copy()
        if nfe >= budget:
            break
    return best_x, nfe


def run_bo(seed, budget, eval_pop=None, n_init=None, refit_every=10):
    """贝叶斯优化：sklearn GP(Matern-2.5) + EI，LHS 初始设计，串行。
    GP 拟合 O(n^3)，每 refit_every 次采集重拟合一次代理模型（标准做法），
    预算单独设置（默认 1000 NFE）。"""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import qmc, norm
    rng = np.random.RandomState(seed)
    n_init = n_init or min(4 * DIM, budget // 3)

    def obj01(x01):
        return fo.eval_med_fast(LO + (HI - LO) * np.asarray(x01))

    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    X = sampler.random(n_init)
    Y = np.array([obj01(x) for x in X])
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(nu=2.5, length_scale=np.full(DIM, 0.2),
                       length_scale_bounds=(1e-2, 1e1))
              + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=0, random_state=seed)
    nfe = n_init
    gp.fit(X, Y)
    steps = 0
    while nfe < budget:
        if steps % refit_every == 0 and steps > 0:
            gp.fit(X, Y)
        cand = rng.rand(4096, DIM)
        mu, sd = gp.predict(cand, return_std=True)
        sd = np.maximum(sd, 1e-12)
        imp = Y.min() - mu
        Z = imp / sd
        ei = imp * norm.cdf(Z) + sd * norm.pdf(Z)
        x_next = cand[np.argmax(ei)]
        y_next = obj01(x_next)
        nfe += 1
        steps += 1
        X = np.vstack([X, x_next]); Y = np.append(Y, y_next)
    x_best = LO + (HI - LO) * X[np.argmin(Y)]
    return x_best, nfe


ALGOS = {
    "DE":     {"fn": run_de,    "note": "scipy differential_evolution, popsize=10, F~(0.5,1.0), CR=0.7 (= 直接40维DE)"},
    "PSO":    {"fn": run_pso,   "note": "40 particles, w:0.9->0.4, c1=c2=1.5"},
    "GA":     {"fn": run_ga,    "note": "pop=100, tournament-3, BLX-0.5, gauss mut 0.1, elite 2"},
    "CMA-ES": {"fn": run_cmaes, "note": "cma 4.5.0, sigma0=0.3, active CMA"},
    "BO":     {"fn": run_bo,    "note": "sklearn GP(Matern-2.5)+EI, LHS init, 串行"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速自检模式")
    ap.add_argument("--budget", type=int, default=16000, help="NFE 预算（默认 16000）")
    ap.add_argument("--runs", type=int, default=20, help="独立运行次数（默认 20）")
    ap.add_argument("--bo-budget", type=int, default=1000, help="BO 的 NFE 预算（默认 1000）")
    ap.add_argument("--bo-runs", type=int, default=5, help="BO 独立运行次数（默认 5）")
    ap.add_argument("--workers", type=int, default=7, help="并行进程数（默认 7）")
    args = ap.parse_args()
    if args.quick:
        args.budget, args.runs, args.bo_budget, args.bo_runs = 800, 2, 300, 1

    state = {"config": {
        "budget_nfe": args.budget, "n_runs": args.runs,
        "bo_budget_nfe": args.bo_budget, "bo_runs": args.bo_runs,
        "search_objective": "eval_global_med (medium: 240 pts, dt=0.02, bisect=35), batched+parallel",
        "final_eval": "eval_global_fine (fine: 400 pts, dt=0.005, bisect=40)",
        "rho_reference_fine": RHO_FINE,
        "algos": {k: v["note"] for k, v in ALGOS.items()},
    }, "results": {}}
    if os.path.exists(OUT_JSON):
        state = json.load(open(OUT_JSON, encoding="utf-8"))

    pool = mp.Pool(args.workers) if args.workers > 1 else None
    eval_pop = pool.imap if pool else map
    try:
        for name, spec_ in ALGOS.items():
            runs_needed = args.bo_runs if name == "BO" else args.runs
            budget = args.bo_budget if name == "BO" else args.budget
            done = state["results"].setdefault(name, [])
            for r in range(len(done), runs_needed):
                seed = 20240000 + r * 977
                t0 = time.time()
                x_best, nfe = spec_["fn"](seed, budget, eval_pop)
                wall = time.time() - t0
                fine = float(eval_global_fine(x_best))
                done.append({"run": r, "seed": seed, "nfe": nfe,
                             "fine_duration_s": round(fine, 6),
                             "wall_time_s": round(wall, 1)})
                print(f"[{name}] run {r+1}/{runs_needed} seed={seed} "
                      f"NFE={nfe} fine={fine:.4f}s wall={wall:.0f}s", flush=True)
                json.dump(state, open(OUT_JSON, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
    finally:
        if pool:
            pool.close(); pool.join()

    # ---------- 汇总统计 ----------
    print("\n========== 公平对比汇总（fine 精度总遮蔽时长 s；RHO = %.6f）==========" % RHO_FINE)
    summary = {}
    for name in ALGOS:
        vals = np.array([d["fine_duration_s"] for d in state["results"].get(name, [])])
        if len(vals) == 0:
            continue
        frac = float(np.mean(vals >= RHO_FINE))
        try:
            pval = float(sstats.wilcoxon(vals - RHO_FINE, alternative="two-sided", method="exact").pvalue)
        except Exception:
            pval = None
        summary[name] = {
            "n_runs": int(len(vals)), "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4), "best": round(float(vals.max()), 4),
            "worst": round(float(vals.min()), 4),
            "frac_ge_rho": round(frac, 3), "wilcoxon_p_vs_rho": pval,
            "mean_wall_s": round(float(np.mean(
                [d["wall_time_s"] for d in state["results"][name]])), 1)}
        print(f"{name:>7}: mean={vals.mean():7.3f} ± {vals.std():5.3f} | "
              f"best={vals.max():7.3f} | worst={vals.min():7.3f} | "
              f">=RHO {frac*100:4.0f}% | p={pval}")
    state["summary"] = summary
    json.dump(state, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {OUT_JSON}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
