# -*- coding: utf-8 -*-
"""带逐步断点的 BO 补跑脚本：每步保存 X/Y，被杀后可从断点继续。"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fast_objective as fo
import run_fair_comparison as rfc

BUDGET = 1000
N_INIT = min(4 * rfc.DIM, BUDGET // 3)
REFIT_EVERY = 10

def run_bo_ckpt(seed, run_idx):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import qmc, norm

    ckpt = os.path.join(HERE, f"bo_ckpt_run{run_idx}.npz")
    rng = np.random.RandomState(seed)

    def obj01(x01):
        return fo.eval_med_fast(rfc.LO + (rfc.HI - rfc.LO) * np.asarray(x01))

    if os.path.exists(ckpt):
        z = np.load(ckpt)
        X, Y, nfe, steps, wall0 = z["X"], z["Y"], int(z["nfe"]), int(z["steps"]), float(z["wall"])
        # 重建随机流状态：直接用新 rng 即可（候选点随机，不影响正确性）
        print(f"[BO run {run_idx}] 从断点恢复: nfe={nfe}", flush=True)
    else:
        sampler = qmc.LatinHypercube(d=rfc.DIM, seed=seed)
        X = sampler.random(N_INIT)
        Y = np.array([obj01(x) for x in X])
        nfe, steps, wall0 = N_INIT, 0, 0.0

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(nu=2.5, length_scale=np.full(rfc.DIM, 0.2),
                       length_scale_bounds=(1e-2, 1e1))
              + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=0, random_state=seed)
    t0 = time.time()
    gp.fit(X, Y)
    while nfe < BUDGET:
        if steps % REFIT_EVERY == 0 and steps > 0:
            gp.fit(X, Y)
        cand = rng.rand(4096, rfc.DIM)
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
        wall = wall0 + time.time() - t0
        np.savez(ckpt, X=X, Y=Y, nfe=nfe, steps=steps, wall=wall)
    wall = wall0 + time.time() - t0
    x_best = rfc.LO + (rfc.HI - rfc.LO) * X[np.argmin(Y)]
    return x_best, nfe, wall

def main():
    state = json.load(open(rfc.OUT_JSON, encoding="utf-8"))
    done = state["results"].setdefault("BO", [])
    for r in range(len(done), 5):
        seed = 20240000 + r * 977
        x_best, nfe, wall = run_bo_ckpt(seed, r)
        fine = float(rfc.eval_global_fine(x_best))
        done.append({"run": r, "seed": seed, "nfe": nfe,
                     "fine_duration_s": round(fine, 6),
                     "wall_time_s": round(wall, 1)})
        os.remove(os.path.join(HERE, f"bo_ckpt_run{r}.npz"))
        print(f"[BO] run {r+1}/5 seed={seed} NFE={nfe} fine={fine:.4f}s wall={wall:.0f}s", flush=True)
        json.dump(state, open(rfc.OUT_JSON, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("BO 全部完成")

if __name__ == "__main__":
    main()
