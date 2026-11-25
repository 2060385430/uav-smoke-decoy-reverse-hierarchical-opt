# -*- coding: utf-8 -*-
"""临时修复脚本：重跑 DE 中被快速模式污染的 run（相同种子、16000 NFE）。"""
import json, time, os, sys
import multiprocessing as mp

def main():
    state = json.load(open("fair_comparison_results.json", encoding="utf-8"))
    bad = [i for i, d in enumerate(state["results"]["DE"]) if d["nfe"] != 16000]
    print("污染 run 索引:", bad, flush=True)
    import run_fair_comparison as rfc
    pool = mp.Pool(7)
    try:
        for i in bad:
            seed = state["results"]["DE"][i]["seed"]
            t0 = time.time()
            x, nfe = rfc.run_de(seed, 16000, pool.imap)
            fine = float(rfc.eval_global_fine(x))
            state["results"]["DE"][i] = {"run": i, "seed": seed, "nfe": nfe,
                                         "fine_duration_s": round(fine, 6),
                                         "wall_time_s": round(time.time() - t0, 1)}
            print(f"修复 run {i}: seed={seed} NFE={nfe} fine={fine:.4f}s", flush=True)
            json.dump(state, open("fair_comparison_results.json", "w",
                                  encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        pool.close(); pool.join()
    print("DE 修复完成")

if __name__ == "__main__":
    mp.freeze_support()
    main()
