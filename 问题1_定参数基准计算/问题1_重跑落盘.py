# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import importlib.util
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # 无显示环境下禁止弹窗

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "问题1.py")
OUT_JSON = os.path.join(HERE, "problem1_results.json")

# ---- 导入 问题1.py（不执行其 __main__ 部分） ----
spec = importlib.util.spec_from_file_location("problem1_src", SRC)
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)

EXPECTED_DURATION = 1.391642
EXPECTED_INTERVAL = [8.056446, 9.448088]


def load_results():
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            os.remove(OUT_JSON)  # 上次中断留下的半截文件，删除重来
    return {}


def save_results(res):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[落盘] 结果已写入 {OUT_JSON}", flush=True)


def run_main(res):
    """环节1：定参数遮蔽时长（N=300、dt=0.01、二分 tol=1e-6）"""
    t0 = time.time()
    dur, intervals, n_samples = p1.calculate_effective_duration_optimized(
        coarse_step=0.01, num_per_circle=300, refine_tol=1e-6,
        visualize=False, savesvg=False)
    elapsed = time.time() - t0
    res["main_result"] = {
        "params": {"num_per_circle": 300, "coarse_step": 0.01, "refine_tol": 1e-6},
        "duration_s": float(dur),
        "intervals_s": [[float(s), float(e)] for s, e in intervals],
        "n_sample_points": int(n_samples),
        "expected_duration_s": EXPECTED_DURATION,
        "expected_interval_s": EXPECTED_INTERVAL,
        "duration_matches_expected": bool(abs(dur - EXPECTED_DURATION) < 5e-7),
        "interval_matches_expected": bool(
            len(intervals) == 1
            and abs(intervals[0][0] - EXPECTED_INTERVAL[0]) < 5e-7
            and abs(intervals[0][1] - EXPECTED_INTERVAL[1]) < 5e-7),
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[主计算] 时长 = {dur:.6f} s，区间 = {intervals}，耗时 {elapsed:.1f}s", flush=True)


def run_conv(res):
    """环节2：采样收敛检验 N = 50,100,200,300,500"""
    n_list = [50, 100, 200, 300, 500]
    t0 = time.time()
    results = p1.sample_convergence_test(n_list=n_list)  # 默认 dt=0.005、tol=1e-6
    elapsed = time.time() - t0
    ref = results[n_list[-1]]
    res["sample_convergence"] = {
        "params": {"n_list": n_list, "coarse_step": 0.005, "refine_tol": 1e-6},
        "durations_s": {str(n): float(d) for n, d in results.items()},
        "deviation_vs_N500_s": {str(n): float(d - ref) for n, d in results.items()},
        "max_abs_deviation_s": float(max(abs(d - ref) for d in results.values())),
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[收敛检验] 完成，耗时 {elapsed:.1f}s", flush=True)


def run_step(res):
    """环节3：步长敏感性 dt = 0.05~0.002（N=200，二分精化）"""
    step_list = [0.05, 0.02, 0.01, 0.005, 0.002]
    t0 = time.time()
    results = p1.coarse_step_sensitivity_test(step_list=step_list)  # 默认 N=200
    elapsed = time.time() - t0
    ref = results[min(step_list)]
    res["step_sensitivity"] = {
        "params": {"step_list": step_list, "num_per_circle": 200, "refine_tol": 1e-6},
        "durations_s": {str(dt): float(d) for dt, d in results.items()},
        "deviation_vs_dt0.002_s": {str(dt): float(d - ref) for dt, d in results.items()},
        "max_abs_deviation_s": float(max(abs(d - ref) for d in results.values())),
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[步长敏感性] 完成，耗时 {elapsed:.1f}s", flush=True)


def main():
    stages = sys.argv[1:] or ["main", "conv", "step"]
    res = load_results()
    res.setdefault("meta", {
        "script": os.path.basename(__file__),
        "source_module": "问题1.py（导入复用，未修改原文件）",
        "problem": "问题1 定参数遮蔽时长（FY1, v=120 m/s, 投放1.5s, 延迟3.6s）",
    })
    for st in stages:
        if st == "main":
            run_main(res)
        elif st == "conv":
            run_conv(res)
        elif st == "step":
            run_step(res)
        elif st == "all":
            run_main(res); run_conv(res); run_step(res)
        else:
            print(f"未知环节: {st}（可选 main / conv / step / all）")
            continue
        save_results(res)  # 每完成一个环节立即落盘，防超时丢进度
    print("全部请求环节完成。", flush=True)


if __name__ == "__main__":
    main()
