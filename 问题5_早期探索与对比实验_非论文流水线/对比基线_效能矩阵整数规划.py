"""
问题五：效能矩阵+整数规划 对比算法（单文件版，导入P2）

算法思路：
  第一步：复用P2的单元优化，构建5×3效能矩阵（每架无人机-导弹组合的最大遮蔽时长）
  第二步：整数规划（枚举法）求解最优分配
    - 每架无人机选择：不参与 / 干扰M1 / 干扰M2 / 干扰M3
    - 4^5 = 1024种方案，筛选三导弹都被覆盖的方案
    - 目标函数：Σ 效能[drone][missile]（简单相加，不考虑区间重叠）
  第三步：用实际遮蔽评价函数计算真实系统总遮蔽时长（考虑区间并集）

与逆推分层的核心区别：
  - 逆推分层：枚举分配时考虑区间并集，目标函数是实际并集时长
  - 效能矩阵+整数规划：目标函数是效能简单相加，忽略多弹协同干扰同一导弹时的区间重叠

依赖：问题5_P2逆推分层.py（需在同一目录下）
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from 问题5_P2逆推分层 import (
    build_unit_optimization_matrix,
    evaluate_deployment,
    DRONES,
    MISSILES,
    TARGET_SAMPLES,
)


# ============================ 1. 构建效能矩阵 ============================
def build_effectiveness_matrix(unit_matrix, verbose=True):
    """
    从单元优化矩阵构建5×3效能矩阵。
    每个元素取该组合最优解的duration（单架无人机对单枚导弹的最大遮蔽时长）。
    """
    eff_matrix = {}
    for drone in DRONES.keys():
        eff_matrix[drone] = {}
        for missile in MISSILES.keys():
            solutions = unit_matrix[drone][missile]
            if solutions:
                eff_matrix[drone][missile] = solutions[0]["duration"]
            else:
                eff_matrix[drone][missile] = 0.0

    if verbose:
        print("\n  效能矩阵（单架无人机对单枚导弹的最大遮蔽时长）:")
        print(f"  {'无人机':<6} {'M1':>8} {'M2':>8} {'M3':>8}")
        for drone in DRONES.keys():
            print(f"  {drone:<6} {eff_matrix[drone]['M1']:>8.2f} "
                  f"{eff_matrix[drone]['M2']:>8.2f} {eff_matrix[drone]['M3']:>8.2f}")

    return eff_matrix


# ============================ 2. 整数规划（枚举法） ============================
def solve_integer_programming(eff_matrix, verbose=True):
    """
    枚举法求解整数规划。
    每架无人机选择：0=不参与, 1=M1, 2=M2, 3=M3
    共4^5 = 1024种方案。
    约束：三枚导弹都至少被一架无人机覆盖。
    目标：最大化 Σ 效能[drone][missile]（简单相加）。
    """
    drone_names = list(DRONES.keys())
    missile_names = list(MISSILES.keys())

    best_assignment = None
    best_eff_sum = -1.0
    n_feasible = 0

    for i in range(4 ** len(drone_names)):
        assignment = {}
        covered = set()
        eff_sum = 0.0
        temp = i
        for drone in drone_names:
            choice = temp % 4
            temp //= 4
            if choice > 0:
                missile = missile_names[choice - 1]
                assignment[drone] = missile
                covered.add(missile)
                eff_sum += eff_matrix[drone][missile]

        # 筛选三导弹都被覆盖的方案
        if len(covered) == 3:
            n_feasible += 1
            if eff_sum > best_eff_sum:
                best_eff_sum = eff_sum
                best_assignment = assignment

    if verbose:
        print(f"\n  枚举方案总数: 4^5 = {4**len(drone_names)}")
        print(f"  满足三导弹覆盖的可行方案: {n_feasible}")
        print(f"  最优效能和（简单相加，忽略区间重叠）: {best_eff_sum:.4f}s")
        print(f"  最优分配方案: ", end="")
        for drone in drone_names:
            if drone in best_assignment:
                missile = best_assignment[drone]
                print(f"{drone}->{missile}({eff_matrix[drone][missile]:.2f}s) ", end="")
        print()

    return best_assignment, best_eff_sum


# ============================ 3. 实际遮蔽评价（考虑并集） ============================
def evaluate_assignment_real(assignment, unit_matrix, verbose=True):
    """
    用实际遮蔽评价函数计算真实系统总遮蔽时长。
    对分配方案中的每架无人机，取其对目标导弹的最优解参数，构建烟幕弹列表，
    然后调用evaluate_deployment计算各导弹区间并集长度。
    """
    smoke_list = []
    for drone, missile in assignment.items():
        solutions = unit_matrix[drone][missile]
        if solutions:
            sol = solutions[0]  # 取该组合的最优解
            smoke_list.append({
                "drone": drone,
                "speed": sol["speed"],
                "direction": sol["direction"],
                "drop_time": sol["drop_time"],
                "det_delay": sol["det_delay"],
            })

    result = evaluate_deployment(smoke_list, TARGET_SAMPLES)

    if verbose:
        print(f"\n  实际遮蔽评价（考虑区间并集）:")
        print(f"  有效烟幕弹数量: {len(smoke_list)}")
        for m in MISSILES.keys():
            dur = result["per_missile"][m]["duration"]
            intervals = result["per_missile"][m]["intervals"]
            print(f"    {m}: 并集时长{dur:.2f}s, 区间{intervals}")
        print(f"  实际系统总遮蔽时长（三导弹并集总和）: {result['system_duration']:.4f}s")

    return result, smoke_list


# ============================ 4. 主函数 ============================
def run_effectiveness_matrix_ip(seed=42, verbose=True):
    """运行效能矩阵+整数规划算法"""
    np.random.seed(seed)
    start_time = time.time()

    if verbose:
        print("=" * 70)
        print("效能矩阵+整数规划 — 问题五对比算法")
        print("=" * 70)

    # 第一步：构建单元优化矩阵（复用P2，15组4维DE优化）
    if verbose:
        print("\n【第一步】构建单元优化矩阵（5×3=15组，DE 4维）")
    unit_matrix = build_unit_optimization_matrix(verbose=verbose)

    # 第二步：构建效能矩阵
    if verbose:
        print("\n【第二步】构建效能矩阵")
    eff_matrix = build_effectiveness_matrix(unit_matrix, verbose=verbose)

    # 第三步：整数规划求解
    if verbose:
        print("\n【第三步】整数规划求解（枚举4^5=1024种方案）")
    best_assignment, best_eff_sum = solve_integer_programming(eff_matrix, verbose=verbose)

    # 第四步：实际遮蔽评价（考虑区间并集）
    if verbose:
        print("\n【第四步】实际遮蔽评价（考虑区间并集，对比效能简单相加）")
    result, smoke_list = evaluate_assignment_real(best_assignment, unit_matrix, verbose=verbose)

    elapsed = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("效能矩阵+整数规划 最终结果")
        print("=" * 70)
        print(f"  总耗时: {elapsed:.1f}s")
        print(f"  投放烟幕弹数量: {len(smoke_list)}")
        print(f"  效能矩阵简单相加（整数规划目标）: {best_eff_sum:.4f}s")
        print(f"  实际系统总遮蔽时长（并集总和）: {result['system_duration']:.4f}s")
        print(f"  区间重叠损失: {best_eff_sum - result['system_duration']:.4f}s "
              f"({(best_eff_sum - result['system_duration']) / best_eff_sum * 100:.1f}%)")
        print(f"  各导弹遮蔽情况:")
        for m in MISSILES.keys():
            print(f"    {m}: 并集时长{result['per_missile'][m]['duration']:.2f}s")

    return {
        "algorithm": "效能矩阵+整数规划",
        "seed": seed,
        "system_duration": result["system_duration"],
        "n_smokes": len(smoke_list),
        "elapsed": elapsed,
        "eff_sum_simple": best_eff_sum,
        "overlap_loss": best_eff_sum - result["system_duration"],
        "assignment": best_assignment,
        "per_missile": {m: info["duration"] for m, info in result["per_missile"].items()},
    }


# ============================ 测试 ============================
if __name__ == "__main__":
    print("问题五 —— 效能矩阵+整数规划 对比算法")
    print("依赖: 问题5_P2逆推分层.py（需在同一目录下）")
    print()

    result = run_effectiveness_matrix_ip(seed=42, verbose=True)

    print(f"\n{'='*70}")
    print(f"最终结果: 时长={result['system_duration']:.2f}s, "
          f"耗时={result['elapsed']:.0f}s, 弹数={result['n_smokes']}")
    print(f"效能简单相加={result['eff_sum_simple']:.2f}s, "
          f"区间重叠损失={result['overlap_loss']:.2f}s")
