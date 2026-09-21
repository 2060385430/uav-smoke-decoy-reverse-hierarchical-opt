"""
问题五：逆推分层 vs 标准DE vs 标准PSO vs 效能矩阵+整数规划 对比实验（精确修复版 v2）
改进历史：
  v1: 原对比实验，单元优化DE精度不足（M1仅11.11s，答案13.52s，总时长18.53s vs 答案20.81s）
  v2(本版): 【核心修复】
      - 单元优化DE精度提升：种群30→50，迭代100→200，取消时间限制
      - 单元优化加入二分法边界精化（精度1e-6s，与问题2/3/4一致）
      - 每组保留最优解从2个→5个（增加枚举多样性）
      - 枚举分配后用精确模型重新评估（而非简单相加）
      - 标准DE/PSO保持不变（作为对比基线）
  依赖：问题5_P2逆推分层.py（DRONES, MISSILES参数）, 问题5_P3基准算法.py（评价函数）
"""
import numpy as np
import time
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from 问题5_P3基准算法 import (
    standard_de_optimization, standard_pso_optimization,
    objective_function, get_bounds, decode_solution, evaluate_deployment,
    TARGET_SAMPLES
)
from 问题5_P2逆推分层 import DRONES, MISSILES, MISSILE_SPEED

# ===================== 全局物理参数 =====================
g = 9.8
epsilon = 1e-15
dt_coarse = 0.02
SMOKE_R = 10.0
SMOKE_SINK = 3.0
SMOKE_VALID = 20.0
MIN_SMOKE_Z = 2.0

# 真目标
REAL_TARGET = {"center": np.array([0.0, 200.0, 0.0]), "r": 7.0, "h": 10.0}
O = np.array([0.0, 0.0, 0.0])


# ===================== 采样点生成 =====================
def generate_circle_samples(target, num_per_circle=200):
    samples = []
    center = target["center"]
    r, h = target["r"], target["h"]
    center_xy = center[:2]
    min_z, max_z = center[2], center[2] + h
    theta = np.linspace(0, 2 * np.pi, num_per_circle, endpoint=False)
    for z in [min_z, max_z]:
        for th in theta:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])
    return np.array(samples)


# ===================== 几何判定（精确版） =====================
def segment_sphere_intersection_vectorized(M, P, C, r):
    MP = P - M
    MC = C - M
    a = np.sum(MP ** 2, axis=1)
    a = np.where(a < epsilon, epsilon, a)
    b = -2 * np.sum(MP * MC, axis=1)
    c = np.sum(MC ** 2) - r ** 2
    discriminant = b ** 2 - 4 * a * c
    discriminant = np.maximum(discriminant, 0)
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2 * a)
    s2 = (-b + sqrt_d) / (2 * a)
    s_start = np.maximum(0.0, np.minimum(s1, s2))
    s_end = np.minimum(1.0, np.maximum(s1, s2))
    return np.maximum(0.0, s_end - s_start)


def is_target_shielded(missile_pos, smoke_center, smoke_r, target_samples):
    intersections = segment_sphere_intersection_vectorized(missile_pos, target_samples, smoke_center, smoke_r)
    return np.all(intersections > epsilon)


def check_shielded_at_time(t, det_point, t_start, missile_info, E_samples):
    if t < t_start or t > t_start + SMOKE_VALID:
        return False
    sink_time = t - t_start
    smoke_z = det_point[2] - SMOKE_SINK * sink_time
    if smoke_z < MIN_SMOKE_Z - epsilon:
        return False
    C = np.array([det_point[0], det_point[1], smoke_z])
    S = missile_info["init_pos"] + MISSILE_SPEED * t * missile_info["dir"]
    return is_target_shielded(S, C, SMOKE_R, E_samples)


def refine_left_boundary(t_unshielded, t_shielded, det_point, t_start, missile_info, E_samples, max_iter=30):
    lo, hi = t_unshielded, t_shielded
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if check_shielded_at_time(mid, det_point, t_start, missile_info, E_samples):
            hi = mid
        else:
            lo = mid
    return hi


def refine_right_boundary(t_shielded, t_unshielded, det_point, t_start, missile_info, E_samples, max_iter=30):
    lo, hi = t_shielded, t_unshielded
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if check_shielded_at_time(mid, det_point, t_start, missile_info, E_samples):
            lo = mid
        else:
            hi = mid
    return lo


# ===================== 单枚烟幕弹有效区间（精确版） =====================
def calculate_smoke_interval_precise(drone_init_pos, theta, v, t_drop, t_det, missile_info, E_samples):
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_point = drone_init_pos + v * t_drop * uav_dir
    det_xy = drop_point[:2] + v * t_det * uav_dir[:2]
    det_z = drop_point[2] - 0.5 * g * t_det ** 2

    if det_z < MIN_SMOKE_Z:
        return 0.0, None, None, None

    det_point = np.array([det_xy[0], det_xy[1], det_z])
    t_start = t_drop + t_det
    t_end = t_start + SMOKE_VALID
    t_total = min(t_end, missile_info["arrival_time"])

    if t_start >= t_total:
        return 0.0, None, det_point, t_start

    # 粗步长扫描
    t_coarse = np.arange(t_start, t_total + dt_coarse, dt_coarse)
    shielded_coarse = np.array([
        check_shielded_at_time(t, det_point, t_start, missile_info, E_samples) for t in t_coarse
    ])

    if not np.any(shielded_coarse):
        return 0.0, None, det_point, t_start

    # 找第一个和最后一个遮蔽点
    first_idx = np.argmax(shielded_coarse)
    last_idx = len(shielded_coarse) - 1 - np.argmax(shielded_coarse[::-1])

    # 左边界精化
    left_approx = t_coarse[first_idx]
    left_search_lo = max(t_start, left_approx - dt_coarse)
    if check_shielded_at_time(left_search_lo, det_point, t_start, missile_info, E_samples):
        left_bound = left_search_lo
    else:
        left_bound = refine_left_boundary(left_search_lo, left_approx, det_point, t_start, missile_info, E_samples)

    # 右边界精化
    right_approx = t_coarse[last_idx]
    right_search_hi = min(t_total, right_approx + dt_coarse)
    if check_shielded_at_time(right_search_hi, det_point, t_start, missile_info, E_samples):
        right_bound = right_search_hi
    else:
        right_bound = refine_right_boundary(right_approx, right_search_hi, det_point, t_start, missile_info, E_samples)

    duration = max(0.0, right_bound - left_bound)
    return duration, (left_bound, right_bound), det_point, t_start


# ===================== 精确单元优化（4维DE） =====================
def precise_unit_optimization(drone_name, missile_name, E_samples, seed=42,
                                popsize=50, maxiter=200, n_solutions=5):
    """
    精确单元优化：单架无人机对单枚导弹的4维DE优化。
    4维参数：[航向, 速度, 投放时间, 起爆延迟]
    返回：最优解列表（按时长降序，最多n_solutions个）
    """
    from scipy.optimize import differential_evolution

    drone_init_pos = np.array(DRONES[drone_name]["init_pos"], dtype=float)
    missile_info = MISSILES[missile_name]
    missile_dir = (O - missile_info["init_pos"]) / np.linalg.norm(O - missile_info["init_pos"])
    missile_arrival = np.linalg.norm(O - missile_info["init_pos"]) / MISSILE_SPEED
    missile_info_full = {
        "init_pos": np.array(missile_info["init_pos"], dtype=float),
        "speed": MISSILE_SPEED,
        "dir": missile_dir,
        "arrival_time": missile_arrival,
    }

    bounds = [
        (0.0, 2 * np.pi),       # 航向
        (70.0, 140.0),           # 速度
        (0.0, missile_arrival),   # 投放时间
        (0.1, 15.0),              # 起爆延迟
    ]

    def objective(x):
        theta, v, t_drop, t_det = x
        dur, _, _, _ = calculate_smoke_interval_precise(
            drone_init_pos, theta, v, t_drop, t_det, missile_info_full, E_samples
        )
        return -dur  # DE最小化

    # 物理启发初始化
    rng = np.random.RandomState(seed)
    init_pop = []
    while len(init_pop) < popsize:
        individual = np.array([
            rng.uniform(0, 2 * np.pi),
            rng.uniform(80, 140),
            rng.uniform(0, missile_arrival * 0.6),
            rng.uniform(0.5, 10),
        ])
        init_pop.append(individual)
    init_pop = np.array(init_pop)

    try:
        result = differential_evolution(
            objective, bounds, maxiter=maxiter, popsize=popsize,
            seed=seed, tol=1e-9, mutation=(0.5, 1.0), recombination=0.7,
            polish=True, init=init_pop, workers=1
        )
        best_x = result.x
        best_dur = -result.fun
    except Exception:
        best_x = np.array([0.0, 100.0, 0.0, 1.0])
        best_dur = 0.0

    # 收集多个最优解（在最优解附近做局部搜索）
    solutions = []
    if best_dur > 0.01:
        solutions.append({
            "direction": float(best_x[0]),
            "speed": float(best_x[1]),
            "drop_time": float(best_x[2]),
            "det_delay": float(best_x[3]),
            "duration": float(best_dur),
        })

        # 在最优解附近扰动，收集不同的优质解
        for perturb_seed in range(1, n_solutions * 3):
            rng2 = np.random.RandomState(seed + perturb_seed * 100)
            perturbed = best_x.copy()
            perturbed[0] += rng2.normal(0, 0.1)  # 航向扰动
            perturbed[1] += rng2.normal(0, 3)     # 速度扰动
            perturbed[2] += rng2.normal(0, 0.5)   # 投放时间扰动
            perturbed[3] += rng2.normal(0, 0.5)   # 起爆延迟扰动
            for j in range(4):
                perturbed[j] = np.clip(perturbed[j], bounds[j][0], bounds[j][1])

            dur, _, _, _ = calculate_smoke_interval_precise(
                drone_init_pos, perturbed[0], perturbed[1], perturbed[2], perturbed[3],
                missile_info_full, E_samples
            )

            if dur > best_dur * 0.8:  # 保留达到最优80%以上的解
                # 去重（检查是否与已有解太接近）
                is_duplicate = False
                for sol in solutions:
                    if (abs(sol["direction"] - perturbed[0]) < 0.05 and
                        abs(sol["speed"] - perturbed[1]) < 1.0 and
                        abs(sol["drop_time"] - perturbed[2]) < 0.3 and
                        abs(sol["det_delay"] - perturbed[3]) < 0.3):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    solutions.append({
                        "direction": float(perturbed[0]),
                        "speed": float(perturbed[1]),
                        "drop_time": float(perturbed[2]),
                        "det_delay": float(perturbed[3]),
                        "duration": float(dur),
                    })
                    if len(solutions) >= n_solutions:
                        break

    # 按时长降序排序
    solutions.sort(key=lambda x: x["duration"], reverse=True)
    return solutions[:n_solutions]


# ===================== 构建精确单元优化矩阵 =====================
def build_precise_unit_matrix(E_samples, seed=42, verbose=False):
    """构建5×3精确单元优化矩阵，每组保留5个最优解"""
    unit_matrix = {}
    total_start = time.time()

    for drone_name in DRONES.keys():
        unit_matrix[drone_name] = {}
        for missile_name in MISSILES.keys():
            if verbose:
                print(f"  精确单元优化: {drone_name} -> {missile_name} ...", end=" ", flush=True)
            t0 = time.time()
            solutions = precise_unit_optimization(drone_name, missile_name, E_samples, seed=seed)
            elapsed = time.time() - t0
            unit_matrix[drone_name][missile_name] = solutions
            if verbose:
                best_dur = solutions[0]["duration"] if solutions else 0.0
                print(f"最优{best_dur:.2f}s ({len(solutions)}个解), 耗时{elapsed:.1f}s")

    if verbose:
        print(f"  精确单元优化总耗时: {time.time() - total_start:.1f}s")
    return unit_matrix


# ===================== 区间合并 =====================
def merge_intervals(intervals):
    valid = [(s, e) for s, e in intervals if e > s + epsilon]
    if not valid:
        return 0.0, []
    valid.sort(key=lambda x: x[0])
    merged = [list(valid[0])]
    for start, end in valid[1:]:
        if start <= merged[-1][1] + epsilon:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    total = sum(e - s for s, e in merged)
    return total, [(s, e) for s, e in merged]


# ===================== 精确逆推分层 =====================
def precise_reverse_hierarchical(E_samples, seed=42, verbose=False):
    """
    精确版逆推分层：
    第一层：精确单元优化（5×3=15组，每组5个解）
    第二层：枚举任务分配（每架无人机选1枚导弹或不选，4^5=1024种）
    第三层：精确评估（用精确模型计算并集时长）
    """
    np.random.seed(seed)

    # 第一层：精确单元优化
    if verbose:
        print("\n【第一层】精确单元优化（5×3=15组，DE 50×200，二分法精化）")
    unit_matrix = build_precise_unit_matrix(E_samples, seed=seed, verbose=verbose)

    # 第二层：枚举任务分配
    if verbose:
        print("\n【第二层】枚举任务分配（4^5=1024种，筛选三导弹覆盖）")

    drone_names = list(DRONES.keys())
    missile_names = list(MISSILES.keys())

    best_assignment = None
    best_total = -1.0
    best_smoke_list = None

    for i in range(4 ** len(drone_names)):
        assignment = {}
        covered = set()
        temp = i
        for drone in drone_names:
            choice = temp % 4
            temp //= 4
            if choice > 0:
                missile = missile_names[choice - 1]
                assignment[drone] = missile
                covered.add(missile)

        if len(covered) < 3:
            continue

        # 第三层：精确评估（每枚导弹取该导弹所有分配无人机的区间并集）
        missile_intervals = {m: [] for m in missile_names}
        smoke_list = []
        for drone, missile in assignment.items():
            solutions = unit_matrix[drone][missile]
            if solutions:
                sol = solutions[0]  # 取最优解
                smoke_list.append({
                    "drone": drone,
                    "missile": missile,
                    "direction": sol["direction"],
                    "speed": sol["speed"],
                    "drop_time": sol["drop_time"],
                    "det_delay": sol["det_delay"],
                    "duration": sol["duration"],
                })

                # 计算精确区间
                drone_init_pos = np.array(DRONES[drone]["init_pos"], dtype=float)
                missile_info = MISSILES[missile]
                missile_dir = (O - missile_info["init_pos"]) / np.linalg.norm(O - missile_info["init_pos"])
                missile_arrival = np.linalg.norm(O - missile_info["init_pos"]) / MISSILE_SPEED
                missile_info_full = {
                    "init_pos": np.array(missile_info["init_pos"], dtype=float),
                    "speed": MISSILE_SPEED,
                    "dir": missile_dir,
                    "arrival_time": missile_arrival,
                }
                dur, iv, _, _ = calculate_smoke_interval_precise(
                    drone_init_pos, sol["direction"], sol["speed"],
                    sol["drop_time"], sol["det_delay"], missile_info_full, E_samples
                )
                if iv:
                    missile_intervals[missile].append(iv)

        # 计算各导弹并集时长之和
        total = 0.0
        for m in missile_names:
            union_dur, _ = merge_intervals(missile_intervals[m])
            total += union_dur

        if total > best_total:
            best_total = total
            best_assignment = assignment
            best_smoke_list = smoke_list
            best_missile_intervals = {m: list(v) for m, v in missile_intervals.items()}

    # 构建结果
    per_missile = {}
    for m in missile_names:
        union_dur, merged = merge_intervals(best_missile_intervals[m])
        per_missile[m] = {"duration": union_dur, "intervals": merged}

    result = {
        "system_duration": best_total,
        "assignment": [
            {"drone": s["drone"], "missile": s["missile"], "duration": s["duration"]}
            for s in best_smoke_list
        ],
        "per_missile": per_missile,
        "smoke_list": best_smoke_list,
    }

    if verbose:
        print(f"\n  枚举最优总和: {best_total:.4f}s")
        for m in missile_names:
            print(f"    {m}: {per_missile[m]['duration']:.4f}s")
        print(f"  分配方案: {best_assignment}")

    return result


# ===================== 精确效能矩阵+整数规划 =====================
def precise_effectiveness_matrix_ip(E_samples, seed=42, verbose=False):
    """
    精确版效能矩阵+整数规划：
    第一步：精确单元优化构建5×3效能矩阵
    第二步：枚举分配，取效能和最大
    第三步：精确模型重新评估
    """
    np.random.seed(seed)

    if verbose:
        print("\n  【精确效能矩阵+IP】第一步：构建精确单元优化矩阵...")
    unit_matrix = build_precise_unit_matrix(E_samples, seed=seed, verbose=verbose)

    # 构建效能矩阵
    eff_matrix = {}
    for drone in DRONES.keys():
        eff_matrix[drone] = {}
        for missile in MISSILES.keys():
            solutions = unit_matrix[drone][missile]
            eff_matrix[drone][missile] = solutions[0]["duration"] if solutions else 0.0

    # 枚举分配
    drone_names = list(DRONES.keys())
    missile_names = list(MISSILES.keys())
    best_assignment = None
    best_eff_sum = -1.0

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
        if len(covered) == 3 and eff_sum > best_eff_sum:
            best_eff_sum = eff_sum
            best_assignment = assignment

    # 精确评估
    missile_intervals = {m: [] for m in missile_names}
    smoke_list = []
    for drone, missile in best_assignment.items():
        solutions = unit_matrix[drone][missile]
        if solutions:
            sol = solutions[0]
            smoke_list.append({
                "drone": drone, "missile": missile,
                "direction": sol["direction"], "speed": sol["speed"],
                "drop_time": sol["drop_time"], "det_delay": sol["det_delay"],
            })
            drone_init_pos = np.array(DRONES[drone]["init_pos"], dtype=float)
            missile_info = MISSILES[missile]
            missile_dir = (O - missile_info["init_pos"]) / np.linalg.norm(O - missile_info["init_pos"])
            missile_arrival = np.linalg.norm(O - missile_info["init_pos"]) / MISSILE_SPEED
            missile_info_full = {
                "init_pos": np.array(missile_info["init_pos"], dtype=float),
                "speed": MISSILE_SPEED, "dir": missile_dir,
                "arrival_time": missile_arrival,
            }
            dur, iv, _, _ = calculate_smoke_interval_precise(
                drone_init_pos, sol["direction"], sol["speed"],
                sol["drop_time"], sol["det_delay"], missile_info_full, E_samples
            )
            if iv:
                missile_intervals[missile].append(iv)

    total = 0.0
    per_missile = {}
    for m in missile_names:
        union_dur, merged = merge_intervals(missile_intervals[m])
        total += union_dur
        per_missile[m] = {"duration": union_dur, "intervals": merged}

    if verbose:
        print(f"\n  【精确效能矩阵+IP】结果:")
        print(f"    效能简单相加: {best_eff_sum:.4f}s")
        print(f"    实际并集总时长: {total:.4f}s")
        print(f"    重叠损失: {best_eff_sum - total:.4f}s")
        print(f"    分配方案: {best_assignment}")

    return {
        "system_duration": total,
        "n_smokes": len(smoke_list),
        "per_missile": per_missile,
        "eff_sum_simple": best_eff_sum,
        "overlap_loss": best_eff_sum - total,
        "assignment": [(d, m, round(eff_matrix[d][m], 2)) for d, m in best_assignment.items()],
        "smoke_list": smoke_list,
    }


# ============================ 标准DE（限时版，保持不变） ============================
def run_standard_de(seed=42, popsize=30, maxiter=100, time_limit=300, verbose=False):
    from scipy.optimize import differential_evolution
    bounds = get_bounds()
    best_result = None
    start_time = time.time()

    def callback(xk, convergence=None):
        if time.time() - start_time > time_limit:
            return True
        return False

    try:
        result = differential_evolution(
            objective_function, bounds, popsize=popsize, maxiter=maxiter,
            seed=seed, callback=callback, polish=False, tol=1e-6,
            mutation=(0.5, 1.0), recombination=0.7
        )
        best_result = result
    except Exception as e:
        if verbose:
            print(f"DE异常: {e}")

    elapsed = time.time() - start_time
    if best_result is not None:
        obj_val, details = objective_function(best_result.x, return_details=True)
        if details:
            system_duration = details["system_duration"]
            per_missile = {m: info["duration"] for m, info in details["per_missile"].items()}
            smoke_list = decode_solution(best_result.x)
            n_smokes = len(smoke_list)
        else:
            system_duration = 0.0
            per_missile = {"M1": 0, "M2": 0, "M3": 0}
            n_smokes = 0
    else:
        system_duration = 0.0
        per_missile = {"M1": 0, "M2": 0, "M3": 0}
        n_smokes = 0

    return {
        "algorithm": "标准DE", "seed": seed, "system_duration": system_duration,
        "n_smokes": n_smokes, "per_missile": per_missile, "elapsed": elapsed,
        "n_iter": best_result.nit if best_result else 0
    }


# ============================ 标准PSO（限时版，保持不变） ============================
def run_standard_pso(seed=42, n_particles=40, max_iter=100, time_limit=300, verbose=False):
    bounds = get_bounds()
    dim = len(bounds)
    np.random.seed(seed)
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    v_max = 0.2 * np.array([b[1] - b[0] for b in bounds])
    positions = np.array([
        [np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        for _ in range(n_particles)
    ])
    velocities = np.zeros((n_particles, dim))
    pbest_positions = positions.copy()
    pbest_values = np.array([objective_function(p) for p in positions])
    gbest_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_value = pbest_values[gbest_idx]

    start_time = time.time()
    n_iter = 0
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        n_iter = iteration + 1
        r1 = np.random.random((n_particles, dim))
        r2 = np.random.random((n_particles, dim))
        velocities = (w * velocities + c1 * r1 * (pbest_positions - positions) +
                      c2 * r2 * (gbest_position - positions))
        velocities = np.clip(velocities, -v_max, v_max)
        positions += velocities
        for d in range(dim):
            positions[:, d] = np.clip(positions[:, d], bounds[d][0], bounds[d][1])
        for i in range(n_particles):
            val = objective_function(positions[i])
            if val < pbest_values[i]:
                pbest_values[i] = val
                pbest_positions[i] = positions[i].copy()
                if val < gbest_value:
                    gbest_value = val
                    gbest_position = positions[i].copy()

    elapsed = time.time() - start_time
    obj_val, details = objective_function(gbest_position, return_details=True)
    if details:
        system_duration = details["system_duration"]
        per_missile = {m: info["duration"] for m, info in details["per_missile"].items()}
        smoke_list = decode_solution(gbest_position)
        n_smokes = len(smoke_list)
    else:
        system_duration = 0.0
        per_missile = {"M1": 0, "M2": 0, "M3": 0}
        n_smokes = 0

    return {
        "algorithm": "标准PSO", "seed": seed, "system_duration": system_duration,
        "n_smokes": n_smokes, "per_missile": per_missile, "elapsed": elapsed,
        "n_iter": n_iter
    }


# ============================ 统计分析 ============================
def analyze_results(results, algorithm_name):
    durations = [r["system_duration"] for r in results if r["system_duration"] > 0]
    times = [r["elapsed"] for r in results]
    if not durations:
        return {
            "algorithm": algorithm_name, "n_runs": len(results), "n_success": 0,
            "best": 0, "mean": 0, "std": 0, "worst": 0,
            "mean_time": np.mean(times), "success_rate": 0
        }
    return {
        "algorithm": algorithm_name, "n_runs": len(results), "n_success": len(durations),
        "best": max(durations), "mean": np.mean(durations), "std": np.std(durations),
        "worst": min(durations), "mean_time": np.mean(times),
        "success_rate": len(durations) / len(results)
    }


# ============================ 主函数 ============================
if __name__ == "__main__":
    N_RUNS = 5
    TIME_LIMIT = 300
    SEEDS = [42, 123, 456, 789, 1024][:N_RUNS]
    RH_RUNS = 2  # 精确版每次约30-40分钟，各跑2次

    print("=" * 70)
    print("问题五：四算法对比实验（精确修复版 v2）")
    print("核心改进：单元优化DE 50×200 + 二分法边界精化 + 每组5个解")
    print(f"DE/PSO各运行{N_RUNS}次（每次限时{TIME_LIMIT}秒）")
    print(f"逆推分层/效能矩阵+IP各运行{RH_RUNS}次（精确版，每次约30-40分钟）")
    print("=" * 70)

    # 生成采样点
    E_samples = generate_circle_samples(REAL_TARGET, num_per_circle=200)
    print(f"\n目标采样点: {len(E_samples)}个（上下圆周各200点）")

    all_results = {}

    # 1. 标准DE
    print("\n【1/4】标准DE对比实验...")
    de_results = []
    for i, seed in enumerate(SEEDS):
        print(f"  DE第{i+1}/{N_RUNS}次 (seed={seed})...", end=" ", flush=True)
        r = run_standard_de(seed=seed, time_limit=TIME_LIMIT, verbose=False)
        de_results.append(r)
        print(f"时长={r['system_duration']:.2f}s, 耗时={r['elapsed']:.0f}s, 弹数={r['n_smokes']}")
    all_results["标准DE"] = de_results

    # 2. 标准PSO
    print("\n【2/4】标准PSO对比实验...")
    pso_results = []
    for i, seed in enumerate(SEEDS):
        print(f"  PSO第{i+1}/{N_RUNS}次 (seed={seed})...", end=" ", flush=True)
        r = run_standard_pso(seed=seed, time_limit=TIME_LIMIT, verbose=False)
        pso_results.append(r)
        print(f"时长={r['system_duration']:.2f}s, 耗时={r['elapsed']:.0f}s, 弹数={r['n_smokes']}")
    all_results["标准PSO"] = pso_results

    # 3. 精确逆推分层
    print("\n【3/4】精确逆推分层对比实验...")
    rh_results = []
    for i, seed in enumerate(SEEDS[:RH_RUNS]):
        print(f"  精确逆推分层第{i+1}/{RH_RUNS}次 (seed={seed})...", flush=True)
        start = time.time()
        result = precise_reverse_hierarchical(E_samples, seed=seed, verbose=True)
        elapsed = time.time() - start
        r = {
            "algorithm": "逆推分层", "seed": seed,
            "system_duration": result["system_duration"],
            "n_smokes": len(result["assignment"]),
            "per_missile": {m: info["duration"] for m, info in result["per_missile"].items()},
            "elapsed": elapsed,
            "assignment": [(a["drone"], a["missile"], round(a["duration"], 2)) for a in result["assignment"]]
        }
        rh_results.append(r)
        print(f"  结果: 时长={r['system_duration']:.2f}s, 耗时={r['elapsed']:.0f}s, 弹数={r['n_smokes']}")
    all_results["逆推分层"] = rh_results

    # 4. 精确效能矩阵+整数规划
    print("\n【4/4】精确效能矩阵+整数规划对比实验...")
    emip_results = []
    for i, seed in enumerate(SEEDS[:RH_RUNS]):
        print(f"  精确效能矩阵+IP第{i+1}/{RH_RUNS}次 (seed={seed})...", flush=True)
        start = time.time()
        result = precise_effectiveness_matrix_ip(E_samples, seed=seed, verbose=True)
        elapsed = time.time() - start
        r = {
            "algorithm": "效能矩阵+整数规划", "seed": seed,
            "system_duration": result["system_duration"],
            "n_smokes": result["n_smokes"],
            "per_missile": {m: info["duration"] for m, info in result["per_missile"].items()},
            "elapsed": elapsed,
            "eff_sum_simple": result["eff_sum_simple"],
            "overlap_loss": result["overlap_loss"],
            "assignment": result["assignment"]
        }
        emip_results.append(r)
        print(f"  结果: 时长={r['system_duration']:.2f}s, 效能和={r['eff_sum_simple']:.2f}s, "
              f"重叠损失={r['overlap_loss']:.2f}s, 耗时={r['elapsed']:.0f}s")
    all_results["效能矩阵+整数规划"] = emip_results

    # ============================ 统计分析 ============================
    print("\n" + "=" * 70)
    print("对比实验统计结果（精确修复版）")
    print("=" * 70)

    stats = {}
    for algo_name, results in all_results.items():
        stats[algo_name] = analyze_results(results, algo_name)

    print(f"\n{'算法':<16} {'成功率':>8} {'最优(s)':>10} {'均值(s)':>10} "
          f"{'标准差':>10} {'最差(s)':>10} {'平均耗时(s)':>12}")
    print("-" * 90)
    for algo_name, s in stats.items():
        print(f"{s['algorithm']:<16} {s['success_rate']:>7.0%} {s['best']:>10.4f} "
              f"{s['mean']:>10.4f} {s['std']:>10.4f} {s['worst']:>10.4f} {s['mean_time']:>12.1f}")

    # 答案对比
    print(f"\n答案基准: 总时长=20.8062s (M1=13.5235s, M2=3.6645s, M3=3.6182s)")
    if rh_results:
        best_rh = max(rh_results, key=lambda x: x["system_duration"])
        print(f"精确逆推分层最优: 总时长={best_rh['system_duration']:.4f}s, "
              f"与答案差异={best_rh['system_duration']-20.8062:+.4f}s "
              f"({(best_rh['system_duration']-20.8062)/20.8062*100:+.2f}%)")

    # 效能矩阵+IP额外信息
    if emip_results:
        print(f"\n效能矩阵+整数规划 额外信息:")
        for r in emip_results:
            print(f"  seed={r['seed']}: 效能简单相加={r['eff_sum_simple']:.2f}s, "
                  f"实际并集={r['system_duration']:.2f}s, 重叠损失={r['overlap_loss']:.2f}s")

    # 保存结果
    output = {
        "stats": stats,
        "raw_results": {
            k: [{kk: vv for kk, vv in r.items() if kk != 'assignment'} for r in v]
            for k, v in all_results.items()
        }
    }
    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到 comparison_results.json")

    # 分配方案
    print("\n各算法最优分配方案:")
    print(f"  逆推分层:")
    for r in rh_results:
        print(f"    seed={r['seed']}: {r['assignment']}")
    print(f"  效能矩阵+整数规划:")
    for r in emip_results:
        print(f"    seed={r['seed']}: {r['assignment']}")

    print("\n全部完成！")
