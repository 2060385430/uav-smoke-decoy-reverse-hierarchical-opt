"""
问题五：多无人机协同烟幕干扰 —— 逆推分层算法（单文件版）
三层结构：
  第一层：单元优化（DE，4维）—— 固定无人机-导弹组合，优化单枚弹参数
  第二层：路径拟合 —— 基于单元优化结果，确定每架无人机的速度和方向
  第三层：全局分配与时序优化 —— 固定速度方向，分配无人机-导弹组合并优化时序

本文件已合并统一评价函数（Part 1），无需外部依赖。
核心修正：
1. 采样点：只用上下底面圆周各50点（充要条件定理），移除侧面采样
2. 烟幕失效：云团中心 z < 2m 时失效
3. 目标函数：各导弹遮蔽区间并集长度的总和（系统总遮蔽时长）
4. 线段-球相交：向量化计算
5. 粗扫描(0.3s)+细步长(0.05s)+二分法精化边界，提升评价速度
"""
import numpy as np
import time
from collections import namedtuple
from scipy.optimize import differential_evolution

# ============================ 1. 全局参数 ============================
TRUE_TARGET = {
    "r": 7.0, "h": 10.0,
    "center": np.array([0.0, 200.0, 0.0]),
    "sample_points": None,
}
MISSILES = {
    "M1": {"init_pos": np.array([20000.0, 0.0, 2000.0])},
    "M2": {"init_pos": np.array([19000.0, 600.0, 2100.0])},
    "M3": {"init_pos": np.array([18000.0, -600.0, 1900.0])},
}
MISSILE_SPEED = 300.0
DRONES = {
    "FY1": {"init_pos": np.array([17800.0, 0.0, 1800.0]), "max_smoke": 3},
    "FY2": {"init_pos": np.array([12000.0, 1400.0, 1400.0]), "max_smoke": 3},
    "FY3": {"init_pos": np.array([6000.0, -3000.0, 700.0]), "max_smoke": 3},
    "FY4": {"init_pos": np.array([11000.0, 2000.0, 1800.0]), "max_smoke": 3},
    "FY5": {"init_pos": np.array([13000.0, -2000.0, 1300.0]), "max_smoke": 3},
}
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_EFFECTIVE_TIME = 20.0
SMOKE_MIN_HEIGHT = 2.0
G = 9.8
TIME_STEP = 0.02
EPSILON = 1e-8

for m_name, m_data in MISSILES.items():
    init_pos = m_data["init_pos"]
    dist = np.linalg.norm(init_pos)
    m_data["dir"] = -init_pos / dist
    m_data["velocity"] = m_data["dir"] * MISSILE_SPEED
    m_data["flight_time"] = dist / MISSILE_SPEED

# ============================ 2. 目标采样点 ============================
def generate_target_samples(n_per_circle=50):
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    theta = np.linspace(0, 2 * np.pi, n_per_circle, endpoint=False)
    bottom = np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        np.full(n_per_circle, center[2]),
    ])
    top = np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        np.full(n_per_circle, center[2] + h),
    ])
    TRUE_TARGET["sample_points"] = np.vstack([bottom, top])
    return TRUE_TARGET["sample_points"]

# ============================ 3. 位置计算 ============================
def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    t_clipped = min(t, m_data["flight_time"])
    return m_data["init_pos"] + m_data["velocity"] * t_clipped

def get_drone_pos(drone_name, t, speed, direction):
    init_pos = DRONES[drone_name]["init_pos"]
    v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0.0])
    return init_pos + v_vec * t

def get_smoke_center(drone_name, speed, direction, drop_time, det_delay, t):
    det_time = drop_time + det_delay
    if t < det_time - EPSILON:
        return None
    if t > det_time + SMOKE_EFFECTIVE_TIME + EPSILON:
        return None
    drop_pos = get_drone_pos(drone_name, drop_time, speed, direction)
    v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0.0])
    det_x = drop_pos[0] + v_vec[0] * det_delay
    det_y = drop_pos[1] + v_vec[1] * det_delay
    det_z = drop_pos[2] - 0.5 * G * det_delay ** 2
    if det_z < SMOKE_MIN_HEIGHT:
        return None
    sink_time = t - det_time
    smoke_z = det_z - SMOKE_SINK_SPEED * sink_time
    if smoke_z < SMOKE_MIN_HEIGHT:
        return None
    return np.array([det_x, det_y, smoke_z])

# ============================ 4. 线段-球相交判定（向量化） ============================
def segment_sphere_intersection_vectorized(missile_pos, target_points, smoke_center, smoke_radius):
    MP = target_points - missile_pos
    MC = smoke_center - missile_pos
    dot_MP_MC = np.sum(MP * MC, axis=1)
    dot_MP_MP = np.sum(MP ** 2, axis=1)
    dot_MP_MP = np.maximum(dot_MP_MP, EPSILON)
    t_proj = dot_MP_MC / dot_MP_MP
    t_clipped = np.clip(t_proj, 0.0, 1.0)
    nearest = missile_pos + t_clipped[:, np.newaxis] * MP
    dist = np.linalg.norm(nearest - smoke_center, axis=1)
    return dist <= smoke_radius + EPSILON

def is_target_fully_shielded(missile_pos, smoke_center, smoke_radius, target_samples):
    intersections = segment_sphere_intersection_vectorized(
        missile_pos, target_samples, smoke_center, smoke_radius
    )
    return np.all(intersections)

# ============================ 5. 单枚弹有效遮蔽区间 ============================
def _check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples):
    smoke_center = get_smoke_center(drone_name, speed, direction, drop_time, det_delay, t)
    if smoke_center is None:
        return False
    missile_pos = get_missile_pos(m_name, t)
    return is_target_fully_shielded(missile_pos, smoke_center, SMOKE_RADIUS, target_samples)

def _refine_boundary_binary_search(t_shielded, t_unshielded, drone_name, speed, direction,
                                     drop_time, det_delay, m_name, target_samples, max_iter=20):
    lo = min(t_shielded, t_unshielded)
    hi = max(t_shielded, t_unshielded)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if _check_shielded_at_time(mid, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples):
            lo = mid
        else:
            hi = mid
    return lo

def calc_single_smoke_interval(drone_name, speed, direction, drop_time, det_delay,
                                m_name, target_samples=None):
    if target_samples is None:
        target_samples = TRUE_TARGET["sample_points"]
    det_time = drop_time + det_delay
    m_flight_time = MISSILES[m_name]["flight_time"]
    t_start_window = max(det_time, 0.0)
    t_end_window = min(det_time + SMOKE_EFFECTIVE_TIME, m_flight_time)
    if t_start_window >= t_end_window - EPSILON:
        return None

    # 粗扫描（0.3s）
    coarse_step = 0.3
    t_coarse = np.arange(t_start_window, t_end_window + coarse_step, coarse_step)
    shielded_coarse = np.array([
        _check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples)
        for t in t_coarse
    ])
    if not np.any(shielded_coarse):
        return None

    coarse_intervals = []
    in_seg = False
    seg_start_idx = 0
    for idx in range(len(t_coarse)):
        if shielded_coarse[idx] and not in_seg:
            seg_start_idx = idx
            in_seg = True
        elif not shielded_coarse[idx] and in_seg:
            coarse_intervals.append((seg_start_idx, idx - 1))
            in_seg = False
    if in_seg:
        coarse_intervals.append((seg_start_idx, len(t_coarse) - 1))

    # 细步长（0.05s）+ 二分法
    fine_intervals = []
    fine_step = 0.05
    for start_idx, end_idx in coarse_intervals:
        fine_start = max(t_start_window, t_coarse[start_idx] - 0.4)
        fine_end = min(t_end_window, t_coarse[end_idx] + 0.4)
        t_fine = np.arange(fine_start, fine_end + fine_step, fine_step)
        shielded_fine = np.array([
            _check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples)
            for t in t_fine
        ])
        if not np.any(shielded_fine):
            continue

        in_seg = False
        seg_start_t = 0.0
        prev_unshielded_t = fine_start
        for idx in range(len(t_fine)):
            if shielded_fine[idx] and not in_seg:
                seg_start_t = t_fine[idx]
                prev_unshielded_t = t_fine[idx - 1] if idx > 0 else fine_start
                in_seg = True
            elif not shielded_fine[idx] and in_seg:
                left_boundary = _refine_boundary_binary_search(
                    seg_start_t, prev_unshielded_t,
                    drone_name, speed, direction, drop_time, det_delay, m_name, target_samples
                )
                right_boundary = _refine_boundary_binary_search(
                    t_fine[idx - 1], t_fine[idx],
                    drone_name, speed, direction, drop_time, det_delay, m_name, target_samples
                )
                fine_intervals.append((left_boundary, right_boundary))
                in_seg = False
        if in_seg:
            left_boundary = seg_start_t
            if seg_start_t > fine_start + EPSILON:
                left_idx = np.argmax(t_fine >= seg_start_t)
                if left_idx > 0:
                    left_boundary = _refine_boundary_binary_search(
                        t_fine[left_idx], t_fine[left_idx - 1],
                        drone_name, speed, direction, drop_time, det_delay, m_name, target_samples
                    )
            fine_intervals.append((left_boundary, t_fine[-1]))

    if not fine_intervals:
        return None
    fine_intervals.sort(key=lambda x: x[0])
    merged = [fine_intervals[0]]
    for current in fine_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + EPSILON:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    if len(merged) == 1:
        return merged[0]
    return merged

def interval_length(interval):
    if interval is None:
        return 0.0
    if isinstance(interval, tuple):
        return max(0.0, interval[1] - interval[0])
    if isinstance(interval, list):
        return sum(max(0.0, iv[1] - iv[0]) for iv in interval)
    return 0.0

# ============================ 6. 多枚弹对单导弹的区间并集 ============================
def merge_intervals(intervals):
    valid = [iv for iv in intervals if iv is not None and iv[1] > iv[0] + EPSILON]
    if not valid:
        return []
    flat = []
    for iv in valid:
        if isinstance(iv, list):
            flat.extend(iv)
        else:
            flat.append(iv)
    flat.sort(key=lambda x: x[0])
    merged = [flat[0]]
    for current in flat[1:]:
        last = merged[-1]
        if current[0] <= last[1] + EPSILON:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def calc_missile_total_shielding(smoke_params_list, m_name, target_samples=None):
    if target_samples is None:
        target_samples = TRUE_TARGET["sample_points"]
    all_intervals = []
    for params in smoke_params_list:
        iv = calc_single_smoke_interval(
            params["drone"], params["speed"], params["direction"],
            params["drop_time"], params["det_delay"], m_name, target_samples
        )
        if iv is not None:
            all_intervals.append(iv)
    merged = merge_intervals(all_intervals)
    total = sum(iv[1] - iv[0] for iv in merged)
    return merged, total

# ============================ 7. 三导弹区间交集 ============================
def intervals_intersection(intervals_list):
    if not intervals_list:
        return [], 0.0
    for ivs in intervals_list:
        if not ivs:
            return [], 0.0
    events = []
    for ivs in intervals_list:
        for start, end in ivs:
            events.append((start, 1))
            events.append((end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    n_lists = len(intervals_list)
    current_cover = 0
    intersection_intervals = []
    seg_start = None
    for t, delta in events:
        if delta == 1:
            current_cover += 1
            if current_cover == n_lists and seg_start is None:
                seg_start = t
        else:
            if current_cover == n_lists and seg_start is not None:
                if t > seg_start + EPSILON:
                    intersection_intervals.append((seg_start, t))
                seg_start = None
            current_cover -= 1
    total = sum(iv[1] - iv[0] for iv in intersection_intervals)
    return intersection_intervals, total

# ============================ 8. 完整评价函数 ============================
SmokeParams = namedtuple("SmokeParams", ["drone", "speed", "direction", "drop_time", "det_delay"])

def evaluate_deployment(smoke_params_list, target_samples=None):
    if target_samples is None:
        target_samples = TRUE_TARGET["sample_points"]
    params_list = []
    for p in smoke_params_list:
        if isinstance(p, SmokeParams):
            params_list.append(p._asdict())
        else:
            params_list.append(p)
    per_missile = {}
    missile_intervals_list = []
    for m_name in MISSILES.keys():
        merged, duration = calc_missile_total_shielding(params_list, m_name, target_samples)
        per_missile[m_name] = {"intervals": merged, "duration": duration}
        missile_intervals_list.append(merged)
    # 三导弹交集（仅作参考，不作为目标函数）
    system_intersection, _ = intervals_intersection(missile_intervals_list)
    # 系统总遮蔽时长 = 各导弹并集长度的总和
    system_duration = sum(info["duration"] for info in per_missile.values())
    per_smoke = []
    for idx, params in enumerate(params_list):
        smoke_info = {"params": params, "per_missile": {}}
        for m_name in MISSILES.keys():
            iv = calc_single_smoke_interval(
                params["drone"], params["speed"], params["direction"],
                params["drop_time"], params["det_delay"], m_name, target_samples
            )
            smoke_info["per_missile"][m_name] = {"interval": iv, "duration": interval_length(iv)}
        per_smoke.append(smoke_info)
    return {
        "per_missile": per_missile,
        "system_intersection": system_intersection,
        "system_duration": system_duration,
        "per_smoke": per_smoke,
    }

# 初始化采样点
if TRUE_TARGET["sample_points"] is None:
    generate_target_samples(n_per_circle=50)
TARGET_SAMPLES = TRUE_TARGET["sample_points"]

# ============================ 第一层：单元优化 ============================
def _grid_search(drone_name, m_name, bounds, n_per_dim):
    """指定密度的网格搜索"""
    v_vals = np.linspace(bounds[0][0], bounds[0][1], n_per_dim)
    theta_vals = np.linspace(bounds[1][0], bounds[1][1], n_per_dim)
    t_drop_vals = np.linspace(bounds[2][0], bounds[2][1], n_per_dim)
    t_bd_vals = np.linspace(bounds[3][0], bounds[3][1], n_per_dim)
    best_dur = 0
    best_params = None
    feasible_points = []
    for v in v_vals:
        for theta in theta_vals:
            for t_drop in t_drop_vals:
                for t_bd in t_bd_vals:
                    drop_pos = get_drone_pos(drone_name, t_drop, v, theta)
                    det_z = drop_pos[2] - 0.5 * G * t_bd ** 2
                    if det_z < SMOKE_MIN_HEIGHT:
                        continue
                    iv = calc_single_smoke_interval(drone_name, v, theta, t_drop, t_bd, m_name, TARGET_SAMPLES)
                    dur = interval_length(iv)
                    if dur > 0:
                        feasible_points.append([v, theta, t_drop, t_bd])
                        if dur > best_dur:
                            best_dur = dur
                            best_params = [v, theta, t_drop, t_bd]
    return best_params, feasible_points, best_dur

def _coarse_grid_search(drone_name, m_name, bounds, n_per_dim=8):
    """两阶段粗网格搜索：先用n_per_dim=8，没找到再用12"""
    best_params, feasible_points, best_dur = _grid_search(drone_name, m_name, bounds, n_per_dim)
    if len(feasible_points) == 0:
        best_params, feasible_points, best_dur = _grid_search(drone_name, m_name, bounds, 15)
    return best_params, feasible_points, best_dur

def unit_optimization(drone_name, m_name, n_top=3, popsize=30, maxiter=150, seed=42):
    m_flight_time = MISSILES[m_name]["flight_time"]
    drone_init_z = DRONES[drone_name]["init_pos"][2]
    max_det_delay_by_height = np.sqrt(2 * max(drone_init_z - SMOKE_MIN_HEIGHT, 0.1) / G)
    max_det_delay = max_det_delay_by_height  # 物理上限：起爆高度>=2m
    bounds = [
        (70.0, 140.0),
        (0.0, 2 * np.pi),
        (0.0, m_flight_time * 0.8),
        (0.1, max(0.2, max_det_delay)),
    ]
    def objective(params):
        v, theta, t_drop, t_bd = params
        drop_pos = get_drone_pos(drone_name, t_drop, v, theta)
        det_z = drop_pos[2] - 0.5 * G * t_bd ** 2
        if det_z < SMOKE_MIN_HEIGHT:
            return 1000.0
        iv = calc_single_smoke_interval(drone_name, v, theta, t_drop, t_bd, m_name, TARGET_SAMPLES)
        dur = interval_length(iv)
        if dur <= 0:
            return 1000.0
        return -dur

    # 第一步：粗网格搜索找到可行点
    best_grid_params, feasible_points, best_grid_dur = _coarse_grid_search(drone_name, m_name, bounds, n_per_dim=8)

    all_solutions = []
    # 如果网格搜索找到可行点，加入解列表
    if best_grid_params is not None:
        v, theta, t_drop, t_bd = best_grid_params
        iv = calc_single_smoke_interval(drone_name, v, theta, t_drop, t_bd, m_name, TARGET_SAMPLES)
        all_solutions.append({
            "speed": v, "direction": theta,
            "drop_time": t_drop, "det_delay": t_bd,
            "interval": iv, "duration": best_grid_dur,
        })

    # 第二步：以可行点为中心生成初始种群，做DE优化
    if feasible_points:
        # 生成初始种群：可行点 + 随机扰动
        rng = np.random.RandomState(seed)
        init_pop = []
        # 加入所有可行点（最多popsize个）
        for fp in feasible_points[:popsize]:
            init_pop.append(fp)
        # 如果可行点不够，用随机扰动填充
        while len(init_pop) < popsize:
            base = feasible_points[rng.randint(len(feasible_points))]
            perturbed = []
            for j in range(4):
                low, high = bounds[j]
                noise = rng.normal(0, (high - low) * 0.1)
                perturbed.append(np.clip(base[j] + noise, low, high))
            init_pop.append(perturbed)
        init_pop = np.array(init_pop[:popsize])

        for run in range(2):
            try:
                result = differential_evolution(
                    objective, bounds, maxiter=maxiter, popsize=popsize,
                    seed=seed + run * 100, tol=1e-6,
                    mutation=(0.5, 1.0), recombination=0.7, polish=True,
                    init=init_pop if run == 0 else 'random',
                )
                if result.fun < 999:
                    v, theta, t_drop, t_bd = result.x
                    iv = calc_single_smoke_interval(drone_name, v, theta, t_drop, t_bd, m_name, TARGET_SAMPLES)
                    dur = interval_length(iv)
                    if dur > 0:
                        all_solutions.append({
                            "speed": v, "direction": theta,
                            "drop_time": t_drop, "det_delay": t_bd,
                            "interval": iv, "duration": dur,
                        })
            except Exception:
                continue
    else:
        # 网格搜索没找到可行点，用默认DE（多跑几次）
        for run in range(5):
            try:
                result = differential_evolution(
                    objective, bounds, maxiter=maxiter, popsize=popsize,
                    seed=seed + run * 100, tol=1e-6,
                    mutation=(0.5, 1.0), recombination=0.7, polish=True,
                )
                if result.fun < 999:
                    v, theta, t_drop, t_bd = result.x
                    iv = calc_single_smoke_interval(drone_name, v, theta, t_drop, t_bd, m_name, TARGET_SAMPLES)
                    dur = interval_length(iv)
                    if dur > 0:
                        all_solutions.append({
                            "speed": v, "direction": theta,
                            "drop_time": t_drop, "det_delay": t_bd,
                            "interval": iv, "duration": dur,
                        })
            except Exception:
                continue

    if not all_solutions:
        return []
    unique = []
    seen = set()
    for sol in sorted(all_solutions, key=lambda x: -x["duration"]):
        key = (round(sol["speed"], 1), round(sol["direction"], 2),
               round(sol["drop_time"], 2), round(sol["det_delay"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(sol)
        if len(unique) >= n_top:
            break
    return unique

def build_unit_optimization_matrix(verbose=True):
    matrix = {}
    total_start = time.time()
    for drone_name in DRONES.keys():
        matrix[drone_name] = {}
        for m_name in MISSILES.keys():
            if verbose:
                print(f"  单元优化: {drone_name} -> {m_name} ...", end=" ", flush=True)
            start = time.time()
            solutions = unit_optimization(drone_name, m_name, n_top=3)
            elapsed = time.time() - start
            matrix[drone_name][m_name] = solutions
            if verbose:
                best_dur = solutions[0]["duration"] if solutions else 0.0
                print(f"最优{best_dur:.2f}s ({len(solutions)}个解), 耗时{elapsed:.1f}s")
    if verbose:
        print(f"  单元优化总耗时: {time.time() - total_start:.1f}s")
    return matrix

# ============================ 第二层：路径拟合 ============================
def _optimize_timing_fixed_trajectory(drone_name, speed, direction, m_name, maxiter=50, init_points=None):
    m_flight_time = MISSILES[m_name]["flight_time"]
    drone_init_z = DRONES[drone_name]["init_pos"][2]
    max_det_delay = min(10.0, np.sqrt(2 * max(drone_init_z - SMOKE_MIN_HEIGHT, 0.1) / G))
    bounds = [(0.0, m_flight_time * 0.8), (0.1, max(0.2, max_det_delay))]
    def objective(params):
        t_drop, t_bd = params
        drop_pos = get_drone_pos(drone_name, t_drop, speed, direction)
        det_z = drop_pos[2] - 0.5 * G * t_bd ** 2
        if det_z < SMOKE_MIN_HEIGHT:
            return 1000.0
        iv = calc_single_smoke_interval(drone_name, speed, direction, t_drop, t_bd, m_name, TARGET_SAMPLES)
        dur = interval_length(iv)
        if dur <= 0:
            return 1000.0
        return -dur
    try:
        # 如果有初始点，以初始点为中心生成种群
        if init_points is not None and len(init_points) > 0:
            rng = np.random.RandomState(42)
            popsize = 15
            init_pop = []
            for ip in init_points[:popsize]:
                init_pop.append(np.clip(ip, [b[0] for b in bounds], [b[1] for b in bounds]))
            # 用随机扰动填充
            while len(init_pop) < popsize:
                base = init_points[rng.randint(len(init_points))]
                perturbed = []
                for j in range(2):
                    low, high = bounds[j]
                    noise = rng.normal(0, (high - low) * 0.1)
                    perturbed.append(np.clip(base[j] + noise, low, high))
                init_pop.append(perturbed)
            init_pop = np.array(init_pop[:popsize])
            result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, seed=42, tol=1e-6, polish=True, init=init_pop)
        else:
            result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=15, seed=42, tol=1e-6, polish=True)
        if result.fun < 999:
            t_drop, t_bd = result.x
            return calc_single_smoke_interval(drone_name, speed, direction, t_drop, t_bd, m_name, TARGET_SAMPLES)
    except Exception:
        pass
    return None

def _refine_trajectory_local(drone_name, init_speed, init_direction, maxiter=20):
    bounds = [
        (max(70, init_speed - 5), min(140, init_speed + 5)),
        (init_direction - np.radians(5), init_direction + np.radians(5)),
    ]
    def objective(params):
        speed, direction = params
        total = 0.0
        for m_name in MISSILES.keys():
            iv = _optimize_timing_fixed_trajectory(drone_name, speed, direction, m_name, maxiter=20)
            total += interval_length(iv)
        return -total if total > 0 else 1000.0
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=10, seed=42, tol=1e-6, polish=True)
        if result.fun < 999:
            speed, direction = result.x
            per_missile = {}
            total = 0.0
            for m_name in MISSILES.keys():
                iv = _optimize_timing_fixed_trajectory(drone_name, speed, direction, m_name, maxiter=30)
                dur = interval_length(iv)
                per_missile[m_name] = {"interval": iv, "duration": dur}
                total += dur
            return {"speed": speed, "direction": direction, "total_duration": total, "per_missile": per_missile}
    except Exception:
        pass
    return {"speed": init_speed, "direction": init_direction, "total_duration": 0.0, "per_missile": {}}

def fit_drone_trajectory(drone_name, unit_matrix, speed_candidates=None, verbose=False):
    """
    路径拟合（极速版）：直接取单元优化最优解的速度方向，不做嵌套DE。
    对每枚导弹在固定速度方向下优化时序，计算总遮蔽时长。
    """
    all_sols = []
    for m_name in MISSILES.keys():
        for sol in unit_matrix[drone_name].get(m_name, []):
            all_sols.append({**sol, "missile": m_name})
    if not all_sols:
        return {"speed": 100.0, "direction": np.pi, "total_duration": 0.0, "per_missile": {}}

    # 取单元优化最优解的速度方向
    best_sol = max(all_sols, key=lambda x: x["duration"])
    speed = best_sol["speed"]
    direction = best_sol["direction"]

    # 对每枚导弹在固定速度方向下优化时序，使用单元优化的t_drop/t_bd作为初始点
    per_missile = {}
    total = 0.0
    for m_name in MISSILES.keys():
        # 收集该无人机对该导弹的单元优化解的t_drop和t_bd
        init_points = []
        for sol in unit_matrix[drone_name].get(m_name, []):
            init_points.append([sol["drop_time"], sol["det_delay"]])
        iv = _optimize_timing_fixed_trajectory(drone_name, speed, direction, m_name, maxiter=30, init_points=init_points if init_points else None)
        dur = interval_length(iv)
        per_missile[m_name] = {"interval": iv, "duration": dur}
        total += dur

    return {"speed": speed, "direction": direction, "total_duration": total, "per_missile": per_missile}

def fit_all_drone_trajectories(unit_matrix, verbose=True):
    trajectories = {}
    for drone_name in DRONES.keys():
        if verbose:
            print(f"  路径拟合: {drone_name} ...", end=" ", flush=True)
        start = time.time()
        traj = fit_drone_trajectory(drone_name, unit_matrix)
        elapsed = time.time() - start
        trajectories[drone_name] = traj
        if verbose:
            print(f"速度{traj['speed']:.1f}m/s, 方向{np.degrees(traj['direction']):.1f}°, "
                  f"总时长{traj['total_duration']:.2f}s, 耗时{elapsed:.1f}s")
    return trajectories

# ============================ 第三层：全局分配与时序优化 ============================
def build_effectiveness_matrix(trajectories, unit_matrix=None, verbose=True):
    matrix = {}
    for drone_name in DRONES.keys():
        matrix[drone_name] = {}
        traj = trajectories[drone_name]
        for m_name in MISSILES.keys():
            if verbose:
                print(f"  效能矩阵: {drone_name} -> {m_name} ...", end=" ", flush=True)
            # 使用单元优化的t_drop/t_bd作为初始点
            init_points = []
            if unit_matrix is not None:
                for sol in unit_matrix[drone_name].get(m_name, []):
                    init_points.append([sol["drop_time"], sol["det_delay"]])
            iv = _optimize_timing_fixed_trajectory(drone_name, traj["speed"], traj["direction"], m_name, maxiter=40, init_points=init_points if init_points else None)
            dur = interval_length(iv)
            matrix[drone_name][m_name] = {"interval": iv, "duration": dur}
            if verbose:
                print(f"{dur:.2f}s")
    return matrix

def _infer_timing_from_interval(drone_name, traj, interval, m_name):
    if interval is None:
        return 0.0, 1.0
    traj_speed = traj["speed"]
    traj_dir = traj["direction"]
    drone_init_z = DRONES[drone_name]["init_pos"][2]
    best_t_drop = 0.0
    best_t_bd = 1.0
    best_diff = float('inf')
    for t_bd in np.linspace(0.5, min(8.0, np.sqrt(2*max(drone_init_z-2,0.1)/G)), 20):
        t_det = interval[0] - 0.5
        t_drop = max(0, t_det - t_bd)
        iv = calc_single_smoke_interval(drone_name, traj_speed, traj_dir, t_drop, t_bd, m_name, TARGET_SAMPLES)
        if iv is not None and isinstance(iv, tuple):
            diff = abs(iv[0] - interval[0]) + abs(iv[1] - interval[1])
            if diff < best_diff:
                best_diff = diff
                best_t_drop = t_drop
                best_t_bd = t_bd
    return best_t_drop, best_t_bd

def enumerate_assignment(unit_matrix, verbose=True):
    """枚举任务分配方案：每架无人机选1枚导弹，用单元优化最优解计算各导弹总和"""
    drone_names = list(DRONES.keys())
    missile_names = list(MISSILES.keys())
    best_total = -1
    best_assignment = []
    best_missile_durations = {}

    # 枚举每架无人机的选择：0=M1, 1=M2, 2=M3, 3=不参与
    from itertools import product
    for choices in product(range(4), repeat=len(drone_names)):
        # 检查三枚导弹都被至少一架无人机干扰
        covered = set()
        for c in choices:
            if c < 3:
                covered.add(missile_names[c])
        if len(covered) < 3:
            continue

        # 收集每枚导弹的所有烟幕弹区间
        missile_intervals = {m: [] for m in missile_names}
        assignment = []
        for i, c in enumerate(choices):
            if c >= 3:
                continue
            drone = drone_names[i]
            missile = missile_names[c]
            sols = unit_matrix[drone].get(missile, [])
            if not sols:
                continue
            sol = sols[0]  # 取最优解
            iv = calc_single_smoke_interval(drone, sol["speed"], sol["direction"],
                                              sol["drop_time"], sol["det_delay"], missile, TARGET_SAMPLES)
            if iv is not None and isinstance(iv, (tuple, list)) and len(iv) == 2:
                missile_intervals[missile].append(iv)
                assignment.append({
                    "drone": drone, "missile": missile,
                    "speed": sol["speed"], "direction": sol["direction"],
                    "drop_time": sol["drop_time"], "det_delay": sol["det_delay"],
                    "interval": iv, "duration": iv[1] - iv[0]
                })

        # 计算各导弹并集长度总和
        total = 0.0
        missile_durations = {}
        for m in missile_names:
            if missile_intervals[m]:
                merged = merge_intervals(missile_intervals[m])
                dur = sum(iv[1] - iv[0] for iv in merged)
                missile_durations[m] = dur
                total += dur
            else:
                missile_durations[m] = 0.0

        if total > best_total:
            best_total = total
            best_assignment = assignment
            best_missile_durations = missile_durations

    if verbose:
        print(f"  枚举任务分配: 最优总和{best_total:.4f}s")
        for m, dur in best_missile_durations.items():
            print(f"    {m}: {dur:.4f}s")
        print(f"  分配方案: ", end="")
        for a in best_assignment:
            print(f"{a['drone']}->{a['missile']}({a['duration']:.2f}s) ", end="")
        print()

    return best_assignment, best_total, best_missile_durations

def greedy_assignment(effectiveness_matrix, trajectories, unit_matrix=None, max_smokes_per_drone=3, verbose=True):
    candidates = []
    for drone_name in DRONES.keys():
        for m_name in MISSILES.keys():
            info = effectiveness_matrix[drone_name][m_name]
            if info["duration"] > 0.1:
                candidates.append({"drone": drone_name, "missile": m_name, "duration": info["duration"], "interval": info["interval"]})
    candidates.sort(key=lambda x: -x["duration"])
    assignment = []
    drone_smoke_count = {d: 0 for d in DRONES.keys()}
    missile_coverage = {m: 0 for m in MISSILES.keys()}
    for cand in candidates:
        drone_name = cand["drone"]
        m_name = cand["missile"]
        if drone_smoke_count[drone_name] >= max_smokes_per_drone:
            continue
        traj = trajectories[drone_name]
        # 直接使用效能矩阵中的interval，不重新优化
        iv = cand["interval"]
        if iv is None or interval_length(iv) <= 0:
            continue
        t_drop, t_bd = _infer_timing_from_interval(drone_name, traj, iv, m_name)
        assignment.append({
            "drone": drone_name, "missile": m_name,
            "speed": traj["speed"], "direction": traj["direction"],
            "drop_time": t_drop, "det_delay": t_bd,
            "interval": iv, "duration": interval_length(iv),
        })
        drone_smoke_count[drone_name] += 1
        missile_coverage[m_name] += 1
        if verbose:
            print(f"  分配: {drone_name} -> {m_name}, 时长{interval_length(iv):.2f}s")
    return assignment

def _evaluate_system_duration(assignment):
    """系统总遮蔽时长 = 各导弹遮蔽区间并集长度的总和（正确答案定义）"""
    if not assignment:
        return 0.0
    # 按导弹分组
    missile_intervals = {m: [] for m in MISSILES.keys()}
    for a in assignment:
        iv = a.get("interval")
        if iv is not None and isinstance(iv, (tuple, list)) and len(iv) == 2:
            missile_intervals[a["missile"]].append(iv)
    # 计算每枚导弹的并集长度，然后求和
    total = 0.0
    for m, ivs in missile_intervals.items():
        if ivs:
            merged = merge_intervals(ivs)
            total += sum(iv[1] - iv[0] for iv in merged)
    return total

def local_exchange_optimization(assignment, trajectories, effectiveness_matrix, n_iterations=10, verbose=True):
    """局部交换优化：使用效能矩阵中已有的interval，不重新调用时序优化"""
    current_assignment = list(assignment)
    current_system_dur = _evaluate_system_duration(current_assignment)
    if verbose:
        print(f"  局部交换优化: 初始系统时长{current_system_dur:.4f}s")
    for iteration in range(n_iterations):
        improved = False
        # 尝试改分配：将某枚弹改分配给另一枚导弹
        for idx in range(len(current_assignment)):
            orig = current_assignment[idx]
            for new_missile in MISSILES.keys():
                if new_missile == orig["missile"]:
                    continue
                # 使用效能矩阵中已有的interval
                eff = effectiveness_matrix[orig["drone"]].get(new_missile, {})
                iv = eff.get("interval")
                if iv is not None and interval_length(iv) > 0:
                    t_drop, t_bd = _infer_timing_from_interval(orig["drone"], trajectories[orig["drone"]], iv, new_missile)
                    new_assignment = list(current_assignment)
                    new_assignment[idx] = {**orig, "missile": new_missile, "drop_time": t_drop, "det_delay": t_bd, "interval": iv, "duration": interval_length(iv)}
                    new_dur = _evaluate_system_duration(new_assignment)
                    if new_dur > current_system_dur + 0.01:
                        current_assignment = new_assignment
                        current_system_dur = new_dur
                        improved = True
                        if verbose:
                            print(f"    迭代{iteration+1}: 弹{idx+1}改分配给{new_missile}, 系统时长{current_system_dur:.4f}s")
                        break
            if improved:
                break
        if not improved:
            if verbose:
                print(f"    迭代{iteration+1}: 无提升，提前终止")
            break
    return current_assignment, current_system_dur

# ============================ 逆推分层主函数 ============================
def reverse_hierarchical_optimization(verbose=True):
    total_start = time.time()
    if verbose:
        print("=" * 70)
        print("逆推分层优化 —— 问题五")
        print("=" * 70)
    if verbose:
        print("\n【第一层】单元优化（5×3=15组，DE 4维）")
    unit_matrix = build_unit_optimization_matrix(verbose=verbose)
    if verbose:
        print("\n【第二层】枚举任务分配（4^5=1024种，筛选三导弹都被覆盖的方案）")
    assignment, best_total, best_missile_durations = enumerate_assignment(unit_matrix, verbose=verbose)
    # 直接使用assignment中的interval计算结果，不重新计算
    missile_intervals = {m: [] for m in MISSILES.keys()}
    per_smoke = []
    for a in assignment:
        iv = a.get("interval")
        if iv is not None and isinstance(iv, (tuple, list)) and len(iv) == 2:
            missile_intervals[a["missile"]].append(iv)
            per_smoke.append({"drone": a["drone"], "missile": a["missile"], "interval": iv, "duration": iv[1]-iv[0]})
        else:
            per_smoke.append({"drone": a["drone"], "missile": a["missile"], "interval": None, "duration": 0})
    missile_unions = {}
    per_missile = {}
    for m, ivs in missile_intervals.items():
        merged = merge_intervals(ivs) if ivs else []
        union_dur = sum(iv[1]-iv[0] for iv in merged)
        missile_unions[m] = merged
        per_missile[m] = {"duration": union_dur, "intervals": merged}
    all_unions = [u for u in missile_unions.values() if u]
    if len(all_unions) == 3:
        system_intersection, _ = intervals_intersection(all_unions)
    else:
        system_intersection = []
    # 系统总遮蔽时长 = 各导弹遮蔽区间并集长度的总和（正确答案定义）
    system_duration = sum(per_missile[m]["duration"] for m in MISSILES.keys())
    intersection_duration = sum(iv[1]-iv[0] for iv in system_intersection)
    total_elapsed = time.time() - total_start
    result = {
        "assignment": assignment,
        "system_duration": system_duration,
        "system_intersection": system_intersection,
        "intersection_duration": intersection_duration,
        "per_missile": per_missile,
        "per_smoke": per_smoke,
        "unit_matrix": unit_matrix,
        "total_time": total_elapsed,
        "n_smokes": len(assignment),
    }
    if verbose:
        print("\n" + "=" * 70)
        print("逆推分层优化结果")
        print("=" * 70)
        print(f"总耗时: {total_elapsed:.1f}s")
        print(f"投放烟幕弹数量: {len(assignment)}")
        print(f"系统总遮蔽时长（各导弹并集总和）: {system_duration:.4f}s")
        print(f"三导弹同时遮蔽（交集）: {intersection_duration:.4f}s")
        print(f"系统遮蔽区间（交集）: {system_intersection}")
        print("\n各导弹遮蔽情况:")
        for m_name, info in per_missile.items():
            print(f"  {m_name}: 并集时长{info['duration']:.2f}s, 区间{info['intervals']}")
        print("\n各烟幕弹详情:")
        for idx, a in enumerate(assignment):
            print(f"  弹{idx+1}: {a['drone']}->{a['missile']}, "
                  f"投放{a['drop_time']:.2f}s, 起爆延迟{a['det_delay']:.2f}s, 时长{a['duration']:.2f}s")
    return result

if __name__ == "__main__":
    result = reverse_hierarchical_optimization(verbose=True)
