import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')
# ============================ 1. 全局参数初始化 ============================
# 真目标参数
TRUE_TARGET = {
    "r": 7,  # 圆柱半径
    "h": 10,  # 圆柱高度
    "center": np.array([0, 200, 0]),  # 底面圆心
    "sample_points": None  # 采样点
}
# 导弹参数
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "dir": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "dir": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "dir": None, "flight_time": None}
}
MISSILE_SPEED = 300  # 导弹速度(m/s)
G = 9.8  # 重力加速度(m/s²)
SMOKE_RADIUS = 10  # 烟幕有效半径(m)
SMOKE_SINK_SPEED = 3  # 起爆后下沉速度(m/s)
SMOKE_EFFECTIVE_TIME = 20  # 起爆后有效时长(s)
# 无人机参数（修改：添加 params_fixed标记）
DRONES = {
    "FY1": {"init_pos": np.array([17800, 0, 1800]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "params_fixed": False, "optimized": False},
    "FY2": {"init_pos": np.array([12000, 1400, 1400]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "params_fixed": False, "optimized": False},
    "FY3": {"init_pos": np.array([6000, -3000, 700]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "params_fixed": False, "optimized": False},
    "FY4": {"init_pos": np.array([11000, 2000, 1800]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "params_fixed": False, "optimized": False},
    "FY5": {"init_pos": np.array([13000, -2000, 1300]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "params_fixed": False, "optimized": False}
}

DROP_INTERVAL = 1  # 同一无人机投放间隔(s)
TIME_STEP = 0.1  # 时间采样步长(s)

# 新增：连续性约束参数
MAX_GAP_ALLOWED = 2.0  # 最大允许间隙时间(s)
CONTINUITY_WEIGHT = 0.7  # 连续性权重
EFFECTIVENESS_WEIGHT = 0.3  # 效果权重

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================ 2. 连续性分析函数 ============================

def calculate_continuous_shielding_time(smoke_intervals):
    """
    计算连续遮蔽时间（考虑间隙）

    Args:
        smoke_intervals: 烟幕时间区间列表 [{'start': t1, 'end': t2}, ...]

    Returns:
        continuous_time: 连续遮蔽时间
        gap_penalty: 间隙惩罚
        merged_intervals: 合并后的区间
    """
    if not smoke_intervals:
        return 0.0, 0.0, []

    # 按开始时间排序
    sorted_intervals = sorted(smoke_intervals, key=lambda x: x['start'])

    merged = []
    current = sorted_intervals[0].copy()
    gap_penalty = 0.0

    for next_interval in sorted_intervals[1:]:
        gap = next_interval['start'] - current['end']

        if gap <= MAX_GAP_ALLOWED:
            # 间隙小于阈值，合并区间
            current['end'] = max(current['end'], next_interval['end'])
            if gap > 0:
                gap_penalty += gap  # 累计间隙惩罚
        else:
            # 间隙过大，保存当前区间，开始新区间
            merged.append(current)
            current = next_interval.copy()
            gap_penalty += MAX_GAP_ALLOWED  # 最大惩罚

    # 添加最后一个区间
    merged.append(current)

    # 计算总连续时间
    continuous_time = sum(interval['end'] - interval['start'] for interval in merged)

    return continuous_time, gap_penalty, merged


def get_missile_existing_intervals(missile_name):
    """
    获取某导弹现有的遮蔽时间区间

    Args:
        missile_name: 导弹名称

    Returns:
        intervals: 现有时间区间列表
    """
    intervals = []

    for drone_name, drone_data in DRONES.items():
        for smoke in drone_data["smokes"]:
            if smoke["missile"] == missile_name:
                intervals.append({
                    'start': smoke["det_time"],
                    'end': smoke["det_time"] + smoke["effective_time"],
                    'source': f"{drone_name}-{len(intervals) + 1}"
                })

    return intervals


def evaluate_continuity_benefit(missile_name, new_start_time, new_duration):
    """
    评估新烟幕弹对连续性的贡献

    Args:
        missile_name: 目标导弹
        new_start_time: 新烟幕弹起爆时间
        new_duration: 新烟幕弹持续时间

    Returns:
        continuity_score: 连续性评分
        gap_reduction: 间隙减少量
    """
    # 获取现有区间
    existing_intervals = get_missile_existing_intervals(missile_name)

    if not existing_intervals:
        # 如果是第一个烟幕弹，返回基础分数
        return new_duration, 0.0

    # 计算添加新烟幕弹前的连续性
    old_continuous_time, old_gap_penalty, _ = calculate_continuous_shielding_time(existing_intervals)

    # 添加新烟幕弹
    new_intervals = existing_intervals + [{
        'start': new_start_time,
        'end': new_start_time + new_duration
    }]

    # 计算添加后的连续性
    new_continuous_time, new_gap_penalty, _ = calculate_continuous_shielding_time(new_intervals)

    # 连续性改善
    continuity_improvement = new_continuous_time - old_continuous_time
    gap_reduction = old_gap_penalty - new_gap_penalty

    # 综合评分
    continuity_score = continuity_improvement + gap_reduction * 0.5

    return continuity_score, gap_reduction


# ============================ 3. 核心工具函数（保持不变） ============================

# 生成真目标采样点
def generate_true_target_samples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]

    # 底面采样
    samples.append(center)
    for theta in np.linspace(0, 2 * np.pi, 15):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))

    # 顶面采样
    top_center = center + np.array([0, 0, h])
    samples.append(top_center)
    for theta in np.linspace(0, 2 * np.pi, 15):
        x = top_center[0] + r * np.cos(theta)
        y = top_center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, top_center[2]]))

    # 侧面采样
    for z in np.linspace(center[2], top_center[2], 5):
        for theta in np.linspace(0, 2 * np.pi, 12):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))

    TRUE_TARGET["sample_points"] = np.array(samples)


# 初始化导弹参数
def init_missiles():
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        dir_vec = -init_pos / np.linalg.norm(init_pos)
        m_data["dir"] = dir_vec * MISSILE_SPEED
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED


# 导弹位置计算
def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    if t > m_data["flight_time"]:
        return m_data["init_pos"] + m_data["dir"] * m_data["flight_time"]
    return m_data["init_pos"] + m_data["dir"] * t


# 无人机位置计算（修改：支持临时参数）
def get_drone_pos(drone_name, t, temp_speed=None, temp_direction=None):
    drone = DRONES[drone_name]

    # 使用临时参数或固化参数
    speed = temp_speed if temp_speed is not None else drone["speed"]
    direction = temp_direction if temp_direction is not None else drone["direction"]

    if speed is None or direction is None:
        return drone["init_pos"]

    v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0])
    return drone["init_pos"] + v_vec * t


# 烟幕弹位置计算（修改：支持临时参数）
def get_smoke_pos(drone_name, drop_time, det_delay, t, temp_speed=None, temp_direction=None):
    if t < drop_time:
        return None

    drop_pos = get_drone_pos(drone_name, drop_time, temp_speed, temp_direction)

    # 使用临时参数或固化参数
    drone = DRONES[drone_name]
    speed = temp_speed if temp_speed is not None else drone["speed"]
    direction = temp_direction if temp_direction is not None else drone["direction"]

    # 投放后到起爆前
    if t < drop_time + det_delay:
        delta_t = t - drop_time
        v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0])
        x = drop_pos[0] + v_vec[0] * delta_t
        y = drop_pos[1] + v_vec[1] * delta_t
        z = drop_pos[2] - 0.5 * G * delta_t ** 2
        return np.array([x, y, max(z, 0.1)])

    # 起爆后
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None

    # 计算起爆位置
    delta_t_det = det_delay
    v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0])
    det_x = drop_pos[0] + v_vec[0] * delta_t_det
    det_y = drop_pos[1] + v_vec[1] * delta_t_det
    det_z = drop_pos[2] - 0.5 * G * delta_t_det ** 2

    if det_z < 0:
        det_z = 0.1

    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])


# 线段与球相交判定
def segment_sphere_intersect(p1, p2, center, radius):
    vec_p = p2 - p1
    vec_c = center - p1
    t = np.dot(vec_c, vec_p) / (np.dot(vec_p, vec_p) + 1e-8)

    if 0 <= t <= 1:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0 else p2

    return np.linalg.norm(nearest - center) <= radius + 1e-8


# 单烟幕弹遮蔽时长计算
def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay, temp_speed=None, temp_direction=None):
    drone = DRONES[drone_name]
    # 使用临时参数或固化参数
    v = temp_speed if temp_speed is not None else drone["speed"]
    theta = temp_direction if temp_direction is not None else drone["direction"]

    if v is None or theta is None:
        return -1000

    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):
        return -1000

    # 检查起爆点有效性
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time, temp_speed, temp_direction)
    delta_t_det = det_delay
    det_z = drop_pos[2] - 0.5 * G * delta_t_det ** 2

    if det_z < -0.5:
        return -1000

    # 检查投放间隔
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000

    # 计算有效时长
    max_t = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILES[m_name]["flight_time"] + 1)
    min_t = max(det_time, 0)

    if min_t >= max_t - 1e-3:
        return 0

    effective_duration = 0
    for t in np.arange(min_t, max_t, TIME_STEP):
        m_pos = get_missile_pos(m_name, t)
        smoke_pos = get_smoke_pos(drone_name, drop_time, det_delay, t, temp_speed, temp_direction)

        if smoke_pos is None:
            continue

        all_intersect = True
        for sample in TRUE_TARGET["sample_points"]:
            if not segment_sphere_intersect(m_pos, sample, smoke_pos, SMOKE_RADIUS):
                all_intersect = False
                break

        if all_intersect:
            effective_duration += TIME_STEP

    return effective_duration


# ============================ 4. 修改的优化函数（增加连续性约束） ============================

def single_smoke_optimization_with_continuity(drone_name, missile_name, speed_fixed=None, direction_fixed=None):
    """单烟幕弹优化（增加连续性约束）"""

    def objective(params):
        drone = DRONES[drone_name]

        # 根据参数固化情况确定优化变量
        if speed_fixed is not None and direction_fixed is not None:
            # 速度和方向都固定，只优化时间参数
            v, theta, drop_time, det_delay = speed_fixed, direction_fixed, params[0], params[1]
        elif speed_fixed is not None:
            # 只固定速度
            v, theta, drop_time, det_delay = speed_fixed, params[0], params[1], params[2]
        elif direction_fixed is not None:
            # 只固定方向
            v, theta, drop_time, det_delay = params[0], direction_fixed, params[1], params[2]
        else:
            # 都不固定
            v, theta, drop_time, det_delay = params[0], params[1], params[2], params[3]

        # 计算基本有效时间
        effective_time = calc_smoke_effective_time(drone_name, missile_name, drop_time, det_delay, v, theta)

        if effective_time <= 0:
            return 1000  # 返回高惩罚值

        # 计算连续性贡献
        det_time = drop_time + det_delay
        continuity_score, gap_reduction = evaluate_continuity_benefit(missile_name, det_time, effective_time)

        # 综合评分（负值因为differential_evolution是最小化）
        total_score = -(EFFECTIVENESS_WEIGHT * effective_time + CONTINUITY_WEIGHT * continuity_score)

        return total_score

    # 设置参数边界（与原版相同）
    drone = DRONES[drone_name]
    missile_flight_time = MISSILES[missile_name]["flight_time"]

    bounds = []

    # 根据固化情况设置边界
    if speed_fixed is not None and direction_fixed is not None:
        bounds = [
            (0, missile_flight_time * 0.8),  # 投放时间
            (0.5, 8.0)  # 起爆延迟
        ]
    elif speed_fixed is not None:
        bounds = [
            (0, 2 * np.pi),  # 方向角
            (0, missile_flight_time * 0.8),  # 投放时间
            (0.5, 8.0)  # 起爆延迟
        ]
    elif direction_fixed is not None:
        bounds = [
            (drone["speed_range"][0], drone["speed_range"][1]),  # 速度
            (0, missile_flight_time * 0.8),  # 投放时间
            (0.5, 8.0)  # 起爆延迟
        ]
    else:
        bounds = [
            (drone["speed_range"][0], drone["speed_range"][1]),  # 速度
            (0, 2 * np.pi),  # 方向角
            (0, missile_flight_time * 0.8),  # 投放时间
            (0.5, 8.0)  # 起爆延迟
        ]

    # 多次优化取最好结果
    best_result = None
    best_value = float('inf')

    for _ in range(5):
        result = differential_evolution(objective, bounds, maxiter=100, seed=np.random.randint(0, 10000))
        if result.fun < best_value:
            best_value = result.fun
            best_result = result

    if best_result is None or best_value >= 100:
        return None

    # 构造返回结果
    if speed_fixed is not None and direction_fixed is not None:
        v, theta, drop_time, det_delay = speed_fixed, direction_fixed, best_result.x[0], best_result.x[1]
    elif speed_fixed is not None:
        v, theta, drop_time, det_delay = speed_fixed, best_result.x[0], best_result.x[1], best_result.x[2]
    elif direction_fixed is not None:
        v, theta, drop_time, det_delay = best_result.x[0], direction_fixed, best_result.x[1], best_result.x[2]
    else:
        v, theta, drop_time, det_delay = best_result.x[0], best_result.x[1], best_result.x[2], best_result.x[3]

    # 计算最终效果
    effective_time = calc_smoke_effective_time(drone_name, missile_name, drop_time, det_delay, v, theta)
    det_time = drop_time + det_delay
    continuity_score, gap_reduction = evaluate_continuity_benefit(missile_name, det_time, effective_time)

    return {
        "speed": v,
        "direction": theta,
        "drop_time": drop_time,
        "det_delay": det_delay,
        "effective_time": effective_time,
        "continuity_score": continuity_score,
        "gap_reduction": gap_reduction
    }


def path_fitting_optimization_with_continuity(drone_name, missile_name, speed_candidates):
    """路径拟合优化（增加连续性考虑）"""
    drone = DRONES[drone_name]

    # 如果参数已固化，只优化时间参数
    if drone["params_fixed"]:
        solution = single_smoke_optimization_with_continuity(drone_name, missile_name,
                                                             speed_fixed=drone["speed"],
                                                             direction_fixed=drone["direction"])
        return solution

    # 参数未固化，进行完整的路径拟合优化
    all_solutions = []

    # 对每个速度候选值进行单弹优化
    for speed in speed_candidates:
        for _ in range(3):
            solution = single_smoke_optimization_with_continuity(drone_name, missile_name, speed_fixed=speed)
            if solution and solution["effective_time"] > 0.5:
                all_solutions.append(solution)

    if not all_solutions:
        return None

    # 选择最好的解（考虑连续性）
    all_solutions.sort(key=lambda x: (x["continuity_score"] + x["effective_time"]), reverse=True)
    best_solution = all_solutions[0]
    return refine_solution_with_continuity(drone_name, missile_name, best_solution)


def refine_solution_with_continuity(drone_name, missile_name, base_solution):
    """波动优化（增加连续性考虑）"""
    drone = DRONES[drone_name]

    # 如果参数已固化，不进行波动优化
    if drone["params_fixed"]:
        return base_solution

    def objective(params):
        v, theta, drop_time, det_delay = params[0], params[1], params[2], params[3]
        effective_time = calc_smoke_effective_time(drone_name, missile_name, drop_time, det_delay, v, theta)

        if effective_time <= 0:
            return 1000

        det_time = drop_time + det_delay
        continuity_score, gap_reduction = evaluate_continuity_benefit(missile_name, det_time, effective_time)

        # 综合评分
        total_score = -(EFFECTIVENESS_WEIGHT * effective_time + CONTINUITY_WEIGHT * continuity_score)
        return total_score

    # 设置波动范围
    missile_flight_time = MISSILES[missile_name]["flight_time"]

    speed_fluctuation = 10
    direction_fluctuation = np.radians(5)
    time_fluctuation = 1.0
    delay_fluctuation = 0.5

    bounds = [
        (max(drone["speed_range"][0], base_solution["speed"] - speed_fluctuation),
         min(drone["speed_range"][1], base_solution["speed"] + speed_fluctuation)),
        (base_solution["direction"] - direction_fluctuation,
         base_solution["direction"] + direction_fluctuation),
        (max(0, base_solution["drop_time"] - time_fluctuation),
         min(missile_flight_time * 0.8, base_solution["drop_time"] + time_fluctuation)),
        (max(0.5, base_solution["det_delay"] - delay_fluctuation),
         min(8.0, base_solution["det_delay"] + delay_fluctuation))
    ]

    result = differential_evolution(objective, bounds, maxiter=50)

    base_score = EFFECTIVENESS_WEIGHT * base_solution["effective_time"] + CONTINUITY_WEIGHT * base_solution[
        "continuity_score"]

    if result.fun < -base_score:
        # 重新计算结果
        v, theta, drop_time, det_delay = result.x[0], result.x[1], result.x[2], result.x[3]
        effective_time = calc_smoke_effective_time(drone_name, missile_name, drop_time, det_delay, v, theta)
        det_time = drop_time + det_delay
        continuity_score, gap_reduction = evaluate_continuity_benefit(missile_name, det_time, effective_time)

        return {
            "speed": v,
            "direction": theta,
            "drop_time": drop_time,
            "det_delay": det_delay,
            "effective_time": effective_time,
            "continuity_score": continuity_score,
            "gap_reduction": gap_reduction
        }
    else:
        return base_solution


# ============================ 5. 修改的任务分配与迭代优化 ============================

def assign_tasks_and_optimize_with_continuity(available_drones, missile_names, max_smokes_per_drone=3):
    """任务分配和优化（增加连续性考虑）"""
    speed_candidates = np.linspace(70, 140, 8)

    # 为每个可用无人机-导弹组合计算最优解
    optimization_matrix = {}
    for drone_name in available_drones:
        optimization_matrix[drone_name] = {}
        for missile_name in missile_names:
            best_solution = path_fitting_optimization_with_continuity(drone_name, missile_name, speed_candidates)
            optimization_matrix[drone_name][missile_name] = best_solution
            if best_solution:
                print(f"  {drone_name} -> {missile_name}: 效果时长 {best_solution['effective_time']:.2f}s, "
                      f"连续性贡献 {best_solution['continuity_score']:.2f}")

    # 改进的分配策略：优先考虑连续性
    assigned_combinations = []
    drone_smoke_count = {drone: len(DRONES[drone]["smokes"]) for drone in available_drones}
    missile_coverage = {missile: 0 for missile in missile_names}

    # 创建所有有效组合的列表
    valid_combinations = []
    for drone_name in available_drones:
        for missile_name in missile_names:
            solution = optimization_matrix[drone_name][missile_name]
            if solution and solution["effective_time"] > 0.5:
                # 综合评分：效果 + 连续性贡献
                combined_score = (EFFECTIVENESS_WEIGHT * solution["effective_time"] +
                                  CONTINUITY_WEIGHT * solution["continuity_score"])
                valid_combinations.append((drone_name, missile_name, solution, combined_score))

    # 按综合评分排序（连续性优先）
    valid_combinations.sort(key=lambda x: x[3], reverse=True)

    # 连续性优先的分配策略
    for drone_name, missile_name, solution, combined_score in valid_combinations:
        if drone_smoke_count[drone_name] < max_smokes_per_drone:
            # 检查连续性贡献
            existing_intervals = get_missile_existing_intervals(missile_name)

            # 如果该导弹还没有烟幕弹，或者新烟幕弹有助于连续性，则分配
            if not existing_intervals or solution["continuity_score"] > 0 or solution["gap_reduction"] > 0:
                assigned_combinations.append((drone_name, missile_name, solution))
                drone_smoke_count[drone_name] += 1
                missile_coverage[missile_name] += 1
                print(f"  分配: {drone_name} -> {missile_name} (综合评分: {combined_score:.2f}, "
                      f"连续性: {solution['continuity_score']:.2f})")

                # 如果所有无人机都达到最大烟幕弹数，停止
                if all(count >= max_smokes_per_drone for count in drone_smoke_count.values()):
                    break

    return assigned_combinations


def iterative_optimization_with_continuity(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3):
    """迭代优化主函数（增加连续性约束）"""
    all_smokes = []
    iteration = 0
    stall_count = 0
    prev_total_score = 0

    print("开始连续性约束的迭代优化...")

    while iteration < max_iterations and stall_count < max_stall_iter:
        print(f"\n=== 第 {iteration + 1} 轮迭代 ===")

        # 找到可用的无人机
        available_drones = [name for name, data in DRONES.items() if not data["optimized"]]
        if not available_drones:
            print("所有无人机已达到最大烟幕弹数")
            break

        print(f"可用无人机: {available_drones}")
        print("计算各组合的连续性优化效果...")

        # 当前轮次的最佳组合
        current_assignments = assign_tasks_and_optimize_with_continuity(
            available_drones,
            list(MISSILES.keys()),
            max_smokes_per_drone=3
        )

        if not current_assignments:
            print("未找到有效的分配方案")
            break

        # 选择最佳组合
        best_assignment = current_assignments[0]
        drone_name, missile_name, solution = best_assignment

        # 设置/固化无人机参数
        if not DRONES[drone_name]["params_fixed"]:
            DRONES[drone_name]["speed"] = solution["speed"]
            DRONES[drone_name]["direction"] = solution["direction"]
            DRONES[drone_name]["params_fixed"] = True
            print(
                f"{drone_name} 参数首次固化：速度{solution['speed']:.1f}m/s，方向{np.degrees(solution['direction']):.1f}°")
        else:
            print(
                f"{drone_name} 使用已固化参数：速度{DRONES[drone_name]['speed']:.1f}m/s，方向{np.degrees(DRONES[drone_name]['direction']):.1f}°")

        # 计算起爆位置
        drop_pos = get_drone_pos(drone_name, solution["drop_time"])
        det_delay = solution["det_delay"]
        v_vec = np.array([DRONES[drone_name]["speed"] * np.cos(DRONES[drone_name]["direction"]),
                          DRONES[drone_name]["speed"] * np.sin(DRONES[drone_name]["direction"]), 0])
        det_x = drop_pos[0] + v_vec[0] * det_delay
        det_y = drop_pos[1] + v_vec[1] * det_delay
        det_z = drop_pos[2] - 0.5 * G * det_delay ** 2
        det_pos = np.array([det_x, det_y, max(det_z, 0.1)])

        # 添加烟幕弹记录
        smoke_record = {
            "drone": drone_name,
            "missile": missile_name,
            "v": DRONES[drone_name]["speed"],
            "theta": DRONES[drone_name]["direction"],
            "drop_time": solution["drop_time"],
            "det_delay": det_delay,
            "det_time": solution["drop_time"] + det_delay,
            "det_pos": det_pos,
            "effective_time": solution["effective_time"],
            "continuity_score": solution["continuity_score"],
            "gap_reduction": solution.get("gap_reduction", 0)
        }

        DRONES[drone_name]["smokes"].append(smoke_record)
        all_smokes.append(smoke_record)

        # 检查是否达到最大烟幕弹数
        if len(DRONES[drone_name]["smokes"]) >= DRONES[drone_name]["max_smoke"]:
            DRONES[drone_name]["optimized"] = True
            print(f"{drone_name} 已达到最大烟幕弹数，标记为已完成")

        # 计算当前总分（连续性优化评分）
        current_total_score = sum([
            EFFECTIVENESS_WEIGHT * s["effective_time"] + CONTINUITY_WEIGHT * s["continuity_score"]
            for s in all_smokes
        ])
        improvement = current_total_score - prev_total_score

        print(f"选择组合：{drone_name} -> {missile_name}")
        print(f"有效遮蔽时长：{solution['effective_time']:.2f}s")
        print(f"连续性贡献：{solution['continuity_score']:.2f}")
        print(f"当前总评分：{current_total_score:.2f} (提升 {improvement:.2f})")

        # 检查是否停滞
        if improvement < improvement_threshold:
            stall_count += 1
            print(f"改善较小，停滞计数：{stall_count}/{max_stall_iter}")
        else:
            stall_count = 0

        prev_total_score = current_total_score
        iteration += 1

    print(f"\n连续性优化完成！共进行 {iteration} 轮迭代")

    # 计算最终连续性统计
    total_effectiveness = sum([s["effective_time"] for s in all_smokes])
    total_continuity_score = sum([s["continuity_score"] for s in all_smokes])

    print(f"最终累计遮蔽时长：{total_effectiveness:.2f}s")
    print(f"最终连续性评分：{total_continuity_score:.2f}")

    return all_smokes


# ============================ 6. 结果输出与可视化（保持原有功能，增加连续性分析） ============================

def save_result_with_continuity(smokes, filename="result3_continuous.xlsx"):
    """保存结果（增加连续性分析）"""
    data = []
    for i, smoke in enumerate(smokes, 1):
        det_pos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0, 0, 0])
        data.append({
            "烟幕弹编号": f"S{i}",
            "无人机编号": smoke["drone"],
            "速度(m/s)": round(smoke["v"], 2),
            "方向(°)": round(np.degrees(smoke["theta"]), 2),
            "投放时刻(s)": round(smoke["drop_time"], 2),
            "起爆延迟(s)": round(smoke["det_delay"], 2),
            "起爆时刻(s)": round(smoke["det_time"], 2),
            "起爆点X(m)": round(det_pos[0], 2),
            "起爆点Y(m)": round(det_pos[1], 2),
            "起爆点Z(m)": round(det_pos[2], 2),
            "干扰导弹": smoke["missile"],
            "有效遮蔽时长(s)": round(smoke["effective_time"], 2),
            "连续性贡献": round(smoke.get("continuity_score", 0), 2),
            "间隙减少(s)": round(smoke.get("gap_reduction", 0), 2)
        })

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"连续性优化结果已保存到 {filename}")

    # 分析各导弹的连续性
    print(f"\n各导弹连续性分析：")
    for missile in MISSILES.keys():
        intervals = get_missile_existing_intervals(missile)
        if intervals:
            continuous_time, gap_penalty, merged_intervals = calculate_continuous_shielding_time(intervals)
            cumulative_time = sum(interval['end'] - interval['start'] for interval in intervals)
            efficiency = continuous_time / cumulative_time if cumulative_time > 0 else 0

            print(f"{missile}:")
            print(f"  烟幕弹数量: {len(intervals)}")
            print(f"  累计遮蔽时间: {cumulative_time:.2f}s")
            print(f"  连续遮蔽时间: {continuous_time:.2f}s")
            print(f"  连续性效率: {efficiency:.1%}")
            print(f"  连续段数量: {len(merged_intervals)}")
            print(f"  间隙惩罚: {gap_penalty:.2f}s")

    return df


def visualize_result_with_continuity(smokes):
    """可视化结果（增加连续性分析）"""
    if not smokes:
        print("无有效数据可可视化")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 绘制真目标、导弹轨迹、无人机轨迹（保持原有）
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", label="真目标投影")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=200, label="真目标中心")

    # 导弹轨迹
    colors = ["red", "green", "blue"]
    linestyles = ["--", "-.", ":"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        t_range = np.linspace(0, m_data["flight_time"], 100)
        pos_list = [get_missile_pos(m_name, t)[:2] for t in t_range]
        pos_arr = np.array(pos_list)
        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], color=colors[i], linestyle=linestyles[i], label=f"{m_name}轨迹")
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=100, label=f"{m_name}初始位置")

    # 无人机轨迹和烟幕
    drone_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (d_name, d_data) in enumerate(DRONES.items()):
        if not d_data["smokes"]:
            continue

        if d_data["params_fixed"]:
            last_smoke = d_data["smokes"][-1]
            t_range = np.linspace(0, last_smoke["drop_time"], 50)
            pos_list = [get_drone_pos(d_name, t) for t in t_range]
            pos_arr = np.array(pos_list)
            ax1.plot(pos_arr[:, 0], pos_arr[:, 1], color=drone_colors[i], linestyle="-",
                     label=f"{d_name}轨迹(v={d_data['speed']:.0f}m/s)")

        ax1.scatter(d_data["init_pos"][0], d_data["init_pos"][1], c=drone_colors[i], s=100, marker="^",
                    label=f"{d_name}初始位置")

        for smoke in d_data["smokes"]:
            det_pos = smoke["det_pos"]
            if det_pos is not None:
                ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=50, alpha=0.7)
                circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.2)
                ax1.add_patch(circle)

    ax1.set_xlabel("X(m)")
    ax1.set_ylabel("Y(m)")
    ax1.set_title("无人机、导弹轨迹及烟幕起爆点")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. 各导弹连续性分析
    missile_data = {}
    for missile in MISSILES.keys():
        intervals = get_missile_existing_intervals(missile)
        if intervals:
            continuous_time, gap_penalty, merged_intervals = calculate_continuous_shielding_time(intervals)
            cumulative_time = sum(interval['end'] - interval['start'] for interval in intervals)
            missile_data[missile] = {
                'cumulative': cumulative_time,
                'continuous': continuous_time,
                'efficiency': continuous_time / cumulative_time if cumulative_time > 0 else 0,
                'segments': len(merged_intervals)
            }
        else:
            missile_data[missile] = {'cumulative': 0, 'continuous': 0, 'efficiency': 0, 'segments': 0}

    missiles = list(missile_data.keys())
    cumulative_times = [missile_data[m]['cumulative'] for m in missiles]
    continuous_times = [missile_data[m]['continuous'] for m in missiles]

    x = np.arange(len(missiles))
    width = 0.35

    ax2.bar(x - width / 2, cumulative_times, width, label='累计遮蔽时间', alpha=0.8, color='lightblue')
    ax2.bar(x + width / 2, continuous_times, width, label='连续遮蔽时间', alpha=0.8, color='darkblue')

    ax2.set_xlabel("导弹编号")
    ax2.set_ylabel("时间(s)")
    ax2.set_title("各导弹遮蔽时间对比（连续性优化）")
    ax2.set_xticks(x)
    ax2.set_xticklabels(missiles)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 添加效率标签
    for i, m in enumerate(missiles):
        efficiency = missile_data[m]['efficiency']
        ax2.text(i, max(cumulative_times[i], continuous_times[i]) + 0.5,
                 f'{efficiency:.1%}', ha='center', va='bottom', fontweight='bold')

    # 3. 连续性效率分析
    efficiencies = [missile_data[m]['efficiency'] * 100 for m in missiles]
    segment_counts = [missile_data[m]['segments'] for m in missiles]

    bars = ax3.bar(missiles, efficiencies, color=colors, alpha=0.7)
    ax3.set_ylabel('连续性效率 (%)')
    ax3.set_title('各导弹连续性效率')
    ax3.set_ylim(0, 100)

    # 添加连续段数标签
    for bar, segments in zip(bars, segment_counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{segments}段', ha='center', va='bottom', fontsize=9)

    ax3.grid(True, alpha=0.3)

    # 4. 连续性贡献分布
    continuity_scores = [smoke.get("continuity_score", 0) for smoke in smokes]
    ax4.hist(continuity_scores, bins=10, color="lightgreen", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("连续性贡献评分")
    ax4.set_ylabel("烟幕弹数量")
    ax4.set_title("烟幕弹连续性贡献分布")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("continuous_smoke_optimization.png", dpi=300, bbox_inches="tight")
    plt.show()


# 初始化所有参数
generate_true_target_samples()
init_missiles()

# ============================ 主函数执行 ============================

if __name__ == "__main__":
    print("=" * 70)
    print("问题5：多无人机协同干扰优化（增加连续性约束）")
    print("=" * 70)
    print(f"连续性参数设置：")
    print(f"  最大允许间隙: {MAX_GAP_ALLOWED} s")
    print(f"  连续性权重: {CONTINUITY_WEIGHT}")
    print(f"  效果权重: {EFFECTIVENESS_WEIGHT}")

    all_smokes = iterative_optimization_with_continuity(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3)

    if all_smokes:
        result_df = save_result_with_continuity(all_smokes, "result3_continuous.xlsx")
        visualize_result_with_continuity(all_smokes)

        print("\n" + "=" * 50)
        print("最终连续性优化结果汇总：")
        print(f"总烟幕弹数量：{len(all_smokes)}")

        # 总体统计
        total_effectiveness = sum([s["effective_time"] for s in all_smokes])
        total_continuity_score = sum([s.get("continuity_score", 0) for s in all_smokes])

        print(f"总累计遮蔽时长：{total_effectiveness:.2f}s")
        print(f"总连续性评分：{total_continuity_score:.2f}")

        print("\n各无人机详情：")
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total_eff = sum([s["effective_time"] for s in d_data["smokes"]])
                total_cont = sum([s.get("continuity_score", 0) for s in d_data["smokes"]])
                print(f"{d_name}：{len(d_data['smokes'])}枚弹，遮蔽时长{total_eff:.2f}s，连续性贡献{total_cont:.2f}")
                if d_data["params_fixed"]:
                    print(f"  固化参数：速度{d_data['speed']:.2f}m/s，方向{np.degrees(d_data['direction']):.1f}°")
            else:
                print(f"{d_name}：未投放烟幕弹")

        print("\n各导弹被干扰情况：")
        for m_name in MISSILES.keys():
            intervals = get_missile_existing_intervals(m_name)
            if intervals:
                continuous_time, gap_penalty, merged_intervals = calculate_continuous_shielding_time(intervals)
                cumulative_time = sum(interval['end'] - interval['start'] for interval in intervals)
                efficiency = continuous_time / cumulative_time if cumulative_time > 0 else 0

                print(
                    f"{m_name}：累计{cumulative_time:.2f}s，连续{continuous_time:.2f}s，效率{efficiency:.1%}，{len(merged_intervals)}段")
            else:
                print(f"{m_name}：未被干扰")

        print("=" * 50)
    else:
        print("未找到有效的烟幕弹投放方案")