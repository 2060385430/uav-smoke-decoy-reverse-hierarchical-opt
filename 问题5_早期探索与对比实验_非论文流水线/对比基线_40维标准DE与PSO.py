"""
问题五：多无人机协同烟幕干扰 —— 基准对比算法（单文件版）
实现两个基准全局优化算法，用于对比逆推分层的效率优势：
  1. 标准DE全局优化：40维向量直接用differential_evolution优化
  2. 标准PSO全局优化：40维向量直接用粒子群优化

决策变量编码（40维）：
  5架无人机 × (速度1 + 方向1 + 3枚弹×(投放时间1 + 起爆延迟1)) = 5 × 8 = 40维
  t_drop < 0 表示该弹不投放

本文件已合并统一评价函数，无需外部依赖。

【修改说明】目标函数已从"三导弹交集"改为"各导弹遮蔽区间并集长度的总和"，
与逆推分层算法的目标函数保持一致，确保对比实验公平。
"""
import numpy as np
import time
from collections import namedtuple
from scipy.optimize import differential_evolution

# ============================ 1. 全局参数 ============================
TRUE_TARGET = {"r": 7.0, "h": 10.0, "center": np.array([0.0, 200.0, 0.0]), "sample_points": None}
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
EPSILON = 1e-8

for m_name, m_data in MISSILES.items():
    init_pos = m_data["init_pos"]
    dist = np.linalg.norm(init_pos)
    m_data["dir"] = -init_pos / dist
    m_data["velocity"] = m_data["dir"] * MISSILE_SPEED
    m_data["flight_time"] = dist / MISSILE_SPEED

# ============================ 2. 评价函数（合并自Part 1） ============================
def generate_target_samples(n_per_circle=50):
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    theta = np.linspace(0, 2 * np.pi, n_per_circle, endpoint=False)
    bottom = np.column_stack([center[0]+r*np.cos(theta), center[1]+r*np.sin(theta), np.full(n_per_circle, center[2])])
    top = np.column_stack([center[0]+r*np.cos(theta), center[1]+r*np.sin(theta), np.full(n_per_circle, center[2]+h)])
    TRUE_TARGET["sample_points"] = np.vstack([bottom, top])
    return TRUE_TARGET["sample_points"]

def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    return m_data["init_pos"] + m_data["velocity"] * min(t, m_data["flight_time"])

def get_drone_pos(drone_name, t, speed, direction):
    init_pos = DRONES[drone_name]["init_pos"]
    v_vec = np.array([speed*np.cos(direction), speed*np.sin(direction), 0.0])
    return init_pos + v_vec * t

def get_smoke_center(drone_name, speed, direction, drop_time, det_delay, t):
    det_time = drop_time + det_delay
    if t < det_time - EPSILON or t > det_time + SMOKE_EFFECTIVE_TIME + EPSILON:
        return None
    drop_pos = get_drone_pos(drone_name, drop_time, speed, direction)
    v_vec = np.array([speed*np.cos(direction), speed*np.sin(direction), 0.0])
    det_z = drop_pos[2] - 0.5*G*det_delay**2
    if det_z < SMOKE_MIN_HEIGHT:
        return None
    smoke_z = det_z - SMOKE_SINK_SPEED*(t - det_time)
    if smoke_z < SMOKE_MIN_HEIGHT:
        return None
    return np.array([drop_pos[0]+v_vec[0]*det_delay, drop_pos[1]+v_vec[1]*det_delay, smoke_z])

def segment_sphere_intersection_vectorized(missile_pos, target_points, smoke_center, smoke_radius):
    MP = target_points - missile_pos
    MC = smoke_center - missile_pos
    dot_MP_MP = np.maximum(np.sum(MP**2, axis=1), EPSILON)
    t_proj = np.sum(MP*MC, axis=1) / dot_MP_MP
    t_clipped = np.clip(t_proj, 0.0, 1.0)
    nearest = missile_pos + t_clipped[:, np.newaxis] * MP
    dist = np.linalg.norm(nearest - smoke_center, axis=1)
    return dist <= smoke_radius + EPSILON

def is_target_fully_shielded(missile_pos, smoke_center, smoke_radius, target_samples):
    return np.all(segment_sphere_intersection_vectorized(missile_pos, target_samples, smoke_center, smoke_radius))

def _check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples):
    smoke_center = get_smoke_center(drone_name, speed, direction, drop_time, det_delay, t)
    if smoke_center is None:
        return False
    return is_target_fully_shielded(get_missile_pos(m_name, t), smoke_center, SMOKE_RADIUS, target_samples)

def _refine_boundary_binary_search(t_shielded, t_unshielded, drone_name, speed, direction,
                                     drop_time, det_delay, m_name, target_samples, max_iter=20):
    lo, hi = min(t_shielded, t_unshielded), max(t_shielded, t_unshielded)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if _check_shielded_at_time(mid, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples):
            lo = mid
        else:
            hi = mid
    return lo

def calc_single_smoke_interval(drone_name, speed, direction, drop_time, det_delay, m_name, target_samples=None):
    if target_samples is None:
        target_samples = TRUE_TARGET["sample_points"]
    det_time = drop_time + det_delay
    t_start_window = max(det_time, 0.0)
    t_end_window = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILES[m_name]["flight_time"])
    if t_start_window >= t_end_window - EPSILON:
        return None
    coarse_step = 0.3
    t_coarse = np.arange(t_start_window, t_end_window + coarse_step, coarse_step)
    shielded_coarse = np.array([_check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples) for t in t_coarse])
    if not np.any(shielded_coarse):
        return None
    coarse_intervals = []
    in_seg = False
    seg_start_idx = 0
    for idx in range(len(t_coarse)):
        if shielded_coarse[idx] and not in_seg:
            seg_start_idx = idx; in_seg = True
        elif not shielded_coarse[idx] and in_seg:
            coarse_intervals.append((seg_start_idx, idx-1)); in_seg = False
    if in_seg:
        coarse_intervals.append((seg_start_idx, len(t_coarse)-1))
    fine_intervals = []
    fine_step = 0.05
    for start_idx, end_idx in coarse_intervals:
        fine_start = max(t_start_window, t_coarse[start_idx] - 0.4)
        fine_end = min(t_end_window, t_coarse[end_idx] + 0.4)
        t_fine = np.arange(fine_start, fine_end + fine_step, fine_step)
        shielded_fine = np.array([_check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples) for t in t_fine])
        if not np.any(shielded_fine):
            continue
        in_seg = False
        seg_start_t = 0.0
        prev_unshielded_t = fine_start
        for idx in range(len(t_fine)):
            if shielded_fine[idx] and not in_seg:
                seg_start_t = t_fine[idx]
                prev_unshielded_t = t_fine[idx-1] if idx > 0 else fine_start
                in_seg = True
            elif not shielded_fine[idx] and in_seg:
                left = _refine_boundary_binary_search(seg_start_t, prev_unshielded_t, drone_name, speed, direction, drop_time, det_delay, m_name, target_samples)
                right = _refine_boundary_binary_search(t_fine[idx-1], t_fine[idx], drone_name, speed, direction, drop_time, det_delay, m_name, target_samples)
                fine_intervals.append((left, right))
                in_seg = False
        if in_seg:
            left = seg_start_t
            if seg_start_t > fine_start + EPSILON:
                left_idx = np.argmax(t_fine >= seg_start_t)
                if left_idx > 0:
                    left = _refine_boundary_binary_search(t_fine[left_idx], t_fine[left_idx-1], drone_name, speed, direction, drop_time, det_delay, m_name, target_samples)
            fine_intervals.append((left, t_fine[-1]))
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
    return merged[0] if len(merged) == 1 else merged

def interval_length(interval):
    if interval is None: return 0.0
    if isinstance(interval, tuple): return max(0.0, interval[1]-interval[0])
    if isinstance(interval, list): return sum(max(0.0, iv[1]-iv[0]) for iv in interval)
    return 0.0

def merge_intervals(intervals):
    valid = [iv for iv in intervals if iv is not None and iv[1] > iv[0] + EPSILON]
    if not valid: return []
    flat = []
    for iv in valid:
        if isinstance(iv, list): flat.extend(iv)
        else: flat.append(iv)
    flat.sort(key=lambda x: x[0])
    merged = [flat[0]]
    for current in flat[1:]:
        last = merged[-1]
        if current[0] <= last[1] + EPSILON: merged[-1] = (last[0], max(last[1], current[1]))
        else: merged.append(current)
    return merged

def calc_missile_total_shielding(smoke_params_list, m_name, target_samples=None):
    if target_samples is None: target_samples = TRUE_TARGET["sample_points"]
    all_intervals = []
    for params in smoke_params_list:
        iv = calc_single_smoke_interval(params["drone"], params["speed"], params["direction"], params["drop_time"], params["det_delay"], m_name, target_samples)
        if iv is not None: all_intervals.append(iv)
    merged = merge_intervals(all_intervals)
    return merged, sum(iv[1]-iv[0] for iv in merged)

def intervals_intersection(intervals_list):
    if not intervals_list: return [], 0.0
    for ivs in intervals_list:
        if not ivs: return [], 0.0
    events = []
    for ivs in intervals_list:
        for start, end in ivs:
            events.append((start, 1)); events.append((end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    n_lists = len(intervals_list)
    current_cover = 0
    intersection_intervals = []
    seg_start = None
    for t, delta in events:
        if delta == 1:
            current_cover += 1
            if current_cover == n_lists and seg_start is None: seg_start = t
        else:
            if current_cover == n_lists and seg_start is not None:
                if t > seg_start + EPSILON: intersection_intervals.append((seg_start, t))
                seg_start = None
            current_cover -= 1
    return intersection_intervals, sum(iv[1]-iv[0] for iv in intersection_intervals)

SmokeParams = namedtuple("SmokeParams", ["drone", "speed", "direction", "drop_time", "det_delay"])

def evaluate_deployment(smoke_params_list, target_samples=None):
    """
    评估部署方案的系统总遮蔽时长。
    【已修改】系统总遮蔽时长 = 各导弹遮蔽区间并集长度的总和（非三导弹交集）
    """
    if target_samples is None: target_samples = TRUE_TARGET["sample_points"]
    params_list = [p._asdict() if isinstance(p, SmokeParams) else p for p in smoke_params_list]
    per_missile = {}
    missile_intervals_list = []
    for m_name in MISSILES.keys():
        merged, duration = calc_missile_total_shielding(params_list, m_name, target_samples)
        per_missile[m_name] = {"intervals": merged, "duration": duration}
        missile_intervals_list.append(merged)
    # 三导弹交集（仅作参考，不作为目标函数）
    system_intersection, _ = intervals_intersection(missile_intervals_list)
    # 【修改】系统总遮蔽时长 = 各导弹并集长度的总和
    system_duration = sum(info["duration"] for info in per_missile.values())
    return {"per_missile": per_missile, "system_intersection": system_intersection, "system_duration": system_duration}

# 初始化
if TRUE_TARGET["sample_points"] is None:
    generate_target_samples(n_per_circle=50)
TARGET_SAMPLES = TRUE_TARGET["sample_points"]

# ============================ 3. 40维编码/解码 ============================
N_DRONES = 5
N_SMOKES_PER_DRONE = 3
DIM_PER_DRONE = 2 + N_SMOKES_PER_DRONE * 2
TOTAL_DIM = N_DRONES * DIM_PER_DRONE
DRONE_NAMES = list(DRONES.keys())

def get_bounds():
    bounds = []
    for _ in range(N_DRONES):
        bounds.extend([
            (70.0, 140.0), (0.0, 2*np.pi),
            (-1.0, 60.0), (0.1, 10.0),
            (-1.0, 60.0), (0.1, 10.0),
            (-1.0, 60.0), (0.1, 10.0),
        ])
    return bounds

def decode_solution(x):
    smoke_list = []
    for i in range(N_DRONES):
        base = i * DIM_PER_DRONE
        v, theta = x[base], x[base+1]
        drop_times = []
        for k in range(N_SMOKES_PER_DRONE):
            t_drop = x[base+2+k*2]
            t_bd = x[base+3+k*2]
            if t_drop >= 0:
                drop_times.append((t_drop, t_bd))
        drop_times.sort(key=lambda x: x[0])
        valid_drops = []
        last_drop = -10.0
        for t_drop, t_bd in drop_times:
            if t_drop - last_drop >= 1.0 - EPSILON:
                valid_drops.append((t_drop, t_bd))
                last_drop = t_drop
        for t_drop, t_bd in valid_drops:
            smoke_list.append({"drone": DRONE_NAMES[i], "speed": v, "direction": theta, "drop_time": t_drop, "det_delay": t_bd})
    return smoke_list

def check_feasibility(x):
    penalty = 0.0
    for i in range(N_DRONES):
        base = i * DIM_PER_DRONE
        v, theta = x[base], x[base+1]
        drop_times = []
        for k in range(N_SMOKES_PER_DRONE):
            t_drop = x[base+2+k*2]
            if t_drop >= 0: drop_times.append(t_drop)
        if not drop_times: continue
        if v < 70.0 or v > 140.0: penalty += 100.0
        if theta < 0.0 or theta > 2*np.pi: penalty += 50.0
        drop_times.sort()
        for j in range(1, len(drop_times)):
            if drop_times[j]-drop_times[j-1] < 1.0-EPSILON: penalty += 50.0
        for k in range(N_SMOKES_PER_DRONE):
            t_drop = x[base+2+k*2]
            t_bd = x[base+3+k*2]
            if t_drop >= 0:
                drop_pos = get_drone_pos(DRONE_NAMES[i], t_drop, v, theta)
                det_z = drop_pos[2] - 0.5*G*t_bd**2
                if det_z < SMOKE_MIN_HEIGHT: penalty += 30.0
    return penalty < 1.0, penalty

def objective_function(x, return_details=False):
    is_feasible, penalty = check_feasibility(x)
    if not is_feasible:
        return (1000.0+penalty, None) if return_details else 1000.0+penalty
    smoke_list = decode_solution(x)
    if not smoke_list:
        return (100.0, None) if return_details else 100.0
    result = evaluate_deployment(smoke_list, TARGET_SAMPLES)
    system_duration = result["system_duration"]
    if system_duration <= 0:
        return (100.0, result) if return_details else 100.0
    return (-system_duration, result) if return_details else -system_duration

# ============================ 4. 标准DE全局优化 ============================
def standard_de_optimization(popsize=1, maxiter=50, seed=42, verbose=True):
    if verbose:
        print("=" * 60)
        print("算法1：标准DE全局优化（40维）")
        print(f"  种群: {popsize}×{TOTAL_DIM}={popsize*TOTAL_DIM}, 迭代: {maxiter}")
        print("=" * 60)
    bounds = get_bounds()
    start_time = time.time()
    convergence_curve = []
    best_obj_tracking = [np.inf]
    def callback(xk, convergence=None):
        current_obj = objective_function(xk)
        best_obj_tracking[0] = min(best_obj_tracking[0], current_obj)
        system_duration = -best_obj_tracking[0] if best_obj_tracking[0] < 100 else 0.0
        convergence_curve.append(system_duration)
        if verbose and len(convergence_curve) % 10 == 0:
            print(f"  迭代 {len(convergence_curve)}/{maxiter}, 当前最优系统时长: {system_duration:.4f}s")
    try:
        result = differential_evolution(
            objective_function, bounds, maxiter=maxiter, popsize=popsize,
            seed=seed, tol=1e-6, mutation=(0.5,1.0), recombination=0.7,
            polish=False, init='random', callback=callback,
        )
        best_x, best_obj = result.x, result.fun
    except Exception as e:
        if verbose: print(f"  DE优化异常: {e}")
        best_x, best_obj = np.zeros(TOTAL_DIM), 1000.0
    elapsed = time.time() - start_time
    final_obj, final_result = objective_function(best_x, return_details=True)
    smoke_list = decode_solution(best_x)
    if verbose:
        print(f"\n  优化完成，耗时: {elapsed:.1f}s")
        print(f"  最优目标值: {best_obj:.4f}")
        print(f"  系统总遮蔽时长: {final_result['system_duration'] if final_result else 0:.4f}s")
        print(f"  有效烟幕弹数量: {len(smoke_list)}")
    return {"algorithm": "标准DE全局优化", "best_x": best_x, "best_objective": best_obj,
            "system_duration": final_result["system_duration"] if final_result else 0.0,
            "per_missile": final_result["per_missile"] if final_result else {},
            "system_intersection": final_result["system_intersection"] if final_result else [],
            "smoke_list": smoke_list, "convergence": convergence_curve,
            "elapsed_time": elapsed, "popsize": popsize, "maxiter": maxiter}

# ============================ 5. 标准PSO全局优化 ============================
class StandardPSO:
    def __init__(self, objective_func, bounds, n_particles=40, max_iter=50,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, seed=42, verbose=True):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2 = c1, c2
        self.verbose = verbose
        self.dim = len(bounds)
        self.rng = np.random.RandomState(seed)
        self.positions = np.zeros((n_particles, self.dim))
        self.velocities = np.zeros((n_particles, self.dim))
        self.pbest_positions = np.zeros((n_particles, self.dim))
        self.pbest_fitness = np.full(n_particles, np.inf)
        self.gbest_position = np.zeros(self.dim)
        self.gbest_fitness = np.inf
        self.convergence = []
        self._initialize()

    def _initialize(self):
        for i in range(self.n_particles):
            for j in range(self.dim):
                low, high = self.bounds[j]
                self.positions[i, j] = self.rng.uniform(low, high)
                self.velocities[i, j] = self.rng.uniform(-(high-low)*0.1, (high-low)*0.1)
            fitness = self.objective_func(self.positions[i])
            self.pbest_positions[i] = self.positions[i].copy()
            self.pbest_fitness[i] = fitness
            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()

    def _constrain(self, position):
        for j in range(self.dim):
            position[j] = np.clip(position[j], self.bounds[j,0], self.bounds[j,1])
        return position

    def optimize(self):
        if self.verbose:
            print("=" * 60)
            print("算法2：标准PSO全局优化（40维）")
            print(f"  粒子: {self.n_particles}, 迭代: {self.max_iter}")
            print("=" * 60)
        start_time = time.time()
        for iteration in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iter)
            for i in range(self.n_particles):
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                for j in range(self.dim):
                    vel_limit = 0.2 * (self.bounds[j,1] - self.bounds[j,0])
                    self.velocities[i,j] = np.clip(self.velocities[i,j], -vel_limit, vel_limit)
                self.positions[i] = self._constrain(self.positions[i] + self.velocities[i])
                fitness = self.objective_func(self.positions[i])
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
            system_dur = -self.gbest_fitness if self.gbest_fitness < 100 else 0.0
            self.convergence.append(system_dur)
            if self.verbose and (iteration+1) % 10 == 0:
                print(f"  迭代 {iteration+1}/{self.max_iter}, 全局最优: {system_dur:.4f}s")
        elapsed = time.time() - start_time
        final_obj, final_result = objective_function(self.gbest_position, return_details=True)
        smoke_list = decode_solution(self.gbest_position)
        if self.verbose:
            print(f"\n  优化完成，耗时: {elapsed:.1f}s")
            print(f"  最优目标值: {self.gbest_fitness:.4f}")
            print(f"  系统总遮蔽时长: {final_result['system_duration'] if final_result else 0:.4f}s")
            print(f"  有效烟幕弹数量: {len(smoke_list)}")
        return {"algorithm": "标准PSO全局优化", "best_x": self.gbest_position, "best_objective": self.gbest_fitness,
                "system_duration": final_result["system_duration"] if final_result else 0.0,
                "per_missile": final_result["per_missile"] if final_result else {},
                "system_intersection": final_result["system_intersection"] if final_result else [],
                "smoke_list": smoke_list, "convergence": self.convergence,
                "elapsed_time": elapsed, "n_particles": self.n_particles, "max_iter": self.max_iter}

def standard_pso_optimization(n_particles=40, max_iter=50, seed=42, verbose=True):
    pso = StandardPSO(objective_func=objective_function, bounds=get_bounds(),
                      n_particles=n_particles, max_iter=max_iter, seed=seed, verbose=verbose)
    return pso.optimize()

# ============================ 测试 ============================
if __name__ == "__main__":
    print("问题五 —— 基准对比算法（单文件版）【已修正目标函数为各导弹并集总和】")
    print(f"决策维度: {TOTAL_DIM}")
    print("\n=== DE测试（popsize=1, maxiter=5）===")
    de_result = standard_de_optimization(popsize=1, maxiter=5, seed=42, verbose=True)
    print(f"\nDE结果: 系统时长{de_result['system_duration']:.4f}s, 耗时{de_result['elapsed_time']:.1f}s")
    print("\n=== PSO测试（n_particles=10, max_iter=5）===")
    pso_result = standard_pso_optimization(n_particles=10, max_iter=5, seed=42, verbose=True)
    print(f"\nPSO结果: 系统时长{pso_result['system_duration']:.4f}s, 耗时{pso_result['elapsed_time']:.1f}s")
    print("\n测试完成！")
