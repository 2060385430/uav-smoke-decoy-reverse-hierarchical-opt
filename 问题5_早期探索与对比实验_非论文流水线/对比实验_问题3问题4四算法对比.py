"""
问题三 + 问题四 对比实验（四算法版）：
  标准DE vs 标准PSO vs 效能矩阵+整数规划 vs 逆推分层

支持两种场景：
  - 问题三：1无人机3弹1导弹，8维决策变量
  - 问题四：3无人机各1弹1导弹，12维决策变量

目标函数统一为：M1的遮蔽区间并集长度（单导弹场景）
所有算法共用同一个遮蔽评价函数，确保对比公平。

【效能矩阵+整数规划的核心缺陷】
  各弹/各无人机独立优化，目标是各自效能最大；
  整数规划选择参与组合，目标是效能简单相加最大；
  不考虑组合后区间重叠，实际并集时长可能小于简单相加。
"""
import numpy as np
import time
import json
from scipy.optimize import differential_evolution

# ============================ 1. 全局参数 ============================
TRUE_TARGET = {"r": 7.0, "h": 10.0, "center": np.array([0.0, 200.0, 0.0]), "sample_points": None}
MISSILE_SPEED = 300.0
G = 9.8
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_EFFECTIVE_TIME = 20.0
SMOKE_MIN_HEIGHT = 2.0
EPSILON = 1e-8

M1_INIT = np.array([20000.0, 0.0, 2000.0])
M1_DIST = np.linalg.norm(M1_INIT)
M1_DIR = -M1_INIT / M1_DIST
M1_VELOCITY = M1_DIR * MISSILE_SPEED
M1_FLIGHT_TIME = M1_DIST / MISSILE_SPEED

DRONE_INITS = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
}

# ============================ 2. 统一遮蔽评价函数 ============================
def generate_target_samples(n_per_circle=50):
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    theta = np.linspace(0, 2 * np.pi, n_per_circle, endpoint=False)
    bottom = np.column_stack([center[0]+r*np.cos(theta), center[1]+r*np.sin(theta), np.full(n_per_circle, center[2])])
    top = np.column_stack([center[0]+r*np.cos(theta), center[1]+r*np.sin(theta), np.full(n_per_circle, center[2]+h)])
    TRUE_TARGET["sample_points"] = np.vstack([bottom, top])
    return TRUE_TARGET["sample_points"]

def get_missile_pos(t):
    return M1_INIT + M1_VELOCITY * min(t, M1_FLIGHT_TIME)

def get_drone_pos(drone_name, t, speed, direction):
    init_pos = DRONE_INITS[drone_name]
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

def _check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, target_samples):
    smoke_center = get_smoke_center(drone_name, speed, direction, drop_time, det_delay, t)
    if smoke_center is None:
        return False
    return is_target_fully_shielded(get_missile_pos(t), smoke_center, SMOKE_RADIUS, target_samples)

def _refine_boundary_binary_search(t_shielded, t_unshielded, drone_name, speed, direction,
                                     drop_time, det_delay, target_samples, max_iter=20):
    lo, hi = min(t_shielded, t_unshielded), max(t_shielded, t_unshielded)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if _check_shielded_at_time(mid, drone_name, speed, direction, drop_time, det_delay, target_samples):
            lo = mid
        else:
            hi = mid
    return lo

def calc_single_smoke_interval(drone_name, speed, direction, drop_time, det_delay, target_samples=None):
    if target_samples is None:
        target_samples = TRUE_TARGET["sample_points"]
    det_time = drop_time + det_delay
    t_start_window = max(det_time, 0.0)
    t_end_window = min(det_time + SMOKE_EFFECTIVE_TIME, M1_FLIGHT_TIME)
    if t_start_window >= t_end_window - EPSILON:
        return None
    coarse_step = 0.3
    t_coarse = np.arange(t_start_window, t_end_window + coarse_step, coarse_step)
    shielded_coarse = np.array([_check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, target_samples) for t in t_coarse])
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
        shielded_fine = np.array([_check_shielded_at_time(t, drone_name, speed, direction, drop_time, det_delay, target_samples) for t in t_fine])
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
                left = _refine_boundary_binary_search(seg_start_t, prev_unshielded_t, drone_name, speed, direction, drop_time, det_delay, target_samples)
                right = _refine_boundary_binary_search(t_fine[idx-1], t_fine[idx], drone_name, speed, direction, drop_time, det_delay, target_samples)
                fine_intervals.append((left, right))
                in_seg = False
        if in_seg:
            left = seg_start_t
            if seg_start_t > fine_start + EPSILON:
                left_idx = np.argmax(t_fine >= seg_start_t)
                if left_idx > 0:
                    left = _refine_boundary_binary_search(t_fine[left_idx], t_fine[left_idx-1], drone_name, speed, direction, drop_time, det_delay, target_samples)
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

def calc_system_shielding(smoke_list, target_samples=None):
    if target_samples is None: target_samples = TRUE_TARGET["sample_points"]
    all_intervals = []
    for s in smoke_list:
        iv = calc_single_smoke_interval(s["drone"], s["speed"], s["direction"], s["drop_time"], s["det_delay"], target_samples)
        if iv is not None: all_intervals.append(iv)
    merged = merge_intervals(all_intervals)
    duration = sum(iv[1]-iv[0] for iv in merged)
    return merged, duration

if TRUE_TARGET["sample_points"] is None:
    generate_target_samples(n_per_circle=50)
TARGET_SAMPLES = TRUE_TARGET["sample_points"]

# ============================ 3. 问题三：8维编码/解码 ============================
P3_DIM = 8
P3_DRONE = "FY1"

def p3_get_bounds():
    return [
        (70.0, 140.0), (0.0, 2*np.pi),
        (-1.0, 60.0), (0.1, 12.0),
        (-1.0, 60.0), (0.1, 12.0),
        (-1.0, 60.0), (0.1, 12.0),
    ]

def p3_decode(x):
    v, theta = x[0], x[1]
    smoke_list = []
    drops = []
    for k in range(3):
        t_drop = x[2 + k*2]
        t_bd = x[3 + k*2]
        if t_drop >= 0:
            drops.append((t_drop, t_bd))
    drops.sort(key=lambda d: d[0])
    last_drop = -10.0
    for t_drop, t_bd in drops:
        if t_drop - last_drop >= 1.0 - EPSILON:
            smoke_list.append({"drone": P3_DRONE, "speed": v, "direction": theta, "drop_time": t_drop, "det_delay": t_bd})
            last_drop = t_drop
    return smoke_list

def p3_check_feasibility(x):
    penalty = 0.0
    v, theta = x[0], x[1]
    if v < 70.0 or v > 140.0: penalty += 100.0
    if theta < 0.0 or theta > 2*np.pi: penalty += 50.0
    drops = []
    for k in range(3):
        t_drop = x[2 + k*2]
        if t_drop >= 0: drops.append(t_drop)
    if not drops: return penalty < 1.0, penalty
    drops.sort()
    for j in range(1, len(drops)):
        if drops[j]-drops[j-1] < 1.0-EPSILON: penalty += 50.0
    for k in range(3):
        t_drop = x[2 + k*2]
        t_bd = x[3 + k*2]
        if t_drop >= 0:
            drop_pos = get_drone_pos(P3_DRONE, t_drop, v, theta)
            det_z = drop_pos[2] - 0.5*G*t_bd**2
            if det_z < SMOKE_MIN_HEIGHT: penalty += 30.0
    return penalty < 1.0, penalty

def p3_objective(x, return_details=False):
    is_feasible, penalty = p3_check_feasibility(x)
    if not is_feasible:
        return (1000.0+penalty, None) if return_details else 1000.0+penalty
    smoke_list = p3_decode(x)
    if not smoke_list:
        return (100.0, None) if return_details else 100.0
    merged, duration = calc_system_shielding(smoke_list, TARGET_SAMPLES)
    if duration <= 0:
        return (100.0, {"intervals": merged, "duration": 0.0}) if return_details else 100.0
    return (-duration, {"intervals": merged, "duration": duration}) if return_details else -duration

# ============================ 4. 问题四：12维编码/解码 ============================
P4_DIM = 12
P4_DRONES = ["FY1", "FY2", "FY3"]

def p4_get_bounds():
    bounds = []
    for _ in range(3):
        bounds.extend([
            (70.0, 140.0), (0.0, 2*np.pi),
            (-1.0, 60.0), (0.1, 12.0),
        ])
    return bounds

def p4_decode(x):
    smoke_list = []
    for i in range(3):
        base = i * 4
        v, theta = x[base], x[base+1]
        t_drop = x[base+2]
        t_bd = x[base+3]
        if t_drop >= 0:
            smoke_list.append({"drone": P4_DRONES[i], "speed": v, "direction": theta, "drop_time": t_drop, "det_delay": t_bd})
    return smoke_list

def p4_check_feasibility(x):
    penalty = 0.0
    for i in range(3):
        base = i * 4
        v, theta = x[base], x[base+1]
        t_drop = x[base+2]
        t_bd = x[base+3]
        if t_drop < 0: continue
        if v < 70.0 or v > 140.0: penalty += 100.0
        if theta < 0.0 or theta > 2*np.pi: penalty += 50.0
        drop_pos = get_drone_pos(P4_DRONES[i], t_drop, v, theta)
        det_z = drop_pos[2] - 0.5*G*t_bd**2
        if det_z < SMOKE_MIN_HEIGHT: penalty += 30.0
    return penalty < 1.0, penalty

def p4_objective(x, return_details=False):
    is_feasible, penalty = p4_check_feasibility(x)
    if not is_feasible:
        return (1000.0+penalty, None) if return_details else 1000.0+penalty
    smoke_list = p4_decode(x)
    if not smoke_list:
        return (100.0, None) if return_details else 100.0
    merged, duration = calc_system_shielding(smoke_list, TARGET_SAMPLES)
    if duration <= 0:
        return (100.0, {"intervals": merged, "duration": 0.0}) if return_details else 100.0
    return (-duration, {"intervals": merged, "duration": duration}) if return_details else -duration

# ============================ 5. 标准DE优化 ============================
def standard_de_optimization(objective_func, bounds, dim, popsize=30, maxiter=100, seed=42, verbose=True, label=""):
    if verbose:
        print(f"{'='*60}")
        print(f"标准DE全局优化（{dim}维）{label}")
        print(f"  种群: {popsize}×{dim}={popsize*dim}, 迭代: {maxiter}")
        print(f"{'='*60}")
    start_time = time.time()
    convergence = []
    best_obj_tracking = [np.inf]
    def callback(xk, convergence_val=None):
        current_obj = objective_func(xk)
        best_obj_tracking[0] = min(best_obj_tracking[0], current_obj)
        system_duration = -best_obj_tracking[0] if best_obj_tracking[0] < 100 else 0.0
        convergence.append(system_duration)
        if verbose and len(convergence) % 20 == 0:
            print(f"  迭代 {len(convergence)}/{maxiter}, 当前最优: {system_duration:.4f}s")
    try:
        result = differential_evolution(
            objective_func, bounds, maxiter=maxiter, popsize=popsize,
            seed=seed, tol=1e-6, mutation=(0.5,1.0), recombination=0.7,
            polish=False, init='random', callback=callback,
        )
        best_x, best_obj = result.x, result.fun
    except Exception as e:
        if verbose: print(f"  DE优化异常: {e}")
        best_x, best_obj = np.zeros(dim), 1000.0
    elapsed = time.time() - start_time
    final_obj, final_result = objective_func(best_x, return_details=True)
    if verbose:
        print(f"\n  优化完成，耗时: {elapsed:.1f}s")
        print(f"  最优目标值: {best_obj:.4f}")
        print(f"  系统总遮蔽时长: {final_result['duration'] if final_result else 0:.4f}s")
    return {"algorithm": "标准DE", "best_x": best_x, "best_objective": best_obj,
            "system_duration": final_result["duration"] if final_result else 0.0,
            "convergence": convergence, "elapsed_time": elapsed}

# ============================ 6. 标准PSO优化 ============================
class StandardPSO:
    def __init__(self, objective_func, bounds, n_particles=30, max_iter=100,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, seed=42, verbose=True, label=""):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2 = c1, c2
        self.verbose = verbose
        self.label = label
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
            print(f"{'='*60}")
            print(f"标准PSO全局优化（{self.dim}维）{self.label}")
            print(f"  粒子: {self.n_particles}, 迭代: {self.max_iter}")
            print(f"{'='*60}")
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
            if self.verbose and (iteration+1) % 20 == 0:
                print(f"  迭代 {iteration+1}/{self.max_iter}, 全局最优: {system_dur:.4f}s")
        elapsed = time.time() - start_time
        final_obj, final_result = self.objective_func(self.gbest_position, return_details=True)
        if self.verbose:
            print(f"\n  优化完成，耗时: {elapsed:.1f}s")
            print(f"  最优目标值: {self.gbest_fitness:.4f}")
            print(f"  系统总遮蔽时长: {final_result['duration'] if final_result else 0:.4f}s")
        return {"algorithm": "标准PSO", "best_x": self.gbest_position, "best_objective": self.gbest_fitness,
                "system_duration": final_result["duration"] if final_result else 0.0,
                "convergence": self.convergence, "elapsed_time": elapsed}

# ============================ 7. 通用工具函数 ============================
def _single_smoke_4d_optimize(drone_name, seed=42, maxiter=80, popsize=20):
    """对单架无人机的单枚弹做4维优化（v,theta,t_drop,tau），返回最优参数和效能"""
    rng = np.random.RandomState(seed)
    # 粗网格找可行点
    best_dur = 0.0
    best_x = None
    for v in np.linspace(70, 140, 10):
        for theta in np.linspace(0, 2*np.pi, 14):
            for t in np.linspace(0, 45, 12):
                for tau in np.linspace(0.5, 10.0, 10):
                    smoke = [{"drone": drone_name, "speed": v, "direction": theta, "drop_time": t, "det_delay": tau}]
                    _, dur = calc_system_shielding(smoke, TARGET_SAMPLES)
                    if dur > best_dur:
                        best_dur = dur
                        best_x = np.array([v, theta, t, tau])
    if best_x is None:
        return np.array([120.0, np.pi, 1.5, 3.6]), 0.0
    # DE精修
    def obj(x):
        smoke = [{"drone": drone_name, "speed": x[0], "direction": x[1], "drop_time": x[2], "det_delay": x[3]}]
        _, dur = calc_system_shielding(smoke, TARGET_SAMPLES)
        return -dur if dur > 0 else 100.0
    bounds = [(70,140),(0,2*np.pi),(0,50),(0.1,12)]
    init = [best_x] + [best_x + rng.normal(0, [3,0.2,1.5,0.8]) for _ in range(popsize-1)]
    init = [np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds]) for p in init]
    try:
        res = differential_evolution(obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed, polish=False, init=init)
        if res.fun < 100:
            return res.x, -res.fun
    except Exception:
        pass
    return best_x, best_dur

# ============================ 8. 效能矩阵+整数规划（问题三） ============================
def emip_p3(seed=42, verbose=True):
    """
    问题三效能矩阵+整数规划：
    1. 第1枚弹：4维优化（v,theta,t1,tau1），获得效能e1
    2. 固定v,theta，第2枚弹：2维独立优化（t2,tau2），目标是自身效能最大（不考虑第1枚弹区间）
    3. 固定v,theta，第3枚弹：2维独立优化（t3,tau3），目标是自身效能最大（不考虑前2枚弹区间）
    4. 整数规划：选择哪些弹参与（2^3=8种），目标是效能简单相加最大
    5. 用实际并集评价：可能因区间重叠而小于简单相加
    """
    if verbose:
        print(f"{'='*60}")
        print(f"效能矩阵+整数规划（问题三，8维）—— 各弹独立优化+简单相加")
        print(f"{'='*60}")
    start_time = time.time()
    rng = np.random.RandomState(seed)
    drone = P3_DRONE

    # 第1枚弹：4维优化
    if verbose: print("  第1枚弹：4维独立优化...")
    x1, e1 = _single_smoke_4d_optimize(drone, seed=seed)
    v1, theta1, t1, tau1 = x1
    if verbose: print(f"    效能e1={e1:.4f}s (v={v1:.1f}, θ={np.degrees(theta1):.1f}°, t={t1:.2f}, τ={tau1:.2f})")

    # 第2枚弹：固定v,theta，2维独立优化（目标自身效能最大，不考虑第1枚弹）
    if verbose: print("  第2枚弹：2维独立优化（固定v,θ，不考虑第1枚弹区间）...")
    best_e2 = 0.0
    best_t2, best_tau2 = t1 + 2.0, 3.0
    for t2 in np.linspace(max(0, t1+1.0), 55, 20):
        for tau2 in np.linspace(0.5, 10.0, 12):
            smoke = [{"drone": drone, "speed": v1, "direction": theta1, "drop_time": t2, "det_delay": tau2}]
            _, dur = calc_system_shielding(smoke, TARGET_SAMPLES)
            if dur > best_e2:
                best_e2 = dur
                best_t2, best_tau2 = t2, tau2
    e2 = best_e2
    t2, tau2 = best_t2, best_tau2
    if verbose: print(f"    效能e2={e2:.4f}s (t={t2:.2f}, τ={tau2:.2f})")

    # 第3枚弹：固定v,theta，2维独立优化（目标自身效能最大，不考虑前2枚弹）
    if verbose: print("  第3枚弹：2维独立优化（固定v,θ，不考虑前2枚弹区间）...")
    last_t = max(t1, t2)
    best_e3 = 0.0
    best_t3, best_tau3 = last_t + 2.0, 3.0
    for t3 in np.linspace(max(0, last_t+1.0), 60, 20):
        for tau3 in np.linspace(0.5, 10.0, 12):
            smoke = [{"drone": drone, "speed": v1, "direction": theta1, "drop_time": t3, "det_delay": tau3}]
            _, dur = calc_system_shielding(smoke, TARGET_SAMPLES)
            if dur > best_e3:
                best_e3 = dur
                best_t3, best_tau3 = t3, tau3
    e3 = best_e3
    t3, tau3 = best_t3, best_tau3
    if verbose: print(f"    效能e3={e3:.4f}s (t={t3:.2f}, τ={tau3:.2f})")

    # 整数规划：2^3=8种组合，目标是效能简单相加最大
    if verbose: print("  整数规划：枚举2^3=8种弹组合，目标=效能简单相加最大...")
    effs = [e1, e2, e3]
    params = [(t1, tau1), (t2, tau2), (t3, tau3)]
    best_combination = None
    best_eff_sum = -1
    for mask in range(8):
        selected = [i for i in range(3) if mask & (1 << i)]
        eff_sum = sum(effs[i] for i in selected)
        # 检查投放间隔约束
        valid = True
        selected_times = sorted([params[i][0] for i in selected])
        for j in range(1, len(selected_times)):
            if selected_times[j] - selected_times[j-1] < 1.0 - EPSILON:
                valid = False
                break
        if valid and eff_sum > best_eff_sum:
            best_eff_sum = eff_sum
            best_combination = selected
    if verbose:
        print(f"    最优组合: 弹{[i+1 for i in best_combination]}, 效能简单相加={best_eff_sum:.4f}s")

    # 构建最终解，用实际并集评价
    final_x = np.array([v1, theta1, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    for i in best_combination:
        final_x[2 + i*2] = params[i][0]
        final_x[3 + i*2] = params[i][1]

    final_obj, final_result = p3_objective(final_x, return_details=True)
    actual_duration = final_result["duration"] if final_result else 0.0
    overlap_loss = best_eff_sum - actual_duration
    elapsed = time.time() - start_time

    result = {"algorithm": "效能矩阵+整数规划", "best_x": final_x, "best_objective": final_obj,
              "system_duration": actual_duration, "convergence": [], "elapsed_time": elapsed,
              "eff_sum_simple": best_eff_sum, "overlap_loss": overlap_loss,
              "individual_effs": [e1, e2, e3]}
    if verbose:
        print(f"\n  效能矩阵+IP完成，耗时: {elapsed:.1f}s")
        print(f"  效能简单相加: {best_eff_sum:.4f}s")
        print(f"  实际并集时长: {actual_duration:.4f}s")
        print(f"  区间重叠损失: {overlap_loss:.4f}s ({overlap_loss/max(best_eff_sum,0.01)*100:.1f}%)")
    return result

# ============================ 9. 效能矩阵+整数规划（问题四） ============================
def emip_p4(seed=42, verbose=True):
    """
    问题四效能矩阵+整数规划：
    1. 3架无人机各做4维独立优化，获得各自效能e1,e2,e3
    2. 整数规划：选择哪些无人机参与（2^3=8种），目标是效能简单相加最大
    3. 用实际并集评价：可能因区间重叠而小于简单相加
    """
    if verbose:
        print(f"{'='*60}")
        print(f"效能矩阵+整数规划（问题四，12维）—— 各无人机独立优化+简单相加")
        print(f"{'='*60}")
    start_time = time.time()

    # 3组4维独立优化
    if verbose: print("  第一阶段：3组4维独立优化（各无人机自身效能最大）...")
    unit_results = {}
    for i, drone_name in enumerate(P4_DRONES):
        if verbose: print(f"    单元优化: {drone_name}...", end=" ")
        x_unit, e_unit = _single_smoke_4d_optimize(drone_name, seed=seed+i*100)
        unit_results[drone_name] = {"x": x_unit, "duration": e_unit}
        if verbose: print(f"效能={e_unit:.2f}s")

    # 整数规划：2^3=8种组合
    if verbose: print("  整数规划：枚举2^3=8种无人机组合，目标=效能简单相加最大...")
    effs = [unit_results[d]["duration"] for d in P4_DRONES]
    best_combination = None
    best_eff_sum = -1
    for mask in range(8):
        selected = [i for i in range(3) if mask & (1 << i)]
        eff_sum = sum(effs[i] for i in selected)
        if eff_sum > best_eff_sum:
            best_eff_sum = eff_sum
            best_combination = selected
    if verbose:
        print(f"    最优组合: {[P4_DRONES[i] for i in best_combination]}, 效能简单相加={best_eff_sum:.4f}s")

    # 构建最终解，用实际并集评价
    combined_x = np.full(12, -1.0)
    for i in best_combination:
        combined_x[i*4:(i+1)*4] = unit_results[P4_DRONES[i]]["x"]

    final_obj, final_result = p4_objective(combined_x, return_details=True)
    actual_duration = final_result["duration"] if final_result else 0.0
    overlap_loss = best_eff_sum - actual_duration
    elapsed = time.time() - start_time

    result = {"algorithm": "效能矩阵+整数规划", "best_x": combined_x, "best_objective": final_obj,
              "system_duration": actual_duration, "convergence": [], "elapsed_time": elapsed,
              "eff_sum_simple": best_eff_sum, "overlap_loss": overlap_loss,
              "individual_effs": effs}
    if verbose:
        print(f"\n  效能矩阵+IP完成，耗时: {elapsed:.1f}s")
        print(f"  效能简单相加: {best_eff_sum:.4f}s")
        print(f"  实际并集时长: {actual_duration:.4f}s")
        print(f"  区间重叠损失: {overlap_loss:.4f}s ({overlap_loss/max(best_eff_sum,0.01)*100:.1f}%)")
    return result

# ============================ 10. 逆推分层（低维退化版） ============================
def reverse_hierarchical_p3(seed=42, verbose=True):
    """问题三逆推分层：逐弹贪心优化（4维→2维→2维），每步考虑并集"""
    if verbose:
        print(f"{'='*60}")
        print(f"逆推分层优化（问题三，8维）—— 逐弹贪心（考虑并集）")
        print(f"{'='*60}")
    start_time = time.time()
    rng = np.random.RandomState(seed)
    drone = P3_DRONE

    # 第1枚弹：4维优化
    if verbose: print("  第1枚弹：4维优化...")
    x1, e1 = _single_smoke_4d_optimize(drone, seed=seed)
    v1, theta1, t1, tau1 = x1
    if verbose: print(f"    累计{e1:.4f}s")

    fixed = [(t1, tau1)]

    # 第2枚弹：2维优化（固定v,theta，目标是累计并集最大）
    if verbose: print("  第2枚弹：2维优化（目标=累计并集最大）...")
    best_t2, best_dur2 = None, e1
    for t2 in np.linspace(max(0, t1+1.0), 55, 20):
        for tau2 in np.linspace(0.5, 10.0, 12):
            smoke = [{"drone": drone, "speed": v1, "direction": theta1, "drop_time": d[0], "det_delay": d[1]} for d in fixed]
            smoke.append({"drone": drone, "speed": v1, "direction": theta1, "drop_time": t2, "det_delay": tau2})
            _, dur = calc_system_shielding(smoke, TARGET_SAMPLES)
            if dur > best_dur2:
                best_dur2 = dur
                best_t2 = (t2, tau2)
    if best_t2 is not None:
        t2, tau2 = best_t2
        fixed.append((t2, tau2))
    else:
        t2, tau2 = -1.0, 1.0
    if verbose: print(f"    累计{best_dur2:.4f}s")

    # 第3枚弹：2维优化（目标=累计并集最大）
    if verbose: print("  第3枚弹：2维优化（目标=累计并集最大）...")
    last_t = max([d[0] for d in fixed])
    best_t3, best_dur3 = None, best_dur2
    for t3 in np.linspace(max(0, last_t+1.0), 60, 20):
        for tau3 in np.linspace(0.5, 10.0, 12):
            smoke = [{"drone": drone, "speed": v1, "direction": theta1, "drop_time": d[0], "det_delay": d[1]} for d in fixed]
            smoke.append({"drone": drone, "speed": v1, "direction": theta1, "drop_time": t3, "det_delay": tau3})
            _, dur = calc_system_shielding(smoke, TARGET_SAMPLES)
            if dur > best_dur3:
                best_dur3 = dur
                best_t3 = (t3, tau3)
    if best_t3 is not None:
        t3, tau3 = best_t3
    else:
        t3, tau3 = -1.0, 1.0
    if verbose: print(f"    累计{best_dur3:.4f}s")

    final_x = np.array([v1, theta1, t1, tau1, t2, tau2, t3, tau3])
    final_obj, final_result = p3_objective(final_x, return_details=True)
    elapsed = time.time() - start_time

    result = {"algorithm": "逆推分层", "best_x": final_x, "best_objective": final_obj,
              "system_duration": final_result["duration"] if final_result else best_dur3,
              "convergence": [], "elapsed_time": elapsed}
    if verbose:
        print(f"  逆推分层完成，耗时: {elapsed:.1f}s")
        print(f"  系统总遮蔽时长: {result['system_duration']:.4f}s")
    return result

def reverse_hierarchical_p4(seed=42, verbose=True):
    """问题四逆推分层：3组4维单元优化 + 直接组合（用实际并集评价）"""
    if verbose:
        print(f"{'='*60}")
        print(f"逆推分层优化（问题四，12维）—— 单元优化直接组合")
        print(f"{'='*60}")
    start_time = time.time()

    if verbose: print("  第一阶段：3组单元优化...")
    unit_results = {}
    for drone_name in P4_DRONES:
        if verbose: print(f"    单元优化: {drone_name}...", end=" ")
        x_unit, e_unit = _single_smoke_4d_optimize(drone_name, seed=seed)
        unit_results[drone_name] = {"x": x_unit, "duration": e_unit}
        if verbose: print(f"{e_unit:.2f}s")

    if verbose: print("  第二阶段：直接组合...")
    combined_x = np.zeros(12)
    for i, drone_name in enumerate(P4_DRONES):
        combined_x[i*4:(i+1)*4] = unit_results[drone_name]["x"]

    final_obj, final_result = p4_objective(combined_x, return_details=True)
    elapsed = time.time() - start_time

    result = {"algorithm": "逆推分层", "best_x": combined_x, "best_objective": final_obj,
              "system_duration": final_result["duration"] if final_result else 0.0,
              "convergence": [], "elapsed_time": elapsed}
    if verbose:
        print(f"  逆推分层完成，耗时: {elapsed:.1f}s")
        print(f"  系统总遮蔽时长: {result['system_duration']:.4f}s")
    return result

# ============================ 11. 对比实验主函数 ============================
def run_comparison(scenario="p3", n_runs=5, seeds=None):
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024][:n_runs]

    if scenario == "p3":
        dim = P3_DIM
        bounds = p3_get_bounds()
        objective = p3_objective
        rh_func = reverse_hierarchical_p3
        emip_func = emip_p3
        label = "问题三（8维）"
    else:
        dim = P4_DIM
        bounds = p4_get_bounds()
        objective = p4_objective
        rh_func = reverse_hierarchical_p4
        emip_func = emip_p4
        label = "问题四（12维）"

    print(f"\n{'#'*70}")
    print(f"# 对比实验：{label}（四算法：DE / PSO / 效能矩阵+IP / 逆推分层）")
    print(f"# 每个算法运行 {n_runs} 次，DE/PSO 种群30 迭代100")
    print(f"{'#'*70}")

    results = {"scenario": label, "dim": dim, "n_runs": n_runs, "algorithms": {}}

    # 1. 标准DE
    print(f"\n【1/4】标准DE对比实验...")
    de_results = []
    for i, seed in enumerate(seeds):
        print(f"  DE第{i+1}/{n_runs}次 (seed={seed})...")
        r = standard_de_optimization(objective, bounds, dim, popsize=30, maxiter=100,
                                       seed=seed, verbose=False, label=f"(seed={seed})")
        de_results.append(r)
        print(f"    时长={r['system_duration']:.4f}s, 耗时={r['elapsed_time']:.1f}s")
    results["algorithms"]["标准DE"] = de_results

    # 2. 标准PSO
    print(f"\n【2/4】标准PSO对比实验...")
    pso_results = []
    for i, seed in enumerate(seeds):
        print(f"  PSO第{i+1}/{n_runs}次 (seed={seed})...")
        pso = StandardPSO(objective_func=objective, bounds=bounds, n_particles=30, max_iter=100,
                           seed=seed, verbose=False, label=f"(seed={seed})")
        r = pso.optimize()
        pso_results.append(r)
        print(f"    时长={r['system_duration']:.4f}s, 耗时={r['elapsed_time']:.1f}s")
    results["algorithms"]["标准PSO"] = pso_results

    # 3. 效能矩阵+整数规划
    print(f"\n【3/4】效能矩阵+整数规划对比实验...")
    emip_results = []
    for i, seed in enumerate(seeds):
        print(f"  效能矩阵+IP第{i+1}/{n_runs}次 (seed={seed})...")
        r = emip_func(seed=seed, verbose=False)
        emip_results.append(r)
        print(f"    实际并集={r['system_duration']:.4f}s, 效能和={r['eff_sum_simple']:.2f}s, "
              f"重叠损失={r['overlap_loss']:.2f}s, 耗时={r['elapsed_time']:.1f}s")
    results["algorithms"]["效能矩阵+整数规划"] = emip_results

    # 4. 逆推分层
    print(f"\n【4/4】逆推分层对比实验...")
    rh_results = []
    for i, seed in enumerate(seeds):
        print(f"  逆推分层第{i+1}/{n_runs}次 (seed={seed})...")
        r = rh_func(seed=seed, verbose=False)
        rh_results.append(r)
        print(f"    时长={r['system_duration']:.4f}s, 耗时={r['elapsed_time']:.1f}s")
    results["algorithms"]["逆推分层"] = rh_results

    # 统计汇总
    print(f"\n{'='*70}")
    print(f"对比实验统计结果 — {label}")
    print(f"{'='*70}")
    print(f"{'算法':<16} {'最优(s)':<10} {'均值(s)':<10} {'标准差':<10} {'最差(s)':<10} {'平均耗时(s)':<12}")
    print("-" * 80)

    stats = {}
    for algo_name, algo_results in results["algorithms"].items():
        durations = [r["system_duration"] for r in algo_results]
        times = [r["elapsed_time"] for r in algo_results]
        mean_dur = np.mean(durations)
        std_dur = np.std(durations)
        best_dur = np.max(durations)
        worst_dur = np.min(durations)
        mean_time = np.mean(times)
        stats[algo_name] = {"best": best_dur, "mean": mean_dur, "std": std_dur,
                             "worst": worst_dur, "mean_time": mean_time}
        print(f"{algo_name:<16} {best_dur:<10.4f} {mean_dur:<10.4f} {std_dur:<10.4f} {worst_dur:<10.4f} {mean_time:<12.1f}")

    # 效能矩阵+IP的额外信息
    emip_avg_eff = np.mean([r["eff_sum_simple"] for r in emip_results])
    emip_avg_loss = np.mean([r["overlap_loss"] for r in emip_results])
    print(f"\n效能矩阵+整数规划 额外信息:")
    print(f"  平均效能简单相加: {emip_avg_eff:.4f}s")
    print(f"  平均实际并集: {stats['效能矩阵+整数规划']['mean']:.4f}s")
    print(f"  平均区间重叠损失: {emip_avg_loss:.4f}s ({emip_avg_loss/max(emip_avg_eff,0.01)*100:.1f}%)")

    results["statistics"] = stats

    # 保存结果
    output_file = f"comparison_{scenario}_results.json"
    serializable = {"scenario": label, "dim": dim, "n_runs": n_runs, "statistics": stats,
                    "emip_extra": {"avg_eff_sum": emip_avg_eff, "avg_overlap_loss": emip_avg_loss},
                    "raw": {algo: [{"system_duration": r["system_duration"], "elapsed_time": r["elapsed_time"],
                                     "eff_sum_simple": r.get("eff_sum_simple"),
                                     "overlap_loss": r.get("overlap_loss"),
                                     "convergence": r.get("convergence", [])[-20:] if r.get("convergence") else []}
                                    for r in algo_results]
                             for algo, algo_results in results["algorithms"].items()}}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_file}")

    return results

# ============================ 主函数 ============================
if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "both"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if scenario in ("p3", "both"):
        run_comparison("p3", n_runs=n_runs)

    if scenario in ("p4", "both"):
        run_comparison("p4", n_runs=n_runs)

    print("\n全部对比实验完成！")
