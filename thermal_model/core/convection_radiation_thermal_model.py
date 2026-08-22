#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy E — 对流 + 非线性辐射 + 一阶滞后热模型
================================================

新并行热模型架构 (EXPERIMENTAL / MODEL-CONSTRUCTION ONLY):

    实测内部温度 T_internal(t)
        -> convection+radiation chip FDM (BARE_TOP_COC_LAYERS)
        -> T_sample_FDM(t)     (控制体积加权空间平均, 不经过滞后)
        -> T_top_FDM(t)        (顶部表面 FDM 状态, 不经过滞后)
        -> 一阶外部滞后 tau_lag
        -> T_top_observed_predicted(t)   (用于与温度计实测比较)

顶部边界能量平衡 (x = 850 um, 裸顶 Top COC 外表面):

    q_cond = q_conv + q_rad

        q_conv = h_conv * (T_surface_C - T_air_C)

        q_rad  = eps * sigma_SB * F_view
                 * ( (T_surface_C + 273.15)^4
                     - (T_surroundings_C + 273.15)^4 )

    辐射第四功率项一律使用开尔文温标。

固定边界参数 (Strategy E 唯一事实来源):

    h_conv          = 10.0   W/(m2 K)
    epsilon_surface = 0.90
    sigma_SB        = 5.670374419e-8   W/(m2 K4)
    F_view          = 1.0

环境温度规则 (关键, 见 CALIBRATION_STRATEGIES.md Strategy E 节):

    T_air_C = T_surroundings_C = 第一个有效实测 Top COC 温度
    (对整次仿真恒定; 是项目级代理假设, 不是独立室温测量)

禁止:
    - 把辐射线性化为固定 h_rad 后当作数值边界条件 (h_rad 仅诊断用);
    - 用实测顶部全迹线作为时变环境边界;
    - 静默回退到 25 C / T_internal[0] / T_top_FDM[0];
    - 修改 heat_model.py 或 lag_augmented_thermal_model.py 的既有物理。

架构说明
--------
- 物理求解器 (run_convection_radiation_fdm) 不读取任何实测 Top COC 数据 /
  校准 CSV; 环境温度由调用方显式传入标量。
- 复用 heat_model 的 Material / Layer / LayerStack / build_layer_stack /
  compute_sample_weights / compute_stable_dt / BARE_TOP_COC_LAYERS 等通用
  基础设施, 不复制、不修改界面有限体积物理。
- 复用 lag_augmented_thermal_model.apply_first_order_lag (分段线性输入的
  精确一阶递推; tau=0 严格恒等)。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from thermal_model.core import heat_model
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag


# ============================================================
# Strategy E 固定边界参数 (唯一事实来源)
# ============================================================

SIGMA_SB_W_M2_K4 = 5.670374419e-8   # Stefan-Boltzmann 常数
H_CONV_STRATEGY_E_W_M2K = 10.0      # 固定自然对流换热系数
EMISSIVITY_STRATEGY_E = 0.90        # 固定表面发射率 (工程假设)
VIEW_FACTOR_STRATEGY_E = 1.0        # 固定视角因子 (降阶假设)
RHO_COC_STRATEGY_E = 1020.0         # COC 密度 (与既有模型一致)

KELVIN_OFFSET = 273.15


# ============================================================
# 参数容器 (不可变)
# ============================================================

@dataclass(frozen=True)
class ConvectionRadiationParameters:
    """策略 E 参数 (不可变)。

    自由参数 (未来标定): k_eff_W_mK / cp_eff_J_kgK / tau_lag_s。
    派生: alpha_eff = k/(rho*cp), effusivity = sqrt(k*rho*cp)。
    固定边界假设: h_conv=10, eps=0.90, sigma=5.670374419e-8, F=1.0。

    环境温度 T_environment_C 不在此容器内 (无通用默认值), 由每次仿真的
    调用方显式提供 (见 run_convection_radiation_lag_model)。
    """
    k_eff_W_mK: float
    cp_eff_J_kgK: float
    tau_lag_s: float
    rho_COC_kg_m3: float = RHO_COC_STRATEGY_E
    h_conv_W_m2K: float = H_CONV_STRATEGY_E_W_M2K
    emissivity: float = EMISSIVITY_STRATEGY_E
    sigma_SB_W_m2K4: float = SIGMA_SB_W_M2_K4
    view_factor: float = VIEW_FACTOR_STRATEGY_E

    def __post_init__(self) -> None:
        _require_finite_positive("k_eff_W_mK", self.k_eff_W_mK)
        _require_finite_positive("cp_eff_J_kgK", self.cp_eff_J_kgK)
        _require_finite_positive("rho_COC_kg_m3", self.rho_COC_kg_m3)
        if not (self.tau_lag_s >= 0 and np.isfinite(self.tau_lag_s)):
            raise ValueError(
                f"tau_lag_s 必须 >= 0 且有限, 收到 {self.tau_lag_s!r}")
        if not (self.h_conv_W_m2K >= 0 and np.isfinite(self.h_conv_W_m2K)):
            raise ValueError(
                f"h_conv_W_m2K 必须 >= 0 且有限, 收到 {self.h_conv_W_m2K!r}")
        if not (0.0 <= self.emissivity <= 1.0 and np.isfinite(self.emissivity)):
            raise ValueError(
                f"emissivity 必须在 [0, 1] 内, 收到 {self.emissivity!r}")
        if not (0.0 <= self.view_factor <= 1.0
                and np.isfinite(self.view_factor)):
            raise ValueError(
                f"view_factor 必须在 [0, 1] 内, 收到 {self.view_factor!r}")
        if not (self.sigma_SB_W_m2K4 > 0 and np.isfinite(self.sigma_SB_W_m2K4)):
            raise ValueError(
                f"sigma_SB_W_m2K4 必须 > 0 且有限, 收到 "
                f"{self.sigma_SB_W_m2K4!r}")

    # --------------------------------------------------------
    # 派生属性
    # --------------------------------------------------------
    @property
    def alpha_eff_m2_s(self) -> float:
        """alpha_eff = k_eff / (rho_COC * cp_eff)。"""
        return self.k_eff_W_mK / (self.rho_COC_kg_m3 * self.cp_eff_J_kgK)

    @property
    def effusivity(self) -> float:
        """热浸透率 e = sqrt(k_eff * rho_COC * cp_eff)。"""
        return float(np.sqrt(
            self.k_eff_W_mK * self.rho_COC_kg_m3 * self.cp_eff_J_kgK))


def _require_finite_positive(field_name: str, value) -> None:
    if not (value > 0 and np.isfinite(value)):
        raise ValueError(
            f"{field_name} 必须 > 0 且有限, 收到 {value!r}")


# ============================================================
# 辐射诊断辅助 (仅供解释; 求解器仍使用非线性 Stefan-Boltzmann)
# ============================================================

def radiative_heat_flux_W_m2(
    surface_temperature_C,
    surroundings_temperature_C,
    emissivity: float = EMISSIVITY_STRATEGY_E,
    view_factor: float = VIEW_FACTOR_STRATEGY_E,
    sigma_SB_W_m2K4: float = SIGMA_SB_W_M2_K4,
) -> float:
    """非线性 Stefan-Boltzmann 辐射热通量 q_rad (W/m2)。

        q_rad = eps*sigma*F * (Ts_K^4 - Tsur_K^4)

    温度以摄氏度输入, 第四功率项内部转换为开尔文。
    """
    Ts_K = float(surface_temperature_C) + KELVIN_OFFSET
    Tsur_K = float(surroundings_temperature_C) + KELVIN_OFFSET
    return float(
        emissivity * sigma_SB_W_m2K4 * view_factor * (Ts_K ** 4 - Tsur_K ** 4))


def equivalent_radiative_heat_transfer_coefficient(
    surface_temperature_C,
    surroundings_temperature_C,
    emissivity: float = EMISSIVITY_STRATEGY_E,
    view_factor: float = VIEW_FACTOR_STRATEGY_E,
    sigma_SB_W_m2K4: float = SIGMA_SB_W_M2_K4,
) -> float:
    """等效辐射换热系数 h_rad (W/(m2 K)) —— 仅诊断。

        h_rad = eps*sigma*F * (Ts_K + Tsur_K) * (Ts_K^2 + Tsur_K^2)

    满足 q_rad = h_rad * (Ts_C - Tsur_C)。绝不允许替代求解器内的
    非线性辐射边界。
    """
    Ts_K = float(surface_temperature_C) + KELVIN_OFFSET
    Tsur_K = float(surroundings_temperature_C) + KELVIN_OFFSET
    return float(
        emissivity * sigma_SB_W_m2K4 * view_factor
        * (Ts_K + Tsur_K) * (Ts_K ** 2 + Tsur_K ** 2))


# ============================================================
# 环境温度解析 (分析层; 不在物理求解器内)
# ============================================================

def infer_environment_from_initial_top_measurement(top_temperature_C,
                                                   time_s=None) -> Dict:
    """分析层辅助: 解析 T_environment = 第一个有效实测 Top COC 温度。

    规则:
        - 取第一个「有限且非 NaN」的实测 Top COC 温度值 (标量);
        - 不是第一个数组元素 (可能为 NaN);
        - 不是前 N 个值的平均;
        - 不来自内部温度;
        - 不拟合环境温度。

    参数:
        top_temperature_C : 实测 Top COC 温度迹线 (一维类数组);
        time_s            : 可选, 与温度等长的测量时间轴 (用于记录来源时间)。

    返回 dict:
        T_environment_C : 标量环境温度 (°C);
        source_index    : 产生该值的数组索引 (int);
        source_time_s   : 产生该值的时间 (float 或 None)。
    """
    top = np.asarray(top_temperature_C, dtype=float)
    if top.ndim != 1:
        raise ValueError("top_temperature_C 必须是一维数组。")
    valid = np.flatnonzero(np.isfinite(top))
    if valid.size == 0:
        raise ValueError(
            "实测 Top COC 温度迹线中没有有效 (有限) 值, 无法解析环境温度。")
    idx = int(valid[0])
    value = float(top[idx])
    src_time = None
    if time_s is not None:
        t = np.asarray(time_s, dtype=float)
        if t.ndim != 1 or t.size != top.size:
            raise ValueError("time_s 必须与 top_temperature_C 等长的一维数组。")
        src_time = float(t[idx])
    return {
        "T_environment_C": value,
        "source_index": idx,
        "source_time_s": src_time,
    }


# ============================================================
# 环境温度验证 (无静默回退)
# ============================================================

def _validate_environment(T_air_C, T_surroundings_C) -> None:
    """校验环境温度: 必须显式提供、有限、开尔文温标 > 0。"""
    for name, value in (("T_air_C", T_air_C),
                        ("T_surroundings_C", T_surroundings_C)):
        if value is None:
            raise ValueError(
                f"{name} 必须显式提供 (无静默回退到 25 C / 内部温度)。")
        v = float(value)
        if not np.isfinite(v):
            raise ValueError(f"{name} 必须有限, 收到 {v!r}")
        if v + KELVIN_OFFSET <= 0.0:
            raise ValueError(
                f"{name} 对应开尔文温标必须 > 0, 收到 {v} C")


# ============================================================
# 材料构造 (仅替换 COC; 不修改 DEFAULT_MATERIALS)
# ============================================================

def make_convection_radiation_materials(k_eff_W_mK, cp_eff_J_kgK,
                                        rho_COC_kg_m3=RHO_COC_STRATEGY_E):
    """构造候选材料库: 仅替换 COC 的 k/cp (rho 显式给定), 其余逐位不变。"""
    mats = heat_model.copy_default_materials()
    coc = mats["COC"]
    mats["COC"] = heat_model.Material(
        name=coc.name,
        k_W_mK=float(k_eff_W_mK),
        rho_kg_m3=float(rho_COC_kg_m3),
        cp_J_kgK=float(cp_eff_J_kgK),
    )
    return mats


# ============================================================
# 非线性顶部边界 (Newton 求解)
# ============================================================

def _convection_only_surface_solution(T_prev_C, T_air_C, k_over_dx,
                                      h_conv) -> float:
    """纯对流顶部解析解 (辐射项被忽略时) —— 用作 Newton 物理初猜。"""
    return (k_over_dx * float(T_prev_C) + h_conv * float(T_air_C)) / (
        k_over_dx + h_conv)


def solve_top_surface_temperature(
    T_prev_C,
    T_air_C,
    T_surroundings_C,
    k_over_dx,
    h_conv_W_m2K,
    emissivity,
    sigma_SB_W_m2K4,
    view_factor,
    T_initial_guess_C=None,
    abs_tolerance_C=1e-10,
    max_iterations=20,
):
    """Newton 迭代求解顶部非线性边界表面温度 Ts (°C)。

    能量平衡 (所有量取正向外流):

        f(Ts) = (k/dx)*(T_prev - Ts)
                - h_conv*(Ts - T_air)
                - eps*sigma*F*((Ts + 273.15)^4 - (Tsur + 273.15)^4)
                = 0

    解析导数:

        df/dTs = -(k/dx) - h_conv
                 - 4*eps*sigma*F*(Ts + 273.15)^3

    在物理范围内 df/dTs < 0, 标量边界函数单调, Newton 迭代稳健。

    初猜: 显式给定, 否则使用纯对流解析解。
    收敛判据: |ΔTs| <= abs_tolerance_C (摄氏温标绝对容差)。
    失败: 达到 max_iterations 仍未收敛 -> 抛 RuntimeError (不静默接受坏根)。

    返回 float: 表面温度 (°C)。
    """
    k_over_dx = float(k_over_dx)
    h_conv = float(h_conv_W_m2K)
    eps = float(emissivity)
    sigma = float(sigma_SB_W_m2K4)
    F = float(view_factor)
    T_prev = float(T_prev_C)
    T_air = float(T_air_C)
    Tsur_K = float(T_surroundings_C) + KELVIN_OFFSET

    if T_initial_guess_C is None:
        Ts = _convection_only_surface_solution(T_prev, T_air, k_over_dx,
                                               h_conv)
    else:
        Ts = float(T_initial_guess_C)

    if not np.isfinite(Ts):
        raise RuntimeError(
            "顶部边界 Newton 初猜非有限, 无法迭代求解表面温度。")

    coeff = eps * sigma * F
    for _ in range(max_iterations):
        Ts_K = Ts + KELVIN_OFFSET
        f = (k_over_dx * (T_prev - Ts)
             - h_conv * (Ts - T_air)
             - coeff * (Ts_K ** 4 - Tsur_K ** 4))
        df = -(k_over_dx + h_conv
               + 4.0 * coeff * Ts_K ** 3)
        if df >= 0.0:
            raise RuntimeError(
                f"顶部边界 Newton 导数非负 (df={df:.6e}), 物理范围外; "
                "无法求解表面温度。")
        step = f / df
        Ts_new = Ts - step
        if abs(step) <= abs_tolerance_C:
            return float(Ts_new)
        Ts = Ts_new

    raise RuntimeError(
        f"顶部非线性边界 Newton 迭代 {max_iterations} 次未收敛 "
        f"(T_prev={T_prev:.6f} C, T_air={T_air:.6f} C, "
        f"T_surroundings={Tsur_K - KELVIN_OFFSET:.6f} C, "
        f"k/dx={k_over_dx:.6e}, h={h_conv:.3f}, eps={eps:.3f}); "
        "不静默接受坏根。")


def boundary_residual_W_m2(T_prev_C, T_surface_C, T_air_C,
                           T_surroundings_C, k_over_dx,
                           h_conv_W_m2K, emissivity, sigma_SB_W_m2K4,
                           view_factor) -> float:
    """诊断能量平衡残差 q_cond - q_conv - q_rad (W/m2)。

    要求: |residual| 很小。用于数值验证与单元测试。
    """
    T_surf = float(T_surface_C)
    Ts_K = T_surf + KELVIN_OFFSET
    Tsur_K = float(T_surroundings_C) + KELVIN_OFFSET
    q_cond = float(k_over_dx) * (float(T_prev_C) - T_surf)
    q_conv = float(h_conv_W_m2K) * (T_surf - float(T_air_C))
    q_rad = (float(emissivity) * float(sigma_SB_W_m2K4)
             * float(view_factor) * (Ts_K ** 4 - Tsur_K ** 4))
    return float(q_cond - q_conv - q_rad)


# ============================================================
# 对流 + 非线性辐射 FDM 求解器
# ============================================================

def run_convection_radiation_fdm(
    time_s,
    bottom_temperature_C,
    materials,
    layers,
    T_air_C,
    T_surroundings_C,
    h_conv_W_m2K=H_CONV_STRATEGY_E_W_M2K,
    emissivity=EMISSIVITY_STRATEGY_E,
    sigma_SB_W_m2K4=SIGMA_SB_W_M2_K4,
    view_factor=VIEW_FACTOR_STRATEGY_E,
    save_dt=0.1,
    T_initial_C=None,
    newton_abs_tolerance_C=1e-10,
    newton_max_iterations=20,
):
    """1D 多层瞬态 FDM —— 底部 Dirichlet + 顶部非线性对流+辐射。

    物理与 heat_model.run_simulation 完全一致, 唯一差异:
    顶部 Robin 边界替换为非线性能量平衡:

        (k/dx_top)*(T_prev - Ts) = h_conv*(Ts - T_air)
                                   + eps*sigma*F*(Ts_K^4 - Tsur_K^4)

    表面温度 Ts 每步用 Newton 迭代求解 (不线性化辐射)。

    本函数不读取任何实测 Top COC / 校准 CSV 数据;
    环境温度 T_air_C / T_surroundings_C 必须由调用方显式提供 (无默认回退)。

    参数:
        time_s               : 边界时间轴 (秒), 严格单调递增;
        bottom_temperature_C : 底部 Dirichlet 温度迹线 (°C);
        materials, layers    : 材料库 / 层叠结构;
        T_air_C              : 环境空气温度 (°C), 必须提供;
        T_surroundings_C     : 辐射周围环境温度 (°C), 必须提供;
        h_conv_W_m2K         : 自然对流换热系数 (策略 E 固定 10.0);
        emissivity / sigma_SB_W_m2K4 / view_factor : 辐射参数;
        save_dt              : 输出下采样间隔 (秒);
        T_initial_C          : 初始均匀温度场 (°C), None -> 第一个底部值;
        newton_abs_tolerance_C / newton_max_iterations : Newton 求解设置。

    返回 dict:
        t_array / T_bottom_arr / T_sample_arr / T_top_surface_arr /
        T_outer_surface_arr / T_final / dt / Nt / Nx / save_interval /
        mesh / time_fdm / bottom_temperature_fdm,
        boundary_residual_arr         : 每个保存点表面能量平衡残差 (W/m2);
                                        [0] 为均匀初始场瞬态 (表面未求解),
                                        可能非零; 后续保存点应 ~0;
        max_abs_boundary_residual_W_m2: 全程最大绝对残差 (W/m2)
                                        (含初始瞬态, 诊断时建议排除 [0]);
        newton_total_iterations       : Newton 总迭代次数 (诊断);
        newton_max_iterations_per_step: 单步最大迭代次数 (诊断);
        newton_convergence_failures   : 收敛失败次数 (应为 0, 失败即抛异常)。
    """
    _validate_environment(T_air_C, T_surroundings_C)

    time_s, bottom_temperature_C = heat_model._validate_boundary_trace(
        time_s, bottom_temperature_C)

    mesh, dt = heat_model.compute_stable_dt(materials, layers)
    if mesh.idx_sample.size == 0:
        raise ValueError(
            "层叠结构中没有 role='sample' 的层, 无法提取样品温度。")

    x = mesh.x
    h = mesh.h
    Nx = mesh.Nx
    idx_top_surface = mesh.idx_top_surface
    has_top_surface_obs = idx_top_surface.size > 0

    if T_initial_C is None:
        T_initial_C = float(bottom_temperature_C[0])
    T_init = float(T_initial_C)
    if not np.isfinite(T_init):
        raise ValueError(f"T_initial_C 必须有限, 收到 {T_init!r}")

    t_total = float(time_s[-1] - time_s[0])
    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)
    bottom_fdm = np.interp(time_fdm, time_s, bottom_temperature_C)

    # ---- 内部节点系数 (与 heat_model.run_simulation 逐位一致) ----
    k_face = mesh.k_face
    rho_cp = mesh.rho_cp
    h_m = h[:-1]
    h_p = h[1:]
    k_m = k_face[:-1]
    k_p = k_face[1:]
    rc_int = rho_cp[1:-1]
    fac = 2 * dt / ((h_m + h_p) * rc_int)
    c_p = fac * k_p / h_p
    c_m = fac * k_m / h_m
    c_c = 1.0 - c_p - c_m

    # ---- 顶部非线性边界参数 ----
    k_over_dx_top = float(k_face[-1] / h[-1])   # (k/dx) 顶部区间
    h_conv = float(h_conv_W_m2K)
    eps = float(emissivity)
    sigma = float(sigma_SB_W_m2K4)
    F = float(view_factor)
    T_air = float(T_air_C)
    Tsur = float(T_surroundings_C)

    # ---- FDM 主循环 ----
    T = np.ones(Nx) * T_init

    save_interval = max(1, int(save_dt / dt))
    plot_times = []
    plot_T_bottom = []
    plot_T_sample = []
    plot_T_outer = []
    plot_T_top_surf = []
    plot_residual = []
    newton_total_iter = 0
    newton_max_per_step = 0

    for n in range(Nt):
        if n % save_interval == 0:
            plot_times.append(time_fdm[n])
            plot_T_bottom.append(bottom_fdm[n])
            plot_T_sample.append(float(np.dot(mesh.sample_weights, T)))
            plot_T_outer.append(float(T[-1]))
            if has_top_surface_obs:
                plot_T_top_surf.append(float(T[idx_top_surface[0]]))
            # 诊断: 表面能量平衡残差 (对当前已求解的表面状态)
            plot_residual.append(boundary_residual_W_m2(
                T[-2], T[-1], T_air, Tsur, k_over_dx_top,
                h_conv, eps, sigma, F))

        T[0] = bottom_fdm[n]                       # 底部 Dirichlet BC
        T[1:-1] = (c_c * T[1:-1] + c_m * T[:-2] + c_p * T[2:])

        # 顶部非线性 BC: 用上一时刻表面温度作为 Newton 初猜
        guess = float(T[-1]) if n > 0 else None
        T[-1], n_iter = _solve_surface_newton(
            T[-2], T_air, Tsur, k_over_dx_top, h_conv, eps, sigma, F,
            guess, newton_abs_tolerance_C, newton_max_iterations)
        newton_total_iter += n_iter
        if n_iter > newton_max_per_step:
            newton_max_per_step = n_iter

    return {
        "time_fdm": time_fdm,
        "bottom_temperature_fdm": bottom_fdm,
        "t_array": np.array(plot_times),
        "T_bottom_arr": np.array(plot_T_bottom),
        "T_sample_arr": np.array(plot_T_sample),
        "T_outer_surface_arr": np.array(plot_T_outer),
        "T_top_surface_arr": np.array(plot_T_top_surf),
        "T_final": T,
        "dt": dt,
        "Nt": Nt,
        "Nx": Nx,
        "save_interval": save_interval,
        "mesh": mesh,
        "boundary_residual_arr": np.array(plot_residual),
        "max_abs_boundary_residual_W_m2": float(
            np.max(np.abs(np.array(plot_residual)))),
        "newton_total_iterations": int(newton_total_iter),
        "newton_max_iterations_per_step": int(newton_max_per_step),
    }


def _solve_surface_newton(T_prev_C, T_air_C, T_surroundings_C, k_over_dx,
                          h_conv, eps, sigma, F, initial_guess, abs_tol,
                          max_iter):
    """Newton 迭代循环 (返回 (Ts, n_iter))。"""
    if initial_guess is None:
        Ts = _convection_only_surface_solution(T_prev_C, T_air_C, k_over_dx,
                                               h_conv)
    else:
        Ts = initial_guess
    Tsur_K = T_surroundings_C + KELVIN_OFFSET
    coeff = eps * sigma * F
    n_iter = 0
    for _ in range(max_iter):
        Ts_K = Ts + KELVIN_OFFSET
        f = (k_over_dx * (T_prev_C - Ts)
             - h_conv * (Ts - T_air_C)
             - coeff * (Ts_K ** 4 - Tsur_K ** 4))
        df = -(k_over_dx + h_conv + 4.0 * coeff * Ts_K ** 3)
        if df >= 0.0:
            raise RuntimeError(
                f"顶部边界 Newton 导数非负 (df={df:.6e}); 物理范围外。")
        step = f / df
        Ts_new = Ts - step
        n_iter += 1
        if abs(step) <= abs_tol:
            return float(Ts_new), n_iter
        Ts = Ts_new
    raise RuntimeError(
        f"顶部非线性边界 Newton 迭代 {max_iter} 次未收敛 "
        f"(T_prev={T_prev_C:.6f} C, T_air={T_air_C:.6f} C); "
        "不静默接受坏根。")


# ============================================================
# 高层 runner: 对流+辐射+滞后
# ============================================================

def run_convection_radiation_lag_model(
    time_s,
    bottom_temperature_C,
    T_environment_C,
    k_eff_W_mK,
    cp_eff_J_kgK,
    tau_lag_s,
    rho_COC_kg_m3=RHO_COC_STRATEGY_E,
    h_conv_W_m2K=H_CONV_STRATEGY_E_W_M2K,
    emissivity=EMISSIVITY_STRATEGY_E,
    sigma_SB_W_m2K4=SIGMA_SB_W_M2_K4,
    view_factor=VIEW_FACTOR_STRATEGY_E,
    save_dt=0.1,
    T_initial_C=None,
    newton_abs_tolerance_C=1e-10,
    newton_max_iterations=20,
):
    """策略 E 高层包装: 对流 + 非线性辐射芯片 FDM + 输出侧一阶滞后。

    流程:
        1. 验证输入; T_environment_C 必须显式提供;
        2. 赋值 T_air_C = T_surroundings_C = T_environment_C (仿真内恒定);
        3. 构造仅替换 COC 的材料库 (k/cp/rho=1020);
        4. 初始 FDM 温度场 = 第一个底部/内部温度 (若未显式给出);
        5. 运行对流+辐射 FDM (BARE_TOP_COC_LAYERS);
        6. 对原始顶部 FDM 施加一阶滞后 (tau=0 严格恒等);
        7. 返回分离的样品 / 原始顶部 / 滞后观察预测迹线。

    滞后只作用于 T_top_observed_predicted_C, 绝不修改 T_sample_FDM_C
    或 T_top_FDM_C。

    返回 dict:
        t_array
        T_sample_FDM_C            : 样品 FDM (内部估计, 非实测, 非滞后);
        T_top_FDM_C               : 原始 Top COC FDM 表面温度;
        T_top_observed_predicted_C: 一阶滞后后的顶部观察预测;
        T_sample_predicted_C      : = T_sample_FDM (明确别名);
        T_air_C / T_surroundings_C: 仿真内使用的恒定环境温度;
        parameters                : ConvectionRadiationParameters 实例;
        result                    : 完整 FDM 结果 dict (诊断用)。
    """
    if T_environment_C is None:
        raise ValueError(
            "T_environment_C 必须显式提供 (无静默回退到 25 C / 内部温度); "
            "运行包含实测 Top COC 数据的实验时, 使用 "
            "infer_environment_from_initial_top_measurement(...) 解析。")
    _validate_environment(T_environment_C, T_environment_C)

    parameters = ConvectionRadiationParameters(
        k_eff_W_mK=k_eff_W_mK,
        cp_eff_J_kgK=cp_eff_J_kgK,
        tau_lag_s=tau_lag_s,
        rho_COC_kg_m3=rho_COC_kg_m3,
        h_conv_W_m2K=h_conv_W_m2K,
        emissivity=emissivity,
        sigma_SB_W_m2K4=sigma_SB_W_m2K4,
        view_factor=view_factor,
    )

    mats = make_convection_radiation_materials(
        k_eff_W_mK, cp_eff_J_kgK, rho_COC_kg_m3)
    layers = heat_model.BARE_TOP_COC_LAYERS

    if T_initial_C is None:
        T_initial_C = float(np.asarray(bottom_temperature_C, dtype=float)[0])

    result = run_convection_radiation_fdm(
        time_s=time_s,
        bottom_temperature_C=bottom_temperature_C,
        materials=mats,
        layers=layers,
        T_air_C=T_environment_C,
        T_surroundings_C=T_environment_C,
        h_conv_W_m2K=h_conv_W_m2K,
        emissivity=emissivity,
        sigma_SB_W_m2K4=sigma_SB_W_m2K4,
        view_factor=view_factor,
        save_dt=save_dt,
        T_initial_C=T_initial_C,
        newton_abs_tolerance_C=newton_abs_tolerance_C,
        newton_max_iterations=newton_max_iterations,
    )

    t_arr = result["t_array"]
    T_sample_FDM = result["T_sample_arr"]
    T_top_FDM = result["T_top_surface_arr"]
    if T_top_FDM.size == 0:
        raise RuntimeError(
            "层叠结构中没有 role='top_surface' 层, 无法提取顶部表面温度。")

    # 输出侧一阶滞后: 仅作用于原始顶部 FDM
    T_top_obs = apply_first_order_lag(
        t_arr, T_top_FDM, tau_lag_s,
        initial_output_C=None,  # y(0) = x(0), 绝不用实测顶部温度
    )

    return {
        "t_array": t_arr,
        "T_sample_FDM_C": T_sample_FDM,
        "T_top_FDM_C": T_top_FDM,
        "T_top_observed_predicted_C": T_top_obs,
        "T_sample_predicted_C": T_sample_FDM,  # 别名: 样品不滞后
        "T_air_C": float(T_environment_C),
        "T_surroundings_C": float(T_environment_C),
        "parameters": parameters,
        "result": result,
    }
