#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy C — 滞后分离 3-DOF 热模型 (lag-separated thermal model)
=================================================================

这是与现有 2-DOF 标定模型**并行**的新模型模块, 不复用/不修改名义标定。

模型结构:
    实测内部温度 T_internal(t)
        -> chip FDM(alpha, k)   [heat_model.run_simulation, BARE_TOP_COC_LAYERS]
        -> T_sample_FDM(t)  (控制体积加权空间平均, 不经过滞后)
        -> T_top_FDM(t)     (顶部表面 FDM 状态)
        -> 一阶外部滞后 tau_ext
        -> T_top_obs(t)     (用于与温度计实测比较的观察预测)

关键约定:
    - cp_eff = k_eff / (rho * alpha_eff), rho = 1020 (派生, 不自由);
    - tau_ext 只作用于 T_top_FDM, 绝不作用于 T_sample_FDM;
    - 滞后用"分段线性输入的精确一阶更新"(非显式 Euler), tau=0 时严格恒等;
    - 滞后初始态 y(0) = x(0) (绝不用实测顶部温度初始化)。

tau_ext 解释: "有效外部/未解析系统热滞后" (降阶唯象参数, 不是温度计时间常数)。
单阶滞后是有意的降阶近似: 两个独立一阶滞后的乘积是二阶, 不能被单个
一阶 tau_ext 精确表示。
"""
from dataclasses import dataclass, field

import numpy as np

from thermal_model.core import heat_model


RHO_COC = 1020.0


# ============================================================
# 参数容器
# ============================================================

@dataclass(frozen=True)
class LagAugmentedParameters:
    """策略 C 参数 (不可变)。

    alpha_eff / k_eff / tau_ext 自由; rho 固定; cp_eff 派生。
    """
    alpha_eff_m2_s: float
    k_eff_W_mK: float
    tau_ext_s: float
    rho_COC_kg_m3: float = RHO_COC

    def __post_init__(self):
        if not (self.alpha_eff_m2_s > 0 and np.isfinite(self.alpha_eff_m2_s)):
            raise ValueError(f"alpha_eff 必须 > 0 且有限, 收到 "
                             f"{self.alpha_eff_m2_s}")
        if not (self.k_eff_W_mK > 0 and np.isfinite(self.k_eff_W_mK)):
            raise ValueError(f"k_eff 必须 > 0 且有限, 收到 {self.k_eff_W_mK}")
        if not (self.rho_COC_kg_m3 > 0 and np.isfinite(self.rho_COC_kg_m3)):
            raise ValueError(f"rho 必须 > 0 且有限, 收到 {self.rho_COC_kg_m3}")
        if not (self.tau_ext_s >= 0 and np.isfinite(self.tau_ext_s)):
            raise ValueError(f"tau_ext 必须 >= 0 且有限, 收到 {self.tau_ext_s}")
        cp = self.cp_eff_J_kgK
        if not (cp > 0 and np.isfinite(cp)):
            raise ValueError(f"派生 cp 必须 > 0 且有限, 收到 {cp}")

    @property
    def cp_eff_J_kgK(self) -> float:
        """cp_eff = k_eff / (rho * alpha_eff)。"""
        return self.k_eff_W_mK / (self.rho_COC_kg_m3 * self.alpha_eff_m2_s)


# ============================================================
# 精确一阶滞后 (分段线性输入的解析更新)
# ============================================================

def apply_first_order_lag(time_s, input_temperature_C, tau_s,
                          initial_output_C=None):
    """一阶滞后 tau*dy/dt + y = x(t), 用分段线性输入的精确更新。

    时间 t_i -> t_{i+1}: dt = t_{i+1} - t_i, x0=x(t_i), x1=x(t_{i+1}),
    斜率 m = (x1-x0)/dt。tau>0 的精确解 (线性输入):
        y1 = x1 - m*tau + (y0 - x0 + m*tau) * exp(-dt/tau)

    特性:
        - 支持非均匀 dt; 要求时间严格递增;
        - 输入必须有限;
        - y(0) = x(0) 默认 (若 initial_output_C 为 None);
        - tau=0 严格恒等 y == x (不经过任何数值近似)。

    返回 np.ndarray (与输入等长)。
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(input_temperature_C, dtype=float)
    tau = float(tau_s)

    if t.ndim != 1 or x.ndim != 1 or t.size != x.size:
        raise ValueError("time_s 与 input_temperature_C 必须是一维且等长。")
    if t.size < 1:
        raise ValueError("输入数组不能为空。")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(x)):
        raise ValueError("时间/输入必须有限。")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s 必须严格递增。")
    if tau < 0 or not np.isfinite(tau):
        raise ValueError(f"tau 必须 >= 0 且有限, 收到 {tau}")

    if tau == 0.0:
        return x.copy()

    y = np.empty_like(x)
    y0 = float(x[0]) if initial_output_C is None else float(initial_output_C)
    if not np.isfinite(y0):
        raise ValueError("初始滞后输出必须有限。")
    y[0] = y0

    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        x0 = x[i - 1]
        x1 = x[i]
        m = (x1 - x0) / dt
        y[i] = x1 - m * tau + (y[i - 1] - x0 + m * tau) * np.exp(-dt / tau)

    return y


# ============================================================
# 材料构造 (仅替换 COC; 不修改 DEFAULT_MATERIALS)
# ============================================================

def make_lag_materials(k_eff_W_mK, cp_eff_J_kgK):
    """构造候选材料库: 仅替换 COC 的 k/cp, rho=1020, 其余逐位不变。"""
    mats = heat_model.copy_default_materials()
    coc = mats["COC"]
    mats["COC"] = heat_model.Material(
        name=coc.name,
        k_W_mK=float(k_eff_W_mK),
        rho_kg_m3=coc.rho_kg_m3,
        cp_J_kgK=float(cp_eff_J_kgK),
    )
    return mats


# ============================================================
# 模型包装
# ============================================================

def run_lag_augmented_model(time_s, T_internal_C, parameters,
                            h_conv=5.0, T_air_ambient=25.0, save_dt=0.1):
    """运行策略 C 模型: 芯片 FDM + 顶部一阶外部滞后。

    返回 dict:
        t_array                    : FDM 下采样时间轴;
        T_sample_FDM_C             : 样品空间平均 (FDM 原始, 不滞后);
        T_top_FDM_C                : 顶部表面 FDM 状态;
        T_top_observed_predicted_C : 滞后后的顶部观察预测;
        T_sample_predicted_C       : = T_sample_FDM (明确别名);
        result                     : 完整 FDM 结果 dict (诊断用)。
    """
    if not isinstance(parameters, LagAugmentedParameters):
        raise TypeError("parameters 必须是 LagAugmentedParameters")

    cp = parameters.cp_eff_J_kgK
    mats = make_lag_materials(parameters.k_eff_W_mK, cp)
    layers = heat_model.BARE_TOP_COC_LAYERS
    T_initial = float(np.asarray(T_internal_C, dtype=float)[0])

    result = heat_model.run_simulation(
        time_s=time_s,
        bottom_temperature_C=T_internal_C,
        materials=mats,
        layers=layers,
        h_conv=h_conv,
        T_air_ambient=T_air_ambient,
        save_dt=save_dt,
        T_initial_C=T_initial,
    )

    t_arr = result["t_array"]
    T_sample_FDM = result["T_sample_arr"]
    T_top_FDM = result["T_top_surface_arr"]

    # 顶部观察预测: 对 FDM 顶部状态施加一阶外部滞后
    T_top_obs = apply_first_order_lag(
        t_arr, T_top_FDM, parameters.tau_ext_s,
        initial_output_C=None,  # y(0) = x(0), 绝不用实测顶部温度
    )

    return {
        "t_array": t_arr,
        "T_sample_FDM_C": T_sample_FDM,
        "T_top_FDM_C": T_top_FDM,
        "T_top_observed_predicted_C": T_top_obs,
        "T_sample_predicted_C": T_sample_FDM,  # 别名: 样品不滞后
        "result": result,
    }


# ============================================================
# 72C 观察目标 (修正测量时间插值)
# ============================================================

def evaluate_72c_objective(time_s, t_internal, t_top_meas, parameters,
                           **kw):
    """对 72C 对齐数据运行策略 C, 比较滞后观察预测 vs 实测顶部。

    查询轴 = 实测时间 time_s (绝不用实测温度值作查询坐标)。
    返回 dict (含 RMSE/MAE/mean/max abs + 派生 cp)。
    """
    out = run_lag_augmented_model(time_s, t_internal, parameters, **kw)
    T_top_obs = np.interp(time_s, out["t_array"],
                          out["T_top_observed_predicted_C"])
    residual = T_top_obs - np.asarray(t_top_meas, dtype=float)
    r = np.asarray(residual, dtype=float)
    return {
        "alpha_eff_m2_s": parameters.alpha_eff_m2_s,
        "k_eff_W_mK": parameters.k_eff_W_mK,
        "derived_cp_eff_J_kgK": parameters.cp_eff_J_kgK,
        "tau_ext_s": parameters.tau_ext_s,
        "RMSE_72C_C": float(np.sqrt(np.mean(r ** 2))),
        "MAE_72C_C": float(np.mean(np.abs(r))),
        "mean_residual_C": float(np.mean(r)),
        "max_abs_residual_C": float(np.max(np.abs(r))),
    }


def evaluate_72c_objective_safe(time_s, t_internal, t_top_meas, parameters,
                                **kw):
    try:
        return evaluate_72c_objective(time_s, t_internal, t_top_meas,
                                      parameters, **kw)
    except Exception as exc:  # noqa: BLE001
        return {
            "alpha_eff_m2_s": parameters.alpha_eff_m2_s,
            "k_eff_W_mK": parameters.k_eff_W_mK,
            "derived_cp_eff_J_kgK": parameters.cp_eff_J_kgK,
            "tau_ext_s": parameters.tau_ext_s,
            "RMSE_72C_C": np.nan, "MAE_72C_C": np.nan,
            "mean_residual_C": np.nan, "max_abs_residual_C": np.nan,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
