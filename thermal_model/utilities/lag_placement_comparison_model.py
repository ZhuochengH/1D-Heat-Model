#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — 滞后放置架构比较模块 (lag-placement comparison model)
================================================================

问题:
    冻结候选 (k=0.055, cp=1200, tau=8.5) 的滞后只放在**输出侧**
    (tau 只作用于 T_top_observed, 不作用于 T_sample)。但一阶滞后物理上
    可能位于:
        - 输出侧 (O): 温度计/观测滞后, FDM 不受影响;
        - 输入侧 (I): 传感器/控制器滞后, 底部 Dirichlet 边界被滤波;
        - 共享单 tau (S): 同一 tau 同时用于输入侧与输出侧。

本模块在**固定 k/cp** (冻结候选) 下扫描单一 tau (0.0-12.0 s, 步长 0.5,
25 值), 为三个架构各选出最优 tau (最小 72C RMSE, 不基于样品)。

架构定义 (严格, 规格 #20-47):
    O (output-side) : 一次 FDM (无滞后边界) + 25 个 tau 的输出侧一阶滞后
                      (仅 T_top_observed, T_sample 不滞后);
    I (input-side)  : 底部 Dirichlet 边界 = 输入侧一阶滞后后的内部温度
                      (初始态 = internal[0]), 无输出滤波器;
                      tau>0 每值一次 FDM (24 次);
    S (shared)      : 同一 tau 同时作用于输入侧底部边界与输出侧顶部;
                      每 tau 一次 FDM (24 次); tau=0 对所有架构严格相同
                      (一阶滞后 tau=0 恒等)。

约束 (规格):
    - 不重拟合 k/cp (固定冻结候选);
    - 不引入独立的 tau_input / tau_output (S 用单一 tau);
    - 不修改对流/辐射物理 (复用 run_convection_radiation_fdm);
    - 不覆盖任何旧输出;
    - 72C 目标: 查询轴 = 实测时间, RMSE/MAE/mean/median/max abs;
    - 每个架构用最小 RMSE 选 tau (绝不基于 PCR 样品预测);
    - 若最优 tau 落在网格上界 (12.0 s) 记录 TAU_BOUNDARY_WARNING。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE


# ============================================================
# 固定网格与物理 (Phase B 唯一事实来源)
# ============================================================

TAU_GRID_S = tuple(float(0.5 * k) for k in range(0, 25))  # 0.0 .. 12.0, 25 值
TAU_MAX_S = 12.0
SAVE_DT = 0.1

# 固定 k/cp (冻结候选, 不重拟合)
FROZEN_K_W_MK = FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK      # 0.055
FROZEN_CP_J_KGK = FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK  # 1200.0
FROZEN_TAU_S = FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s        # 8.5 (仅参考)


# ============================================================
# 指标 (查询轴 = 实测时间)
# ============================================================

def top_metrics_at_measurement_times(pred_on_meas_time, t_top_meas):
    """72C 目标: 残差 = pred - meas (查询轴 = 实测时间)。"""
    r = np.asarray(pred_on_meas_time, dtype=float) - np.asarray(
        t_top_meas, dtype=float)
    return {
        "RMSE_C": float(np.sqrt(np.mean(r ** 2))),
        "MAE_C": float(np.mean(np.abs(r))),
        "mean_residual_C": float(np.mean(r)),
        "median_abs_residual_C": float(np.median(np.abs(r))),
        "max_abs_residual_C": float(np.max(np.abs(r))),
    }


# ============================================================
# 三个架构的评估
# ============================================================

def run_arch_output_side(t_proto, t_int, t_env, tau,
                         k=FROZEN_K_W_MK, cp=FROZEN_CP_J_KGK,
                         save_dt=SAVE_DT):
    """O: 一次 FDM + 输出侧滞后。tau 任意 (0 或 >0)。

    返回 dict: t_array / T_sample_fdm / T_top_fdm / T_top_obs (滞后后)。
    """
    mats = cr.make_convection_radiation_materials(k, cp)
    result = cr.run_convection_radiation_fdm(
        time_s=t_proto, bottom_temperature_C=t_int, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=save_dt,
        T_initial_C=float(np.asarray(t_int, dtype=float)[0]))
    t_arr = result["t_array"]
    T_top_fdm = result["T_top_surface_arr"]
    T_sample_fdm = result["T_sample_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top_fdm, tau)
    return {"t_array": t_arr, "T_sample_fdm": T_sample_fdm,
            "T_top_fdm": T_top_fdm, "T_top_obs": T_top_obs}


def run_arch_input_side(t_proto, t_int, t_env, tau,
                        k=FROZEN_K_W_MK, cp=FROZEN_CP_J_KGK,
                        save_dt=SAVE_DT):
    """I: 输入侧滞后底部边界 (初始态 = internal[0]), 无输出滤波器。

    tau>0 每值一次 FDM; tau=0 时与 O(tau=0) 严格相同 (恒等)。
    """
    if tau == 0.0:
        base = run_arch_output_side(t_proto, t_int, t_env, 0.0,
                                    k=k, cp=cp, save_dt=save_dt)
        return dict(base, T_top_obs=base["T_top_fdm"].copy())
    t_int_lag = apply_first_order_lag(
        t_proto, np.asarray(t_int, dtype=float), tau,
        initial_output_C=float(np.asarray(t_int, dtype=float)[0]))
    mats = cr.make_convection_radiation_materials(k, cp)
    result = cr.run_convection_radiation_fdm(
        time_s=t_proto, bottom_temperature_C=t_int_lag, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=save_dt,
        T_initial_C=float(np.asarray(t_int, dtype=float)[0]))
    t_arr = result["t_array"]
    return {"t_array": t_arr,
            "T_sample_fdm": result["T_sample_arr"],
            "T_top_fdm": result["T_top_surface_arr"],
            "T_top_obs": result["T_top_surface_arr"].copy()}


def run_arch_shared(t_proto, t_int, t_env, tau,
                    k=FROZEN_K_W_MK, cp=FROZEN_CP_J_KGK,
                    save_dt=SAVE_DT):
    """S: 同一 tau 输入侧 + 输出侧。

    tau>0 每值一次 FDM; tau=0 时与 O(tau=0) 严格相同 (恒等)。
    """
    if tau == 0.0:
        base = run_arch_output_side(t_proto, t_int, t_env, 0.0,
                                    k=k, cp=cp, save_dt=save_dt)
        return dict(base, T_top_obs=base["T_top_fdm"].copy())
    t_int_lag = apply_first_order_lag(
        t_proto, np.asarray(t_int, dtype=float), tau,
        initial_output_C=float(np.asarray(t_int, dtype=float)[0]))
    mats = cr.make_convection_radiation_materials(k, cp)
    result = cr.run_convection_radiation_fdm(
        time_s=t_proto, bottom_temperature_C=t_int_lag, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=save_dt,
        T_initial_C=float(np.asarray(t_int, dtype=float)[0]))
    t_arr = result["t_array"]
    T_top_fdm = result["T_top_surface_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top_fdm, tau)
    return {"t_array": t_arr,
            "T_sample_fdm": result["T_sample_arr"],
            "T_top_fdm": T_top_fdm,
            "T_top_obs": T_top_obs}


# ============================================================
# 架构注册表
# ============================================================

ARCHITECTURES = {
    "O": {
        "label": "output-side (tau on T_top_observed only)",
        "n_fdm_for_tau_gt_0": 0,   # 复用基线 FDM
        "runner": run_arch_output_side,
    },
    "I": {
        "label": "input-side (tau filters bottom Dirichlet)",
        "n_fdm_for_tau_gt_0": 1,
        "runner": run_arch_input_side,
    },
    "S": {
        "label": "shared single tau (input + output)",
        "n_fdm_for_tau_gt_0": 1,
        "runner": run_arch_shared,
    },
}


def evaluate_architecture(arch, tau, t_proto, t_int, t_top_meas, t_env,
                          save_dt=SAVE_DT):
    """单个架构在单个 tau 下的 72C 指标 (查询轴 = 实测时间)。"""
    runner = ARCHITECTURES[arch]["runner"]
    out = runner(t_proto, t_int, t_env, tau, save_dt=save_dt)
    t_arr = out["t_array"]
    T_top_obs = out["T_top_obs"]
    pred = np.interp(t_proto, t_arr, T_top_obs)
    m = top_metrics_at_measurement_times(pred, t_top_meas)
    return {
        "architecture": arch,
        "tau_lag_s": float(tau),
        "RMSE_72C_C": m["RMSE_C"],
        "MAE_72C_C": m["MAE_C"],
        "mean_residual_C": m["mean_residual_C"],
        "median_abs_residual_C": m["median_abs_residual_C"],
        "max_abs_residual_C": m["max_abs_residual_C"],
    }


def scan_architecture(arch, t_proto, t_int, t_top_meas, t_env,
                      taus=TAU_GRID_S, save_dt=SAVE_DT):
    """扫描一个架构的全部 tau (25 值), 返回行列表。"""
    rows = []
    for tau in taus:
        rows.append(evaluate_architecture(
            arch, tau, t_proto, t_int, t_top_meas, t_env, save_dt=save_dt))
    return rows


def select_best_tau(rows, arch, tau_max=TAU_MAX_S):
    """每个架构用最小 RMSE 选最佳 tau (绝不基于 PCR 样品预测)。"""
    best = min(rows, key=lambda r: r["RMSE_72C_C"])
    return {
        "architecture": arch,
        "best_tau_s": float(best["tau_lag_s"]),
        "best_RMSE_C": float(best["RMSE_72C_C"]),
        "best_MAE_C": float(best["MAE_72C_C"]),
        "best_mean_residual_C": float(best["mean_residual_C"]),
        "best_median_abs_residual_C": float(best["median_abs_residual_C"]),
        "best_max_abs_residual_C": float(best["max_abs_residual_C"]),
        "TAU_BOUNDARY_WARNING": bool(best["tau_lag_s"] >= tau_max - 1e-12),
    }


def run_72c_comparison(t_proto, t_int, t_top_meas, t_env,
                       taus=TAU_GRID_S, save_dt=SAVE_DT):
    """扫描 O/I/S 三个架构, 返回 (rows, summary)。

    rows: 3 架构 x 25 tau = 75 行 CSV 行。
    summary: 每架构最佳 tau 列表。
    """
    rows = []
    for arch in ("O", "I", "S"):
        rows.extend(scan_architecture(arch, t_proto, t_int, t_top_meas,
                                      t_env, taus=taus, save_dt=save_dt))
    summary = [select_best_tau([r for r in rows if r["architecture"] == a], a)
               for a in ("O", "I", "S")]
    return rows, summary
