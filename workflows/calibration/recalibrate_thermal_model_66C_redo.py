#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
66C REDO — 重新标定裸顶热模型候选 + 绝缘 3s 样品前向预测
========================================================

科学目的
--------
当前冻结模型 (strategy_G_conservative_cross_protocol_v1, k=0.055/cp=1200/
tau=8.5) 此前在另一组 Top COC 数据上校准并外部验证。实验者现提供了更高质量
的同步热数据集:

    修正后实测 Top COC (extension 66°C_redo.xls)
    +
    实测内部温度 (08.17 COC top_66°C_zone1_temperature_analysis.xlsx)

本脚本用该数据集创建一个新的重新标定候选:

    66C_RECALIBRATED_CANDIDATE_V1

随后:

    冻结新参数 -> 独立 3s-extension 裸顶验证 (不重拟合)
             -> 绝缘几何 3s 样品温度前向预测 (不重拟合)
             -> 重复周期样品峰 >=86 C 描述性统计 (事后检查)

反循环规则 (任务 #34, 架构上强制):
    PHASE A: 仅用 66C 实测 Top + 66C 内部标定 k/cp/tau。结束即存参数。
    PHASE C: 锁定参数, 在独立 3s Top COC 上验证。无参数更新。
    PHASE D: 锁定参数, 绝缘样品预测。无参数更新。
    PHASE E: 锁定参数, >=86 C 统计。无参数更新。
    不存在 样品峰 -> 参数搜索 的反馈路径 (assert_parameters_locked 强制)。

qPCR 信息使用规则:
    - ~86-87 C 功能性参考为独立生物化学证据;
    - 绝不允许进入标定目标 (目标 ONLY: 预测 Top COC vs 实测 Top COC);
    - 仅作为事后描述性参考 (>=85/86/87/90/92/95 C 计数)。

固定物理 (任务 #6, 不拟合):
    h_conv = 10.0 W/(m2 K); epsilon = 0.90; sigma = 5.670374419e-8;
    F_view = 1.0; 非线性 Stefan-Boltzmann 辐射; rho_COC = 1020 kg/m3。

标定参数 (任务 #7, 仅拟合):
    k_eff_COC, cp_eff_COC, tau_top (tau 只作用于 Top 观测, 绝不作用于样品)。

几何 (任务 #5 / #26):
    标定  : BARE_TOP_COC_LAYERS (850 um, 无 Air/PDMS, 外热损在 Top COC 外表面)
    绝缘  : LEGACY_INSULATED_LAYERS (4050 um, 含 3 mm 密封 Air + 200 um PDMS;
            外热损只在 PDMS 外表面; 密封空气仅纯导热)

同步 (任务 #10):
    两个记录各自首点 = 相对 t=0; applied_time_shift_s = 0.0;
    无交叉相关 / 无峰值对齐 / 无手动偏移 / 无时移搜索。

输出 (不覆盖任何历史目录):
    calibrated_model_output/66C_recalibrated_candidate_v1/
        calibration_66C/
        validation_3s_extension/
        insulated_3s_sample_prediction/
        comparison/

用法:
    uv run python recalibrate_thermal_model_66C_redo.py [--n-workers 8]
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE
from thermal_model.utilities.predict_sample_from_internal_temperature import load_internal_data
from thermal_model.utilities.validate_frozen_model_two_new_bare_top_datasets import (
    load_top_series,
    load_internal_series,
    _regime_labels,
)
from thermal_model.utilities.analyze_frozen_sample_peak import detect_repeated_cycles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"
REC_START_DIR = CALIBRATION_DIR / "Recording at the start"

# ------------------------------------------------------------
# 数据源
# ------------------------------------------------------------
DS1_TOP_66C = CALIBRATION_DIR / "extension 66°C_redo.xls"
DS1_INT_66C = REC_START_DIR / (
    "08.17 COC top_66°C_zone1_temperature_analysis.xlsx")
DS2_TOP_3S = CALIBRATION_DIR / "PCR 3s extension.xls"
DS2_INT_3S = REC_START_DIR / (
    "08.17 COC top_pm_3s extension_zone1_temperature_analysis.xlsx")

OUTPUT_ROOT = (PROJECT_ROOT / "calibrated_model_output"
               / "66C_recalibrated_candidate_v1")
DIR_CALIB = OUTPUT_ROOT / "calibration_66C"
DIR_VALID = OUTPUT_ROOT / "validation_3s_extension"
DIR_INSUL = OUTPUT_ROOT / "insulated_3s_sample_prediction"
DIR_COMP = OUTPUT_ROOT / "comparison"

SAVE_DT = 0.1

# ------------------------------------------------------------
# 固定物理 (不拟合)
# ------------------------------------------------------------
RHO_COC = cr.RHO_COC_STRATEGY_E          # 1020 kg/m3
H_CONV = cr.H_CONV_STRATEGY_E_W_M2K      # 10.0
EPS = cr.EMISSIVITY_STRATEGY_E           # 0.90
SIGMA = cr.SIGMA_SB_W_M2_K4              # 5.670374419e-8
F_VIEW = cr.VIEW_FACTOR_STRATEGY_E       # 1.0

# ------------------------------------------------------------
# 旧冻结模型 (只读参考; 绝不修改 frozen_strategy_G_candidate.py)
# ------------------------------------------------------------
OLD_K = FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK      # 0.055
OLD_CP = FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK   # 1200
OLD_TAU = FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s     # 8.5
OLD_66C_RMSE_HISTORICAL = 3.0134      # 历史外部验证值 (只读参考)
OLD_3S_RMSE_HISTORICAL = 2.3941       # 历史外部验证值 (只读参考)
OLD_72C_RMSE_HISTORICAL = 0.8891597125869538

# ------------------------------------------------------------
# 粗网格 (任务 #13)
# ------------------------------------------------------------
K_COARSE = np.array([0.030, 0.040, 0.050, 0.060, 0.070, 0.080,
                     0.090, 0.100, 0.120, 0.150])
CP_COARSE = np.array([700.0, 900.0, 1100.0, 1300.0, 1500.0, 1800.0, 2200.0])
TAU_COARSE = np.arange(0.0, 13.0, 1.0)   # 0..12 s

# 扩展方向 (物理合理方向)
K_EXTEND_UP = np.array([0.18, 0.22, 0.28])
K_EXTEND_DOWN = np.array([0.020, 0.025])
CP_EXTEND_UP = np.array([2600.0, 3000.0, 3500.0])
CP_EXTEND_DOWN = np.array([500.0, 600.0])
TAU_EXTEND_UP = np.array([13.0, 14.0, 16.0, 18.0, 20.0])

# 局部细化 (任务 #15)
K_REFINE_STEP = 0.0025
K_REFINE_HALF = 4
CP_REFINE_STEP = 100.0
CP_REFINE_HALF = 3
TAU_REFINE_STEP = 0.5
TAU_REFINE_HALF = 4

# 物理下限 (cp 不能低于 500; k 不能低于 0.01; tau >= 0)
K_FLOOR = 0.01
CP_FLOOR = 500.0

# ------------------------------------------------------------
# 诊断 / 分类阈值 (描述性, 显式记录)
# ------------------------------------------------------------
NEAR_OPTIMAL_RMSE_TOL = 0.10            # 近最优区域: RMSE <= best + 0.10 C
CALIB_IMPROVE_ABS_C = 0.50              # 标定"实质改善"绝对阈值
CALIB_IMPROVE_FRAC = 0.15               # 标定"实质改善"相对阈值
VAL_ACCEPTABLE_RMSE_C = 4.0             # 独立验证"可接受" (项目 MODERATE 上限)
VAL_DEGRADE_ABS_C = 0.50                # 独立验证"实质退化"绝对阈值
VAL_DEGRADE_FRAC = 0.30                 # 独立验证"实质退化"相对阈值

# 描述性 PCR 热参考阈值 (只作事后统计; 绝不进入标定)
DESCRIPTIVE_THRESHOLDS = (85.0, 86.0, 87.0, 90.0, 92.0, 95.0)
QPCR_FUNCTIONAL_REFERENCE_C = 86.0      # 独立 qPCR 功能性参考 (仅事后)
MAJORITY_FRACTION = 0.50                # "majority >=86 C" = >50%

CANDIDATE_ID = "66C_RECALIBRATED_CANDIDATE_V1"
SYNC_RULE = "SIMULTANEOUS_START_RELATIVE_T0"


# ============================================================
# 数据加载 (复用既有权威加载器)
# ============================================================

def load_66c_dataset():
    """加载 66C redo 标定数据集 (修正后 Top + 内部)。"""
    top = load_top_series(DS1_TOP_66C)
    internal = load_internal_series(DS1_INT_66C)
    return top, internal


def load_3s_dataset():
    """加载 3s-extension 独立验证数据集 (修正后 Top + 内部)。"""
    top = load_top_series(DS2_TOP_3S)
    internal = load_internal_series(DS2_INT_3S)
    return top, internal


# ============================================================
# 单候选评估 (单一权威路径; 查询轴 = 实测 Top 时间)
# ============================================================

def evaluate_top_for_tau(t_arr_c, T_fdm_c, t_top_c, T_top_c, tau_top):
    """对给定 raw Top FDM 迹线应用 tau 并计算目标指标。

    residual_i = T_top_predicted(t_measured_i) - T_top_measured(t_measured_i)
    RMSE = sqrt(mean(residual_i^2))  (任务 #9)

    查询轴 = 实测 Top COC 时间坐标 t_top_c; 温度值绝不作插值坐标。
    """
    if tau_top < 0:
        raise ValueError(f"tau 必须 >= 0, 收到 {tau_top!r}")
    T_obs = apply_first_order_lag(t_arr_c, T_fdm_c, tau_top)
    T_pred = np.interp(t_top_c, t_arr_c, T_obs)
    resid = T_pred - T_top_c
    n = len(resid)
    if n == 0:
        raise ValueError("评估窗口内无数据点。")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    mean_r = float(np.mean(resid))
    med_abs = float(np.median(np.abs(resid)))
    std_r = float(np.std(resid))
    max_abs = float(np.max(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((T_top_c - np.mean(T_top_c)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {
        "RMSE_C": rmse, "MAE_C": mae, "mean_residual_C": mean_r,
        "median_abs_residual_C": med_abs, "residual_std_C": std_r,
        "max_abs_residual_C": max_abs, "R_squared": r2,
        "n_points": int(n),
        "T_pred": T_pred, "residual": resid,
    }


def run_bare_fdm(k_eff, cp_eff, internal, layers=None, save_dt=SAVE_DT):
    """一次裸顶 FDM (底部 Dirichlet = 实测内部; 顶部非线性对流+辐射)。

    环境 = 第一个有效实测 Top; 初始 = 第一个内部 (调用方传入)。
    返回 (t_arr, T_top_fdm, T_sample, T_env, T_init)。
    """
    t_int = internal["t_rel"]
    T_int = internal["T"]
    T_env = float(internal.get("_env_C", np.nan))
    if not np.isfinite(T_env):
        raise ValueError("run_bare_fdm 需要 internal['_env_C'] 环境温度。")
    T_init = float(internal["T"][0])
    if layers is None:
        layers = heat_model.BARE_TOP_COC_LAYERS
    mats = cr.make_convection_radiation_materials(k_eff, cp_eff, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=layers, T_air_C=T_env, T_surroundings_C=T_env,
        save_dt=save_dt, T_initial_C=T_init)
    t_arr = result["t_array"]
    T_top = result["T_top_surface_arr"]
    T_sample = result["T_sample_arr"]
    if T_top.size == 0:
        raise RuntimeError("裸顶 FDM 未返回顶部表面温度 (缺 role='top_surface')。")
    return t_arr, T_top, T_sample, T_env, T_init


def evaluate_top_prediction(top, internal, k_eff, cp_eff, tau_top,
                            save_dt=SAVE_DT):
    """完整评估: FDM + 输出侧滞后 + 插值到实测 Top 时间 + 指标。

    同时返回重叠窗口数组供绘图 / regime 诊断 / trace CSV。
    """
    t_top = top["t_rel"]
    T_top = top["T"]
    t_int = internal["t_rel"]
    T_int = internal["T"]
    t0 = max(t_top[0], t_int[0])
    t1 = min(t_top[-1], t_int[-1])
    if t1 <= t0:
        raise ValueError("Top 与内部时间范围无重叠。")

    internal_env = dict(internal)
    internal_env["_env_C"] = float(T_top[0])   # INITIAL_MEASURED_TOP
    t_arr, T_top_fdm, T_sample, T_env, T_init = run_bare_fdm(
        k_eff, cp_eff, internal_env, layers=heat_model.BARE_TOP_COC_LAYERS,
        save_dt=save_dt)

    m_arr = (t_arr >= t0 - 1e-9) & (t_arr <= t1 + 1e-9)
    m_top = (t_top >= t0 - 1e-9) & (t_top <= t1 + 1e-9)
    t_arr_c = t_arr[m_arr]
    T_fdm_c = T_top_fdm[m_arr]
    t_top_c = t_top[m_top]
    T_top_c = T_top[m_top]

    met = evaluate_top_for_tau(t_arr_c, T_fdm_c, t_top_c, T_top_c, tau_top)

    # 内部温度插值到 Top 时间 (regime 诊断用)
    T_int_at_top = np.interp(t_top_c, t_int, T_int)
    regimes = _regime_labels(t_top_c, T_int_at_top, T_top_c)

    return {
        "t_overlap_s": float(t1 - t0),
        "t_top": t_top_c, "T_top_measured": T_top_c,
        "T_internal_at_top": T_int_at_top,
        "T_top_fdm_raw": np.interp(t_top_c, t_arr_c, T_fdm_c),
        "T_top_predicted_lagged": met["T_pred"],
        "residual": met["residual"],
        "regimes": regimes,
        "T_env_C": T_env, "T_initial_C": T_init,
        "environment_source": "INITIAL_MEASURED_TOP",
        "synchronization_rule": SYNC_RULE,
        "time_shift_applied_s": 0.0,
        "metrics": {k: v for k, v in met.items()
                    if k not in ("T_pred", "residual")},
    }


# ============================================================
# 并行 FDM 批量 (每 (k, cp) 一次 FDM; tau 不重跑 FDM)
# ============================================================

def _fdm_worker(args):
    k, cp, t_int, T_int, T_env, save_dt = args
    mats = cr.make_convection_radiation_materials(k, cp, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=T_env,
        T_surroundings_C=T_env, save_dt=save_dt,
        T_initial_C=float(T_int[0]))
    return (k, cp, result["t_array"], result["T_top_surface_arr"])


def run_fdm_batch(top, internal, k_grid, cp_grid, n_workers=8,
                  save_dt=SAVE_DT, cache=None):
    """对每个 (k, cp) 运行一次 FDM; 返回/更新 cache {(k,cp): (t_arr, T_fdm)}。

    返回重叠窗口裁剪后的 cache 视图: {(k, cp): (t_arr_c, T_fdm_c)}。
    """
    if cache is None:
        cache = {}
    t_top = top["t_rel"]
    t_int = internal["t_rel"]
    t0 = max(t_top[0], t_int[0])
    t1 = min(t_top[-1], t_int[-1])
    T_env = float(top["T"][0])
    pending = [(float(k), float(cp)) for k in k_grid for cp in cp_grid
               if (float(k), float(cp)) not in cache]
    if pending:
        args_list = [(k, cp, t_int, internal["T"], T_env, save_dt)
                     for k, cp in pending]
        n_workers = max(1, min(n_workers, len(args_list)))
        if n_workers > 1 and len(args_list) > 1:
            with mp.Pool(n_workers) as pool:
                for i, (k, cp, t_arr, T_fdm) in enumerate(
                        pool.imap_unordered(_fdm_worker, args_list, chunksize=1)):
                    cache[(float(k), float(cp))] = (t_arr, T_fdm)
                    if i % 10 == 0 or i == len(args_list) - 1:
                        print(f"    FDM 完成 {i + 1}/{len(args_list)} "
                              f"(k={k}, cp={cp})", flush=True)
        else:
            for i, a in enumerate(args_list):
                k, cp, t_arr, T_fdm = _fdm_worker(a)
                cache[(float(k), float(cp))] = (t_arr, T_fdm)
                print(f"    FDM 完成 {i + 1}/{len(args_list)} "
                      f"(k={k}, cp={cp})", flush=True)

    clipped = {}
    for (k, cp), (t_arr, T_fdm) in cache.items():
        m_arr = (t_arr >= t0 - 1e-9) & (t_arr <= t1 + 1e-9)
        clipped[(float(k), float(cp))] = (t_arr[m_arr], T_fdm[m_arr])
    return cache, clipped


def search_stage(top, internal, k_grid, cp_grid, tau_grid, stage_name,
                 n_workers=8, cache=None, results=None):
    """对网格 (k x cp x tau) 搜索; FDM 每 (k, cp) 一次。

    results: 列表, 每元素为 dict (k, cp, tau, RMSE, MAE, mean, n, stage)。
    """
    if results is None:
        results = []
    t_top = top["t_rel"]
    T_top = top["T"]
    t_int = internal["t_rel"]
    t0 = max(t_top[0], t_int[0])
    t1 = min(t_top[-1], t_int[-1])
    m_top = (t_top >= t0 - 1e-9) & (t_top <= t1 + 1e-9)
    t_top_c = t_top[m_top]
    T_top_c = T_top[m_top]

    print(f"[search_stage] {stage_name}: {len(k_grid)}k x {len(cp_grid)}cp "
          f"x {len(tau_grid)}tau", flush=True)
    cache, clipped = run_fdm_batch(top, internal, k_grid, cp_grid,
                                   n_workers=n_workers, cache=cache)
    for (k, cp) in sorted(clipped):
        t_arr_c, T_fdm_c = clipped[(k, cp)]
        for tau in tau_grid:
            met = evaluate_top_for_tau(t_arr_c, T_fdm_c, t_top_c, T_top_c,
                                       float(tau))
            results.append({
                "k_eff_W_mK": float(k),
                "cp_eff_J_kgK": float(cp),
                "tau_top_s": float(tau),
                "RMSE_C": met["RMSE_C"],
                "MAE_C": met["MAE_C"],
                "mean_residual_C": met["mean_residual_C"],
                "n_points": met["n_points"],
                "stage": stage_name,
            })
    return cache, results


# ============================================================
# 边界检测 / 扩展 (任务 #14)
# ============================================================

def check_boundary(best, k_grid, cp_grid, tau_grid):
    """检测最优是否落在搜索边界。返回 (warnings, unresolved)。

    warnings: 字符串列表, 如 "K_MAX" / "CP_MIN" / "TAU_MAX"。
    unresolved: True 若边界为物理边缘且无扩展空间 (如 tau=0)。
    """
    warnings = []
    if abs(best["k_eff_W_mK"] - min(k_grid)) < 1e-12:
        warnings.append("K_MIN")
    if abs(best["k_eff_W_mK"] - max(k_grid)) < 1e-12:
        warnings.append("K_MAX")
    if abs(best["cp_eff_J_kgK"] - min(cp_grid)) < 1e-12:
        warnings.append("CP_MIN")
    if abs(best["cp_eff_J_kgK"] - max(cp_grid)) < 1e-12:
        warnings.append("CP_MAX")
    if abs(best["tau_top_s"] - min(tau_grid)) < 1e-12:
        warnings.append("TAU_MIN")
    if abs(best["tau_top_s"] - max(tau_grid)) < 1e-12:
        warnings.append("TAU_MAX")
    return warnings


def expand_grids(k_grid, cp_grid, tau_grid, warnings):
    """按边界警告在物理合理方向扩展网格。返回 (新k, 新cp, 新tau)。"""
    new_k = list(k_grid)
    new_cp = list(cp_grid)
    new_tau = list(tau_grid)
    if "K_MAX" in warnings:
        new_k.extend(K_EXTEND_UP.tolist())
    if "K_MIN" in warnings:
        new_k = K_EXTEND_DOWN.tolist() + new_k
    if "CP_MAX" in warnings:
        new_cp.extend(CP_EXTEND_UP.tolist())
    if "CP_MIN" in warnings:
        new_cp = CP_EXTEND_DOWN.tolist() + new_cp
    if "TAU_MAX" in warnings:
        new_tau.extend(TAU_EXTEND_UP.tolist())
    # TAU_MIN = 0 是物理地板, 无向下扩展
    new_k = sorted(set(round(float(x), 6) for x in new_k))
    new_cp = sorted(set(round(float(x), 6) for x in new_cp))
    new_tau = sorted(set(round(float(x), 6) for x in new_tau))
    return (np.array(new_k), np.array(new_cp), np.array(new_tau))


# ============================================================
# 局部细化网格 (任务 #15)
# ============================================================

def build_refined_grid(best, k_step=K_REFINE_STEP, k_half=K_REFINE_HALF,
                       cp_step=CP_REFINE_STEP, cp_half=CP_REFINE_HALF,
                       tau_step=TAU_REFINE_STEP, tau_half=TAU_REFINE_HALF):
    """以粗最优为中心构建局部网格; 必须包含粗最优及两侧邻居。

    物理裁剪: k >= K_FLOOR, cp >= CP_FLOOR, tau >= 0。
    """
    k_pts = sorted({round(best["k_eff_W_mK"] + i * k_step, 6)
                    for i in range(-k_half, k_half + 1)
                    if best["k_eff_W_mK"] + i * k_step >= K_FLOOR})
    cp_pts = sorted({round(best["cp_eff_J_kgK"] + i * cp_step, 6)
                     for i in range(-cp_half, cp_half + 1)
                     if best["cp_eff_J_kgK"] + i * cp_step >= CP_FLOOR})
    tau_pts = sorted({round(best["tau_top_s"] + i * tau_step, 6)
                      for i in range(-tau_half, tau_half + 1)
                      if best["tau_top_s"] + i * tau_step >= 0.0})
    return np.array(k_pts), np.array(cp_pts), np.array(tau_pts)


# ============================================================
# 可辨识性诊断 (任务 #16)
# ============================================================

def near_optimal_region(results_df, best_rmse, tol=NEAR_OPTIMAL_RMSE_TOL):
    """RMSE <= best_rmse + tol 的所有搜索点。"""
    return results_df[results_df["RMSE_C"] <= best_rmse + tol].copy()


def per_k_best_rmse(results_df):
    """每个 k 的最小 RMSE (跨 cp/tau)。"""
    return (results_df.sort_values("RMSE_C")
            .groupby("k_eff_W_mK", sort=True)
            .head(1)[["k_eff_W_mK", "RMSE_C", "cp_eff_J_kgK",
                      "tau_top_s"]])


def per_cp_best_rmse(results_df):
    """每个 cp 的最小 RMSE (跨 k/tau)。"""
    return (results_df.sort_values("RMSE_C")
            .groupby("cp_eff_J_kgK", sort=True)
            .head(1)[["cp_eff_J_kgK", "RMSE_C", "k_eff_W_mK",
                      "tau_top_s"]])


def top_n_candidates(results_df, n=10):
    """按 RMSE 升序取前 n 个候选 (含 best 与 second-best)。"""
    if n < 1:
        raise ValueError("n 必须 >= 1")
    cols = ["k_eff_W_mK", "cp_eff_J_kgK", "tau_top_s", "RMSE_C",
            "MAE_C", "mean_residual_C", "n_points", "stage"]
    df = results_df.sort_values("RMSE_C", ascending=True).reset_index(drop=True)
    return df[cols].head(n).copy()


def parameter_correlation(results_df):
    """log 参数之间的 Pearson 相关 (可辨识性诊断)。

    tau=0 的行无法取 log; 诊断相关时仅在 tau>0 的子集上计算, 避免
    log(0) 警告与 -inf 污染。k/cp 恒为正, 直接取 log。
    """
    d = results_df.copy()
    d["log_k_eff_W_mK"] = np.log(d["k_eff_W_mK"])
    d["log_cp_eff_J_kgK"] = np.log(d["cp_eff_J_kgK"])
    d["log_tau_top_s"] = np.log(d["tau_top_s"])
    # 排除 tau=0 (log 无定义) 后的子集
    d_pos = d[d["tau_top_s"] > 0]
    cols = ["log_k_eff_W_mK", "log_cp_eff_J_kgK", "log_tau_top_s"]
    if len(d_pos) < 3:
        return pd.DataFrame(np.nan, index=cols, columns=cols)
    return d_pos[cols].corr()


# ============================================================
# 标定度量 (任务 #18): regime 细分
# ============================================================

def regime_metrics_from_eval(ev):
    """从评估结果提取 heating/cooling/settling RMSE 与 mean residual。"""
    out = {}
    resid = ev["residual"]
    regimes = ev["regimes"]
    mapping = {
        "heating": "TRANSIENT_HEATING",
        "cooling": "TRANSIENT_COOLING",
        "settling": "SETTLING",
    }
    for short, rg in mapping.items():
        m = regimes == rg
        if m.sum():
            out[f"{short}_n"] = int(m.sum())
            out[f"{short}_RMSE_C"] = float(
                np.sqrt(np.mean(resid[m] ** 2)))
            out[f"{short}_mean_residual_C"] = float(np.mean(resid[m]))
        else:
            out[f"{short}_n"] = 0
            out[f"{short}_RMSE_C"] = np.nan
            out[f"{short}_mean_residual_C"] = np.nan
    return out


# ============================================================
# 图 (标定)
# ============================================================

def plot_best_fit(ev, out_path_png, out_path_pdf, title_extra=""):
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(ev["t_top"], ev["T_internal_at_top"], color="#7f7f7f", lw=1.1,
            ls=":", label="Measured internal temperature")
    ax.plot(ev["t_top"], ev["T_top_measured"], color="#1f77b4", lw=1.6,
            label="Corrected measured Top COC")
    ax.plot(ev["t_top"], ev["T_top_predicted_lagged"], color="#d62728",
            lw=1.8, label="Predicted Top COC (new candidate, output-side lag)")
    ax.plot(ev["t_top"], ev["T_top_fdm_raw"], color="#2ca02c", lw=0.9,
            ls="--", alpha=0.7, label="Raw Top COC FDM (no lag)")
    ax.set_xlabel("Elapsed time [s] (simultaneous start, t=0)")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(f"66C redo calibration — best fit Top COC validation"
                 f"{title_extra}\n(h=10 + nonlinear radiation, BARE TOP)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def plot_residual(ev, out_path_png, title_extra=""):
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.plot(ev["t_top"], ev["residual"], color="#d62728", lw=1.0,
            label="Residual (predicted - measured)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Residual [C]")
    ax.set_title(f"Residual vs time — 66C redo calibration{title_extra}")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)


def plot_rmse_vs_k(results_df, best, old_rmse, out_path_png):
    pk = per_k_best_rmse(results_df)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(pk["k_eff_W_mK"], pk["RMSE_C"], "o-", color="#1f77b4",
            label="Best RMSE (min over cp, tau)")
    ax.axvline(OLD_K, color="#8c564b", ls="--", lw=1.2,
               label=f"old frozen k={OLD_K}")
    ax.plot([OLD_K], [old_rmse], "s", color="#8c564b", ms=8,
            label=f"old candidate RMSE={old_rmse:.3f} C")
    ax.axvline(best["k_eff_W_mK"], color="#d62728", ls="--", lw=1.2,
               label=f"new optimum k={best['k_eff_W_mK']}")
    ax.plot([best["k_eff_W_mK"]], [best["RMSE_C"]], "D", color="#d62728",
            ms=9, label=f"new optimum RMSE={best['RMSE_C']:.3f} C")
    ax.set_xlabel("k_eff [W/(m K)]")
    ax.set_ylabel("Best RMSE [C]")
    ax.set_title("66C calibration — best RMSE vs k (min over cp, tau)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)


def plot_k_cp_landscape(results_df, best, old_rmse, out_path_png):
    pivot = (results_df.sort_values("RMSE_C")
             .groupby(["k_eff_W_mK", "cp_eff_J_kgK"], sort=True)
             .head(1)
             .pivot(index="cp_eff_J_kgK", columns="k_eff_W_mK",
                    values="RMSE_C"))
    pivot = pivot.reindex(index=sorted(pivot.index),
                          columns=sorted(pivot.columns))
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#dddddd")
    pc = ax.pcolormesh(pivot.columns.values, pivot.index.values,
                       pivot.values, cmap=cmap, shading="auto")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("Best RMSE (min over tau) [C]")
    ax.plot([best["k_eff_W_mK"]], [best["cp_eff_J_kgK"]], "D", color="#d62728",
            ms=10, label=f"new optimum ({best['k_eff_W_mK']}, "
                         f"{best['cp_eff_J_kgK']:.0f})")
    ax.plot([OLD_K], [OLD_CP], "s", color="#8c564b", ms=9,
            label=f"old frozen ({OLD_K}, {OLD_CP:.0f})")
    ax.set_xlabel("k_eff [W/(m K)]")
    ax.set_ylabel("cp_eff [J/(kg K)]")
    ax.set_title("66C calibration — RMSE landscape (k vs cp, min over tau)")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)


# ============================================================
# 绝缘样品前向预测 (任务 #25-31)
# ============================================================

def run_insulated_sample(k_eff, cp_eff, internal, save_dt=SAVE_DT):
    """绝缘几何 (LEGACY_INSULATED_LAYERS) 样品层前向预测。

    外部边界 = PDMS 外表面 (h=10 + 非线性辐射);
    环境/初始 = 第一个内部温度 (INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT);
    样品 = CV 加权空间平均 (raw FDM, 无滞后)。
    """
    t_int = internal["t_rel"]
    T_int = internal["T"]
    T_env = float(T_int[0])
    T_init = T_env
    mats = cr.make_convection_radiation_materials(k_eff, cp_eff, RHO_COC)
    layers = heat_model.LEGACY_INSULATED_LAYERS
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=layers, T_air_C=T_env, T_surroundings_C=T_env,
        save_dt=save_dt, T_initial_C=T_init)
    t_arr = result["t_array"]
    T_sample = result["T_sample_arr"]
    T_outer = result["T_outer_surface_arr"]
    return {
        "t_arr": t_arr,
        "T_sample": T_sample,
        "T_outer_pdms": T_outer,
        "T_environment_C": T_env,
        "environment_source": "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT",
        "T_initial_C": T_init,
        "geometry": "LEGACY_INSULATED_LAYERS",
    }


def repeated_cycle_stats(peaks):
    """重复周期样品峰统计。"""
    arr = np.asarray(peaks, dtype=float)
    if arr.size == 0:
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan,
                "median": np.nan, "std": np.nan}
    return {"n": int(arr.size), "min": float(np.min(arr)),
            "max": float(np.max(arr)), "mean": float(np.mean(arr)),
            "median": float(np.median(arr)), "std": float(np.std(arr))}


def threshold_counts(peaks, thresholds=DESCRIPTIVE_THRESHOLDS):
    """重复周期样品峰 >= 各阈值的 个数/分数/百分比。"""
    arr = np.asarray(peaks, dtype=float)
    out = {}
    for th in thresholds:
        n_ge = int(np.sum(arr >= th)) if arr.size else 0
        frac = float(n_ge / arr.size) if arr.size else np.nan
        out[f"ge{int(th)}_count"] = n_ge
        out[f"ge{int(th)}_fraction"] = frac
        out[f"ge{int(th)}_percent"] = float(frac * 100.0) if arr.size else np.nan
    return out


# ============================================================
# 反循环参数锁定
# ============================================================

def assert_parameters_locked(k, cp, tau, ref_k, ref_cp, ref_tau, phase):
    """强制: 下游阶段参数必须与 66C 最优完全一致 (无反馈路径)。"""
    for name, val, ref in (("k_eff", float(k), float(ref_k)),
                           ("cp_eff", float(cp), float(ref_cp)),
                           ("tau_top", float(tau), float(ref_tau))):
        if abs(val - ref) > 1e-12:
            raise RuntimeError(
                f"[{phase}] 参数锁定断言失败: {name}={val} != {ref}; "
                "下游阶段禁止任何参数更新。")


def assert_no_qPCR_in_calibration():
    """结构断言: 标定目标函数内不使用 86 C / 样品温度 / 绝缘仿真。"""
    import inspect
    src_eval = inspect.getsource(evaluate_top_for_tau)
    src_run = inspect.getsource(run_bare_fdm)
    src_search = inspect.getsource(search_stage)
    for name, src in (("evaluate_top_for_tau", src_eval),
                      ("run_bare_fdm", src_run),
                      ("search_stage", src_search)):
        if "86" in src or "QPCR" in src or "qPCR" in src:
            raise RuntimeError(
                f"反循环结构断言失败: {name} 包含 86/qPCR 引用。")
        if "insulated" in src or "LEGACY_INSULATED" in src:
            raise RuntimeError(
                f"反循环结构断言失败: {name} 引用绝缘仿真。")
        if "T_sample" in src and "sample" in src and "objective" in src:
            raise RuntimeError(
                f"反循环结构断言失败: {name} 目标引用样品温度。")


# ============================================================
# 分类 (任务 #43)
# ============================================================

def classify_candidate(calib_old_rmse, calib_new_rmse,
                       val_old_rmse, val_new_rmse, boundary_warnings):
    """基于 ONLY 热拟合/转移的分类 (绝不用样品峰 >=86 C)。"""
    calib_improve = calib_old_rmse - calib_new_rmse
    calib_improve_frac = (calib_improve / calib_old_rmse
                          if calib_old_rmse > 0 else 0.0)
    val_delta = val_new_rmse - val_old_rmse
    val_delta_frac = (val_delta / val_old_rmse if val_old_rmse > 0 else 0.0)

    improved = (calib_improve >= CALIB_IMPROVE_ABS_C
                and calib_improve_frac >= CALIB_IMPROVE_FRAC)
    degraded = (val_delta >= VAL_DEGRADE_ABS_C
                and val_delta_frac >= VAL_DEGRADE_FRAC)

    if boundary_warnings:
        cls = "UNRESOLVED_BOUNDARY_OPTIMUM"
        reason = (f"k/cp/tau 最优仍位于搜索边界: "
                  f"{', '.join(boundary_warnings)}")
    elif improved and not degraded:
        cls = "PROMISING_NEW_CANDIDATE"
        reason = (f"66C 标定实质改善 (Δ {calib_improve:.3f} C, "
                  f"{calib_improve_frac * 100:.1f}%), 独立 3s 验证未实质退化 "
                  f"(Δ {val_delta:+.3f} C)")
    elif improved and degraded:
        cls = "OVERFIT_TO_66C"
        reason = (f"66C 标定实质改善 (Δ {calib_improve:.3f} C) 但独立 3s "
                  f"验证实质退化 (Δ {val_delta:+.3f} C, "
                  f"{val_delta_frac * 100:.1f}%)")
    else:
        cls = "NO_MEANINGFUL_IMPROVEMENT"
        reason = (f"66C 标定改善有限 (Δ {calib_improve:.3f} C, "
                  f"{calib_improve_frac * 100:.1f}%)")
    return cls, reason, {
        "calib_improve_abs_C": float(calib_improve),
        "calib_improve_fraction": float(calib_improve_frac),
        "val_delta_abs_C": float(val_delta),
        "val_delta_fraction": float(val_delta_frac),
        "improved": bool(improved),
        "degraded": bool(degraded),
        "thresholds": {
            "CALIB_IMPROVE_ABS_C": CALIB_IMPROVE_ABS_C,
            "CALIB_IMPROVE_FRAC": CALIB_IMPROVE_FRAC,
            "VAL_ACCEPTABLE_RMSE_C": VAL_ACCEPTABLE_RMSE_C,
            "VAL_DEGRADE_ABS_C": VAL_DEGRADE_ABS_C,
            "VAL_DEGRADE_FRAC": VAL_DEGRADE_FRAC,
        },
    }


# ============================================================
# 输出辅助
# ============================================================

def _round_float(x, nd=6):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def save_json(obj, path):
    (path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_round_float),
        encoding="utf-8")


# ============================================================
# PHASE A — 66C 标定
# ============================================================

def phase_a_calibration(n_workers=8):
    """PHASE A: 仅用 66C Top + 66C 内部标定 k/cp/tau。"""
    assert_no_qPCR_in_calibration()
    print("=" * 74)
    print("PHASE A — 66C REDO CALIBRATION (k, cp, tau)")
    print("=" * 74)
    top, internal = load_66c_dataset()
    print(f"66C Top: n={top['n_valid']} dur={top['duration_s']:.1f}s "
          f"T[{top['first_value_C']:.2f}, {top['last_value_C']:.2f}]")
    print(f"66C internal: n={internal['n_valid']} "
          f"dur={internal['duration_s']:.1f}s")

    cache = {}
    results = []

    # 1) 粗搜索
    t_start = time.time()
    cache, results = search_stage(
        top, internal, K_COARSE, CP_COARSE, TAU_COARSE, "coarse",
        n_workers=n_workers, cache=cache, results=results)
    df = pd.DataFrame(results)
    best = df.loc[df["RMSE_C"].idxmin()].to_dict()
    print(f"[coarse] best: k={best['k_eff_W_mK']}, "
          f"cp={best['cp_eff_J_kgK']}, tau={best['tau_top_s']}, "
          f"RMSE={best['RMSE_C']:.4f} "
          f"({time.time() - t_start:.0f}s)")

    # 2) 边界检测 + 扩展 (最多 2 轮)
    k_grid, cp_grid, tau_grid = K_COARSE, CP_COARSE, TAU_COARSE
    rounds = 0
    while rounds < 2:
        warnings = check_boundary(best, k_grid, cp_grid, tau_grid)
        if not warnings:
            break
        print(f"[boundary] 最优位于边界: {warnings} -> 扩展网格")
        k_grid, cp_grid, tau_grid = expand_grids(
            k_grid, cp_grid, tau_grid, warnings)
        cache, results = search_stage(
            top, internal, k_grid, cp_grid, tau_grid, "expanded",
            n_workers=n_workers, cache=cache, results=results)
        df = pd.DataFrame(results)
        best = df.loc[df["RMSE_C"].idxmin()].to_dict()
        print(f"[expanded] best: k={best['k_eff_W_mK']}, "
              f"cp={best['cp_eff_J_kgK']}, tau={best['tau_top_s']}, "
              f"RMSE={best['RMSE_C']:.4f}")
        rounds += 1

    final_warnings = check_boundary(best, k_grid, cp_grid, tau_grid)
    if final_warnings:
        print(f"[boundary] 最终警告: {final_warnings}")

    # 3) 局部细化 (必须包含粗最优及两侧邻居)
    k_ref, cp_ref, tau_ref = build_refined_grid(best)
    print(f"[refine] 局部网格 k={k_ref.tolist()}")
    print(f"[refine] 局部网格 cp={cp_ref.tolist()}")
    print(f"[refine] 局部网格 tau={tau_ref.tolist()}")
    cache, results = search_stage(
        top, internal, k_ref, cp_ref, tau_ref, "refined",
        n_workers=n_workers, cache=cache, results=results)
    # 去重: 同一 (k, cp, tau) 保留 RMSE 最小的一行 (扩展/细化阶段会重算重复点)
    df = (pd.DataFrame(results)
          .sort_values("RMSE_C", ascending=True)
          .drop_duplicates(["k_eff_W_mK", "cp_eff_J_kgK", "tau_top_s"],
                           keep="first")
          .sort_values(["k_eff_W_mK", "cp_eff_J_kgK", "tau_top_s"])
          .reset_index(drop=True))
    best = df.loc[df["RMSE_C"].idxmin()].to_dict()
    print(f"[refined] best: k={best['k_eff_W_mK']}, "
          f"cp={best['cp_eff_J_kgK']}, tau={best['tau_top_s']}, "
          f"RMSE={best['RMSE_C']:.4f}")

    # 细化后再次检查边界 (若细化网格内最优仍在细化边界, 记录但不自动扩展)
    refined_warnings = check_boundary(best, k_ref, cp_ref, tau_ref)

    # 4) 旧冻结模型在相同 66C 数据上的同管线评估
    old_ev = evaluate_top_prediction(top, internal, OLD_K, OLD_CP, OLD_TAU)
    old_rmse = old_ev["metrics"]["RMSE_C"]

    # 5) 新最优完整评估
    new_ev = evaluate_top_prediction(top, internal,
                                     best["k_eff_W_mK"],
                                     best["cp_eff_J_kgK"],
                                     best["tau_top_s"])
    new_rmse = new_ev["metrics"]["RMSE_C"]

    # 6) 可辨识性诊断
    best_rmse = best["RMSE_C"]
    near = near_optimal_region(df, best_rmse)
    pk = per_k_best_rmse(df)
    pc = per_cp_best_rmse(df)
    corr = parameter_correlation(df)

    # 近最优区域内 k 的锐度: 每个 k 的最佳 RMSE 是否 <= best + tol
    k_in_near = pk[pk["RMSE_C"] <= best_rmse + NEAR_OPTIMAL_RMSE_TOL][
        "k_eff_W_mK"].tolist()
    cp_in_near = pc[pc["RMSE_C"] <= best_rmse + NEAR_OPTIMAL_RMSE_TOL][
        "cp_eff_J_kgK"].tolist()
    k_sharp = (max(k_in_near) - min(k_in_near)
               <= 2 * K_REFINE_STEP) if k_in_near else True
    cp_sharp = (max(cp_in_near) - min(cp_in_near)
                <= 2 * CP_REFINE_STEP) if cp_in_near else True

    return {
        "top": top, "internal": internal,
        "results_df": df, "cache": cache,
        "best": best, "new_ev": new_ev, "old_ev": old_ev,
        "old_rmse": old_rmse, "new_rmse": new_rmse,
        "final_warnings": final_warnings,
        "refined_warnings": refined_warnings,
        "near": near, "pk": pk, "pc": pc, "corr": corr,
        "k_in_near": k_in_near, "cp_in_near": cp_in_near,
        "k_sharp": k_sharp, "cp_sharp": cp_sharp,
    }


# ============================================================
# PHASE C — 独立 3s 裸顶验证 (无重拟合)
# ============================================================

def phase_c_validation(best, n_workers=1):
    """PHASE C: 锁定参数在独立 3s Top COC 上验证。无参数更新。"""
    print("=" * 74)
    print("PHASE C — INDEPENDENT 3S BARE-TOP VALIDATION (no refit)")
    print("=" * 74)
    top3, internal3 = load_3s_dataset()
    k = best["k_eff_W_mK"]
    cp = best["cp_eff_J_kgK"]
    tau = best["tau_top_s"]
    assert_parameters_locked(k, cp, tau,
                             best["k_eff_W_mK"], best["cp_eff_J_kgK"],
                             best["tau_top_s"], "PHASE_C")
    new_ev = evaluate_top_prediction(top3, internal3, k, cp, tau)
    old_ev = evaluate_top_prediction(top3, internal3, OLD_K, OLD_CP, OLD_TAU)
    print(f"3s new: RMSE={new_ev['metrics']['RMSE_C']:.4f} C | "
          f"old: RMSE={old_ev['metrics']['RMSE_C']:.4f} C")
    return {"top3": top3, "internal3": internal3,
            "new_ev": new_ev, "old_ev": old_ev}


# ============================================================
# PHASE D — 绝缘 3s 样品前向预测 (无重拟合)
# ============================================================

def phase_d_insulated_prediction(best, internal3, n_workers=1):
    """PHASE D: 锁定参数, 绝缘几何 3s 样品预测。无参数更新。"""
    print("=" * 74)
    print("PHASE D — INSULATED 3S SAMPLE PREDICTION (no refit)")
    print("=" * 74)
    k = best["k_eff_W_mK"]
    cp = best["cp_eff_J_kgK"]
    tau = best["tau_top_s"]
    assert_parameters_locked(k, cp, tau,
                             best["k_eff_W_mK"], best["cp_eff_J_kgK"],
                             best["tau_top_s"], "PHASE_D")

    new_run = run_insulated_sample(k, cp, internal3)
    old_run = run_insulated_sample(OLD_K, OLD_CP, internal3)

    # 样品插值到内部测量时间轴 (与既有检测器一致)
    elapsed = internal3["t_rel"]
    tint = internal3["T"]
    T_sample_new_meas = np.interp(elapsed, new_run["t_arr"],
                                  new_run["T_sample"])
    T_sample_old_meas = np.interp(elapsed, old_run["t_arr"],
                                  old_run["T_sample"])

    cyc_new = detect_repeated_cycles(elapsed, tint, T_sample_new_meas)
    cyc_old = detect_repeated_cycles(elapsed, tint, T_sample_old_meas)
    rep_new = cyc_new["repeated_cycles"]
    rep_old = cyc_old["repeated_cycles"]
    peaks_new = [float(c["sample_high_peak_C"]) for c in rep_new]
    peaks_old = [float(c["sample_high_peak_C"]) for c in rep_old]
    stats_new = repeated_cycle_stats(peaks_new)
    stats_old = repeated_cycle_stats(peaks_old)

    i_max = int(np.argmax(tint))
    s_max_idx = int(np.argmax(new_run["T_sample"]))
    internal_max = float(np.max(tint))
    sample_max = float(np.max(new_run["T_sample"]))
    t_sample_max = float(new_run["t_arr"][s_max_idx])

    thresholds_new = threshold_counts(peaks_new)
    thresholds_old = threshold_counts(peaks_old)

    n86 = thresholds_new["ge86_count"]
    n_cyc = stats_new["n"]
    frac86 = float(n86 / n_cyc) if n_cyc else np.nan
    majority86 = bool(frac86 > MAJORITY_FRACTION) if n_cyc else False
    mean_ge86 = bool(stats_new["mean"] >= QPCR_FUNCTIONAL_REFERENCE_C) \
        if np.isfinite(stats_new["mean"]) else False
    median_ge86 = bool(stats_new["median"] >= QPCR_FUNCTIONAL_REFERENCE_C) \
        if np.isfinite(stats_new["median"]) else False

    return {
        "new_run": new_run, "old_run": old_run,
        "elapsed": elapsed, "tint": tint,
        "T_sample_new_meas": T_sample_new_meas,
        "T_sample_old_meas": T_sample_old_meas,
        "cyc_new": cyc_new, "cyc_old": cyc_old,
        "rep_new": rep_new, "rep_old": rep_old,
        "peaks_new": peaks_new, "peaks_old": peaks_old,
        "stats_new": stats_new, "stats_old": stats_old,
        "internal_max_C": internal_max,
        "sample_max_C": sample_max,
        "time_of_sample_max_s": t_sample_max,
        "thresholds_new": thresholds_new,
        "thresholds_old": thresholds_old,
        "n86": n86, "n_cycles": n_cyc, "frac86": frac86,
        "majority86": majority86,
        "mean_ge86": mean_ge86, "median_ge86": median_ge86,
        "activation": cyc_new["activation"],
    }


# ============================================================
# 输出写入
# ============================================================

def write_calibration_outputs(cal):
    """写入 calibration_66C/ 全部文件。"""
    d = DIR_CALIB
    d.mkdir(parents=True, exist_ok=True)
    best = cal["best"]
    new_ev = cal["new_ev"]
    old_ev = cal["old_ev"]

    # parameter_search_results.csv
    cal["results_df"].to_csv(d / "parameter_search_results.csv", index=False)

    # top-10 candidates (任务 #16)
    top10 = top_n_candidates(cal["results_df"], n=10)
    top10.to_csv(d / "top_10_candidates.csv", index=False)

    # best_fit_summary.csv
    m = new_ev["metrics"]
    reg = regime_metrics_from_eval(new_ev)
    alpha = best["k_eff_W_mK"] / (RHO_COC * best["cp_eff_J_kgK"])
    effus = float(np.sqrt(best["k_eff_W_mK"] * RHO_COC
                          * best["cp_eff_J_kgK"]))
    summary = {
        "candidate_id": CANDIDATE_ID,
        "k_eff_W_mK": best["k_eff_W_mK"],
        "cp_eff_J_kgK": best["cp_eff_J_kgK"],
        "tau_top_s": best["tau_top_s"],
        "rho_COC_kg_m3": RHO_COC,
        "alpha_eff_m2_s": alpha,
        "effusivity": effus,
        "h_conv_W_m2K": H_CONV,
        "emissivity": EPS,
        "sigma_SB": SIGMA,
        "view_factor": F_VIEW,
        "geometry": "BARE_TOP_COC_LAYERS",
        "RMSE_C": m["RMSE_C"],
        "MAE_C": m["MAE_C"],
        "mean_residual_C": m["mean_residual_C"],
        "median_abs_residual_C": m["median_abs_residual_C"],
        "residual_std_C": m["residual_std_C"],
        "max_abs_residual_C": m["max_abs_residual_C"],
        "R_squared": m["R_squared"],
        "n_points": m["n_points"],
        **{f"heating_n": reg["heating_n"],
           "heating_RMSE_C": reg["heating_RMSE_C"],
           "heating_mean_residual_C": reg["heating_mean_residual_C"],
           "cooling_n": reg["cooling_n"],
           "cooling_RMSE_C": reg["cooling_RMSE_C"],
           "cooling_mean_residual_C": reg["cooling_mean_residual_C"],
           "settling_n": reg["settling_n"],
           "settling_RMSE_C": reg["settling_RMSE_C"],
           "settling_mean_residual_C": reg["settling_mean_residual_C"]},
        "old_frozen_RMSE_C": cal["old_rmse"],
        "improvement_abs_C": cal["old_rmse"] - m["RMSE_C"],
        "improvement_fraction": (cal["old_rmse"] - m["RMSE_C"])
        / cal["old_rmse"],
        "environment_source": new_ev["environment_source"],
        "synchronization_rule": new_ev["synchronization_rule"],
        "time_shift_applied_s": new_ev["time_shift_applied_s"],
        "qPCR_used_in_objective": False,
        "sample_temperature_used_in_objective": False,
    }
    pd.DataFrame([summary]).to_csv(d / "best_fit_summary.csv", index=False)

    # best_fit_trace.csv
    trace = pd.DataFrame({
        "measured_top_time_s": new_ev["t_top"],
        "measured_top_C": new_ev["T_top_measured"],
        "internal_interpolated_C": new_ev["T_internal_at_top"],
        "top_FDM_raw_C": new_ev["T_top_fdm_raw"],
        "top_predicted_lagged_C": new_ev["T_top_predicted_lagged"],
        "residual_C": new_ev["residual"],
    })
    trace.to_csv(d / "best_fit_trace.csv", index=False)

    # 图
    plot_best_fit(new_ev, d / "best_fit_66C_top_validation.png",
                  d / "best_fit_66C_top_validation.pdf",
                  title_extra=f"\n(k={best['k_eff_W_mK']}, "
                              f"cp={best['cp_eff_J_kgK']:.0f}, "
                              f"tau={best['tau_top_s']:.2f} s)")
    plot_residual(new_ev, d / "residual_vs_time_66C.png",
                  title_extra=f"\n(k={best['k_eff_W_mK']}, "
                              f"cp={best['cp_eff_J_kgK']:.0f}, "
                              f"tau={best['tau_top_s']:.2f} s)")
    plot_rmse_vs_k(cal["results_df"], best, cal["old_rmse"],
                   d / "rmse_vs_k.png")
    plot_k_cp_landscape(cal["results_df"], best, cal["old_rmse"],
                        d / "k_cp_rmse_landscape.png")


def write_validation_outputs(val):
    """写入 validation_3s_extension/ 全部文件。"""
    d = DIR_VALID
    d.mkdir(parents=True, exist_ok=True)
    new_ev = val["new_ev"]
    old_ev = val["old_ev"]
    m = new_ev["metrics"]

    summary = {
        "candidate_id": CANDIDATE_ID,
        "parameters_refitted": False,
        "k_eff_W_mK": None, "cp_eff_J_kgK": None, "tau_top_s": None,
        "RMSE_C": m["RMSE_C"], "MAE_C": m["MAE_C"],
        "mean_residual_C": m["mean_residual_C"],
        "median_abs_residual_C": m["median_abs_residual_C"],
        "residual_std_C": m["residual_std_C"],
        "max_abs_residual_C": m["max_abs_residual_C"],
        "R_squared": m["R_squared"],
        "n_points": m["n_points"],
        "old_frozen_RMSE_C": old_ev["metrics"]["RMSE_C"],
        "old_frozen_historical_RMSE_C": OLD_3S_RMSE_HISTORICAL,
        "environment_source": new_ev["environment_source"],
        "synchronization_rule": new_ev["synchronization_rule"],
        "time_shift_applied_s": new_ev["time_shift_applied_s"],
    }
    # 回填锁定参数 (由 main 在 PHASE A 结束后设置; 保证与 66C 最优一致)
    summary["k_eff_W_mK"] = _LOCKED_K
    summary["cp_eff_J_kgK"] = _LOCKED_CP
    summary["tau_top_s"] = _LOCKED_TAU
    pd.DataFrame([summary]).to_csv(d / "validation_summary.csv", index=False)

    trace = pd.DataFrame({
        "measured_top_time_s": new_ev["t_top"],
        "measured_top_C": new_ev["T_top_measured"],
        "internal_interpolated_C": new_ev["T_internal_at_top"],
        "top_FDM_raw_C": new_ev["T_top_fdm_raw"],
        "top_predicted_lagged_C": new_ev["T_top_predicted_lagged"],
        "residual_C": new_ev["residual"],
    })
    trace.to_csv(d / "validation_trace.csv", index=False)

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(new_ev["t_top"], new_ev["T_internal_at_top"], color="#7f7f7f",
            lw=1.1, ls=":", label="Measured internal temperature")
    ax.plot(new_ev["t_top"], new_ev["T_top_measured"], color="#1f77b4",
            lw=1.6, label="Corrected measured Top COC")
    ax.plot(new_ev["t_top"], new_ev["T_top_predicted_lagged"],
            color="#d62728", lw=1.8,
            label="Predicted Top COC (new candidate, no refit)")
    ax.plot(new_ev["t_top"], new_ev["T_top_fdm_raw"], color="#2ca02c",
            lw=0.9, ls="--", alpha=0.7, label="Raw Top COC FDM (no lag)")
    ax.set_xlabel("Elapsed time [s] (simultaneous start, t=0)")
    ax.set_ylabel("Temperature [C]")
    ax.set_title("3s-extension independent validation — new 66C candidate\n"
                 "(no refit, no time shift; h=10 + nonlinear radiation, "
                 "BARE TOP)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(d / "new_candidate_3s_top_validation.png", dpi=150)
    fig.savefig(d / "new_candidate_3s_top_validation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.plot(new_ev["t_top"], new_ev["residual"], color="#d62728", lw=1.0,
            label="Residual (predicted - measured)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Residual [C]")
    ax.set_title("Residual vs time — 3s-extension independent validation")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "residual_vs_time.png", dpi=150)
    plt.close(fig)


def write_insulated_outputs(ins):
    """写入 insulated_3s_sample_prediction/ 全部文件。"""
    d = DIR_INSUL
    d.mkdir(parents=True, exist_ok=True)
    elapsed = ins["elapsed"]
    tint = ins["tint"]
    new_run = ins["new_run"]
    t_arr = new_run["t_arr"]

    # sample_temperature_trace.csv (测量时间轴)
    trace = pd.DataFrame({
        "time_s": elapsed,
        "T_internal_measured_C": tint,
        "T_sample_predicted_C": np.interp(elapsed, t_arr,
                                          new_run["T_sample"]),
        "T_outer_PDMS_C": np.interp(elapsed, t_arr,
                                    new_run["T_outer_pdms"]),
    })
    trace.to_csv(d / "sample_temperature_trace.csv", index=False)

    # repeated_cycle_sample_peaks.csv
    rows = []
    for c in ins["rep_new"]:
        rows.append({
            "cycle_number": int(c["cycle_number"]),
            "cycle_peak_time_s": float(c["sample_peak_time_s"]),
            "cycle_sample_peak_C": float(c["sample_high_peak_C"]),
            "internal_peak_C": float(c["internal_high_peak_C"]),
            "cycle_start_time_s": float(c["cycle_start_time_s"]),
        })
    pd.DataFrame(rows).to_csv(d / "repeated_cycle_sample_peaks.csv",
                              index=False)

    # 主图
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(elapsed, tint, color="#7f7f7f", lw=1.1, ls=":",
            label="Measured internal temperature")
    ax.plot(elapsed, np.interp(elapsed, t_arr, new_run["T_sample"]),
            color="#2ca02c", lw=2.0,
            label="Predicted insulated sample temperature")
    ax.axhline(QPCR_FUNCTIONAL_REFERENCE_C, color="#d62728", ls="--", lw=1.2,
               alpha=0.8,
               label="qPCR functional reference (~86 C)")
    ax.annotate(f"predicted insulated sample max = "
                f"{ins['sample_max_C']:.2f} C",
                xy=(ins["time_of_sample_max_s"], ins["sample_max_C"]),
                xytext=(ins["time_of_sample_max_s"] - 50,
                        ins["sample_max_C"] - 10),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                fontsize=9, color="black")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title("3s-extension insulated forward prediction (new 66C "
                 "candidate)\n(k, cp locked; tau_top does NOT affect sample; "
                 "h=10 + nonlinear radiation on PDMS)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(d / "insulated_sample_prediction.png", dpi=150)
    fig.savefig(d / "insulated_sample_prediction.pdf")
    plt.close(fig)

    # 重复周期样品峰图
    if ins["peaks_new"]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        nums = [int(c["cycle_number"]) for c in ins["rep_new"]]
        ax.plot(nums, ins["peaks_new"], "o-", color="#2ca02c", lw=1.5,
                label="Predicted insulated sample peak (per repeated cycle)")
        ax.axhline(QPCR_FUNCTIONAL_REFERENCE_C, color="#d62728", ls="--",
                   lw=1.2, alpha=0.8,
                   label="qPCR functional reference (~86 C)")
        ax.set_xlabel("Repeated cycle number")
        ax.set_ylabel("Predicted insulated sample peak [C]")
        ax.set_title("3s-extension — repeated-cycle insulated sample peaks")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(d / "repeated_cycle_sample_peaks.png", dpi=150)
        plt.close(fig)

    # sample_prediction_summary.txt
    stats = ins["stats_new"]
    L = []
    A = L.append
    A("=" * 70)
    A("INSULATED 3S FORWARD SAMPLE PREDICTION SUMMARY (66C candidate)")
    A("=" * 70)
    A(f"candidate: {CANDIDATE_ID}; parameters locked (no further tuning)")
    A(f"geometry: LEGACY_INSULATED_LAYERS (Bottom COC 180 um / Sample 20 um "
      "/ Oil 50 um / Top COC 600 um / Air 3000 um / PDMS 200 um)")
    A(f"external boundary: h={H_CONV} + nonlinear radiation on outer PDMS")
    A(f"sealed-air assumption: pure conduction, no internal air convection / "
      "no internal surface-to-surface radiation (simplified)")
    A(f"environment/initial: INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT "
      f"({ins['new_run']['T_environment_C']:.2f} C)")
    A(f"sample: CV-weighted average, raw FDM, NOT lagged")
    A(f"internal maximum: {ins['internal_max_C']:.2f} C")
    A(f"overall predicted sample maximum: {ins['sample_max_C']:.2f} C "
      f"@ t={ins['time_of_sample_max_s']:.1f} s")
    A(f"repeated cycles detected: {stats['n']} "
      f"(activation excluded: "
      f"{'yes, first phase counted as activation' if ins['activation'] else 'none'})")
    A(f"repeated sample peak min/mean/median/max/std: "
      f"{stats['min']:.2f} / {stats['mean']:.2f} / {stats['median']:.2f} / "
      f"{stats['max']:.2f} / {stats['std']:.2f} C")
    A(f"86 C functional reference: {ins['n86']}/{stats['n']} peaks >= 86 C "
      f"({ins['frac86'] * 100:.1f}%); majority (>50%): "
      f"{'YES' if ins['majority86'] else 'NO'}")
    A("86 C used to fit any thermal parameter: NO")
    A("qPCR outcome used to select k/cp/tau: NO")
    A("insulated simulation used during calibration: NO")
    (d / "sample_prediction_summary.txt").write_text("\n".join(L) + "\n",
                                                     encoding="utf-8")


def write_comparison_outputs(cal, val, ins, cls, reason, cls_detail):
    """写入 comparison/ 全部文件。"""
    d = DIR_COMP
    d.mkdir(parents=True, exist_ok=True)
    best = cal["best"]

    # old_vs_new_model_parameters.csv
    alpha_new = best["k_eff_W_mK"] / (RHO_COC * best["cp_eff_J_kgK"])
    alpha_old = OLD_K / (RHO_COC * OLD_CP)
    eff_new = float(np.sqrt(best["k_eff_W_mK"] * RHO_COC
                            * best["cp_eff_J_kgK"]))
    eff_old = float(np.sqrt(OLD_K * RHO_COC * OLD_CP))
    pd.DataFrame([
        {"model": "old_frozen", "k_eff_W_mK": OLD_K, "cp_eff_J_kgK": OLD_CP,
         "tau_top_s": OLD_TAU, "rho_COC": RHO_COC,
         "alpha_eff_m2_s": alpha_old, "effusivity": eff_old},
        {"model": "66C_RECALIBRATED_CANDIDATE_V1",
         "k_eff_W_mK": best["k_eff_W_mK"],
         "cp_eff_J_kgK": best["cp_eff_J_kgK"],
         "tau_top_s": best["tau_top_s"], "rho_COC": RHO_COC,
         "alpha_eff_m2_s": alpha_new, "effusivity": eff_new},
    ]).to_csv(d / "old_vs_new_model_parameters.csv", index=False)

    # old_vs_new_66C_fit.csv
    pd.DataFrame([
        {"dataset": "66C_redo", "model": "old_frozen",
         "RMSE_C": cal["old_rmse"]},
        {"dataset": "66C_redo", "model": "new_candidate",
         "RMSE_C": cal["new_rmse"]},
    ]).to_csv(d / "old_vs_new_66C_fit.csv", index=False)

    # old_vs_new_3s_validation.csv
    pd.DataFrame([
        {"dataset": "PCR_3s_extension", "model": "old_frozen",
         "RMSE_C": val["old_ev"]["metrics"]["RMSE_C"],
         "historical_RMSE_C": OLD_3S_RMSE_HISTORICAL},
        {"dataset": "PCR_3s_extension", "model": "new_candidate",
         "RMSE_C": val["new_ev"]["metrics"]["RMSE_C"]},
    ]).to_csv(d / "old_vs_new_3s_validation.csv", index=False)

    # old_vs_new_insulated_sample_peaks.csv
    pd.DataFrame([
        {"geometry": "insulated", "model": "old_frozen",
         "repeated_mean_peak_C": ins["stats_old"]["mean"],
         "repeated_median_peak_C": ins["stats_old"]["median"],
         "n_cycles": ins["stats_old"]["n"],
         "frac_ge86": ins["thresholds_old"]["ge86_fraction"],
         "sample_max_C": float(np.max(ins["old_run"]["T_sample"]))},
        {"geometry": "insulated", "model": "new_candidate",
         "repeated_mean_peak_C": ins["stats_new"]["mean"],
         "repeated_median_peak_C": ins["stats_new"]["median"],
         "n_cycles": ins["stats_new"]["n"],
         "frac_ge86": ins["thresholds_new"]["ge86_fraction"],
         "sample_max_C": ins["sample_max_C"]},
    ]).to_csv(d / "old_vs_new_insulated_sample_peaks.csv", index=False)

    # metadata
    meta = {
        "candidate_id": CANDIDATE_ID,
        "status": "66C_RECALIBRATED_CANDIDATE_V1 (NOT promoted; "
                  "pending ChatGPT review)",
        "current_frozen_model_unchanged": True,
        "frozen_model_file": "frozen_strategy_G_candidate.py (untouched)",
        "phases": {
            "A_calibration": {"dataset": "66C redo corrected Top + internal",
                              "fitted": ["k_eff", "cp_eff", "tau_top"],
                              "objective": "predicted Top COC vs measured "
                                           "Top COC RMSE (measured-time "
                                           "interpolation)"},
            "C_3s_validation": {"parameters_refitted": False,
                                "dataset": "PCR 3s extension corrected Top "
                                           "+ internal"},
            "D_insulated_prediction": {"parameters_refitted": False,
                                       "geometry": "LEGACY_INSULATED_LAYERS"},
            "E_86C_statistics": {"reference": "post-hoc descriptive only",
                                 "used_in_calibration": False},
        },
        "anti_circularity": {
            "feedback_path_sample_peak_to_parameters": False,
            "qPCR_used_in_objective": False,
            "sample_temperature_used_in_objective": False,
            "insulated_simulation_used_in_calibration": False,
            "time_shift_optimization": False,
            "applied_time_shift_s": 0.0,
        },
        "fixed_physics": {
            "h_conv_W_m2K": H_CONV, "emissivity": EPS,
            "sigma_SB": SIGMA, "view_factor": F_VIEW,
            "radiation": "nonlinear Stefan-Boltzmann",
            "rho_COC_kg_m3": RHO_COC,
        },
        "calibration": {
            "best_k_eff_W_mK": best["k_eff_W_mK"],
            "best_cp_eff_J_kgK": best["cp_eff_J_kgK"],
            "best_tau_top_s": best["tau_top_s"],
            "alpha_eff_m2_s": alpha_new, "effusivity": eff_new,
            "RMSE_C": cal["new_rmse"],
            "old_frozen_RMSE_C": cal["old_rmse"],
            "improvement_abs_C": cal["old_rmse"] - cal["new_rmse"],
            "boundary_warnings": cal["final_warnings"],
            "refined_boundary_warnings": cal["refined_warnings"],
            "near_optimal_tol_C": NEAR_OPTIMAL_RMSE_TOL,
        },
        "validation_3s": {
            "new_RMSE_C": val["new_ev"]["metrics"]["RMSE_C"],
            "old_frozen_RMSE_C": val["old_ev"]["metrics"]["RMSE_C"],
            "old_frozen_historical_RMSE_C": OLD_3S_RMSE_HISTORICAL,
            "parameters_refitted": False,
        },
        "classification": {
            "class": cls, "reason": reason, **cls_detail,
            "based_only_on_thermal_fit_and_transfer": True,
            "sample_peaks_86C_not_used_in_classification": True,
        },
        "insulated_sample": {
            "sample_max_C": ins["sample_max_C"],
            "time_of_sample_max_s": ins["time_of_sample_max_s"],
            "internal_max_C": ins["internal_max_C"],
            "n_repeated_cycles": ins["stats_new"]["n"],
            "repeated_peak_mean_C": ins["stats_new"]["mean"],
            "repeated_peak_median_C": ins["stats_new"]["median"],
            "ge86_count": ins["n86"],
            "ge86_fraction": ins["frac86"],
            "majority_ge86": ins["majority86"],
            "thresholds": {
                str(th): {"count": ins["thresholds_new"][f"ge{int(th)}_count"],
                          "fraction": ins["thresholds_new"][
                              f"ge{int(th)}_fraction"]}
                for th in DESCRIPTIVE_THRESHOLDS},
            "note": "86 C is a descriptive assay-specific functional "
                    "reference (independent qPCR evidence); NOT a "
                    "calibration target and NOT proof of absolute sample "
                    "temperature",
        },
    }
    save_json(meta, d / "recalibrated_candidate_metadata.json")

    # final_recalibration_summary.txt
    txt = _final_summary_text(cal, val, ins, cls, reason)
    (d / "final_recalibration_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


def _final_summary_text(cal, val, ins, cls, reason):
    best = cal["best"]
    new_m = cal["new_ev"]["metrics"]
    old_m = cal["old_ev"]["metrics"]
    val_m = val["new_ev"]["metrics"]
    val_old_m = val["old_ev"]["metrics"]
    reg = regime_metrics_from_eval(cal["new_ev"])
    st = ins["stats_new"]
    L = []
    A = L.append
    A("=" * 74)
    A("66C RECALIBRATION + INSULATED 3S SAMPLE-PREDICTION SUMMARY")
    A("=" * 74)
    A(f"candidate: {CANDIDATE_ID} (NOT promoted; pending ChatGPT review)")
    A("")
    A("PHASE A — 66C CALIBRATION (k/cp/tau fit ONLY on 66C Top+internal)")
    A(f"  old frozen on 66C: k={OLD_K}, cp={OLD_CP}, tau={OLD_TAU} -> "
      f"RMSE {cal['old_rmse']:.4f} C (historical ext-val 3.0134 C)")
    A(f"  new optimum: k={best['k_eff_W_mK']}, "
      f"cp={best['cp_eff_J_kgK']:.0f}, tau={best['tau_top_s']} s -> "
      f"RMSE {cal['new_rmse']:.4f} C")
    A(f"  MAE {new_m['MAE_C']:.4f} | mean {new_m['mean_residual_C']:+.4f} | "
      f"median_abs {new_m['median_abs_residual_C']:.4f} | "
      f"std {new_m['residual_std_C']:.4f} | "
      f"max_abs {new_m['max_abs_residual_C']:.4f} | "
      f"R2 {new_m['R_squared']:.4f}")
    A(f"  regime: heating n={reg['heating_n']} "
      f"RMSE={reg['heating_RMSE_C']:.3f} mean={reg['heating_mean_residual_C']:+.3f}; "
      f"cooling n={reg['cooling_n']} RMSE={reg['cooling_RMSE_C']:.3f} "
      f"mean={reg['cooling_mean_residual_C']:+.3f}; "
      f"settling n={reg['settling_n']} RMSE={reg['settling_RMSE_C']:.3f} "
      f"mean={reg['settling_mean_residual_C']:+.3f}")
    A(f"  improvement vs old: {cal['old_rmse'] - cal['new_rmse']:.3f} C "
      f"({(cal['old_rmse'] - cal['new_rmse']) / cal['old_rmse'] * 100:.1f}%)")
    A(f"  boundary warnings: {cal['final_warnings'] or 'none'}; "
      f"refined-grid warnings: {cal['refined_warnings'] or 'none'}")
    A(f"  near-optimal region (RMSE <= {cal['new_rmse']:.3f} + "
      f"{NEAR_OPTIMAL_RMSE_TOL} C): {len(cal['near'])} points; "
      f"k range [{min(cal['k_in_near']):.4f}, {max(cal['k_in_near']):.4f}] "
      f"(sharp={cal['k_sharp']}); cp range [{min(cal['cp_in_near']):.0f}, "
      f"{max(cal['cp_in_near']):.0f}] (sharp={cal['cp_sharp']})")
    A("")
    A("PHASE C — INDEPENDENT 3S BARE-TOP VALIDATION (no refit)")
    A(f"  new: RMSE {val_m['RMSE_C']:.4f} | MAE {val_m['MAE_C']:.4f} | "
      f"mean {val_m['mean_residual_C']:+.4f} | R2 {val_m['R_squared']:.4f}")
    A(f"  old frozen: RMSE {val_old_m['RMSE_C']:.4f} C "
      f"(historical 2.3941 C)")
    A(f"  transfer delta: {val_m['RMSE_C'] - val_old_m['RMSE_C']:+.4f} C")
    A("")
    A("CLASSIFICATION (thermal fit/transfer only):")
    A(f"  {cls}")
    A(f"  reason: {reason}")
    A("")
    A("PHASE D — INSULATED 3S SAMPLE PREDICTION (no refit)")
    A(f"  internal max {ins['internal_max_C']:.2f} C; sample max "
      f"{ins['sample_max_C']:.2f} C @ t={ins['time_of_sample_max_s']:.1f} s")
    A(f"  repeated cycles: {st['n']}; peak min/mean/median/max/std = "
      f"{st['min']:.2f} / {st['mean']:.2f} / {st['median']:.2f} / "
      f"{st['max']:.2f} / {st['std']:.2f} C")
    A(f"  >=86 C: {ins['n86']}/{st['n']} ({ins['frac86'] * 100:.1f}%); "
      f"majority: {'YES' if ins['majority86'] else 'NO'}")
    A("")
    A("Anti-circularity: qPCR NOT in objective; sample NOT in objective; "
      "insulation NOT in calibration; time shift = 0.0 s; "
      "parameters locked across all downstream phases.")
    A("Current frozen model replaced automatically: NO")
    return "\n".join(L)


# ============================================================
# 全局锁定参数 (写入阶段回填; 由 main 设置)
# ============================================================
_LOCKED_K = None
_LOCKED_CP = None
_LOCKED_TAU = None


# ============================================================
# main
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="66C redo recalibration + insulated 3s sample prediction")
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args(argv)

    global _LOCKED_K, _LOCKED_CP, _LOCKED_TAU

    # ---------- PHASE A ----------
    cal = phase_a_calibration(n_workers=args.n_workers)
    best = cal["best"]
    # 冻结参数 (66C 最优; 之后绝不允许修改)
    _LOCKED_K = best["k_eff_W_mK"]
    _LOCKED_CP = best["cp_eff_J_kgK"]
    _LOCKED_TAU = best["tau_top_s"]

    # ---------- PHASE C ----------
    val = phase_c_validation(best)

    # ---------- PHASE D ----------
    ins = phase_d_insulated_prediction(best, val["internal3"])

    # ---------- 分类 (PHASE A+C 结果, 不用样品峰) ----------
    cls, reason, cls_detail = classify_candidate(
        cal["old_rmse"], cal["new_rmse"],
        val["old_ev"]["metrics"]["RMSE_C"],
        val["new_ev"]["metrics"]["RMSE_C"],
        cal["final_warnings"] + cal["refined_warnings"])

    # ---------- 输出 ----------
    for d in (DIR_CALIB, DIR_VALID, DIR_INSUL, DIR_COMP):
        d.mkdir(parents=True, exist_ok=True)
    write_calibration_outputs(cal)
    write_validation_outputs(val)
    write_insulated_outputs(ins)
    write_comparison_outputs(cal, val, ins, cls, reason, cls_detail)

    print("=" * 74)
    print(f"DONE. 输出目录: {OUTPUT_ROOT}")
    print(f"classification: {cls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
