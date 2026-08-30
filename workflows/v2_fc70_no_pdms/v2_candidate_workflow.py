#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THERMAL MODEL V2 CANDIDATE — FC-70 + no-PDMS insulated geometry
完整工作流: 数值检查 -> 重标定 -> 三数据集验证 -> 样品预测 -> 分解 -> 对比
====================================================================

本脚本构造并评估 V2 候选模型 (NOT final / NOT frozen), 与 V1 冻结模型
(FINAL_FROZEN_THERMAL_MODEL_V1) 对比。

两个科学变更:
  A. 材料: 通用矿物油 -> 3M Fluorinert FC-70 (k=0.070, rho=1940, cp=1050);
  B. 绝缘几何: 移除 PDMS 层 (仅绝缘几何; 裸顶标定/验证几何仅 oil->FC-70)。

绝不:
  - 修改 FINAL_FROZEN_THERMAL_MODEL_V1 / heat_model 的 V1 权威对象;
  - 重新拟合 FC-70 / h / epsilon / 水 / 空气 / COC 密度;
  - 提交 / 推送 / 修改 Git tag。

标定自由参数 (与 V1 一致): k_eff, cp_eff, tau_top (tau 仅作用 Top 观测)。

同步规则:
  - 标定 66C / 验证 3s: SIMULTANEOUS_START_RELATIVE_T0;
  - 验证 60C / 72C: SETPOINT_90C_EVENT_PLUS_1S (t90 + 1.0 s)。

输出 (不覆盖任何历史):
  outputs/v2_fc70_no_pdms_comparison/
"""
import json
import multiprocessing as mp
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
from thermal_model.config import thermal_model_v2_candidate as v2
from thermal_model.config.final_frozen_model import FINAL_FROZEN_THERMAL_MODEL_V1
from thermal_model.utilities.validate_frozen_model_two_new_bare_top_datasets import (
    load_top_series,
    load_internal_series,
    _regime_labels,
)
from workflows.validation.validate_66C_candidate_known_offset import (
    find_setpoint90_transition,
)
from workflows.diagnostics.analyze_insulated_sample_sensitivity import (
    define_cycle_windows,
    evaluate_cycles,
    load_setpoint,
    summarize_cycles,
)
from workflows.prediction.predict_sample_temperature_frozen_model import (
    load_internal_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"
REC_START_DIR = CALIBRATION_DIR / "Recording at the start"
REC_REACH_DIR = CALIBRATION_DIR / "Recording when reach setting"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_fc70_no_pdms_comparison"

# ------------------------------------------------------------
# 数据集 (权威, 与 V1 完全一致)
# ------------------------------------------------------------
DS_CALIB_66C_TOP = CALIBRATION_DIR / "extension 66°C_redo.xls"
DS_CALIB_66C_INT = REC_START_DIR / "08.17 COC top_66°C_zone1_temperature_analysis.xlsx"

DS_VAL = {
    "60C": {
        "top": CALIBRATION_DIR / "extension 60°C_redo.xls",
        "int": REC_REACH_DIR / "08.17 COC top_60°C_zone1_temperature_analysis.xlsx",
        "sync": "SETPOINT_90C_EVENT_PLUS_1S",
    },
    "72C": {
        "top": CALIBRATION_DIR / "extension 72°C_redo.xls",
        "int": REC_REACH_DIR / "08.17 COC top_72°C_zone1_temperature_analysis.xlsx",
        "sync": "SETPOINT_90C_EVENT_PLUS_1S",
    },
    "3s": {
        "top": CALIBRATION_DIR / "PCR 3s extension.xls",
        "int": REC_START_DIR / "08.17 COC top_pm_3s extension_zone1_temperature_analysis.xlsx",
        "sync": "SIMULTANEOUS_START_RELATIVE_T0",
    },
}

DS_PREDICT_08_24 = (CALIBRATION_DIR
                    / "08.24 am_15x primers, no holding_zone1_temperature_analysis.xlsx")

SAVE_DT = 0.1
SAVE_DT_PREDICT = 0.02

# 固定物理 (不拟合; 与 V1 一致)
RHO_COC = 1020.0
H_CONV = cr.H_CONV_STRATEGY_E_W_M2K
EPS = cr.EMISSIVITY_STRATEGY_E
SIGMA = cr.SIGMA_SB_W_M2_K4
F_VIEW = cr.VIEW_FACTOR_STRATEGY_E
EXPERIMENTAL_OFFSET_S = 1.0
SETPOINT_90_C = 90.0

# ------------------------------------------------------------
# V1 冻结基线 (只读权威值)
# ------------------------------------------------------------
V1 = {
    "k_eff": float(FINAL_FROZEN_THERMAL_MODEL_V1.k_eff_W_mK),   # 0.0675
    "cp_eff": float(FINAL_FROZEN_THERMAL_MODEL_V1.cp_eff_J_kgK),  # 700
    "tau_top": float(FINAL_FROZEN_THERMAL_MODEL_V1.tau_top_s),   # 8.0
    "rho": float(FINAL_FROZEN_THERMAL_MODEL_V1.rho_COC_kg_m3),   # 1020
    "calib": {"RMSE": 0.6368, "MAE": 0.4860, "mean_residual": +0.0316,
              "R2": 0.9903},
    "validation": {
        "60C": {"RMSE": 1.3749, "MAE": 1.1702, "mean_residual": -0.5124,
                "R2": 0.9595},
        "72C": {"RMSE": 3.0817, "MAE": 2.4851, "mean_residual": -0.0012,
                "R2": 0.7818},
        "3s": {"RMSE": 1.0643, "MAE": 0.9099, "mean_residual": +0.0359,
               "R2": 0.9739},
    },
}

# ------------------------------------------------------------
# 标定网格 (与 V1 recalibrate_thermal_model_66C_redo 一致)
# ------------------------------------------------------------
K_COARSE = np.array([0.030, 0.040, 0.050, 0.060, 0.070, 0.080,
                     0.090, 0.100, 0.120, 0.150])
CP_COARSE = np.array([700.0, 900.0, 1100.0, 1300.0, 1500.0, 1800.0, 2200.0])
TAU_COARSE = np.arange(0.0, 13.0, 1.0)

K_EXTEND_UP = np.array([0.18, 0.22, 0.28])
K_EXTEND_DOWN = np.array([0.020, 0.025])
CP_EXTEND_UP = np.array([2600.0, 3000.0, 3500.0])
CP_EXTEND_DOWN = np.array([500.0, 600.0])
TAU_EXTEND_UP = np.array([13.0, 14.0, 16.0, 18.0, 20.0])

K_REFINE_STEP = 0.0025
K_REFINE_HALF = 4
CP_REFINE_STEP = 100.0
CP_REFINE_HALF = 3
TAU_REFINE_STEP = 0.5
TAU_REFINE_HALF = 4
K_FLOOR = 0.01
CP_FLOOR = 500.0


# ============================================================
# 指标辅助
# ============================================================

def compute_metrics(resid, T_measured):
    resid = np.asarray(resid, dtype=float)
    T_measured = np.asarray(T_measured, dtype=float)
    n = len(resid)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((T_measured - np.mean(T_measured)) ** 2))
    return {
        "n_points": n,
        "RMSE_C": float(np.sqrt(np.mean(resid ** 2))),
        "MAE_C": float(np.mean(np.abs(resid))),
        "mean_residual_C": float(np.mean(resid)),
        "median_abs_residual_C": float(np.median(np.abs(resid))),
        "residual_std_C": float(np.std(resid)),
        "max_abs_residual_C": float(np.max(np.abs(resid))),
        "R_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def evaluate_top_for_tau(t_arr_c, T_fdm_c, t_top_c, T_top_c, tau_top):
    """对 raw Top 有限体积输出施加 tau 并插值到实测 Top 时间。

    注: 变量名中的 fdm 仅为历史命名; 数值方法是一维节点中心有限体积。
    """
    T_obs = apply_first_order_lag(t_arr_c, T_fdm_c, float(tau_top))
    T_pred = np.interp(t_top_c, t_arr_c, T_obs)
    resid = T_pred - T_top_c
    met = compute_metrics(resid, T_top_c)
    met["T_pred"] = T_pred
    met["residual"] = resid
    return met


# ============================================================
# V2 裸顶 FDM (标定/验证共用)
# ============================================================

def run_bare_fdm_v2(k_eff, cp_eff, internal, t_env, save_dt=SAVE_DT):
    """一次 V2 裸顶有限体积求解 (FC-70; 底部 Dirichlet = 实测内部;
    顶部非线性辐射)。函数名保留历史命名。"""
    t_int = internal["t_rel"]
    T_int = internal["T"]
    mats = v2.make_v2_materials(k_eff, cp_eff, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=v2.V2_BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=save_dt,
        T_initial_C=float(T_int[0]))
    return (result["t_array"], result["T_top_surface_arr"],
            result["T_sample_arr"], result)


# ============================================================
# PHASE A — 66C 重标定 (网格搜索 k/cp/tau)
# ============================================================

def _fdm_worker_v2(args):
    k, cp, t_int, T_int, T_env, save_dt = args
    mats = v2.make_v2_materials(k, cp, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=v2.V2_BARE_TOP_COC_LAYERS, T_air_C=T_env,
        T_surroundings_C=T_env, save_dt=save_dt,
        T_initial_C=float(T_int[0]))
    return (k, cp, result["t_array"], result["T_top_surface_arr"])


def run_fdm_batch_v2(top, internal, k_grid, cp_grid, n_workers, cache=None,
                     save_dt=SAVE_DT):
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
                        pool.imap_unordered(_fdm_worker_v2, args_list,
                                            chunksize=1)):
                    cache[(float(k), float(cp))] = (t_arr, T_fdm)
                    if i % 20 == 0 or i == len(args_list) - 1:
                        print(f"    FDM {i + 1}/{len(args_list)} (k={k}, "
                              f"cp={cp})", flush=True)
        else:
            for i, a in enumerate(args_list):
                k, cp, t_arr, T_fdm = _fdm_worker_v2(a)
                cache[(float(k), float(cp))] = (t_arr, T_fdm)
    clipped = {}
    for (k, cp), (t_arr, T_fdm) in cache.items():
        m = (t_arr >= t0 - 1e-9) & (t_arr <= t1 + 1e-9)
        clipped[(float(k), float(cp))] = (t_arr[m], T_fdm[m])
    return cache, clipped


def search_stage_v2(top, internal, k_grid, cp_grid, tau_grid, stage_name,
                    n_workers, cache=None, results=None):
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
    print(f"[{stage_name}] {len(k_grid)}k x {len(cp_grid)}cp x "
          f"{len(tau_grid)}tau", flush=True)
    cache, clipped = run_fdm_batch_v2(top, internal, k_grid, cp_grid,
                                      n_workers=n_workers, cache=cache)
    for (k, cp) in sorted(clipped):
        t_arr_c, T_fdm_c = clipped[(k, cp)]
        for tau in tau_grid:
            met = evaluate_top_for_tau(t_arr_c, T_fdm_c, t_top_c, T_top_c,
                                       float(tau))
            results.append({
                "k_eff_W_mK": float(k), "cp_eff_J_kgK": float(cp),
                "tau_top_s": float(tau), "RMSE_C": met["RMSE_C"],
                "MAE_C": met["MAE_C"],
                "mean_residual_C": met["mean_residual_C"],
                "n_points": met["n_points"], "stage": stage_name,
            })
    return cache, results


def check_boundary(best, k_grid, cp_grid, tau_grid):
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
    return (np.array(sorted(set(round(float(x), 6) for x in new_k))),
            np.array(sorted(set(round(float(x), 6) for x in new_cp))),
            np.array(sorted(set(round(float(x), 6) for x in new_tau))))


def build_refined_grid(best):
    k_pts = sorted({round(best["k_eff_W_mK"] + i * K_REFINE_STEP, 6)
                    for i in range(-K_REFINE_HALF, K_REFINE_HALF + 1)
                    if best["k_eff_W_mK"] + i * K_REFINE_STEP >= K_FLOOR})
    cp_pts = sorted({round(best["cp_eff_J_kgK"] + i * CP_REFINE_STEP, 6)
                     for i in range(-CP_REFINE_HALF, CP_REFINE_HALF + 1)
                     if best["cp_eff_J_kgK"] + i * CP_REFINE_STEP >= CP_FLOOR})
    tau_pts = sorted({round(best["tau_top_s"] + i * TAU_REFINE_STEP, 6)
                      for i in range(-TAU_REFINE_HALF, TAU_REFINE_HALF + 1)
                      if best["tau_top_s"] + i * TAU_REFINE_STEP >= 0.0})
    return np.array(k_pts), np.array(cp_pts), np.array(tau_pts)


def phase_a_calibration(n_workers=12):
    print("=" * 74)
    print("PHASE A — 66C REDO RECALIBRATION (V2: FC-70 bare geometry)")
    print("=" * 74)
    top = load_top_series(DS_CALIB_66C_TOP)
    internal = load_internal_series(DS_CALIB_66C_INT)
    print(f"66C Top: n={top['n_valid']} dur={top['duration_s']:.1f}s "
          f"T[{top['first_value_C']:.2f}, {top['last_value_C']:.2f}]")
    print(f"66C internal: n={internal['n_valid']} "
          f"dur={internal['duration_s']:.1f}s")

    cache, results = {}, []
    t_start = time.time()
    cache, results = search_stage_v2(top, internal, K_COARSE, CP_COARSE,
                                     TAU_COARSE, "coarse", n_workers,
                                     cache, results)
    df = pd.DataFrame(results)
    best = df.loc[df["RMSE_C"].idxmin()].to_dict()
    print(f"[coarse] best k={best['k_eff_W_mK']}, cp={best['cp_eff_J_kgK']}, "
          f"tau={best['tau_top_s']}, RMSE={best['RMSE_C']:.4f} "
          f"({time.time() - t_start:.0f}s)")

    k_grid, cp_grid, tau_grid = K_COARSE, CP_COARSE, TAU_COARSE
    for _round in range(2):
        warnings = check_boundary(best, k_grid, cp_grid, tau_grid)
        if not warnings:
            break
        print(f"[boundary] {warnings} -> expand")
        k_grid, cp_grid, tau_grid = expand_grids(k_grid, cp_grid, tau_grid,
                                                 warnings)
        cache, results = search_stage_v2(top, internal, k_grid, cp_grid,
                                         tau_grid, "expanded", n_workers,
                                         cache, results)
        df = pd.DataFrame(results)
        best = df.loc[df["RMSE_C"].idxmin()].to_dict()
        print(f"[expanded] best k={best['k_eff_W_mK']}, "
              f"cp={best['cp_eff_J_kgK']}, tau={best['tau_top_s']}, "
              f"RMSE={best['RMSE_C']:.4f}")

    final_warnings = check_boundary(best, k_grid, cp_grid, tau_grid)

    k_ref, cp_ref, tau_ref = build_refined_grid(best)
    print(f"[refine] k={k_ref.tolist()}")
    print(f"[refine] cp={cp_ref.tolist()}")
    print(f"[refine] tau={tau_ref.tolist()}")
    cache, results = search_stage_v2(top, internal, k_ref, cp_ref, tau_ref,
                                     "refined", n_workers, cache, results)
    df = (pd.DataFrame(results)
          .sort_values("RMSE_C", ascending=True)
          .drop_duplicates(["k_eff_W_mK", "cp_eff_J_kgK", "tau_top_s"],
                           keep="first")
          .sort_values(["k_eff_W_mK", "cp_eff_J_kgK", "tau_top_s"])
          .reset_index(drop=True))
    best = df.loc[df["RMSE_C"].idxmin()].to_dict()
    print(f"[refined] best k={best['k_eff_W_mK']}, cp={best['cp_eff_J_kgK']}, "
          f"tau={best['tau_top_s']}, RMSE={best['RMSE_C']:.4f}")

    refined_warnings = check_boundary(best, k_ref, cp_ref, tau_ref)

    # 旧 V1 在同数据上的 V1 管线评估 (oil 材料 + 裸顶)
    old_ev = _evaluate_v1_66c(top, internal)
    new_ev = _evaluate_v2_bare(top, internal, best["k_eff_W_mK"],
                               best["cp_eff_J_kgK"], best["tau_top_s"])

    return {
        "top": top, "internal": internal, "results_df": df, "cache": cache,
        "best": best, "new_ev": new_ev, "old_ev": old_ev,
        "final_warnings": final_warnings,
        "refined_warnings": refined_warnings,
    }


def _evaluate_v1_66c(top, internal):
    """V1 (oil 材料) 在同 66C 数据上的裸顶评估 (用于直接对比)。"""
    return _evaluate_bare_with(top, internal, V1["k_eff"], V1["cp_eff"],
                               V1["tau_top"], use_fc70=False)


def _evaluate_v2_bare(top, internal, k, cp, tau):
    return _evaluate_bare_with(top, internal, k, cp, tau, use_fc70=True)


def _evaluate_bare_with(top, internal, k, cp, tau, use_fc70):
    """SIMULTANEOUS_START_RELATIVE_T0 裸顶评估 (标定 / 3s 验证共用)。"""
    t_top = top["t_rel"]
    T_top = top["T"]
    t_int = internal["t_rel"]
    T_int = internal["T"]
    t0 = max(t_top[0], t_int[0])
    t1 = min(t_top[-1], t_int[-1])
    if t1 <= t0:
        raise ValueError("Top 与内部时间范围无重叠。")
    t_env = float(T_top[0])
    if use_fc70:
        mats = v2.make_v2_materials(k, cp, RHO_COC)
        layers = v2.V2_BARE_TOP_COC_LAYERS
    else:
        mats = cr.make_convection_radiation_materials(k, cp, RHO_COC)
        layers = heat_model.BARE_TOP_COC_LAYERS
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=layers, T_air_C=t_env, T_surroundings_C=t_env,
        save_dt=SAVE_DT, T_initial_C=float(T_int[0]))
    t_arr = result["t_array"]
    T_top_fdm = result["T_top_surface_arr"]
    m_arr = (t_arr >= t0 - 1e-9) & (t_arr <= t1 + 1e-9)
    m_top = (t_top >= t0 - 1e-9) & (t_top <= t1 + 1e-9)
    t_arr_c = t_arr[m_arr]
    T_fdm_c = T_top_fdm[m_arr]
    t_top_c = t_top[m_top]
    T_top_c = T_top[m_top]
    met = evaluate_top_for_tau(t_arr_c, T_fdm_c, t_top_c, T_top_c, tau)
    T_int_at_top = np.interp(t_top_c, t_int, T_int)
    regimes = _regime_labels(t_top_c, T_int_at_top, T_top_c)
    return {
        "t_top": t_top_c, "T_top_measured": T_top_c,
        "T_internal_at_top": T_int_at_top,
        "T_top_fdm_raw": np.interp(t_top_c, t_arr_c, T_fdm_c),
        "T_top_predicted_lagged": met["T_pred"],
        "residual": met["residual"], "regimes": regimes,
        "T_env_C": t_env, "T_initial_C": float(T_int[0]),
        "metrics": {kk: vv for kk, vv in met.items()
                    if kk not in ("T_pred", "residual")},
    }


# ============================================================
# PHASE C — 三数据集验证 (零重拟合)
# ============================================================

def evaluate_known_offset_v2(top, internal, k, cp, tau, t90_rel,
                             use_fc70=True):
    """SETPOINT_90C_EVENT_PLUS_1S 验证 (60C/72C)。"""
    t_int = internal["t_rel"]
    T_int = internal["T"]
    t_top_rel = top["t_rel"]
    T_top = top["T"]
    t_top_start = t90_rel + EXPERIMENTAL_OFFSET_S
    t_mapped = t_top_start + t_top_rel
    m = (t_mapped >= t_int[0] - 1e-9) & (t_mapped <= t_int[-1] + 1e-9)
    t_top_used = t_top_rel[m]
    T_top_used = T_top[m]
    t_mapped_used = t_mapped[m]
    t_env = float(T_top[0])
    T_init = float(T_int[0])
    if use_fc70:
        mats = v2.make_v2_materials(k, cp, RHO_COC)
        layers = v2.V2_BARE_TOP_COC_LAYERS
    else:
        mats = cr.make_convection_radiation_materials(k, cp, RHO_COC)
        layers = heat_model.BARE_TOP_COC_LAYERS
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=layers, T_air_C=t_env, T_surroundings_C=t_env,
        save_dt=SAVE_DT, T_initial_C=T_init)
    t_arr = result["t_array"]
    T_top_fdm = result["T_top_surface_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top_fdm, tau)
    T_pred = np.interp(t_mapped_used, t_arr, T_top_obs)
    resid = T_pred - T_top_used
    met = compute_metrics(resid, T_top_used)
    return {
        "t_mapped": t_mapped_used, "T_top_measured": T_top_used,
        "T_pred": T_pred, "residual": resid, "metrics": met,
        "T_env_C": t_env, "T_initial_C": T_init,
        "t90_rel_s": t90_rel,
    }


def phase_c_validation(best, n_workers=12):
    print("=" * 74)
    print("PHASE C — THREE EXTERNAL VALIDATION DATASETS (no refit)")
    print("=" * 74)
    k = best["k_eff_W_mK"]
    cp = best["cp_eff_J_kgK"]
    tau = best["tau_top_s"]
    results = {}
    for label, cfg in DS_VAL.items():
        top = load_top_series(cfg["top"])
        internal = load_internal_series(cfg["int"])
        if cfg["sync"] == "SETPOINT_90C_EVENT_PLUS_1S":
            anchor = find_setpoint90_transition(cfg["int"])
            t90 = anchor["t90_rel_s"]
            ev_v2 = evaluate_known_offset_v2(top, internal, k, cp, tau, t90,
                                             use_fc70=True)
            ev_v1 = evaluate_known_offset_v2(top, internal, V1["k_eff"],
                                             V1["cp_eff"], V1["tau_top"], t90,
                                             use_fc70=False)
            sync = "SETPOINT_90C_EVENT_PLUS_1S"
        else:
            ev_v2 = _evaluate_bare_with(top, internal, k, cp, tau,
                                        use_fc70=True)
            ev_v1 = _evaluate_bare_with(top, internal, V1["k_eff"],
                                        V1["cp_eff"], V1["tau_top"],
                                        use_fc70=False)
            sync = "SIMULTANEOUS_START_RELATIVE_T0"
        results[label] = {
            "v2": ev_v2, "v1": ev_v1, "sync": sync,
        }
        print(f"{label}: V2 RMSE={ev_v2['metrics']['RMSE_C']:.4f} C | "
              f"V1 RMSE={ev_v1['metrics']['RMSE_C']:.4f} C "
              f"(sync={sync})", flush=True)
    return results


# ============================================================
# PHASE D — 08.24 无保持样品预测 (V2 绝缘无 PDMS) + Setpoint 周期分析
# ============================================================

def setpoint_cycle_summary_from_fdm(t_abs, T_sample, windows):
    """复用权威 evaluate_cycles: 在密集有限体积输出时间轴上做窗口 max/min。"""
    cyc = evaluate_cycles(t_abs, T_sample, windows)
    summary = summarize_cycles(cyc)
    highs = np.array(cyc["highs_C"], dtype=float)
    lows = np.array(cyc["lows_C"], dtype=float)
    amps = np.array(cyc["amps_C"], dtype=float)
    return {
        "n_complete_cycles": int(highs.size),
        "mean_high_C": float(np.mean(highs)) if highs.size else np.nan,
        "mean_low_C": float(np.mean(lows)) if lows.size else np.nan,
        "mean_range_C": float(np.mean(amps)) if amps.size else np.nan,
        "median_high_C": float(np.median(highs)) if highs.size else np.nan,
        "median_low_C": float(np.median(lows)) if lows.size else np.nan,
        "highs": highs, "lows": lows,
        "summary": summary,
    }


def _load_prediction_windows():
    t_sp, sp = load_setpoint(DS_PREDICT_08_24)
    cw = define_cycle_windows(t_sp, sp)
    return cw, cw["windows"]


def run_insulated_fdm(mats, layers, t_src, T_internal, save_dt):
    """绝缘几何有限体积求解, 返回 (t_abs, T_sample) 权威密集输出轴。
    函数名保留历史命名; 数值方法是一维节点中心有限体积。

    t_abs = t_src[0] + res['t_array'] (恢复绝对时间轴, 与权威脚本一致)。
    """
    T_init = float(T_internal[0])
    res = cr.run_convection_radiation_fdm(
        time_s=t_src, bottom_temperature_C=T_internal, materials=mats,
        layers=layers, T_air_C=T_init, T_surroundings_C=T_init,
        save_dt=save_dt, T_initial_C=T_init)
    t_abs = float(t_src[0]) + res["t_array"]
    return t_abs, res["T_sample_arr"], res


def run_sample_prediction(best):
    """08.24 无保持: V2 绝缘(无 PDMS)样品预测 + Setpoint 周期分析。

    V1 基线用 V1 参数 (0.0675/700) + LEGACY_INSULATED_LAYERS;
    V2 用 V2 参数 (best) + V2_INSULATED_NO_PDMS_LAYERS。
    周期统计复用权威 evaluate_cycles (密集有限体积输出轴, 不插值)。
    """
    print("=" * 74)
    print("PHASE D — 08.24 NO-HOLDING SAMPLE PREDICTION (V2 insulated)")
    print("=" * 74)
    data = load_internal_data(DS_PREDICT_08_24, sheet="Extracted_Data",
                              time_col="Time(s)",
                              temp_col="Zone 1 Avg (°C)")
    t = data["source_time_s"]
    T = data["T_internal_C"]
    cw, windows = _load_prediction_windows()

    k, cp, tau = best["k_eff_W_mK"], best["cp_eff_J_kgK"], best["tau_top_s"]

    # V2 绝缘 (FC-70, 无 PDMS, V2 参数)
    mats_v2 = v2.make_v2_materials(k, cp, RHO_COC)
    t_abs_v2, T_sample_v2, res_v2 = run_insulated_fdm(
        mats_v2, v2.V2_INSULATED_NO_PDMS_LAYERS, t, T, SAVE_DT_PREDICT)

    # V1 绝缘 (oil + PDMS, V1 参数)
    mats_v1 = cr.make_convection_radiation_materials(V1["k_eff"],
                                                     V1["cp_eff"], RHO_COC)
    t_abs_v1, T_sample_v1, res_v1 = run_insulated_fdm(
        mats_v1, heat_model.LEGACY_INSULATED_LAYERS, t, T, SAVE_DT_PREDICT)

    cyc_v2 = setpoint_cycle_summary_from_fdm(t_abs_v2, T_sample_v2, windows)
    cyc_v1 = setpoint_cycle_summary_from_fdm(t_abs_v1, T_sample_v1, windows)
    return {
        "t_internal_abs": t, "T_internal": T,
        "T_sample_v2": T_sample_v2, "T_sample_v1": T_sample_v1,
        "t_abs_v1": t_abs_v1, "t_abs_v2": t_abs_v2,
        "cyc_v2": cyc_v2, "cyc_v1": cyc_v1,
        "activation": cw["activation"],
        "sample_max_v2_C": float(np.max(T_sample_v2)),
        "sample_max_v1_C": float(np.max(T_sample_v1)),
        "newton_max_iter": int(res_v2["newton_max_iterations_per_step"]),
        "newton_residual": float(res_v2["max_abs_boundary_residual_W_m2"]),
    }


# ============================================================
# 分解 (固定 V1 参数, 隔离 FC-70 / PDMS / 重校准)
# ============================================================

def run_decomposition(best):
    """A/B/C/D 预测分解 (固定 V1 参数 0.0675/700/8.0; 样品不依赖 tau)。"""
    print("=" * 74)
    print("DECOMPOSITION — FC-70 vs PDMS removal vs recalibration")
    print("=" * 74)
    data = load_internal_data(DS_PREDICT_08_24, sheet="Extracted_Data",
                              time_col="Time(s)",
                              temp_col="Zone 1 Avg (°C)")
    t = data["source_time_s"]
    T = data["T_internal_C"]
    cw, windows = _load_prediction_windows()
    k_fixed = V1["k_eff"]
    cp_fixed = V1["cp_eff"]

    def _run(mat_key, no_pdms):
        """mat_key in {'oil', 'FC70'}; no_pdms 控制是否移除 PDMS 层。"""
        use_fc70 = (mat_key == "FC70")
        if use_fc70:
            mats = v2.make_v2_materials(k_fixed, cp_fixed, RHO_COC)
        else:
            mats = cr.make_convection_radiation_materials(k_fixed, cp_fixed,
                                                          RHO_COC)
        if no_pdms:
            layers = heat_model.copy_layers(v2.V2_INSULATED_NO_PDMS_LAYERS)
            if not use_fc70:
                for layer in layers:
                    if layer.material == "FC70":
                        layer.material = "Oil"
        else:
            layers = heat_model.copy_layers(heat_model.LEGACY_INSULATED_LAYERS)
            if use_fc70:
                for layer in layers:
                    if layer.material == "Oil":
                        layer.material = "FC70"
        t_abs, T_sample, _ = run_insulated_fdm(mats, layers, t, T,
                                               SAVE_DT_PREDICT)
        return t_abs, T_sample

    t_abs_A, A = _run("oil", no_pdms=False)    # V1 几何 + V1 oil
    _, B = _run("FC70", no_pdms=False)          # FC-70, PDMS 保留
    _, C = _run("oil", no_pdms=True)            # oil, PDMS 移除
    _, D = _run("FC70", no_pdms=True)           # FC-70 + PDMS 移除 (V1 参数)

    # 重校准效应: D(V2 参数) - D(V1 参数)
    mats_v2k = v2.make_v2_materials(best["k_eff_W_mK"], best["cp_eff_J_kgK"],
                                    RHO_COC)
    _, D_v2, _ = run_insulated_fdm(mats_v2k, v2.V2_INSULATED_NO_PDMS_LAYERS,
                                   t, T, SAVE_DT_PREDICT)

    def _stats(arr):
        cyc = setpoint_cycle_summary_from_fdm(t_abs_A, arr, windows)
        return {
            "mean_high_C": cyc["mean_high_C"],
            "mean_low_C": cyc["mean_low_C"],
            "sample_max_C": float(np.max(arr)),
        }

    sA, sB, sC, sD, sDv2 = (_stats(A), _stats(B), _stats(C), _stats(D),
                             _stats(D_v2))
    out = {
        "A_V1_oil_PDMS": sA,
        "B_FC70_PDMS": sB,
        "C_oil_noPDMS": sC,
        "D_FC70_noPDMS_V1params": sD,
        "D_FC70_noPDMS_V2params": sDv2,
        "FC70_effect_mean_high_C": sB["mean_high_C"] - sA["mean_high_C"],
        "PDMS_effect_mean_high_C": sC["mean_high_C"] - sA["mean_high_C"],
        "recalibration_effect_mean_high_C":
            sDv2["mean_high_C"] - sD["mean_high_C"],
        "combined_mean_high_C": sDv2["mean_high_C"] - sA["mean_high_C"],
        "sum_of_isolated_C": (sB["mean_high_C"] - sA["mean_high_C"])
        + (sC["mean_high_C"] - sA["mean_high_C"])
        + (sDv2["mean_high_C"] - sD["mean_high_C"]),
    }
    return out


# ============================================================
# 数值检查
# ============================================================

def numerical_check():
    mats = v2.make_v2_materials(V1["k_eff"], V1["cp_eff"], RHO_COC)
    mb, db = heat_model.compute_stable_dt(mats, v2.V2_BARE_TOP_COC_LAYERS)
    mi, di = heat_model.compute_stable_dt(mats, v2.V2_INSULATED_NO_PDMS_LAYERS)
    mats_v1 = cr.make_convection_radiation_materials(V1["k_eff"],
                                                     V1["cp_eff"], RHO_COC)
    _, dv1_bare = heat_model.compute_stable_dt(mats_v1,
                                               heat_model.BARE_TOP_COC_LAYERS)
    _, dv1_ins = heat_model.compute_stable_dt(
        mats_v1, heat_model.LEGACY_INSULATED_LAYERS)
    return {
        "v2_bare_nodes": int(mb.Nx),
        "v2_bare_boundaries_um": [round(float(b) * 1e6, 1)
                                  for b in mb.boundaries],
        "v2_bare_dt_s": float(db),
        "v2_insulated_nodes": int(mi.Nx),
        "v2_insulated_boundaries_um": [round(float(b) * 1e6, 1)
                                       for b in mi.boundaries],
        "v2_insulated_dt_s": float(di),
        "v2_insulated_layer_names": list(mi.layer_names),
        "v1_bare_dt_s": float(dv1_bare),
        "v1_insulated_dt_s": float(dv1_ins),
        "dt_unchanged_from_v1": bool(abs(db - dv1_bare) < 1e-15
                                     and abs(di - dv1_ins) < 1e-15),
    }


# ============================================================
# 输出写入
# ============================================================

def _round(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return [float(x) for x in o]
    return o


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=_round), encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_workers = 12

    # ---- 数值检查 ----
    num = numerical_check()
    print(json.dumps(num, indent=2))
    save_json(num, OUTPUT_ROOT / "numerical_check.json")

    # ---- 重标定 ----
    cal = phase_a_calibration(n_workers=n_workers)
    best = cal["best"]
    v2_params = {
        "model_id": v2.MODEL_ID, "status": v2.STATUS,
        "k_eff_W_mK": best["k_eff_W_mK"],
        "cp_eff_J_kgK": best["cp_eff_J_kgK"],
        "tau_top_s": best["tau_top_s"],
        "rho_COC_kg_m3": RHO_COC,
        "fc70": {"k": v2.FC70_K_W_MK, "rho": v2.FC70_RHO_KG_M3,
                 "cp": v2.FC70_CP_J_KGK},
        "calibration_RMSE_C": best["RMSE_C"],
        "final_warnings": cal["final_warnings"],
        "refined_warnings": cal["refined_warnings"],
        "pct_change_vs_V1": {
            "k_eff": 100.0 * (best["k_eff_W_mK"] - V1["k_eff"]) / V1["k_eff"],
            "cp_eff": 100.0 * (best["cp_eff_J_kgK"] - V1["cp_eff"])
            / V1["cp_eff"],
            "tau_top": 100.0 * (best["tau_top_s"] - V1["tau_top"])
            / V1["tau_top"],
        },
    }
    save_json(v2_params, OUTPUT_ROOT / "v2_calibration_params.json")
    cal["results_df"].to_csv(OUTPUT_ROOT / "calibration_search_results.csv",
                             index=False)
    print("V2 calibrated:", {kk: v2_params[kk] for kk in
                             ("k_eff_W_mK", "cp_eff_J_kgK", "tau_top_s")})

    # ---- 验证 ----
    val = phase_c_validation(best, n_workers=n_workers)
    val_summary = {}
    for label, r in val.items():
        val_summary[label] = {
            "sync": r["sync"],
            "V2": r["v2"]["metrics"],
            "V1_reproduced": r["v1"]["metrics"],
        }
    save_json(val_summary, OUTPUT_ROOT / "validation_metrics.json")

    # ---- 预测 ----
    pred = run_sample_prediction(best)
    save_json({"cyc_v2": {k: v for k, v in pred["cyc_v2"].items()
                          if k not in ("highs", "lows", "summary")},
               "cyc_v1": {k: v for k, v in pred["cyc_v1"].items()
                          if k not in ("highs", "lows", "summary")},
               "sample_max_v2_C": pred["sample_max_v2_C"],
               "sample_max_v1_C": pred["sample_max_v1_C"],
               "newton_max_iter": pred["newton_max_iter"],
               "newton_residual_W_m2": pred["newton_residual"]},
              OUTPUT_ROOT / "sample_prediction_summary.json")

    # ---- 分解 ----
    decomp = run_decomposition(best)
    save_json(decomp, OUTPUT_ROOT / "decomposition.json")

    # ---- 对比表 ----
    write_comparison(cal, val, pred, decomp, num, v2_params)

    # ---- 图 ----
    write_figures(cal, val, pred)

    print("=" * 74)
    print("DONE. Output ->", OUTPUT_ROOT)


def write_comparison(cal, val, pred, decomp, num, v2_params):
    best = cal["best"]
    rows = []
    def add(metric, v1v, v2v, change):
        rows.append({"metric": metric, "V1": v1v, "V2_candidate": v2v,
                     "change": change})

    add("k_eff_W_mK", V1["k_eff"], best["k_eff_W_mK"],
        best["k_eff_W_mK"] - V1["k_eff"])
    add("cp_eff_J_kgK", V1["cp_eff"], best["cp_eff_J_kgK"],
        best["cp_eff_J_kgK"] - V1["cp_eff"])
    add("tau_top_s", V1["tau_top"], best["tau_top_s"],
        best["tau_top_s"] - V1["tau_top"])
    add("CALIB_RMSE_C", V1["calib"]["RMSE"], best["RMSE_C"],
        best["RMSE_C"] - V1["calib"]["RMSE"])
    m = cal["new_ev"]["metrics"]
    add("CALIB_MAE_C", V1["calib"]["MAE"], m["MAE_C"],
        m["MAE_C"] - V1["calib"]["MAE"])
    add("CALIB_mean_residual_C", V1["calib"]["mean_residual"],
        m["mean_residual_C"], m["mean_residual_C"] - V1["calib"]["mean_residual"])
    add("CALIB_R2", V1["calib"]["R2"], m["R_squared"],
        m["R_squared"] - V1["calib"]["R2"])

    for label in ("60C", "72C", "3s"):
        r = val[label]
        m2 = r["v2"]["metrics"]
        add(f"VAL_{label}_RMSE_C", V1["validation"][label]["RMSE"],
            m2["RMSE_C"], m2["RMSE_C"] - V1["validation"][label]["RMSE"])
        add(f"VAL_{label}_MAE_C", V1["validation"][label]["MAE"],
            m2["MAE_C"], m2["MAE_C"] - V1["validation"][label]["MAE"])
        add(f"VAL_{label}_bias_C", V1["validation"][label]["mean_residual"],
            m2["mean_residual_C"],
            m2["mean_residual_C"] - V1["validation"][label]["mean_residual"])
        add(f"VAL_{label}_R2", V1["validation"][label]["R2"],
            m2["R_squared"], m2["R_squared"] - V1["validation"][label]["R2"])

    # 验证汇总 RMSE
    v1_rmse = [V1["validation"][l]["RMSE"] for l in ("60C", "72C", "3s")]
    v2_rmse = [val[l]["v2"]["metrics"]["RMSE_C"] for l in ("60C", "72C", "3s")]
    add("VAL_mean_RMSE_C", float(np.mean(v1_rmse)), float(np.mean(v2_rmse)),
        float(np.mean(v2_rmse)) - float(np.mean(v1_rmse)))
    add("VAL_median_RMSE_C", float(np.median(v1_rmse)),
        float(np.median(v2_rmse)),
        float(np.median(v2_rmse)) - float(np.median(v1_rmse)))
    add("VAL_worst_RMSE_C", float(np.max(v1_rmse)), float(np.max(v2_rmse)),
        float(np.max(v2_rmse)) - float(np.max(v1_rmse)))

    # 样品预测
    c2 = pred["cyc_v2"]
    c1 = pred["cyc_v1"]
    add("SAMPLE_complete_cycles", c1["n_complete_cycles"],
        c2["n_complete_cycles"],
        c2["n_complete_cycles"] - c1["n_complete_cycles"])
    add("SAMPLE_mean_high_C", c1["mean_high_C"], c2["mean_high_C"],
        c2["mean_high_C"] - c1["mean_high_C"])
    add("SAMPLE_mean_low_C", c1["mean_low_C"], c2["mean_low_C"],
        c2["mean_low_C"] - c1["mean_low_C"])
    add("SAMPLE_mean_range_C", c1["mean_range_C"], c2["mean_range_C"],
        c2["mean_range_C"] - c1["mean_range_C"])
    add("SAMPLE_median_high_C", c1["median_high_C"], c2["median_high_C"],
        c2["median_high_C"] - c1["median_high_C"])
    add("SAMPLE_median_low_C", c1["median_low_C"], c2["median_low_C"],
        c2["median_low_C"] - c1["median_low_C"])

    add("NUMERICS_stable_dt_s", num["v1_bare_dt_s"], num["v2_bare_dt_s"],
        num["v2_bare_dt_s"] - num["v1_bare_dt_s"])

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_ROOT / "v1_vs_v2_comparison.csv", index=False)
    print(df.to_string(index=False))


def write_figures(cal, val, pred):
    # 标定拟合图
    ev = cal["new_ev"]
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(ev["t_top"], ev["T_internal_at_top"], color="#7f7f7f", lw=1.1,
            ls=":", label="Measured internal temperature")
    ax.plot(ev["t_top"], ev["T_top_measured"], color="#1f77b4", lw=1.6,
            label="Measured Top COC")
    ax.plot(ev["t_top"], ev["T_top_predicted_lagged"], color="#d62728",
            lw=1.8, label="V2 candidate predicted Top COC (lagged)")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title("66C redo calibration — V2 candidate (FC-70 bare)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "calibration_fit.png", dpi=150)
    fig.savefig(OUTPUT_ROOT / "calibration_fit.pdf")
    plt.close(fig)

    # 验证图
    for label in ("60C", "72C", "3s"):
        r = val[label]
        ev = r["v2"]
        fig, ax = plt.subplots(figsize=(12.5, 6.0))
        if "t_mapped" in ev:
            t = ev["t_mapped"]
        else:
            t = ev["t_top"]
        T_pred = ev.get("T_pred", ev.get("T_top_predicted_lagged"))
        ax.plot(t, ev["T_top_measured"], color="#1f77b4", lw=1.6,
                label="Measured Top COC")
        ax.plot(t, T_pred, color="#d62728", lw=1.6,
                label="V2 candidate predicted Top COC")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [C]")
        ax.set_title(f"{label} external validation — V2 candidate")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / f"validation_{label}.png", dpi=150)
        plt.close(fig)

    # 样品预测图 (密集 FDM 轴)
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(pred["t_internal_abs"], pred["T_internal"], color="#7f7f7f",
            lw=1.1, ls=":", label="Measured internal temperature")
    ax.plot(pred["t_abs_v1"], pred["T_sample_v1"], color="#2ca02c", lw=1.6,
            label="V1 sample (oil + PDMS)")
    ax.plot(pred["t_abs_v2"], pred["T_sample_v2"], color="#d62728", lw=1.6,
            label="V2 sample (FC-70, no PDMS)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title("08.24 no-holding — V1 vs V2 predicted sample temperature")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "sample_prediction_v1_vs_v2.png", dpi=150)
    fig.savefig(OUTPUT_ROOT / "sample_prediction_v1_vs_v2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
