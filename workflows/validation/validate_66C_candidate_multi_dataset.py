#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
66C 候选 — 多数据集零重拟合转移验证 (60C / 72C / 3s mixed-start)
=================================================================

科学目的
--------
验证 66C_RECALIBRATED_CANDIDATE_V1 (k=0.0675, cp=700, rho=1020, tau=8.0)
是否能在三个额外的数据集对上零重拟合地预测实测 Top COC 温度。

本任务为验证 ONLY: 禁止任何参数拟合 / 网格搜索 / 时移优化 / 交叉相关。

锁定参数 (运行时断言):
    k_eff  = 0.0675 W/(m K)
    cp_eff = 700 J/(kg K)
    rho    = 1020 kg/m3
    tau_top= 8.0 s  (输出侧滞后, 只作用于 Top 观测)
    h_conv = 10.0 W/(m2 K)
    epsilon= 0.90
    sigma  = 5.670374419e-8
    F_view = 1.0
    非线性 Stefan-Boltzmann 辐射
    几何    = BARE_TOP_COC_LAYERS (无 Air / PDMS)

架构:
    实测内部温度 (完整热历史, 直接 Dirichlet 底部边界)
        -> 裸顶 1D 多层 FDM
        -> raw Top COC
        -> 输出侧一阶有效滞后 tau_top=8.0 s
        -> 预测实测 Top COC

数据集
------
DS1 VALIDATION_60C_REDO:
    internal: Calibration/Recording when reach setting/
              08.17 COC top_60°C_zone1_temperature_analysis.xlsx
    top     : Calibration/extension 60°C_redo.xls
DS2 VALIDATION_72C_REDO:
    internal: Calibration/Recording when reach setting/
              08.17 COC top_72°C_zone1_temperature_analysis.xlsx
    top     : Calibration/extension 72°C_redo.xls
DS3 VALIDATION_3S_MIXED_RECORDING_START:
    internal: Calibration/Recording at the start/
              08.17 COC top_pm_3s extension_zone1_temperature_analysis.xlsx
    top     : Calibration/PCR 3s extension.xls
              (注意: 任务书中路径 "Recording when reach setting/PCR 3s
               extension.xls" 在磁盘上不存在; 实际文件位于 Calibration 根,
               与上一任务中已建立 SIMULTANEOUS_START_RELATIVE_T0 的
               "previous synchronized 3s validation" 是同一对文件。)

同步判定 (基于实验计时信息, 绝不拟合温度曲线)
----------------------------------------------
- 所有 internal 文件仅含相对时间列 (Time(s) / Relative time(s)),
  无绝对时间戳; Top 文件含绝对 RECTime。
- 因此 ABSOLUTE_TIMESTAMP 对齐对所有数据集均不可能。
- DS1 (60C) 与 DS2 (72C): internal 位于 "Recording when reach setting",
  无同项目文档建立同时启动; 按任务规则标记
      synchronization_status = UNCERTAIN
      validation_role       = DIAGNOSTIC_ONLY
      authoritative_validation = NO
  在明确声明的 SIMULTANEOUS_START_RELATIVE_T0 假设下评估 (项目惯例;
  两条迹线均从冷环境启动, 初始温度差一致), 但不计入权威验证均值。
- DS3 (3s): internal 位于 "Recording at the start" (文件夹名即记录启动语义),
  且该文件对已在上一任务中建立并文档化 SIMULTANEOUS_START_RELATIVE_T0
  (CALIBRATION_HISTORY.md, RMSE=1.0643 C)。任务书关于 "Top 位于
  Recording when reach setting" 的前提与实际文件系统不符 (文件在 Calibration
  根)。因此:
      synchronization_status = SIMULTANEOUS_START_RELATIVE_T0
      validation_role       = AUTHORITATIVE
      authoritative_validation = YES
  RMSE 应与上一任务 1.0643 C 位级一致 (回归检查)。

环境 (任务 #14):
    每个数据集 Top 迹线均从冷环境开始 (<40 C), 因此
    T_environment = 第一个有效实测 Top COC 温度 (INITIAL_MEASURED_TOP)。
初始: T_initial = 第一个有效内部温度 (均匀场), 绝不在 Top 记录起点重置。
底部: 实测内部迹线直接 Dirichlet, 完整热历史。

指标: RMSE / MAE / mean / median_abs / std / max_abs / R2;
      regime (heating/cooling/settling) + 温度带诊断 (描述性)。
分类: 仅基于权威验证 (DS3) 与参考 (66C 校准); 60C/72C 同步不确定
      时不计入权威均值, 单独作为诊断报告。

输出 (不覆盖历史):
    calibrated_model_output/66C_candidate_multi_dataset_validation_v1/
        60C/ 72C/ 3s_mixed_start/ comparison/

用法:
    uv run python validate_66C_candidate_multi_dataset.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag
from thermal_model.utilities.validate_frozen_model_two_new_bare_top_datasets import (
    load_top_series,
    load_internal_series,
    _regime_labels,
)
from thermal_model.utilities.predict_sample_from_internal_temperature import _find_column

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"
REC_START_DIR = CALIBRATION_DIR / "Recording at the start"
REC_REACH_DIR = CALIBRATION_DIR / "Recording when reach setting"

OUTPUT_ROOT = (PROJECT_ROOT / "calibrated_model_output"
               / "66C_candidate_multi_dataset_validation_v1")

SAVE_DT = 0.1

# ------------------------------------------------------------
# 锁定候选参数 (唯一事实来源; 运行时断言)
# ------------------------------------------------------------
CANDIDATE_ID = "66C_RECALIBRATED_CANDIDATE_V1"
K_EFF = 0.0675
CP_EFF = 700.0
RHO_COC = 1020.0
TAU_TOP = 8.0
H_CONV = cr.H_CONV_STRATEGY_E_W_M2K          # 10.0
EPS = cr.EMISSIVITY_STRATEGY_E               # 0.90
SIGMA = cr.SIGMA_SB_W_M2_K4                  # 5.670374419e-8
F_VIEW = cr.VIEW_FACTOR_STRATEGY_E           # 1.0

# 参考 (只读)
REF_66C_CALIB_RMSE = 0.6368
REF_3S_PREVIOUS_RMSE = 1.0643

# 质量分级 (描述性)
QUALITY_EXCELLENT = 1.5
QUALITY_GOOD = 2.5
QUALITY_MODERATE = 4.0

# 温度带诊断
TEMPERATURE_BANDS = ((0.0, 50.0, "LT50"), (50.0, 60.0, "50_60"),
                     (60.0, 70.0, "60_70"), (70.0, 1e9, "GE70"))

AMBIENT_UPPER_C = 40.0   # 判定 Top 首点是否为环境温度 (非加热后)


def assert_locked_parameters():
    """运行时断言: 参数与 66C 最优完全一致, 任何下游阶段禁止修改。"""
    checks = [
        ("k_eff", K_EFF, 0.0675),
        ("cp_eff", CP_EFF, 700.0),
        ("rho_COC", RHO_COC, 1020.0),
        ("tau_top", TAU_TOP, 8.0),
        ("h_conv", H_CONV, 10.0),
        ("epsilon", EPS, 0.90),
        ("sigma_SB", SIGMA, 5.670374419e-8),
        ("view_factor", F_VIEW, 1.0),
    ]
    for name, val, ref in checks:
        if abs(float(val) - float(ref)) > 1e-12:
            raise RuntimeError(
                f"锁定参数断言失败: {name}={val} != {ref}; 验证阶段禁止 "
                "任何参数修改。")


# ------------------------------------------------------------
# 数据集定义 (路径 + 同步元数据)
# ------------------------------------------------------------

DATASETS = {
    "VALIDATION_60C_REDO": {
        "label": "60C",
        "top_path": CALIBRATION_DIR / "extension 60°C_redo.xls",
        "int_path": REC_REACH_DIR / (
            "08.17 COC top_60°C_zone1_temperature_analysis.xlsx"),
        "internal_recording_mode": "RECORDING_WHEN_REACH_SETTING",
        "top_recording_mode": "EXTENSION_60C_REDO (Calibration root)",
        "synchronization_status": "UNCERTAIN",
        "synchronization_rule": "SIMULTANEOUS_START_RELATIVE_T0_ASSUMED",
        "validation_role": "DIAGNOSTIC_ONLY",
        "authoritative": False,
        "sync_note": ("internal 无绝对时间戳; 位于 'Recording when reach "
                      "setting'; 无同项目文档建立同时启动; 仅按项目惯例 "
                      "以相对 t0 假设评估 (两条迹线均从冷环境启动)。"),
    },
    "VALIDATION_72C_REDO": {
        "label": "72C",
        "top_path": CALIBRATION_DIR / "extension 72°C_redo.xls",
        "int_path": REC_REACH_DIR / (
            "08.17 COC top_72°C_zone1_temperature_analysis.xlsx"),
        "internal_recording_mode": "RECORDING_WHEN_REACH_SETTING",
        "top_recording_mode": "EXTENSION_72C_REDO (Calibration root)",
        "synchronization_status": "UNCERTAIN",
        "synchronization_rule": "SIMULTANEOUS_START_RELATIVE_T0_ASSUMED",
        "validation_role": "DIAGNOSTIC_ONLY",
        "authoritative": False,
        "sync_note": ("同 60C: internal 无绝对时间戳, 位于 'Recording when "
                      "reach setting', 无同时启动文档; 相对 t0 假设评估。"),
    },
    "VALIDATION_3S_MIXED_RECORDING_START": {
        "label": "3s",
        "top_path": CALIBRATION_DIR / "PCR 3s extension.xls",
        "int_path": REC_START_DIR / (
            "08.17 COC top_pm_3s extension_zone1_temperature_analysis.xlsx"),
        "internal_recording_mode": "RECORDING_AT_START",
        "top_recording_mode": ("PCR_3S_EXTENSION (Calibration root; 任务书中 "
                               "'Recording when reach setting/PCR 3s "
                               "extension.xls' 不存在)"),
        "synchronization_status": "SIMULTANEOUS_START_RELATIVE_T0",
        "synchronization_rule": "SIMULTANEOUS_START_RELATIVE_T0",
        "validation_role": "AUTHORITATIVE",
        "authoritative": True,
        "sync_note": ("internal 位于 'Recording at the start'; 该文件对与 "
                      "上一任务已建立并文档化的同步 3s 验证为同一对 "
                      "(CALIBRATION_HISTORY.md, RMSE=1.0643 C); "
                      "相对 t0 对齐, 无时移优化。"),
    },
}


# ------------------------------------------------------------
# 文件检查 (任务 #7) — 六个工作簿的结构与统计
# ------------------------------------------------------------

def inspect_internal_file(path):
    """检查 internal 工作簿 (Extracted_Data)。"""
    p = Path(path)
    df = pd.read_excel(p, sheet_name="Extracted_Data")
    tc = _find_column(df, "Time(s)")
    gc = _find_column(df, "Zone 1 Avg (°C)")
    t = pd.to_numeric(df[tc], errors="coerce").to_numpy(float)
    T = pd.to_numeric(df[gc], errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(T)
    t, T = t[ok], T[ok]
    t_rel = t - t[0]
    return {
        "path": str(p.resolve()), "sheet": "Extracted_Data",
        "time_column": tc, "temp_column": gc,
        "first_time_s": float(t[0]), "first_temp_C": float(T[0]),
        "n_valid": int(len(t)),
        "median_dt_s": float(np.median(np.diff(t_rel))),
        "duration_s": float(t_rel[-1] - t_rel[0]),
        "min_temp_C": float(np.min(T)), "max_temp_C": float(np.max(T)),
        "absolute_timestamps_available": False,
    }


def inspect_top_file(path):
    """检查 Top 工作簿 (Data, T Avg 列; 含绝对 RECTime)。"""
    p = Path(path)
    df = pd.read_excel(p, sheet_name="Data")
    rt = pd.to_datetime(df["RECTime"], errors="coerce")
    gc = _find_column(df, "T Avg")
    T = pd.to_numeric(df[gc], errors="coerce").to_numpy(float)
    t = (rt - rt.iloc[0]).dt.total_seconds().to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(T) & (T > 0)
    t, T, rtv = t[ok], T[ok], rt[ok]
    t_rel = t - t[0]
    return {
        "path": str(p.resolve()), "sheet": "Data",
        "time_column": "RECTime", "temp_column": gc,
        "first_time_s": float(t[0]),
        "first_absolute_time": str(rtv[0]),
        "first_temp_C": float(T[0]),
        "n_valid": int(len(t)),
        "median_dt_s": float(np.median(np.diff(t_rel))),
        "duration_s": float(t_rel[-1] - t_rel[0]),
        "min_temp_C": float(np.min(T)), "max_temp_C": float(np.max(T)),
        "absolute_timestamps_available": True,
    }


def build_inspection_report():
    """六文件检查报告 (CSV + 可读文本)。"""
    rows = []
    for ds_id, cfg in DATASETS.items():
        rows.append({"dataset": ds_id, "role": "internal",
                     **inspect_internal_file(cfg["int_path"])})
        rows.append({"dataset": ds_id, "role": "top",
                     **inspect_top_file(cfg["top_path"])})
    df = pd.DataFrame(rows)
    return df


# ------------------------------------------------------------
# 单数据集验证 (与上一任务评估管线逐位一致)
# ------------------------------------------------------------

def evaluate_candidate(top, internal, k_eff=K_EFF, cp_eff=CP_EFF,
                       tau_top=TAU_TOP, save_dt=SAVE_DT):
    """锁定参数裸顶 FDM + 输出侧滞后 + 插值到实测 Top 时间。

    返回 dict 含重叠窗口数组、指标、regime、温度带。
    """
    assert_locked_parameters()
    t_top = top["t_rel"]
    T_top = top["T"]
    t_int = internal["t_rel"]
    T_int = internal["T"]
    t0 = max(t_top[0], t_int[0])
    t1 = min(t_top[-1], t_int[-1])
    if t1 <= t0:
        raise ValueError("Top 与内部时间范围无重叠。")

    # 环境 = 第一个有效实测 Top (必须为环境温度, 非加热后值)
    t_env = float(T_top[0])
    if t_env >= AMBIENT_UPPER_C:
        raise ValueError(
            f"Top 首点 {t_env:.1f} C >= {AMBIENT_UPPER_C} C: 不能用作环境 "
            "温度 (INITIAL_MEASURED_TOP 规则)。")
    T_init = float(T_int[0])

    mats = cr.make_convection_radiation_materials(k_eff, cp_eff, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=save_dt, T_initial_C=T_init)
    t_arr = result["t_array"]
    T_top_fdm = result["T_top_surface_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top_fdm, tau_top)

    # 重叠窗口
    m_arr = (t_arr >= t0 - 1e-9) & (t_arr <= t1 + 1e-9)
    m_top = (t_top >= t0 - 1e-9) & (t_top <= t1 + 1e-9)
    t_arr_c = t_arr[m_arr]
    T_fdm_c = T_top_fdm[m_arr]
    T_obs_c = T_top_obs[m_arr]
    t_top_c = t_top[m_top]
    T_top_c = T_top[m_top]

    # 插值到实测 Top 时间 (查询轴 = 实测时间; 温度绝不作插值坐标)
    T_pred = np.interp(t_top_c, t_arr_c, T_obs_c)
    T_fdm_at_top = np.interp(t_top_c, t_arr_c, T_fdm_c)
    T_int_at_top = np.interp(t_top_c, t_int, T_int)

    resid = T_pred - T_top_c
    n = len(resid)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((T_top_c - np.mean(T_top_c)) ** 2))
    metrics = {
        "n_points": n,
        "RMSE_C": float(np.sqrt(np.mean(resid ** 2))),
        "MAE_C": float(np.mean(np.abs(resid))),
        "mean_residual_C": float(np.mean(resid)),
        "median_abs_residual_C": float(np.median(np.abs(resid))),
        "residual_std_C": float(np.std(resid)),
        "max_abs_residual_C": float(np.max(np.abs(resid))),
        "R_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "measured_top_min_C": float(np.min(T_top_c)),
        "measured_top_max_C": float(np.max(T_top_c)),
        "predicted_top_min_C": float(np.min(T_pred)),
        "predicted_top_max_C": float(np.max(T_pred)),
    }

    # regime 诊断
    regimes = _regime_labels(t_top_c, T_int_at_top, T_top_c)
    regime_metrics = {}
    for short, rg in (("heating", "TRANSIENT_HEATING"),
                      ("cooling", "TRANSIENT_COOLING"),
                      ("settling", "SETTLING")):
        m = regimes == rg
        if m.sum():
            regime_metrics[f"{short}_n"] = int(m.sum())
            regime_metrics[f"{short}_RMSE_C"] = float(
                np.sqrt(np.mean(resid[m] ** 2)))
            regime_metrics[f"{short}_mean_residual_C"] = float(
                np.mean(resid[m]))
        else:
            regime_metrics[f"{short}_n"] = 0
            regime_metrics[f"{short}_RMSE_C"] = np.nan
            regime_metrics[f"{short}_mean_residual_C"] = np.nan

    # 温度带诊断
    band_metrics = {}
    for lo, hi, name in TEMPERATURE_BANDS:
        m = (T_top_c >= lo) & (T_top_c < hi)
        if m.sum():
            band_metrics[f"{name}_n"] = int(m.sum())
            band_metrics[f"{name}_RMSE_C"] = float(
                np.sqrt(np.mean(resid[m] ** 2)))
            band_metrics[f"{name}_mean_residual_C"] = float(
                np.mean(resid[m]))
        else:
            band_metrics[f"{name}_n"] = 0
            band_metrics[f"{name}_RMSE_C"] = np.nan
            band_metrics[f"{name}_mean_residual_C"] = np.nan

    return {
        "t_overlap_s": float(t1 - t0),
        "t_top": t_top_c,
        "T_top_measured": T_top_c,
        "T_internal_at_top": T_int_at_top,
        "T_top_fdm_raw": T_fdm_at_top,
        "T_top_predicted_lagged": T_pred,
        "residual": resid,
        "regimes": regimes,
        "T_env_C": t_env,
        "T_initial_C": T_init,
        "environment_source": "INITIAL_MEASURED_TOP",
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "band_metrics": band_metrics,
    }


# ------------------------------------------------------------
# 输出: 每数据集
# ------------------------------------------------------------

def write_dataset_outputs(ds_id, cfg, top, internal, ev, top_abs_time):
    """写入 60C/72C/3s_mixed_start 目录文件。"""
    d = OUTPUT_ROOT / ds_id.split("_", 1)[1].lower().replace("_redo", "")
    if ds_id == "VALIDATION_3S_MIXED_RECORDING_START":
        d = OUTPUT_ROOT / "3s_mixed_start"
    d.mkdir(parents=True, exist_ok=True)

    label = cfg["label"]
    m = ev["metrics"]

    # ---- validation_trace.csv ----
    trace = pd.DataFrame({
        "measured_top_time_s": ev["t_top"],
        "measured_top_C": ev["T_top_measured"],
        "internal_interpolated_C": ev["T_internal_at_top"],
        "top_FDM_raw_C": ev["T_top_fdm_raw"],
        "top_predicted_lagged_C": ev["T_top_predicted_lagged"],
        "residual_C": ev["residual"],
    })
    if top_abs_time is not None:
        # 绝对时间戳 (Top RECTime) 与重叠窗口的实测 Top 时间对齐
        t_top_full = top["t_rel"]
        idx = np.searchsorted(t_top_full, ev["t_top"])
        abs_t = np.asarray([str(x) for x in top_abs_time[idx]])
        trace.insert(0, "absolute_timestamp", abs_t)
        trace.insert(1, "simulation_time_since_internal_start_s",
                     ev["t_top"])
        trace.insert(2, "top_recording_relative_time_s", ev["t_top"])
    trace.to_csv(d / "validation_trace.csv", index=False)

    # ---- validation_summary.csv ----
    summary = {
        "dataset_id": ds_id,
        "label": label,
        "dataset_role": "VALIDATION",
        "authoritative_validation": cfg["authoritative"],
        "synchronization_status": cfg["synchronization_status"],
        "synchronization_rule": cfg["synchronization_rule"],
        "fitted_parameters": False,
        "k_eff": K_EFF, "cp_eff": CP_EFF, "tau_top": TAU_TOP,
        "h_conv": H_CONV, "epsilon": EPS,
        "n_top_points": m["n_points"],
        "overlapping_duration_s": ev["t_overlap_s"],
        **m,
        **ev["regime_metrics"],
        **{f"band_{k}": v for k, v in ev["band_metrics"].items()},
        "environment_C": ev["T_env_C"],
        "environment_source": ev["environment_source"],
        "T_initial_C": ev["T_initial_C"],
        "time_shift_optimized": False,
        "time_shift_applied_s": 0.0,
        "internal_source": str(cfg["int_path"].resolve()),
        "top_source": str(cfg["top_path"].resolve()),
        "validation_quality": quality_label(m["RMSE_C"]),
    }
    pd.DataFrame([summary]).to_csv(d / "validation_summary.csv", index=False)

    # ---- top_COC_validation 图 ----
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(ev["t_top"], ev["T_internal_at_top"], color="#7f7f7f", lw=1.1,
            ls=":", label="Measured internal temperature")
    ax.plot(ev["t_top"], ev["T_top_measured"], color="#1f77b4", lw=1.6,
            label="Corrected measured Top COC")
    ax.plot(ev["t_top"], ev["T_top_predicted_lagged"], color="#d62728",
            lw=1.8, label="Predicted Top COC (66C candidate, no refit)")
    ax.plot(ev["t_top"], ev["T_top_fdm_raw"], color="#2ca02c", lw=0.9,
            ls="--", alpha=0.7, label="Raw Top COC FDM (no lag)")
    ax.set_xlabel("Elapsed time since internal recording start [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(
        f"{ds_id} — zero-refit Top COC validation (66C candidate)\n"
        f"(k={K_EFF}, cp={CP_EFF:.0f}, tau={TAU_TOP:.1f} s; h={H_CONV} + "
        "nonlinear radiation; BARE TOP)\n"
        f"sync={cfg['synchronization_status']} "
        f"({cfg['synchronization_rule']}); role={cfg['validation_role']}")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(d / "top_COC_validation.png", dpi=150)
    fig.savefig(d / "top_COC_validation.pdf")
    plt.close(fig)

    # ---- residual_vs_time 图 ----
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.plot(ev["t_top"], ev["residual"], color="#d62728", lw=1.0,
            label="Residual (predicted - measured)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Elapsed time since internal recording start [s]")
    ax.set_ylabel("Residual [C]")
    ax.set_title(f"{ds_id} — residual vs time (zero-refit 66C candidate)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "residual_vs_time.png", dpi=150)
    plt.close(fig)

    return summary


def quality_label(rmse):
    if rmse <= QUALITY_EXCELLENT:
        return "EXCELLENT"
    if rmse <= QUALITY_GOOD:
        return "GOOD"
    if rmse <= QUALITY_MODERATE:
        return "MODERATE"
    return "POOR"


# ------------------------------------------------------------
# Top 绝对时间戳 (供 trace CSV)
# ------------------------------------------------------------

def load_top_absolute_times(path):
    """返回与 load_top_series 相同有效/去重规则下的绝对时间戳数组。"""
    df = pd.read_excel(path, sheet_name="Data")
    rt = pd.to_datetime(df["RECTime"], errors="coerce")
    T = pd.to_numeric(df["T Avg"], errors="coerce").to_numpy(float)
    t = (rt - rt.iloc[0]).dt.total_seconds().to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(T) & (T > 0)
    rt = rt[ok]
    t = t[ok] - t[ok][0]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return rt[keep]


# ------------------------------------------------------------
# 组合输出
# ------------------------------------------------------------

def combined_summary_rows(results, inspection_df):
    """多数据集验证汇总表 (CSV)。"""
    rows = []
    # 66C 校准参考 (只读)
    rows.append({
        "dataset_id": "CALIBRATION_66C",
        "dataset_role": "CALIBRATION",
        "authoritative_validation": True,
        "synchronization_status": "SIMULTANEOUS_START_RELATIVE_T0",
        "synchronization_rule": "SIMULTANEOUS_START_RELATIVE_T0",
        "fitted_parameters": True,
        "k_eff": K_EFF, "cp_eff": CP_EFF, "tau_top": TAU_TOP,
        "h_conv": H_CONV, "epsilon": EPS,
        "n_top_points": np.nan, "overlapping_duration_s": np.nan,
        "RMSE_C": REF_66C_CALIB_RMSE, "MAE_C": np.nan,
        "mean_residual_C": np.nan, "median_abs_residual_C": np.nan,
        "max_abs_residual_C": np.nan, "R_squared": np.nan,
        "heating_RMSE_C": np.nan, "cooling_RMSE_C": np.nan,
        "settling_RMSE_C": np.nan,
        "note": "read-only reference from 66C recalibration task",
    })
    # 上一任务同步 3s 验证参考 (只读; 与 DS3 同对文件)
    rows.append({
        "dataset_id": "VALIDATION_3S_PREVIOUS_CORRECTED",
        "dataset_role": "VALIDATION_REFERENCE",
        "authoritative_validation": True,
        "synchronization_status": "SIMULTANEOUS_START_RELATIVE_T0",
        "synchronization_rule": "SIMULTANEOUS_START_RELATIVE_T0",
        "fitted_parameters": False,
        "k_eff": K_EFF, "cp_eff": CP_EFF, "tau_top": TAU_TOP,
        "h_conv": H_CONV, "epsilon": EPS,
        "n_top_points": np.nan, "overlapping_duration_s": np.nan,
        "RMSE_C": REF_3S_PREVIOUS_RMSE, "MAE_C": np.nan,
        "mean_residual_C": np.nan, "median_abs_residual_C": np.nan,
        "max_abs_residual_C": np.nan, "R_squared": np.nan,
        "heating_RMSE_C": np.nan, "cooling_RMSE_C": np.nan,
        "settling_RMSE_C": np.nan,
        "note": "read-only reference from previous task (same file pair "
                "as DS3); regression check only",
    })
    # 当前三数据集
    for ds_id in ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO",
                  "VALIDATION_3S_MIXED_RECORDING_START"):
        r = results[ds_id]["summary"]
        rows.append({
            "dataset_id": ds_id,
            "dataset_role": "VALIDATION",
            "authoritative_validation": r["authoritative_validation"],
            "synchronization_status": r["synchronization_status"],
            "synchronization_rule": r["synchronization_rule"],
            "fitted_parameters": False,
            "k_eff": r["k_eff"], "cp_eff": r["cp_eff"],
            "tau_top": r["tau_top"], "h_conv": r["h_conv"],
            "epsilon": r["epsilon"],
            "n_top_points": r["n_top_points"],
            "overlapping_duration_s": r["overlapping_duration_s"],
            "RMSE_C": r["RMSE_C"], "MAE_C": r["MAE_C"],
            "mean_residual_C": r["mean_residual_C"],
            "median_abs_residual_C": r["median_abs_residual_C"],
            "max_abs_residual_C": r["max_abs_residual_C"],
            "R_squared": r["R_squared"],
            "heating_RMSE_C": r["heating_RMSE_C"],
            "cooling_RMSE_C": r["cooling_RMSE_C"],
            "settling_RMSE_C": r["settling_RMSE_C"],
            "note": r["synchronization_note"]
            if "synchronization_note" in r else "",
        })
    return pd.DataFrame(rows)


def plot_rmse_overview(results):
    """RMSE 总览图: 权威 = 实心; 诊断 = 空心/条纹; 参考 = 灰色。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    order = [
        ("CALIBRATION_66C", REF_66C_CALIB_RMSE, "calibration", "#1f77b4",
         True),
        ("VALIDATION_3S_PREVIOUS", REF_3S_PREVIOUS_RMSE,
         "validation (ref)", "#2ca02c", True),
        ("VALIDATION_60C_REDO", results["VALIDATION_60C_REDO"]["summary"][
            "RMSE_C"], "diagnostic (60C)", "#ff7f0e", False),
        ("VALIDATION_72C_REDO", results["VALIDATION_72C_REDO"]["summary"][
            "RMSE_C"], "diagnostic (72C)", "#ff7f0e", False),
        ("VALIDATION_3S_MIXED", results[
            "VALIDATION_3S_MIXED_RECORDING_START"]["summary"]["RMSE_C"],
         "validation (3s)", "#2ca02c", True),
    ]
    labels = []
    for name, rmse, role, color, authoritative in order:
        labels.append(f"{name}\n({role})")
        if authoritative:
            ax.bar(name, rmse, color=color, width=0.55,
                   label="authoritative" if role == "calibration" else None)
        else:
            ax.bar(name, rmse, color=color, width=0.55, hatch="//",
                   alpha=0.85, label="diagnostic (sync uncertain)")
    for name, rmse, *_ in order:
        ax.text(name, rmse + 0.03, f"{rmse:.3f}", ha="center", fontsize=9)
    ax.axhline(QUALITY_GOOD, color="gray", ls=":", lw=1,
               label=f"GOOD boundary ({QUALITY_GOOD} C)")
    ax.set_ylabel("Top COC RMSE [C]")
    ax.set_ylim(0, max(REF_66C_CALIB_RMSE, REF_3S_PREVIOUS_RMSE,
                       results["VALIDATION_60C_REDO"]["summary"]["RMSE_C"],
                       results["VALIDATION_72C_REDO"]["summary"]["RMSE_C"],
                       results["VALIDATION_3S_MIXED_RECORDING_START"][
                           "summary"]["RMSE_C"]) * 1.35 + 0.5)
    ax.set_title("66C candidate — multi-dataset zero-refit Top COC RMSE\n"
                 "(hatched = diagnostic, synchronization uncertain)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "comparison" / "multi_dataset_RMSE_overview.png",
                dpi=150)
    plt.close(fig)


def plot_mean_residual_vs_temperature(results):
    """跨温度平均残差 (任务 #33)。"""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pts = [
        ("60C (diagnostic)", results["VALIDATION_60C_REDO"]["summary"][
            "mean_residual_C"], "#ff7f0e", False),
        ("66C (calibration ref)", 0.0316, "#1f77b4", True),
        ("72C (diagnostic)", results["VALIDATION_72C_REDO"]["summary"][
            "mean_residual_C"], "#ff7f0e", False),
        ("3s (authoritative)", results[
            "VALIDATION_3S_MIXED_RECORDING_START"]["summary"][
            "mean_residual_C"], "#2ca02c", True),
    ]
    xs = range(len(pts))
    for x, (label, mr, color, auth) in enumerate(pts):
        marker = "o" if auth else "D"
        ax.plot(x, mr, marker, color=color, ms=9 if auth else 8,
                label=f"{label} ({'+' if mr >= 0 else ''}{mr:.3f} C)")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([p[0].split(" ")[0] for p in pts])
    ax.set_xlabel("Protocol / nominal temperature level")
    ax.set_ylabel("Mean residual (predicted - measured) [C]")
    ax.set_title("Cross-temperature mean residual — 66C candidate\n"
                 "(diagnostic markers = synchronization uncertain)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "comparison" /
                "mean_residual_vs_temperature.png", dpi=150)
    plt.close(fig)


def plot_measured_vs_predicted(results):
    """权威验证点实测 vs 预测 (y=x 参考, 不拟合回归)。"""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = {"VALIDATION_60C_REDO": "#ff7f0e",
              "VALIDATION_72C_REDO": "#9467bd",
              "VALIDATION_3S_MIXED_RECORDING_START": "#2ca02c"}
    for ds_id in ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO",
                  "VALIDATION_3S_MIXED_RECORDING_START"):
        ev = results[ds_id]["ev"]
        ls = "o" if results[ds_id]["cfg"]["authoritative"] else "x"
        ax.scatter(ev["T_top_measured"], ev["T_top_predicted_lagged"],
                   s=8 if results[ds_id]["cfg"]["authoritative"] else 14,
                   marker=ls[0], alpha=0.5, color=colors[ds_id],
                   label=results[ds_id]["cfg"]["label"]
                   + (" (auth)" if results[ds_id]["cfg"]["authoritative"]
                      else " (diag)"))
    lo = min(np.min(results[ds_id]["ev"]["T_top_measured"]) for ds_id in
             ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO",
              "VALIDATION_3S_MIXED_RECORDING_START"))
    hi = max(np.max(results[ds_id]["ev"]["T_top_measured"]) for ds_id in
             ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO",
              "VALIDATION_3S_MIXED_RECORDING_START"))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Corrected measured Top COC [C]")
    ax.set_ylabel("Predicted Top COC [C]")
    ax.set_title("Measured vs predicted Top COC — 66C candidate\n"
                 "(no regression fitted)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "comparison" /
                "measured_vs_predicted_top_COC.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# 分类 (任务 #25-26)
# ------------------------------------------------------------

def classify_transfer(results):
    """基于 ONLY 热学验证的转移分类。

    规则:
      - 60C/72C 同步不确定 -> 不计入权威均值 (任务 #26)。
      - 若可客观同步的额外数据集 (3s) 与参考均低误差且无系统性温度偏置
         -> 依据 RMSE 判定 STRONG / ACCEPTABLE。
      - 若附加数据集中 >=2 个无法客观同步 -> INSUFFICIENT_SYNCHRONIZED_DATA。
      - 若权威验证误差随温度系统性恶化 -> TEMPERATURE_DEPENDENT_...
    """
    auth_rmse = []
    for ds_id, cfg in DATASETS.items():
        if cfg["authoritative"]:
            auth_rmse.append(results[ds_id]["summary"]["RMSE_C"])

    uncertain = [ds_id for ds_id, cfg in DATASETS.items()
                 if cfg["synchronization_status"] == "UNCERTAIN"]

    if len(uncertain) >= 2:
        cls = "INSUFFICIENT_SYNCHRONIZED_DATA"
        reason = (f"{len(uncertain)} of 3 additional datasets "
                  f"({', '.join(sorted(uncertain))}) cannot be objectively "
                  "synchronized (internal files lack absolute timestamps; no "
                  "documented simultaneous start). Only the 3s pair is "
                  "authoritative. Diagnostic RMSE under the stated "
                  "relative-t0 assumption: "
                  + "; ".join(f"{cfg['label']} "
                              f"{results[ds_id]['summary']['RMSE_C']:.3f} C"
                              for ds_id, cfg in DATASETS.items()
                              if cfg["synchronization_status"] == "UNCERTAIN")
                  + ".")
        return cls, reason, {
            "authoritative_validation_datasets":
                [ds_id for ds_id, cfg in DATASETS.items()
                 if cfg["authoritative"]],
            "uncertain_datasets": uncertain,
            "mean_authoritative_validation_RMSE_C":
                float(np.mean(auth_rmse)) if auth_rmse else np.nan,
            "worst_authoritative_validation_RMSE_C":
                float(np.max(auth_rmse)) if auth_rmse else np.nan,
        }

    if auth_rmse:
        mean_auth = float(np.mean(auth_rmse))
        worst_auth = float(np.max(auth_rmse))
        if worst_auth <= QUALITY_EXCELLENT:
            cls = "STRONG_MULTI_DATASET_TRANSFER"
            reason = (f"all authoritative validation RMSE <= "
                      f"{QUALITY_EXCELLENT} C (3s {auth_rmse[0]:.3f} C); "
                      "no systematic cross-temperature bias")
        elif worst_auth <= QUALITY_GOOD:
            cls = "ACCEPTABLE_MULTI_DATASET_TRANSFER"
            reason = (f"authoritative validation RMSE within a few degrees "
                      f"(3s {auth_rmse[0]:.3f} C <= {QUALITY_GOOD} C); "
                      "no major systematic breakdown")
        else:
            cls = "TEMPERATURE_DEPENDENT_TRANSFER_LIMITATION"
            reason = (f"authoritative validation error exceeds "
                      f"{QUALITY_GOOD} C (worst {worst_auth:.3f} C)")
        return cls, reason, {
            "authoritative_validation_datasets":
                [ds_id for ds_id, cfg in DATASETS.items()
                 if cfg["authoritative"]],
            "uncertain_datasets": uncertain,
            "mean_authoritative_validation_RMSE_C": mean_auth,
            "worst_authoritative_validation_RMSE_C": worst_auth,
        }

    return ("INSUFFICIENT_SYNCHRONIZED_DATA",
            "no objectively synchronized additional validation datasets",
            {"authoritative_validation_datasets": [],
             "uncertain_datasets": uncertain,
             "mean_authoritative_validation_RMSE_C": np.nan,
             "worst_authoritative_validation_RMSE_C": np.nan})


def cross_temperature_bias(results):
    """跨温度偏置分析 (任务 #24)。"""
    rows = []
    for ds_id, cfg in DATASETS.items():
        s = results[ds_id]["summary"]
        rows.append({
            "dataset": ds_id, "label": cfg["label"],
            "authoritative": cfg["authoritative"],
            "mean_residual_C": s["mean_residual_C"],
            "RMSE_C": s["RMSE_C"],
            "measured_top_max_C": s["measured_top_max_C"],
        })
    # 66C 校准参考 mean residual +0.0316 (来自上一任务)
    rows.append({"dataset": "CALIBRATION_66C", "label": "66C",
                 "authoritative": True, "mean_residual_C": 0.0316,
                 "RMSE_C": REF_66C_CALIB_RMSE, "measured_top_max_C": np.nan})
    df = pd.DataFrame(rows)
    auth = df[df["authoritative"]].sort_values("label")
    bias = "INCONCLUSIVE"
    evidence = []
    if len(auth) >= 2:
        diffs = np.diff(auth["mean_residual_C"].to_numpy())
        if np.all(diffs > 0.3):
            bias = "YES (mean residual increases with temperature level)"
        elif np.all(diffs < -0.3):
            bias = "YES (mean residual decreases with temperature level)"
        else:
            bias = "NO (no monotonic systematic bias across temperature)"
    for _, r in df.iterrows():
        evidence.append(
            f"{r['label']}: mean residual "
            f"{r['mean_residual_C']:+.3f} C (RMSE {r['RMSE_C']:.3f} C, "
            f"{'authoritative' if r['authoritative'] else 'diagnostic'})")
    return {"bias_verdict": bias, "evidence": evidence}


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    assert_locked_parameters()
    comp = OUTPUT_ROOT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)

    # ---- 六文件检查报告 ----
    insp = build_inspection_report()
    insp.to_csv(comp / "file_inspection_report.csv", index=False)
    print("=== FILE INSPECTION REPORT ===")
    print(insp.to_string(index=False))

    # ---- 每数据集验证 ----
    results = {}
    for ds_id, cfg in DATASETS.items():
        print(f"\n=== {ds_id} ({cfg['label']}) ===")
        top = load_top_series(cfg["top_path"])
        internal = load_internal_series(cfg["int_path"])
        ev = evaluate_candidate(top, internal)
        top_abs = load_top_absolute_times(cfg["top_path"])
        summary = write_dataset_outputs(ds_id, cfg, top, internal, ev,
                                        top_abs)
        # 追加同步说明
        summary["synchronization_note"] = cfg["sync_note"]
        print(f"  Top valid={top['n_valid']} int valid={internal['n_valid']}")
        print(f"  sync={cfg['synchronization_status']} "
              f"({cfg['synchronization_rule']}); role={cfg['validation_role']}")
        print(f"  RMSE={ev['metrics']['RMSE_C']:.4f} | "
              f"MAE={ev['metrics']['MAE_C']:.4f} | "
              f"mean={ev['metrics']['mean_residual_C']:+.4f} | "
              f"R2={ev['metrics']['R_squared']:.4f}")
        print(f"  regime: heating={ev['regime_metrics']['heating_RMSE_C']:.3f} "
              f"cooling={ev['regime_metrics']['cooling_RMSE_C']:.3f} "
              f"settling={ev['regime_metrics']['settling_RMSE_C']:.3f}")
        results[ds_id] = {"cfg": cfg, "ev": ev, "summary": summary,
                          "top": top, "internal": internal}

    # ---- 组合表 ----
    comb = combined_summary_rows(results, insp)
    comb.to_csv(comp / "multi_dataset_validation_summary.csv", index=False)

    # ---- 图 ----
    plot_rmse_overview(results)
    plot_mean_residual_vs_temperature(results)
    plot_measured_vs_predicted(results)

    # ---- 跨温度偏置 ----
    bias = cross_temperature_bias(results)

    # ---- 分类 ----
    cls, reason, cls_detail = classify_transfer(results)

    # ---- 元数据 ----
    meta = {
        "candidate_id": CANDIDATE_ID,
        "parameters": {"k_eff_W_mK": K_EFF, "cp_eff_J_kgK": CP_EFF,
                       "rho_COC": RHO_COC, "tau_top_s": TAU_TOP,
                       "h_conv_W_m2K": H_CONV, "emissivity": EPS,
                       "sigma_SB": SIGMA, "view_factor": F_VIEW,
                       "radiation": "nonlinear Stefan-Boltzmann",
                       "geometry": "BARE_TOP_COC_LAYERS",
                       "parameters_changed_during_validation": False},
        "reference": {"66C_calibration_RMSE_C": REF_66C_CALIB_RMSE,
                      "previous_3s_RMSE_C": REF_3S_PREVIOUS_RMSE},
        "synchronization": {ds_id: {
            "status": cfg["synchronization_status"],
            "rule": cfg["synchronization_rule"],
            "role": cfg["validation_role"],
            "authoritative": cfg["authoritative"],
            "note": cfg["sync_note"],
            "top_path": str(cfg["top_path"].resolve()),
            "internal_path": str(cfg["int_path"].resolve()),
        } for ds_id, cfg in DATASETS.items()},
        "no_parameter_fitting": True,
        "no_grid_search": True,
        "no_time_shift_optimization": True,
        "no_cross_correlation": True,
        "no_qPCR_or_sample_information_used": True,
        "classification": {"class": cls, "reason": reason, **cls_detail},
        "cross_temperature_bias": bias,
        "per_dataset": {ds_id: {
            "RMSE_C": results[ds_id]["summary"]["RMSE_C"],
            "MAE_C": results[ds_id]["summary"]["MAE_C"],
            "mean_residual_C": results[ds_id]["summary"][
                "mean_residual_C"],
            "R_squared": results[ds_id]["summary"]["R_squared"],
            "validation_quality": results[ds_id]["summary"][
                "validation_quality"],
        } for ds_id in DATASETS},
    }
    (comp / "validation_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = _summary_text(results, comb, cls, reason, bias)
    (comp / "validation_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


def _summary_text(results, comb, cls, reason, bias):
    L = []
    A = L.append
    A("=" * 74)
    A("66C CANDIDATE — MULTI-DATASET ZERO-REFIT VALIDATION SUMMARY")
    A("=" * 74)
    A(f"candidate: {CANDIDATE_ID}; k={K_EFF}, cp={CP_EFF:.0f}, "
      f"rho={RHO_COC}, tau={TAU_TOP} s; h={H_CONV} + nonlinear radiation")
    A(f"reference: 66C calib RMSE {REF_66C_CALIB_RMSE} C; previous "
      f"synchronized 3s RMSE {REF_3S_PREVIOUS_RMSE} C")
    A("")
    for ds_id, cfg in DATASETS.items():
        s = results[ds_id]["summary"]
        A(f"[{ds_id}] ({cfg['label']})")
        A(f"  sync: {cfg['synchronization_status']} "
          f"({cfg['synchronization_rule']}); role={cfg['validation_role']}; "
          f"authoritative={cfg['authoritative']}")
        A(f"  RMSE {s['RMSE_C']:.4f} | MAE {s['MAE_C']:.4f} | "
          f"mean {s['mean_residual_C']:+.4f} | "
          f"median_abs {s['median_abs_residual_C']:.4f} | "
          f"max_abs {s['max_abs_residual_C']:.4f} | "
          f"R2 {s['R_squared']:.4f}")
        A(f"  regime: heating {s['heating_RMSE_C']:.3f} "
          f"(n={s['heating_n']}), cooling {s['cooling_RMSE_C']:.3f} "
          f"(n={s['cooling_n']}), settling {s['settling_RMSE_C']:.3f} "
          f"(n={s['settling_n']})")
        A(f"  measured Top range [{s['measured_top_min_C']:.2f}, "
          f"{s['measured_top_max_C']:.2f}] C; quality: "
          f"{s['validation_quality']}")
        A(f"  note: {cfg['sync_note']}")
        A("")
    A(f"classification: {cls}")
    A(f"reason: {reason}")
    A("")
    A(f"cross-temperature bias: {bias['bias_verdict']}")
    for e in bias["evidence"]:
        A(f"  {e}")
    A("")
    A("No parameter fitting: CONFIRMED | No time-shift optimization: "
      "CONFIRMED | No qPCR/sample information used: CONFIRMED")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    sys.exit(main())
