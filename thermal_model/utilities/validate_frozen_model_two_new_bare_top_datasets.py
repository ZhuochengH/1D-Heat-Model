#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冻结裸顶热模型 — 两个修正后同步数据集的样本外验证 (从零重建)
=================================================================

背景修正:
    此前的外部验证使用了未经温度转换/修正的 Top COC 源文件, 结果无效
    (伪尖峰/掉线伪影/伪 RMSE ~12-14 C)。实验者已修正 Top COC 数据
    (同路径, 现为与 72C 校准一致的 T Avg 列)。

本脚本从零重建, 基于**修正后的原始 Top COC 数据**:
    - 不做任何伪影过滤 (无 spike/median/clean-band 过滤);
    - 不做基于模型分歧的点剔除;
    - 不做时移优化 / 交叉相关;
    - 唯一排除: 结构性无效值 (NaN / 空 / 非数字), 以及文件中 RECTime
      缺失的尾部残留空行 (T Avg=0 且无时间戳)。

冻结模型 (绝对不改):
    architecture   : OUTPUT-SIDE EFFECTIVE LAG
    k_eff          : 0.055 W/(m K)
    cp_eff         : 1200 J/(kg K)
    rho_COC        : 1020 kg/m3
    tau_top        : 8.5 s
    h_conv         : 10 W/(m2 K)
    epsilon        : 0.90
    sigma_SB       : 5.670374419e-8
    F_view         : 1.0
    辐射            : 非线性 Stefan-Boltzmann
    几何            : BARE_TOP_COC_LAYERS

数据集 (同路径, 修正后数据):
    DS1 66C redo      : Calibration/extension 66°C_redo.xls
    DS1 internal      : Calibration/Recording at the start/
                        08.17 COC top_66°C_zone1_temperature_analysis.xlsx
    DS2 PCR 3s ext    : Calibration/PCR 3s extension.xls
    DS2 internal      : Calibration/Recording at the start/
                        08.17 COC top_pm_3s extension_zone1_temperature_analysis.xlsx

同步 (SIMULTANEOUS_START_RELATIVE_T0):
    各自首点 = t=0; applied_time_shift_s = 0.0; 无任何附加时间校正。

环境: 第一个有效修正后实测 Top 温度 (INITIAL_MEASURED_TOP)。
初始: 第一个有效实测内部温度 (均匀场)。
底部边界: 实测内部迹线直接 Dirichlet。

主验证: 修正后实测 Top COC (原始有效值) vs 滞后预测 Top COC。
    residual = predicted - measured; 主指标 RMSE。

输出 (gitignored, 完全重建):
    calibrated_model_output/frozen_output_lag_external_validation_v1/
        66C_redo/     validation_trace.csv + trace png/pdf + residual png
        PCR_3s_extension/ (同上)
        comparison/   validation_summary.csv + overview png +
                      measured_vs_predicted_top_COC.png +
                      external_validation_metadata.json +
                      external_validation_summary.txt

元数据记录:
    previous_validation_invalidated = true
    reason = TOP_COC_SOURCE_TEMPERATURE_NOT_PREVIOUSLY_CONVERTED
    current_validation_uses_corrected_top_data = true
    no_artifact_filtering = true
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
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE
from thermal_model.utilities.predict_sample_from_internal_temperature import load_internal_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"
REC_START_DIR = CALIBRATION_DIR / "Recording at the start"

DS1_TOP = CALIBRATION_DIR / "extension 66°C_redo.xls"
DS1_INT = REC_START_DIR / (
    "08.17 COC top_66°C_zone1_temperature_analysis.xlsx")
DS2_TOP = CALIBRATION_DIR / "PCR 3s extension.xls"
DS2_INT = REC_START_DIR / (
    "08.17 COC top_pm_3s extension_zone1_temperature_analysis.xlsx")

OUTPUT_ROOT = (PROJECT_ROOT / "calibrated_model_output"
               / "frozen_output_lag_external_validation_v1")

SAVE_DT = 0.1

# 冻结参数
K_EFF = FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK
CP_EFF = FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK
RHO_COC = FROZEN_STRATEGY_G_CANDIDATE.rho_COC_kg_m3
TAU_TOP = FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s
H_CONV = cr.H_CONV_STRATEGY_E_W_M2K
EPS = cr.EMISSIVITY_STRATEGY_E
SIGMA = cr.SIGMA_SB_W_M2_K4
F_VIEW = cr.VIEW_FACTOR_STRATEGY_E

# 校准参考 (只读)
CALIB_RMSE_72C = 0.8891597125869538


# ============================================================
# 加载修正后 Top (T Avg 列, 与 72C 校准格式一致)
# ============================================================

def load_top_series(path):
    """加载修正后 Top COC 序列 (T Avg 列)。

    仅排除结构性无效值:
        - NaN / 非数字 T Avg;
        - 无有效时间戳的行 (RECTime 缺失 — 文件中尾部残留空行);
        - 非正温度 (T Avg <= 0, 无时间戳空行残留)。
    绝不做基于模型分歧的过滤。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Top 文件不存在: {p}")
    df = pd.read_excel(p, sheet_name="Data")
    if "T Avg" not in df.columns:
        raise KeyError(f"{p}: 修正后文件缺少 'T Avg' 列; 可用: {list(df.columns)}")
    rt = pd.to_datetime(df["RECTime"], errors="coerce")
    T_avg = pd.to_numeric(df["T Avg"], errors="coerce").to_numpy(float)
    t_abs = (rt - rt.iloc[0]).dt.total_seconds().to_numpy(float)
    # 结构性有效: 时间戳存在 且 温度有限 且 温度 > 0 (排除无时间戳残留空行)
    ok = np.isfinite(t_abs) & np.isfinite(T_avg) & (T_avg > 0.0)
    t_rel = t_abs[ok]
    T = T_avg[ok]
    if len(t_rel) < 2:
        raise ValueError(f"{p}: 有效 Top 数据点不足 ({len(t_rel)})。")
    t_rel = t_rel - t_rel[0]
    keep = np.concatenate([[True], np.diff(t_rel) > 0])
    t_rel = t_rel[keep]
    T = T[keep]
    return {
        "t_rel": t_rel,
        "T": T,
        "n_valid": int(len(T)),
        "median_dt": float(np.median(np.diff(t_rel))),
        "first_value_C": float(T[0]),
        "last_value_C": float(T[-1]),
        "duration_s": float(t_rel[-1] - t_rel[0]),
        "source": str(p.resolve()),
        "column": "T Avg (corrected)",
        "time_source": "RECTime -> relative seconds",
    }


def load_internal_series(path):
    """加载内部温度序列 (Extracted_Data, 复用权威加载器)。"""
    data = load_internal_data(path, sheet="Extracted_Data",
                              time_col="Time(s)",
                              temp_col="Zone 1 Avg (°C)")
    t = data["source_time_s"]
    T = data["T_internal_C"]
    t_rel = t - t[0]
    return {
        "t_rel": t_rel,
        "T": T,
        "n_valid": int(data["n_valid"]),
        "median_dt": float(data["median_dt"]),
        "first_value_C": float(T[0]),
        "last_value_C": float(T[-1]),
        "duration_s": float(t_rel[-1] - t_rel[0]),
        "source": str(Path(path).resolve()),
        "column": "Zone 1 Avg (°C)",
        "time_source": "Time(s) -> relative seconds",
    }


# ============================================================
# 冻结模型前向
# ============================================================

def run_frozen_validation(top, internal, save_dt=SAVE_DT):
    """同步 + FDM + 输出侧滞后 + 预测到 Top 测量时间 (无伪影过滤)。"""
    t_top = top["t_rel"]
    T_top = top["T"]
    t_int = internal["t_rel"]
    T_int = internal["T"]

    t0 = max(t_top[0], t_int[0])
    t1 = min(t_top[-1], t_int[-1])
    if t1 <= t0:
        raise ValueError("Top 与内部时间范围无重叠。")

    t_env = float(T_top[0])
    T_init = float(T_int[0])

    # 冻结参数运行时断言
    for name, val, ref in (("k_eff", K_EFF, 0.055),
                           ("cp_eff", CP_EFF, 1200.0),
                           ("tau_top", TAU_TOP, 8.5),
                           ("h_conv", H_CONV, 10.0),
                           ("epsilon", EPS, 0.90)):
        if abs(float(val) - float(ref)) > 1e-12:
            raise RuntimeError(f"冻结参数断言失败: {name}={val} != {ref}")

    mats = cr.make_convection_radiation_materials(K_EFF, CP_EFF, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=save_dt, T_initial_C=T_init)
    t_arr = result["t_array"]
    T_top_fdm = result["T_top_surface_arr"]

    # 输出侧滞后 (tau=8.5, 仅顶部观测)
    T_top_obs = apply_first_order_lag(t_arr, T_top_fdm, TAU_TOP)

    # 重叠区间裁剪
    m_arr = (t_arr >= t0 - 1e-9) & (t_arr <= t1 + 1e-9)
    m_top = (t_top >= t0 - 1e-9) & (t_top <= t1 + 1e-9)
    t_arr_c = t_arr[m_arr]
    T_fdm_c = T_top_fdm[m_arr]
    T_obs_c = T_top_obs[m_arr]
    t_top_c = t_top[m_top]
    T_top_c = T_top[m_top]

    # 预测插值到实测 Top 时间 (查询轴 = 实测时间)
    T_pred = np.interp(t_top_c, t_arr_c, T_obs_c)
    T_fdm_at_top = np.interp(t_top_c, t_arr_c, T_fdm_c)
    T_int_at_top = np.interp(t_top_c, t_int, T_int)

    resid = T_pred - T_top_c
    n = len(resid)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    mean_r = float(np.mean(resid))
    med_abs = float(np.median(np.abs(resid)))
    max_abs = float(np.max(np.abs(resid)))
    std_r = float(np.std(resid))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((T_top_c - np.mean(T_top_c)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # ---- regime 诊断 (复用项目分类器阈值; 仅诊断, 不改权重) ----
    regimes = _regime_labels(t_top_c, T_int_at_top, T_top_c)
    regime_metrics = {}
    for rg in ("TRANSIENT_HEATING", "TRANSIENT_COOLING", "STEADY",
               "SETTLING", "TRANSITION_OTHER"):
        m = regimes == rg
        if m.sum():
            regime_metrics[f"{rg}_n"] = int(m.sum())
            regime_metrics[f"{rg}_RMSE_C"] = float(
                np.sqrt(np.mean(resid[m] ** 2)))
            regime_metrics[f"{rg}_mean_residual_C"] = float(np.mean(resid[m]))
        else:
            regime_metrics[f"{rg}_n"] = 0
            regime_metrics[f"{rg}_RMSE_C"] = np.nan
            regime_metrics[f"{rg}_mean_residual_C"] = np.nan

    # ---- 温度带诊断 (仅解释, 不用于评分/排除) ----
    band_metrics = {}
    for lo, hi, name in ((0.0, 50.0, "LT50"), (50.0, 65.0, "50_65"),
                         (65.0, 75.0, "65_75"), (75.0, 1e9, "GE75")):
        m = (T_top_c >= lo) & (T_top_c < hi)
        if m.sum():
            band_metrics[f"{name}_n"] = int(m.sum())
            band_metrics[f"{name}_RMSE_C"] = float(
                np.sqrt(np.mean(resid[m] ** 2)))
            band_metrics[f"{name}_mean_residual_C"] = float(np.mean(resid[m]))
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
        "t_env_C": t_env,
        "T_initial_C": T_init,
        "delta_initial_C": T_init - float(T_top[0]),
        "environment_source": "INITIAL_MEASURED_TOP",
        "synchronization_rule": "SIMULTANEOUS_START_RELATIVE_T0",
        "time_shift_applied_s": 0.0,
        "metrics": {
            "n_points": n,
            "RMSE_C": rmse,
            "MAE_C": mae,
            "mean_residual_C": mean_r,
            "median_abs_residual_C": med_abs,
            "max_abs_residual_C": max_abs,
            "residual_std_C": std_r,
            "R_squared": r2,
            "measured_top_min_C": float(np.min(T_top_c)),
            "measured_top_max_C": float(np.max(T_top_c)),
            "predicted_top_min_C": float(np.min(T_pred)),
            "predicted_top_max_C": float(np.max(T_pred)),
        },
        "regime_metrics": regime_metrics,
        "band_metrics": band_metrics,
    }


def _regime_labels(t, T_int, T_top):
    """简化 regime 分类 (与项目 classify_temperature_regimes 阈值一致)。"""
    dT_int = np.gradient(T_int, t)
    dT_top = np.gradient(T_top, t)
    s = pd.Series(dT_int)
    dT_int_s = s.rolling(window=5, center=True,
                         min_periods=1).mean().to_numpy()
    s2 = pd.Series(dT_top)
    dT_top_s = s2.rolling(window=5, center=True,
                          min_periods=1).mean().to_numpy()
    regimes = np.array(["TRANSITION_OTHER"] * len(t), dtype=object)
    for i in range(len(t)):
        di = dT_int_s[i]
        dt_ = dT_top_s[i]
        if di >= 0.40:
            regimes[i] = "TRANSIENT_HEATING"
        elif di <= -0.40:
            regimes[i] = "TRANSIENT_COOLING"
        elif abs(di) <= 0.20 and abs(dt_) <= 0.15:
            regimes[i] = "STEADY"
        elif abs(di) <= 0.20 and abs(dt_) > 0.15:
            regimes[i] = "SETTLING"
    return regimes


# ============================================================
# 单数据集处理
# ============================================================

def process_dataset(label, top_path, int_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    top = load_top_series(top_path)
    internal = load_internal_series(int_path)
    v = run_frozen_validation(top, internal)

    # ---- 迹线 CSV (无伪影列) ----
    trace = pd.DataFrame({
        "measured_top_time_s": v["t_top"],
        "measured_top_C": v["T_top_measured"],
        "internal_interpolated_C": v["T_internal_at_top"],
        "top_FDM_raw_C": v["T_top_fdm_raw"],
        "top_predicted_lagged_C": v["T_top_predicted_lagged"],
        "residual_C": v["residual"],
    })
    trace.to_csv(out_dir / "validation_trace.csv", index=False)

    # ---- 迹线图 ----
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(v["t_top"], v["T_internal_at_top"], color="#7f7f7f", lw=1.1,
            ls=":", label="Measured internal temperature")
    ax.plot(v["t_top"], v["T_top_measured"], color="#1f77b4", lw=1.6,
            label="Corrected measured Top COC")
    ax.plot(v["t_top"], v["T_top_predicted_lagged"], color="#d62728", lw=1.8,
            label="Predicted Top COC (frozen model, output-side lag)")
    ax.plot(v["t_top"], v["T_top_fdm_raw"], color="#2ca02c", lw=0.9, ls="--",
            alpha=0.7, label="Raw Top COC FDM (no lag)")
    ax.set_xlabel("Elapsed time [s] (simultaneous start, t=0)")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(f"{label} — corrected out-of-sample validation (frozen "
                 f"model)\n(k={K_EFF}, cp={CP_EFF:.0f}, tau={TAU_TOP:.1f} s; "
                 f"h={H_CONV} + nonlinear radiation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{_slug(label)}_validation_trace.png", dpi=150)
    fig.savefig(out_dir / f"{_slug(label)}_validation_trace.pdf")
    plt.close(fig)

    # ---- 残差图 ----
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.plot(v["t_top"], v["residual"], color="#d62728", lw=1.0,
            label="Residual (predicted - measured)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Residual [C]")
    ax.set_title(f"{label} — residual vs time (corrected frozen model)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{_slug(label)}_residual_vs_time.png", dpi=150)
    plt.close(fig)

    return {
        "role": "VALIDATION",
        "k_eff": K_EFF, "cp_eff": CP_EFF, "tau_top": TAU_TOP,
        "h_conv": H_CONV, "epsilon": EPS,
        "top_source": str(top_path.resolve()),
        "internal_source": str(int_path.resolve()),
        "top_sheet": "Data", "internal_sheet": "Extracted_Data",
        "top_column": top["column"], "internal_column": internal["column"],
        "top_time_col": top["time_source"],
        "internal_time_col": internal["time_source"],
        "synchronization_assumption": "simultaneous recording start",
        "synchronization_rule": v["synchronization_rule"],
        "time_shift_applied_s": v["time_shift_applied_s"],
        "top_valid_points": top["n_valid"],
        "internal_valid_points": internal["n_valid"],
        "overlap_duration_s": v["t_overlap_s"],
        "top_median_dt_s": top["median_dt"],
        "internal_median_dt_s": internal["median_dt"],
        "top_first_value_C": top["first_value_C"],
        "internal_first_value_C": internal["first_value_C"],
        "initial_internal_minus_top_C": v["delta_initial_C"],
        "environment_C": v["t_env_C"],
        "environment_source": v["environment_source"],
        **v["metrics"],
        **v["regime_metrics"],
        **{f"band_{k}": val for k, val in v["band_metrics"].items()},
        "residual_bias": ("positive" if v["metrics"]["mean_residual_C"] > 0.05
                          else "negative"
                          if v["metrics"]["mean_residual_C"] < -0.05
                          else "near_zero"),
    }, v


def _slug(label):
    return label.lower().replace(" ", "_").replace("/", "_")


# ============================================================
# 描述性转移质量
# ============================================================

def _transfer_quality(rmse):
    if rmse <= 1.5:
        return "EXCELLENT"
    if rmse <= 2.5:
        return "GOOD"
    if rmse <= 4.0:
        return "MODERATE"
    return "POOR"


# ============================================================
# 主流程
# ============================================================

def main():
    comp_dir = OUTPUT_ROOT / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    out1 = OUTPUT_ROOT / "66C_redo"
    out2 = OUTPUT_ROOT / "PCR_3s_extension"
    s1, v1 = process_dataset("66C_redo", DS1_TOP, DS1_INT, out1)
    s2, v2 = process_dataset("PCR_3s_extension", DS2_TOP, DS2_INT, out2)

    # ---- 汇总表 ----
    calib_row = {
        "role": "CALIBRATION",
        "k_eff": K_EFF, "cp_eff": CP_EFF, "tau_top": TAU_TOP,
        "h_conv": H_CONV, "epsilon": EPS,
        "n_points": np.nan, "duration_s": np.nan,
        "environment_C": np.nan, "initial_internal_C": np.nan,
        "initial_top_C": np.nan, "initial_internal_minus_top_C": np.nan,
        "RMSE_C": CALIB_RMSE_72C, "MAE_C": np.nan,
        "mean_residual_C": np.nan, "median_abs_residual_C": np.nan,
        "residual_std_C": np.nan, "max_abs_residual_C": np.nan,
        "R_squared": np.nan,
        "measured_top_min_C": np.nan, "measured_top_max_C": np.nan,
        "predicted_top_min_C": np.nan, "predicted_top_max_C": np.nan,
        "time_shift_applied_s": 0.0,
    }
    rows = [calib_row]
    for s in (s1, s2):
        rows.append({
            "role": s["role"],
            "k_eff": s["k_eff"], "cp_eff": s["cp_eff"],
            "tau_top": s["tau_top"], "h_conv": s["h_conv"],
            "epsilon": s["epsilon"],
            "n_points": s["n_points"], "duration_s": s["overlap_duration_s"],
            "environment_C": s["environment_C"],
            "initial_internal_C": s["internal_first_value_C"],
            "initial_top_C": s["top_first_value_C"],
            "initial_internal_minus_top_C": s["initial_internal_minus_top_C"],
            "RMSE_C": s["RMSE_C"], "MAE_C": s["MAE_C"],
            "mean_residual_C": s["mean_residual_C"],
            "median_abs_residual_C": s["median_abs_residual_C"],
            "residual_std_C": s["residual_std_C"],
            "max_abs_residual_C": s["max_abs_residual_C"],
            "R_squared": s["R_squared"],
            "measured_top_min_C": s["measured_top_min_C"],
            "measured_top_max_C": s["measured_top_max_C"],
            "predicted_top_min_C": s["predicted_top_min_C"],
            "predicted_top_max_C": s["predicted_top_max_C"],
            "time_shift_applied_s": s["time_shift_applied_s"],
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(comp_dir / "validation_summary.csv", index=False)

    # ---- RMSE 总览图 ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = ["72C\ncalibration", "66C redo\n(corrected)", "PCR 3s ext\n(corrected)"]
    rmses = [CALIB_RMSE_72C, s1["RMSE_C"], s2["RMSE_C"]]
    colors = ["#1f77b4", "#2ca02c", "#2ca02c"]
    bars = ax.bar(labels, rmses, color=colors, width=0.55)
    for b, r in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02,
                f"{r:.3f} C", ha="center", fontsize=9)
    ax.set_ylabel("Top COC RMSE [C]")
    ax.set_title("Frozen model — calibration vs corrected external validation\n"
                 "(same frozen parameters, no refit, no time shift, "
                 "no artifact filtering)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(comp_dir / "model_validation_overview.png", dpi=150)
    plt.close(fig)

    # ---- measured vs predicted 散点 ----
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for v, lbl, c in ((v1, "66C redo", "#2ca02c"),
                      (v2, "PCR 3s ext", "#d62728")):
        ax.scatter(v["T_top_measured"], v["T_top_predicted_lagged"],
                   s=8, alpha=0.5, color=c, label=lbl)
    lo = min(np.min(v1["T_top_measured"]), np.min(v2["T_top_measured"]),
             np.min(v1["T_top_predicted_lagged"]),
             np.min(v2["T_top_predicted_lagged"]))
    hi = max(np.max(v1["T_top_measured"]), np.max(v2["T_top_measured"]),
             np.max(v1["T_top_predicted_lagged"]),
             np.max(v2["T_top_predicted_lagged"]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Corrected measured Top COC [C]")
    ax.set_ylabel("Predicted Top COC [C]")
    ax.set_title("Measured vs predicted Top COC (corrected data, frozen "
                 "model)\nvalidation datasets only")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(comp_dir / "measured_vs_predicted_top_COC.png", dpi=150)
    plt.close(fig)

    # ---- 元数据 ----
    meta = {
        "validation_version": "v2-corrected-top",
        "previous_validation_invalidated": True,
        "reason": "TOP_COC_SOURCE_TEMPERATURE_NOT_PREVIOUSLY_CONVERTED",
        "current_validation_uses_corrected_top_data": True,
        "model_id": "frozen_output_lag (strategy_G_conservative_cross_protocol_v1)",
        "model": {"architecture": "OUTPUT-SIDE EFFECTIVE LAG",
                  "k_eff": K_EFF, "cp_eff": CP_EFF, "rho": RHO_COC,
                  "tau_top": TAU_TOP, "h_conv": H_CONV, "epsilon": EPS,
                  "sigma": SIGMA, "F_view": F_VIEW,
                  "geometry": "BARE_TOP_COC_LAYERS"},
        "calibration_reference": {"72C_RMSE_C": CALIB_RMSE_72C},
        "no_artifact_filtering": True,
        "no_model_based_point_rejection": True,
        "no_time_shift_optimization": True,
        "no_cross_correlation": True,
        "no_parameter_fitting": True,
        "environment_rule": "first valid corrected measured Top COC "
                            "(INITIAL_MEASURED_TOP)",
        "initial_condition_rule": "first valid measured internal (uniform)",
        "bottom_boundary": "measured internal trace (direct Dirichlet)",
        "datasets": {
            "66C_redo": {"top_source": s1["top_source"],
                         "internal_source": s1["internal_source"],
                         "top_sheet": s1["top_sheet"],
                         "internal_sheet": s1["internal_sheet"],
                         "top_column": s1["top_column"],
                         "internal_column": s1["internal_column"],
                         "synchronization_rule": s1["synchronization_rule"],
                         "time_shift_applied_s": s1["time_shift_applied_s"],
                         "metrics": {k: s1[k] for k in (
                             "n_points", "RMSE_C", "MAE_C",
                             "mean_residual_C", "median_abs_residual_C",
                             "residual_std_C", "max_abs_residual_C",
                             "R_squared", "measured_top_min_C",
                             "measured_top_max_C", "predicted_top_min_C",
                             "predicted_top_max_C")},
                         "regime_metrics": {k: s1[k] for k in s1
                                            if k.startswith(
                                                ("TRANSIENT_HEATING_",
                                                 "TRANSIENT_COOLING_",
                                                 "STEADY_", "SETTLING_",
                                                 "TRANSITION_OTHER_"))},
                         "band_metrics": {k: s1[k] for k in s1
                                          if k.startswith("band_")}},
            "PCR_3s_extension": {"top_source": s2["top_source"],
                                 "internal_source": s2["internal_source"],
                                 "top_sheet": s2["top_sheet"],
                                 "internal_sheet": s2["internal_sheet"],
                                 "top_column": s2["top_column"],
                                 "internal_column": s2["internal_column"],
                                 "synchronization_rule":
                                     s2["synchronization_rule"],
                                 "time_shift_applied_s":
                                     s2["time_shift_applied_s"],
                                 "metrics": {k: s2[k] for k in (
                                     "n_points", "RMSE_C", "MAE_C",
                                     "mean_residual_C",
                                     "median_abs_residual_C",
                                     "residual_std_C", "max_abs_residual_C",
                                     "R_squared", "measured_top_min_C",
                                     "measured_top_max_C",
                                     "predicted_top_min_C",
                                     "predicted_top_max_C")},
                                 "regime_metrics": {k: s2[k] for k in s2
                                                    if k.startswith(
                                                        ("TRANSIENT_HEATING_",
                                                         "TRANSIENT_COOLING_",
                                                         "STEADY_",
                                                         "SETTLING_",
                                                         "TRANSITION_OTHER_"))},
                                 "band_metrics": {k: s2[k] for k in s2
                                                  if k.startswith("band_")}},
        },
        "validation_basis": "corrected measured Top COC only; "
                            "sample/PCR outcome not used",
        "historical_60C_transfer_check": {
            "included_in_score": False,
            "reason": "historical synchronization uncertainty"},
    }
    (comp_dir / "external_validation_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = _summary_text(s1, s2, v1, v2)
    (comp_dir / "external_validation_summary.txt").write_text(
        txt, encoding="utf-8")
    print(txt)


def _summary_text(s1, s2, v1, v2):
    L = []
    A = L.append
    A("=" * 74)
    A("CORRECTED FROZEN MODEL — TWO-DATASET EXTERNAL VALIDATION")
    A("=" * 74)
    A("先前验证已作废 (Top COC 源温度未转换); 本验证基于修正后 T Avg 数据。")
    A(f"模型: OUTPUT-SIDE EFFECTIVE LAG; k={K_EFF}, cp={CP_EFF}, "
      f"tau={TAU_TOP} s; h={H_CONV} + 非线性辐射; BARE_TOP")
    A(f"参考: 72C 校准 RMSE = {CALIB_RMSE_72C:.4f} C (只读)")
    A("主验证: 修正后原始有效 Top COC vs 滞后预测; 无伪影过滤。")
    A("")
    for s, v, label in ((s1, v1, "66C REDO (CORRECTED VALIDATION)"),
                        (s2, v2, "PCR 3S EXTENSION (CORRECTED VALIDATION)")):
        A(f"[{label}]")
        A(f"  Top: {s['top_source']}")
        A(f"  Internal: {s['internal_source']}")
        A(f"  同步: {s['synchronization_rule']}; 时移 = "
          f"{s['time_shift_applied_s']:.1f} s")
        A(f"  Top 有效点 {s['top_valid_points']} (median dt "
          f"{s['top_median_dt_s']:.3f} s); Internal 有效点 "
          f"{s['internal_valid_points']} (median dt "
          f"{s['internal_median_dt_s']:.3f} s); 重叠 "
          f"{s['overlap_duration_s']:.1f} s")
        A(f"  初始: Top {s['top_first_value_C']:.2f} C, Internal "
          f"{s['internal_first_value_C']:.2f} C, "
          f"Δ={s['initial_internal_minus_top_C']:+.2f} C; 环境 "
          f"{s['environment_C']:.2f} C ({s['environment_source']})")
        A(f"  实测 Top 范围 [{s['measured_top_min_C']:.2f}, "
          f"{s['measured_top_max_C']:.2f}] C; 预测 Top 范围 "
          f"[{s['predicted_top_min_C']:.2f}, {s['predicted_top_max_C']:.2f}] C")
        A(f"  RMSE = {s['RMSE_C']:.4f} C | MAE = {s['MAE_C']:.4f} | "
          f"mean = {s['mean_residual_C']:+.4f} | "
          f"median_abs = {s['median_abs_residual_C']:.4f} | "
          f"std = {s['residual_std_C']:.4f} | "
          f"max_abs = {s['max_abs_residual_C']:.4f} | "
          f"R² = {s['R_squared']:.4f}")
        A(f"  regime: HEATING n={s['TRANSIENT_HEATING_n']} "
          f"RMSE={s['TRANSIENT_HEATING_RMSE_C']:.3f} "
          f"mean={s['TRANSIENT_HEATING_mean_residual_C']:+.3f}; "
          f"COOLING n={s['TRANSIENT_COOLING_n']} "
          f"RMSE={s['TRANSIENT_COOLING_RMSE_C']:.3f} "
          f"mean={s['TRANSIENT_COOLING_mean_residual_C']:+.3f}; "
          f"SETTLING n={s['SETTLING_n']} "
          f"RMSE={s['SETTLING_RMSE_C']:.3f} "
          f"mean={s['SETTLING_mean_residual_C']:+.3f}")
        A(f"  温度带诊断 (仅解释, 不评分): "
          f"<50 n={s['band_LT50_n']} RMSE={s['band_LT50_RMSE_C']:.3f} "
          f"mean={s['band_LT50_mean_residual_C']:+.3f}; "
          f"50-65 n={s['band_50_65_n']} RMSE={s['band_50_65_RMSE_C']:.3f} "
          f"mean={s['band_50_65_mean_residual_C']:+.3f}; "
          f"65-75 n={s['band_65_75_n']} RMSE={s['band_65_75_RMSE_C']:.3f} "
          f"mean={s['band_65_75_mean_residual_C']:+.3f}; "
          f">=75 n={s['band_GE75_n']} RMSE={s['band_GE75_RMSE_C']:.3f} "
          f"mean={s['band_GE75_mean_residual_C']:+.3f}")
        A(f"  残差偏置: {s['residual_bias']}; "
          f"描述性转移质量: {_transfer_quality(s['RMSE_C'])}")
        A("")
    A("跨数据集: 校准 {:.3f}; 验证66C {:.3f}; 验证3s {:.3f}; "
      "均值 {:.3f}; 最差 {:.3f}".format(
          CALIB_RMSE_72C, s1["RMSE_C"], s2["RMSE_C"],
          (s1["RMSE_C"] + s2["RMSE_C"]) / 2.0,
          max(s1["RMSE_C"], s2["RMSE_C"])))
    A("验证/校准 RMSE 比: DS1 {:.2f}x, DS2 {:.2f}x".format(
        s1["RMSE_C"] / CALIB_RMSE_72C, s2["RMSE_C"] / CALIB_RMSE_72C))
    A("无伪影过滤; 无基于模型分歧的点剔除; 无时移优化; 无参数重拟合。")
    A("验证仅基于修正后实测 Top COC; 样品/PCR 结果未用于评判或修改模型。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
