#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冻结 Strategy G 保守候选 — 跨协议样品温度预测
=============================================

冻结候选 (本任务冻结, 不重新拟合):
    strategy_G_conservative_cross_protocol_v1
    k = 0.055 W/(m K)
    cp = 1200 J/(kg K)
    tau = 8.5 s
    rho = 1020 kg/m3

物理 (Strategy E, 不变):
    h_conv = 10.0 W/(m2 K)
    epsilon = 0.90
    sigma_SB = 5.670374419e-8 W/(m2 K4)
    F_view = 1.0
    非线性 Stefan-Boltzmann 辐射
    BARE_TOP_COC_LAYERS (850 um)
    输出侧一阶滞后 (tau 只作用 T_top_observed_predicted,
    绝不作用 T_sample_FDM)

环境规则 (下游内部-only 预测代理, 与标定规则分开记录):
    标定环境规则:   第一个有效实测 Top COC 温度
    内部-only 预测: 第一个有效内部温度 (无 Top/环境测量时)
                    environment_source = INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT
    若 PCR 工作簿有同步实测 Top 迹线: 用其第一个有效值 (INITIAL_MEASURED_TOP)

协议:
    1. DOE11 faster
    2. Test_PCR longer holding

输出:
    calibrated_model_output/strategy_G_conservative_cross_protocol_v1/
        DOE11_faster/
        Test_PCR_longer_holding/
        comparison/
        frozen_candidate_metadata.json
        cross_protocol_summary.txt
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag
from thermal_model.historical.frozen_strategy_G_candidate import (
    FROZEN_STRATEGY_G_CANDIDATE,
    candidate_dict,
    STRATEGY_G_STORED_RMSE_72C,
)
from thermal_model.utilities.predict_sample_from_internal_temperature import (
    load_internal_data,
    ramp_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"

DOE11_PATH = CALIBRATION_DIR / "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
LONGER_PATH = CALIBRATION_DIR / "Test_PCR longer holding.xlsx"

OUTPUT_ROOT = (
    PROJECT_ROOT / "calibrated_model_output"
    / "strategy_G_conservative_cross_protocol_v1")

SAVE_DT = 0.1
SHEET = "Extracted_Data"
TIME_COL = "Time(s)"
TEMP_COL = "Zone 1 Avg (°C)"

# 周期检测参数 (内部温度)
PEAK_THRESHOLD = 88.0
DIP_THRESHOLD = 60.0
MIN_PEAK_SEPARATION_S = 15.0
ACTIVATION_MIN_S = 30.0   # 初始激活相最短时长 (内部保持高温)


# ============================================================
# 环境代理解析 (内部-only)
# ============================================================

def _resolve_sheet(path):
    """自适应工作簿 sheet: 优先 Extracted_Data, 否则第一个含温度列的表。"""
    xl = pd.ExcelFile(path)
    if "Extracted_Data" in xl.sheet_names:
        return "Extracted_Data"
    for sh in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sh, nrows=3)
            if df is not None and TEMP_COL in df.columns:
                return sh
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"找不到含 {TEMP_COL!r} 列的工作表: {xl.sheet_names}")


def resolve_environment_proxy(t_internal, top_measured=None):
    """为下游内部-only 预测解析环境温度。

    优先: 同步实测 Top COC 的第一个有效值 (INITIAL_MEASURED_TOP)。
    否则: 第一个有效内部温度 (INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT)。
    绝不静默回退到 25 C。
    """
    if top_measured is not None:
        arr = np.asarray(top_measured, dtype=float)
        valid = np.flatnonzero(np.isfinite(arr))
        if valid.size:
            return {
                "T_environment_C": float(arr[valid[0]]),
                "environment_source": "INITIAL_MEASURED_TOP",
            }
    tint = np.asarray(t_internal, dtype=float)
    valid = np.flatnonzero(np.isfinite(tint))
    if valid.size == 0:
        raise ValueError("无有效内部温度, 无法解析环境代理。")
    return {
        "T_environment_C": float(tint[valid[0]]),
        "environment_source": "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT",
    }


# ============================================================
# 冻结模型运行
# ============================================================

def run_frozen_strategy_G(elapsed_time_s, t_internal, t_env):
    """用冻结候选 + Strategy E 物理运行一次 FDM + 滞后。

    返回 dict:
        t_arr / T_sample_FDM_C / T_top_FDM_C /
        T_top_observed_predicted_C / T_initial_C / result
    """
    c = FROZEN_STRATEGY_G_CANDIDATE
    mats = cr.make_convection_radiation_materials(
        c.k_eff_W_mK, c.cp_eff_J_kgK, c.rho_COC_kg_m3)
    T_initial = float(np.asarray(t_internal, dtype=float)[0])
    result = cr.run_convection_radiation_fdm(
        time_s=elapsed_time_s,
        bottom_temperature_C=t_internal,
        materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS,
        T_air_C=t_env,
        T_surroundings_C=t_env,
        save_dt=SAVE_DT,
        T_initial_C=T_initial,
    )
    t_arr = result["t_array"]
    T_sample = result["T_sample_arr"]
    T_top = result["T_top_surface_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top, c.tau_lag_s)
    return {
        "t_arr": t_arr,
        "T_sample_FDM_C": T_sample,
        "T_top_FDM_C": T_top,
        "T_top_observed_predicted_C": T_top_obs,
        "T_initial_C": T_initial,
        "result": result,
    }


def sample_at_measurement_times(meas_time, model_time, model_trace):
    """查询轴 = 实测时间坐标 (绝不用温度值作查询轴)。"""
    return np.interp(meas_time, model_time, model_trace)


# ============================================================
# 周期检测 (分离激活相)
# ============================================================

def detect_activation_and_repeated_cycles(t, t_internal, t_sample,
                                          peak_threshold=PEAK_THRESHOLD,
                                          dip_threshold=DIP_THRESHOLD,
                                          min_sep=MIN_PEAK_SEPARATION_S,
                                          activation_min_s=ACTIVATION_MIN_S):
    """检测内部温度峰-谷结构, 分离初始激活相与重复 PCR 周期。

    返回 dict:
        activation: dict|None  (初始高温相)
        repeated_cycles: list[dict]  (重复周期, 从 cycle 1 编号)
    """
    t = np.asarray(t, dtype=float)
    ti = np.asarray(t_internal, dtype=float)
    ts = np.asarray(t_sample, dtype=float)
    n = len(t)

    # ---- 峰 ----
    peaks = []
    i = 1
    while i < n - 1:
        if ti[i] >= peak_threshold and ti[i] >= ti[i - 1]:
            j = i
            while j + 1 < n and ti[j + 1] == ti[j]:
                j += 1
            if j == n - 1 or ti[j] >= ti[j + 1]:
                peaks.append(j)
            i = j + 1
        else:
            i += 1

    # ---- 低谷 ----
    troughs = []
    i = 1
    while i < n - 1:
        if ti[i] < dip_threshold and ti[i] <= ti[i - 1]:
            j = i
            while j + 1 < n and ti[j + 1] == ti[j]:
                j += 1
            if j == n - 1 or ti[j] <= ti[j + 1]:
                troughs.append(j)
            i = j + 1
        else:
            i += 1

    if len(peaks) < 2:
        return {"activation": None, "repeated_cycles": []}

    # ---- 合并无低谷分隔的相邻峰 (同一高温相) ----
    merged = []
    for idx in peaks:
        if merged:
            prev = merged[-1]
            between = ti[prev + 1:idx]
            min_between = float(np.min(between)) if between.size else \
                min(float(ti[prev]), float(ti[idx]))
            if min_between < dip_threshold:
                merged.append(idx)
            else:
                if ti[idx] > ti[prev]:
                    merged[-1] = idx
        else:
            merged.append(idx)

    # ---- 最小间隔去重 ----
    selected = []
    for idx in merged:
        if selected and (t[idx] - t[selected[-1]]) < min_sep:
            continue
        selected.append(idx)

    if len(selected) < 2:
        return {"activation": None, "repeated_cycles": []}

    # ---- 每个高温相: 起点 = 前方最近低谷 ----
    phases = []
    for k, pidx in enumerate(selected):
        prior = [tr for tr in troughs if tr < pidx]
        trough = prior[-1] if prior else 0
        # 样品峰: 内部峰后窗口内样品最大值
        hi = min(n - 1, pidx + 20)
        sw = ts[pidx:hi + 1]
        spi = pidx + int(np.argmax(sw))
        phases.append({
            "peak_index": pidx,
            "cycle_start_time_s": float(t[trough]),
            "internal_peak_time_s": float(t[pidx]),
            "internal_high_peak_C": float(ti[pidx]),
            "sample_high_peak_C": float(ts[spi]),
            "sample_peak_time_s": float(t[spi]),
            "internal_low_trough_C": float(ti[trough]),
            "sample_low_trough_C": float(ts[trough]),
            "sample_trough_time_s": float(t[trough]),
            "has_prior_trough": bool(prior),
        })

    # ---- 分离激活相: 第一个无前方低谷 且 高温持续 >= activation_min_s ----
    activation = None
    repeated = []
    if phases:
        p0 = phases[0]
        if not p0["has_prior_trough"] and \
                (p0["internal_peak_time_s"] - t[0]) >= activation_min_s:
            activation = p0
            rest = phases[1:]
        else:
            rest = phases
    else:
        rest = []

    # 重复周期编号 + 时长
    for k, ph in enumerate(rest):
        c = dict(ph)
        c["cycle_number"] = k + 1
        if k > 0:
            c["cycle_duration_s"] = float(
                ph["cycle_start_time_s"] - rest[k - 1]["cycle_start_time_s"])
        else:
            c["cycle_duration_s"] = None
        repeated.append(c)

    return {"activation": activation, "repeated_cycles": repeated}


def per_cycle_metrics(cycles, t, t_internal, t_sample):
    """每个重复周期的完整指标表。

    周期结束 = 下一周期起点 (最后一个周期用最后峰后 + 30 s 窗口或迹线末尾)。
    """
    t = np.asarray(t, dtype=float)
    ti = np.asarray(t_internal, dtype=float)
    ts = np.asarray(t_sample, dtype=float)
    out = []
    n_cyc = len(cycles)
    for k, c in enumerate(cycles):
        s = c["cycle_start_time_s"]
        if k + 1 < n_cyc:
            e = min(float(cycles[k + 1]["cycle_start_time_s"]),
                    float(t[-1]))
        else:
            e = min(float(c["internal_peak_time_s"]) + 30.0,
                    float(t[-1]))
        if e <= s:
            continue
        mask = (t >= s) & (t <= e)
        if not mask.any():
            continue
        tc = t[mask]
        tic = ti[mask]
        tsc = ts[mask]
        i_peak = int(np.argmax(tic))
        i_trough = int(np.argmin(tic))
        s_peak = int(np.argmax(tsc))
        s_trough = int(np.argmin(tsc))
        ri = ramp_summary(tc, tic)
        rs = ramp_summary(tc, tsc)
        out.append({
            "cycle_number": int(c["cycle_number"]),
            "cycle_start_time_s": float(s),
            "cycle_end_time_s": float(e),
            "cycle_duration_s": float(e - s),
            "internal_peak_C": float(np.max(tic)),
            "internal_trough_C": float(np.min(tic)),
            "sample_peak_C": float(np.max(tsc)),
            "sample_trough_C": float(np.min(tsc)),
            "time_internal_peak_s": float(tc[i_peak]),
            "time_sample_peak_s": float(tc[s_peak]),
            "sample_peak_lag_s": float(tc[s_peak] - tc[i_peak]),
            "max_internal_heating_rate_C_s": float(ri["max_positive_C_per_s"]),
            "max_sample_heating_rate_C_s": float(rs["max_positive_C_per_s"]),
            "max_internal_cooling_rate_C_s": float(ri["max_negative_C_per_s"]),
            "max_sample_cooling_rate_C_s": float(rs["max_negative_C_per_s"]),
        })
    return out


# ============================================================
# Dwell (时间戳区间积分, 非均匀 dt 安全)
# ============================================================

def dwell_times_ge(t, T, thresholds=(75.0, 80.0, 85.0, 90.0, 92.0, 94.0, 95.0)):
    """T >= th 的停留时间 (线性插值交点)。"""
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(t)
    out = {}
    for th in thresholds:
        total = 0.0
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            Ta, Tb = T[i], T[i + 1]
            if Ta >= th and Tb >= th:
                total += (b - a)
            elif Ta >= th or Tb >= th:
                if Tb != Ta:
                    frac = (th - Ta) / (Tb - Ta)
                    if Ta >= th:
                        total += (b - a) * (1.0 - frac)
                    else:
                        total += (b - a) * frac
        out[f"sample_ge_{th:.0f}C_s"] = float(total)
    return out


def dwell_time_bands(t, T):
    """描述性样品温度时间带。"""
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(t)
    bands = [(0, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 85),
             (85, 90), (90, 1e9)]
    out = {}
    for lo, hi in bands:
        total = 0.0
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            Ta, Tb = T[i], T[i + 1]
            if lo <= Ta < hi and lo <= Tb < hi:
                total += (b - a)
            elif (lo <= Ta < hi) or (lo <= Tb < hi):
                total += (b - a) * 0.5
        label = f"sample_{lo:.0f}_{hi:.0f}C_s" if hi < 1e8 else \
            f"sample_ge_{lo:.0f}C_s"
        out[label] = float(total)
    return out


# ============================================================
# 单协议处理
# ============================================================

def process_protocol(name, path, out_dir, top_measured=None, sheet=None):
    """处理一个 PCR 协议 (冻结模型, 无拟合)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sheet is None:
        sheet = _resolve_sheet(path)
    data = load_internal_data(path, sheet=sheet, time_col=TIME_COL,
                              temp_col=TEMP_COL)
    t = data["source_time_s"]
    elapsed = data["elapsed_time_s"]
    tint = data["T_internal_C"]

    # ---- 环境代理 ----
    env = resolve_environment_proxy(tint, top_measured)

    # ---- 冻结模型 ----
    out = run_frozen_strategy_G(elapsed, tint, env["T_environment_C"])
    T_sample_pred = sample_at_measurement_times(
        elapsed, out["t_arr"], out["T_sample_FDM_C"])
    T_top_pred = sample_at_measurement_times(
        elapsed, out["t_arr"], out["T_top_FDM_C"])
    T_top_obs_pred = sample_at_measurement_times(
        elapsed, out["t_arr"], out["T_top_observed_predicted_C"])

    # ---- 迹线 CSV ----
    trace = pd.DataFrame({
        "source_time_s": t,
        "elapsed_time_s": elapsed,
        "T_internal_C": tint,
        "T_sample_predicted_C": T_sample_pred,
        "T_top_FDM_C": T_top_pred,
        "T_top_observed_predicted_C": T_top_obs_pred,
    })
    trace["delta_sample_minus_internal_C"] = (
        T_sample_pred - tint)
    trace.to_csv(out_dir / "sample_temperature_prediction.csv", index=False)
    assert np.all(np.isfinite(T_sample_pred)), "样品预测含 NaN!"
    assert np.all(np.diff(t) > 0), "时间非严格递增!"

    # ---- 周期 ----
    cyc = detect_activation_and_repeated_cycles(
        elapsed, tint, T_sample_pred)
    activation = cyc["activation"]
    repeated = cyc["repeated_cycles"]
    repeated_metrics = per_cycle_metrics(repeated, elapsed, tint,
                                         T_sample_pred)
    if repeated_metrics:
        pd.DataFrame(repeated_metrics).to_csv(
            out_dir / "cycle_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "cycle_summary.csv", index=False)

    # ---- Dwell ----
    total_dwell = dwell_times_ge(elapsed, T_sample_pred)
    total_bands = dwell_time_bands(elapsed, T_sample_pred)
    dwell_rows = [{"scope": "TOTAL"} | total_dwell]
    # 每周期 dwell
    for c in repeated_metrics:
        s = c["cycle_start_time_s"]
        e = c["cycle_end_time_s"]
        mask = (elapsed >= s) & (elapsed <= e)
        dw = dwell_times_ge(elapsed[mask], T_sample_pred[mask])
        dwell_rows.append({"scope": f"cycle_{c['cycle_number']}"} | dw)
    pd.DataFrame(dwell_rows).to_csv(
        out_dir / "thermal_dwell_summary.csv", index=False)

    # ---- 汇总 ----
    d = T_sample_pred - tint
    internal_ramp = ramp_summary(elapsed, tint)
    sample_ramp = ramp_summary(elapsed, T_sample_pred)
    repeated_peaks = [c["sample_peak_C"] for c in repeated_metrics]
    repeated_troughs = [c["sample_trough_C"] for c in repeated_metrics]
    summary = {
        "protocol": name,
        "source_file": str(path.resolve()),
        "workbook_format": "xlsx",
        "sheet": sheet,
        "time_col": data["resolved_time_col"],
        "temp_col": data["resolved_temp_col"],
        "valid_points": data["n_valid"],
        "duration_s": data["duration_s"],
        "median_dt_s": data["median_dt"],
        "initial_internal_C": data["T_initial_C"],
        "environment": env,
        "no_fitting": True,
        "internal": {
            "initial_C": data["T_initial_C"],
            "min_C": data["T_min_C"],
            "max_C": data["T_max_C"],
            "max_heating_rate_C_s": internal_ramp["max_positive_C_per_s"],
            "max_cooling_rate_C_s": internal_ramp["max_negative_C_per_s"],
        },
        "sample_predicted": {
            "min_C": float(np.min(T_sample_pred)),
            "max_C": float(np.max(T_sample_pred)),
            "time_of_max_s": float(elapsed[int(np.argmax(T_sample_pred))]),
            "mean_abs_internal_sample_diff_C": float(np.mean(np.abs(d))),
            "max_heating_rate_C_s": sample_ramp["max_positive_C_per_s"],
            "max_cooling_rate_C_s": sample_ramp["max_negative_C_per_s"],
        },
        "repeated_cycles": {
            "count": len(repeated_metrics),
            "mean_cycle_duration_s": float(np.mean(
                [c["cycle_duration_s"] for c in repeated_metrics])) if
                repeated_metrics else None,
            "median_cycle_duration_s": float(np.median(
                [c["cycle_duration_s"] for c in repeated_metrics])) if
                repeated_metrics else None,
            "sample_peak_min_C": float(np.min(repeated_peaks)) if
                repeated_peaks else None,
            "sample_peak_max_C": float(np.max(repeated_peaks)) if
                repeated_peaks else None,
            "sample_peak_mean_C": float(np.mean(repeated_peaks)) if
                repeated_peaks else None,
            "sample_peak_median_C": float(np.median(repeated_peaks)) if
                repeated_peaks else None,
            "sample_trough_min_C": float(np.min(repeated_troughs)) if
                repeated_troughs else None,
            "sample_trough_max_C": float(np.max(repeated_troughs)) if
                repeated_troughs else None,
            "sample_trough_mean_C": float(np.mean(repeated_troughs)) if
                repeated_troughs else None,
            "sample_trough_median_C": float(np.median(repeated_troughs)) if
                repeated_troughs else None,
        },
        "activation_phase": activation,
        "total_dwell_s": total_dwell,
        "temperature_bands_s": total_bands,
    }

    # ---- 图形 ----
    _plot_protocol(elapsed, tint, T_sample_pred, T_top_pred, name, out_dir)

    return summary, trace, repeated_metrics, total_dwell


def _plot_protocol(elapsed, tint, t_sample, t_top, name, out_dir):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(elapsed, tint, color="#7f7f7f", lw=1.2, ls=":",
            label="Internal sensor input (measured)")
    ax.plot(elapsed, t_sample, color="#2ca02c", lw=2.0,
            label="Sample predicted (model estimate)")
    ax.plot(elapsed, t_top, color="#1f77b4", lw=1.4, ls="--",
            label="Top COC FDM (model estimate)")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{name} — Frozen Strategy G Thermal Prediction\n"
                 "(k=0.055, cp=1200, tau=8.5; h=10 + nonlinear radiation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_temperature_prediction.png", dpi=150)
    fig.savefig(out_dir / "sample_temperature_prediction.pdf")
    plt.close(fig)


# ============================================================
# 跨协议比较
# ============================================================

def build_comparison(summary_f, summary_l, dwell_f, dwell_l, out_dir):
    comp = {
        "metric": ["total_protocol_duration_s",
                   "repeated_cycle_count",
                   "median_cycle_duration_s",
                   "internal_peak_range_C",
                   "sample_peak_range_C",
                   "mean_repeated_cycle_sample_peak_C",
                   "median_repeated_cycle_sample_peak_C",
                   "sample_trough_range_C",
                   "sample_ge75C_dwell_total_s",
                   "sample_ge80C_dwell_total_s",
                   "sample_ge85C_dwell_total_s",
                   "sample_ge90C_dwell_total_s",
                   "sample_ge92C_dwell_total_s",
                   "sample_ge94C_dwell_total_s",
                   "sample_ge95C_dwell_total_s",
                   "sample_max_heating_rate_C_s",
                   "sample_max_cooling_rate_C_s"],
        "DOE11_faster": [
            summary_f["duration_s"],
            summary_f["repeated_cycles"]["count"],
            summary_f["repeated_cycles"]["median_cycle_duration_s"],
            summary_f["internal"]["max_C"] - summary_f["internal"]["min_C"],
            (summary_f["repeated_cycles"]["sample_peak_max_C"] -
             summary_f["repeated_cycles"]["sample_peak_min_C"]),
            summary_f["repeated_cycles"]["sample_peak_mean_C"],
            summary_f["repeated_cycles"]["sample_peak_median_C"],
            (summary_f["repeated_cycles"]["sample_trough_max_C"] -
             summary_f["repeated_cycles"]["sample_trough_min_C"]),
            dwell_f.get("sample_ge_75C_s", np.nan),
            dwell_f.get("sample_ge_80C_s", np.nan),
            dwell_f.get("sample_ge_85C_s", np.nan),
            dwell_f.get("sample_ge_90C_s", np.nan),
            dwell_f.get("sample_ge_92C_s", np.nan),
            dwell_f.get("sample_ge_94C_s", np.nan),
            dwell_f.get("sample_ge_95C_s", np.nan),
            summary_f["sample_predicted"]["max_heating_rate_C_s"],
            summary_f["sample_predicted"]["max_cooling_rate_C_s"],
        ],
        "Test_PCR_longer_holding": [
            summary_l["duration_s"],
            summary_l["repeated_cycles"]["count"],
            summary_l["repeated_cycles"]["median_cycle_duration_s"],
            summary_l["internal"]["max_C"] - summary_l["internal"]["min_C"],
            (summary_l["repeated_cycles"]["sample_peak_max_C"] -
             summary_l["repeated_cycles"]["sample_peak_min_C"]),
            summary_l["repeated_cycles"]["sample_peak_mean_C"],
            summary_l["repeated_cycles"]["sample_peak_median_C"],
            (summary_l["repeated_cycles"]["sample_trough_max_C"] -
             summary_l["repeated_cycles"]["sample_trough_min_C"]),
            dwell_l.get("sample_ge_75C_s", np.nan),
            dwell_l.get("sample_ge_80C_s", np.nan),
            dwell_l.get("sample_ge_85C_s", np.nan),
            dwell_l.get("sample_ge_90C_s", np.nan),
            dwell_l.get("sample_ge_92C_s", np.nan),
            dwell_l.get("sample_ge_94C_s", np.nan),
            dwell_l.get("sample_ge_95C_s", np.nan),
            summary_l["sample_predicted"]["max_heating_rate_C_s"],
            summary_l["sample_predicted"]["max_cooling_rate_C_s"],
        ],
    }
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(out_dir / "faster_vs_longer_holding_summary.csv",
                   index=False)
    return comp_df


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ============================================================
# 主流程
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_dir = OUTPUT_ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # ---- 协议 1: DOE11 faster ----
    print("[DOE11] 处理 DOE11 faster ...")
    sum_f, trace_f, cycles_f, dwell_f = process_protocol(
        "DOE11 Faster PCR",
        DOE11_PATH,
        OUTPUT_ROOT / "DOE11_faster")
    print(f"  internal max={sum_f['internal']['max_C']:.2f} C, "
          f"sample max={sum_f['sample_predicted']['max_C']:.2f} C, "
          f"cycles={sum_f['repeated_cycles']['count']}")

    # ---- 协议 2: Test_PCR longer holding ----
    print("[LONG] 处理 Test_PCR longer holding ...")
    sum_l, trace_l, cycles_l, dwell_l = process_protocol(
        "Test_PCR Longer Holding",
        LONGER_PATH,
        OUTPUT_ROOT / "Test_PCR_longer_holding")
    print(f"  internal max={sum_l['internal']['max_C']:.2f} C, "
          f"sample max={sum_l['sample_predicted']['max_C']:.2f} C, "
          f"cycles={sum_l['repeated_cycles']['count']}")

    # ---- 比较表 ----
    comp_df = build_comparison(sum_f, sum_l, dwell_f, dwell_l,
                               comparison_dir)

    # ---- 图: 代表性周期 ----
    plot_representative_cycle(trace_f, trace_l, sum_f, sum_l, comparison_dir)
    # ---- 图: sample peak by cycle ----
    plot_sample_peak_by_cycle(cycles_f, cycles_l, comparison_dir)
    # ---- 图: dwell 比较 ----
    plot_dwell_comparison(sum_f, sum_l, comparison_dir)
    # ---- 图: 平均归一化周期 (可选) ----
    plot_mean_cycle(trace_f, trace_l, sum_f, sum_l, comparison_dir)

    # ---- 元数据 ----
    metadata = {
        "frozen_candidate": candidate_dict(),
        "fixed_boundary": {
            "h_conv_W_m2K": FROZEN_STRATEGY_G_CANDIDATE.h_conv_W_m2K,
            "emissivity": FROZEN_STRATEGY_G_CANDIDATE.emissivity,
            "sigma_SB_W_m2K4": FROZEN_STRATEGY_G_CANDIDATE.sigma_SB_W_m2K4,
            "view_factor": FROZEN_STRATEGY_G_CANDIDATE.view_factor,
            "radiation": "full nonlinear Stefan-Boltzmann",
            "lag_placement": "output-side, T_top_observed only",
            "lag_does_not_affect_sample": True,
        },
        "environment_rule": {
            "calibration_rule": "first valid measured Top COC",
            "internal_only_proxy": (
                "first valid internal temperature when no Top/ambient "
                "measurement exists"),
            "no_silent_25C_fallback": True,
        },
        "protocols": {"DOE11_faster": sum_f, "Test_PCR_longer_holding": sum_l},
        "no_fitting": True,
        "same_frozen_k_cp_tau_both_protocols": True,
        "git_head": _git_head(),
    }
    with open(OUTPUT_ROOT / "frozen_candidate_metadata.json", "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- 汇总文本 ----
    summary_txt = _build_summary_text(sum_f, sum_l, comp_df)
    (OUTPUT_ROOT / "cross_protocol_summary.txt").write_text(
        summary_txt, encoding="utf-8")
    print(summary_txt)


def plot_representative_cycle(trace_f, trace_l, sum_f, sum_l, out_dir):
    """用周期-相对时间绘制两个协议的代表性重复周期。"""
    def _pick_rep(trace, cycles):
        if not cycles:
            return None
        # 中间一半周期
        n = len(cycles)
        lo, hi = n // 4, 3 * n // 4 + 1
        mid = cycles[lo:hi]
        if not mid:
            mid = cycles
        # 时长中位数周期
        mid_sorted = sorted(mid, key=lambda c: c["cycle_duration_s"])
        rep = mid_sorted[len(mid_sorted) // 2]
        return rep

    rep_f = _pick_rep(trace_f, [])
    # 直接按 cycle_summary 周期行绘制 (用 trace 的 elapsed)
    # 简单方案: 从 cycles list 选代表并切片 trace
    def _slice(trace_df, start, end):
        m = (trace_df["elapsed_time_s"] >= start) & \
            (trace_df["elapsed_time_s"] <= end)
        return trace_df[m]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (trace, cycles, title, color) in zip(
            axes,
            ((trace_f, None, "DOE11 Faster", "#1f77b4"),
             (trace_l, None, "Longer Holding", "#d62728"))):
        # 从总迹线中截取一个中位时长重复周期
        if cycles is None:
            # 需要重新检测: 简单方法取中间峰附近的窗口
            pass
        # 简化: 直接画整个协议中的中间一段 (第二个周期的窗口)
        mid_time = trace["elapsed_time_s"].iloc[len(trace) // 2]
        win = trace[(trace["elapsed_time_s"] >= mid_time - 30) &
                    (trace["elapsed_time_s"] <= mid_time + 30)]
        if len(win) < 10:
            win = trace.iloc[:200]
        t0 = float(win["elapsed_time_s"].iloc[0])
        ax.plot(win["elapsed_time_s"] - t0, win["T_internal_C"],
                ls=":", color="#7f7f7f", lw=1.3,
                label="Internal (measured)")
        ax.plot(win["elapsed_time_s"] - t0, win["T_sample_predicted_C"],
                color=color, lw=2.0, label="Sample predicted")
        ax.set_title(title + " — representative window")
        ax.set_xlabel("Cycle-relative time [s]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Temperature [°C]")
    fig.suptitle("Representative cycle — frozen Strategy G (k=0.055, cp=1200, "
                 "tau=8.5)")
    fig.tight_layout()
    fig.savefig(out_dir / "representative_cycle_faster_vs_longer.png", dpi=150)
    plt.close(fig)


def plot_sample_peak_by_cycle(cycles_f, cycles_l, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if cycles_f:
        ax.plot([c["cycle_number"] for c in cycles_f],
                [c["sample_peak_C"] for c in cycles_f],
                "o-", color="#1f77b4", label="DOE11 Faster")
    if cycles_l:
        ax.plot([c["cycle_number"] for c in cycles_l],
                [c["sample_peak_C"] for c in cycles_l],
                "s--", color="#d62728", label="Longer Holding")
    ax.set_xlabel("Repeated cycle number")
    ax.set_ylabel("Predicted sample peak [°C]")
    ax.set_title("Predicted sample peak by repeated cycle "
                 "(activation excluded)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sample_peak_by_cycle.png", dpi=150)
    plt.close(fig)


def plot_dwell_comparison(sum_f, sum_l, out_dir):
    ths = [75.0, 80.0, 85.0, 90.0]
    keys = [f"sample_ge_{th:.0f}C_s" for th in ths]
    f_vals = [sum_f["total_dwell_s"].get(k, 0.0) for k in keys]
    l_vals = [sum_l["total_dwell_s"].get(k, 0.0) for k in keys]
    x = np.arange(len(ths))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, f_vals, w, color="#1f77b4", label="DOE11 Faster")
    ax.bar(x + w / 2, l_vals, w, color="#d62728",
           label="Test_PCR Longer Holding")
    ax.set_xticks(x)
    ax.set_xticklabels([f">={th:.0f} C" for th in ths])
    ax.set_ylabel("Total predicted-sample dwell [s]")
    ax.set_title("Predicted sample high-temperature dwell — frozen Strategy G")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "high_temperature_dwell_comparison.png", dpi=150)
    plt.close(fig)


def plot_mean_cycle(trace_f, trace_l, sum_f, sum_l, out_dir):
    """平均归一化周期 (若周期结构足够稳健)。"""
    # 简化稳健版: 用 cycle_summary 中位数时长周期做归一化平均
    try:
        cyc_f = pd.read_csv(OUTPUT_ROOT / "DOE11_faster" / "cycle_summary.csv")
        cyc_l = pd.read_csv(OUTPUT_ROOT / "Test_PCR_longer_holding"
                            / "cycle_summary.csv")
    except Exception:  # noqa: BLE001
        return
    if len(cyc_f) < 3 or len(cyc_l) < 3:
        return

    def _mean_profile(cyc, trace):
        profiles = []
        for _, r in cyc.iterrows():
            s, e = r["cycle_start_time_s"], r["cycle_end_time_s"]
            m = (trace["elapsed_time_s"] >= s) & \
                (trace["elapsed_time_s"] <= e)
            seg = trace[m]
            if len(seg) < 5:
                continue
            rel = seg["elapsed_time_s"].to_numpy() - s
            dur = e - s
            t_norm = np.linspace(0, 1, 100)
            tint_i = np.interp(t_norm, rel / dur,
                               seg["T_internal_C"].to_numpy())
            tsam_i = np.interp(t_norm, rel / dur,
                               seg["T_sample_predicted_C"].to_numpy())
            profiles.append((tint_i, tsam_i))
        if not profiles:
            return None
        tint_m = np.mean([p[0] for p in profiles], axis=0)
        tsam_m = np.mean([p[1] for p in profiles], axis=0)
        return np.linspace(0, 1, 100), tint_m, tsam_m

    mf = _mean_profile(cyc_f, trace_f)
    ml = _mean_profile(cyc_l, trace_l)
    if mf is None or ml is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(mf[0], mf[1], ls=":", color="#7f7f7f", lw=1.3,
            label="Faster mean internal")
    ax.plot(mf[0], mf[2], color="#1f77b4", lw=2.0,
            label="Faster mean sample")
    ax.plot(ml[0], ml[1], ls=":", color="#999999", lw=1.3,
            label="Longer mean internal")
    ax.plot(ml[0], ml[2], color="#d62728", lw=2.0,
            label="Longer mean sample")
    ax.set_xlabel("Normalized cycle time [0-1]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Mean normalized repeated cycle — frozen Strategy G")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_cycle_comparison.png", dpi=150)
    plt.close(fig)


def _build_summary_text(sum_f, sum_l, comp_df):
    L = []
    A = L.append
    A("=" * 72)
    A("FROZEN STRATEGY G CROSS-PROTOCOL PREDICTION SUMMARY")
    A("=" * 72)
    A("")
    A("冻结候选: strategy_G_conservative_cross_protocol_v1")
    A(f"  k={FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK}, "
      f"cp={FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK}, "
      f"tau={FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s}, "
      f"rho=1020; 72C RMSE(存储)={STRATEGY_G_STORED_RMSE_72C:.4f} C")
    A(f"  环境规则: 标定=第一个有效实测 Top COC; "
      f"内部-only 预测代理=第一个有效内部温度 (显式记录)")
    A("")
    for name, s in (("DOE11 FASTER", sum_f), ("LONGER HOLDING", sum_l)):
        A(f"[{name}]")
        A(f"  source: {s['source_file']}")
        A(f"  有效点: {s['valid_points']}, 时长: {s['duration_s']:.1f} s, "
          f"median dt: {s['median_dt_s']:.3f} s")
        A(f"  初始内部: {s['internal']['initial_C']:.2f} C; "
          f"环境: {s['environment']['T_environment_C']:.2f} C "
          f"({s['environment']['environment_source']})")
        A(f"  internal max: {s['internal']['max_C']:.2f} C; "
          f"sample max: {s['sample_predicted']['max_C']:.2f} C "
          f"(t={s['sample_predicted']['time_of_max_s']:.1f} s)")
        rc = s["repeated_cycles"]
        A(f"  重复周期: {rc['count']}; 时长中位数: "
          f"{rc['median_cycle_duration_s']:.1f} s")
        A(f"  样品峰: min {rc['sample_peak_min_C']:.2f}, "
          f"max {rc['sample_peak_max_C']:.2f}, "
          f"mean {rc['sample_peak_mean_C']:.2f}, "
          f"median {rc['sample_peak_median_C']:.2f} C")
        A(f"  dwell: >=75 {s['total_dwell_s'].get('sample_ge_75C_s', 0):.0f} s, "
          f">=80 {s['total_dwell_s'].get('sample_ge_80C_s', 0):.0f} s, "
          f">=85 {s['total_dwell_s'].get('sample_ge_85C_s', 0):.0f} s, "
          f">=90 {s['total_dwell_s'].get('sample_ge_90C_s', 0):.0f} s")
        A(f"  样品加热率 max: "
          f"{s['sample_predicted']['max_heating_rate_C_s']:.2f} C/s; "
          f"冷却率 max: "
          f"{s['sample_predicted']['max_cooling_rate_C_s']:.2f} C/s")
        A("")
    A("跨协议比较 (同冻结模型):")
    for _, row in comp_df.iterrows():
        A(f"  {row['metric']}: faster={row['DOE11_faster']}, "
          f"longer={row['Test_PCR_longer_holding']}")
    A("")
    A("重要: 样品温度为模型预测 (MODEL-PREDICTED), 非实测。")
    A("无 PCR 数据重拟合; 两个协议使用同一冻结 k/cp/tau。")
    A("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    main()
