#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化下游热解释 — 仅预测样品峰温度 (frozen output-side-lag model)
=================================================================

项目决定: 停止扩展模型复杂度。保留 ONLY 输出侧有效滞后架构:

    实测内部温度 -> 1D 多层 FDM -> 预测样品温度
    (滞后不作用于样品)

    原始预测 Top COC -> 输出侧一阶滞后 tau_top -> 预测实测 Top COC

主下游指标 = 预测样品峰温度 (不再是 dwell / 相位分类)。

冻结参数 (不重拟合 / 不扫描):
    k_eff = 0.055 W/(m K)
    cp_eff = 1200 J/(kg K)
    rho_COC = 1020 kg/m3
    tau_top = 8.5 s

固定顶部边界 (Strategy E, 不变):
    h_conv = 10.0 W/(m2 K)
    epsilon = 0.90
    sigma_SB = 5.670374419e-8
    F_view = 1.0
    非线性 Stefan-Boltzmann 辐射
    几何 = BARE_TOP_COC_LAYERS

环境规则:
    72C 标定: 第一个有效实测 Top COC 温度
    PCR 内部-only: 第一个有效内部温度 (INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT)

初始条件: 每个 PCR 运行 T_initial = 第一个有效内部温度 (整场均匀)。

输出 (gitignored):
    calibrated_model_output/frozen_output_lag_sample_peak_analysis_v1/
        DOE11_faster/
            sample_peak_summary.csv
            sample_temperature_trace.csv
            sample_peak_temperature.png/.pdf
            repeated_cycle_sample_peaks.png
        Test_PCR_longer_holding/ (同上)
        comparison/
            sample_peak_comparison.csv
            faster_vs_longer_sample_peak.png
            frozen_sample_peak_metadata.json
            frozen_sample_peak_summary.txt

不覆盖任何历史输出; 不重跑滞后放置比较; 不引入 dwell 新指标。
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
DOE11_PATH = CALIBRATION_DIR / (
    "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx")
LONGER_PATH = CALIBRATION_DIR / "Test_PCR longer holding.xlsx"

OUTPUT_ROOT = (PROJECT_ROOT / "calibrated_model_output"
               / "frozen_output_lag_sample_peak_analysis_v1")

SAVE_DT = 0.1
PEAK_THRESHOLD = 88.0
DIP_THRESHOLD = 60.0
MIN_PEAK_SEPARATION_S = 15.0
ACTIVATION_MIN_S = 30.0

# 描述性 PCR 热参考阈值 (仅作为热学合理性参考, 不是校准/通过标准)
THRESHOLDS = (85.0, 90.0, 92.0, 95.0)


# ============================================================
# 环境 / 加载
# ============================================================

def resolve_environment_proxy(t_internal):
    """内部-only PCR 预测环境代理 (显式记录, 无静默 25 C)。"""
    tint = np.asarray(t_internal, dtype=float)
    valid = np.flatnonzero(np.isfinite(tint))
    if valid.size == 0:
        raise ValueError("无有效内部温度。")
    return {"T_environment_C": float(tint[valid[0]]),
            "environment_source":
                "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT"}


def _resolve_sheet(path):
    xl = pd.ExcelFile(path)
    if "Extracted_Data" in xl.sheet_names:
        return "Extracted_Data"
    for sh in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sh, nrows=3)
            if "Zone 1 Avg (°C)" in df.columns:
                return sh
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"找不到含温度列的工作表: {xl.sheet_names}")


# ============================================================
# 冻结输出侧滞后模型 (单一权威路径)
# ============================================================

def run_frozen_output_lag(elapsed, tint, t_env):
    """输出侧滞后模型: FDM + 顶部输出滞后 (样品不滞后)。

    返回 dict: t_arr / T_sample / T_top_fdm / T_top_obs / T_initial。
    """
    c = FROZEN_STRATEGY_G_CANDIDATE
    mats = cr.make_convection_radiation_materials(c.k_eff_W_mK,
                                                  c.cp_eff_J_kgK)
    T_initial = float(np.asarray(tint, dtype=float)[0])
    result = cr.run_convection_radiation_fdm(
        time_s=elapsed, bottom_temperature_C=tint, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=SAVE_DT, T_initial_C=T_initial)
    t_arr = result["t_array"]
    T_sample = result["T_sample_arr"]
    T_top_fdm = result["T_top_surface_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top_fdm, c.tau_lag_s)
    return {"t_arr": t_arr, "T_sample": T_sample, "T_top_fdm": T_top_fdm,
            "T_top_obs": T_top_obs, "T_initial": T_initial}


# ============================================================
# 周期检测 (复用现有逻辑, 不新增启发式)
# ============================================================

def detect_repeated_cycles(t, t_internal, t_sample):
    """峰-谷检测 + 激活相分离 (与 v1/v2 相同逻辑, 保持可复用)。"""
    t = np.asarray(t, dtype=float)
    ti = np.asarray(t_internal, dtype=float)
    ts = np.asarray(t_sample, dtype=float)
    n = len(t)

    peaks = []
    i = 1
    while i < n - 1:
        if ti[i] >= PEAK_THRESHOLD and ti[i] >= ti[i - 1]:
            j = i
            while j + 1 < n and ti[j + 1] == ti[j]:
                j += 1
            if j == n - 1 or ti[j] >= ti[j + 1]:
                peaks.append(j)
            i = j + 1
        else:
            i += 1

    troughs = []
    i = 1
    while i < n - 1:
        if ti[i] < DIP_THRESHOLD and ti[i] <= ti[i - 1]:
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

    merged = []
    for idx in peaks:
        if merged:
            prev = merged[-1]
            between = ti[prev + 1:idx]
            min_between = float(np.min(between)) if between.size else \
                min(float(ti[prev]), float(ti[idx]))
            if min_between < DIP_THRESHOLD:
                merged.append(idx)
            else:
                if ti[idx] > ti[prev]:
                    merged[-1] = idx
        else:
            merged.append(idx)

    selected = []
    for idx in merged:
        if selected and (t[idx] - t[selected[-1]]) < MIN_PEAK_SEPARATION_S:
            continue
        selected.append(idx)

    if len(selected) < 2:
        return {"activation": None, "repeated_cycles": []}

    phases = []
    for pidx in selected:
        prior = [tr for tr in troughs if tr < pidx]
        trough = prior[-1] if prior else 0
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
            "has_prior_trough": bool(prior),
        })

    activation = None
    repeated = []
    if phases:
        p0 = phases[0]
        if not p0["has_prior_trough"] and \
                (p0["internal_peak_time_s"] - t[0]) >= ACTIVATION_MIN_S:
            activation = p0
            rest = phases[1:]
        else:
            rest = phases
    else:
        rest = []

    for k, ph in enumerate(rest):
        c = dict(ph)
        c["cycle_number"] = k + 1
        repeated.append(c)

    return {"activation": activation, "repeated_cycles": repeated}


# ============================================================
# 指标
# ============================================================

def threshold_flags(T, thresholds=THRESHOLDS):
    """样品整体最大是否 >= 各阈值 (YES/NO)。"""
    mx = float(np.max(T))
    return {f"overall_ge{int(th)}": bool(mx >= th) for th in thresholds}


def repeated_peak_counts(peaks, thresholds=THRESHOLDS):
    """重复周期样品峰 >= 各阈值的个数。"""
    arr = np.asarray(peaks, dtype=float)
    return {f"repeated_peaks_ge{int(th)}_count": int(np.sum(arr >= th))
            for th in thresholds}


def summarize_protocol(name, path, out_dir):
    """单协议: 冻结输出侧滞后模型 + 简化样品峰分析。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = _resolve_sheet(path)
    data = load_internal_data(path, sheet=sheet, time_col="Time(s)",
                              temp_col="Zone 1 Avg (°C)")
    elapsed = data["elapsed_time_s"]
    tint = data["T_internal_C"]
    env = resolve_environment_proxy(tint)

    r = run_frozen_output_lag(elapsed, tint, env["T_environment_C"])
    t_arr = r["t_arr"]
    T_sample = r["T_sample"]
    T_top_fdm = r["T_top_fdm"]
    T_top_obs = r["T_top_obs"]

    # ---- 迹线 (在测量时间插值, 保持实际时间戳) ----
    trace = pd.DataFrame({
        "time_s": elapsed,
        "T_internal_measured_C": tint,
        "T_sample_predicted_C": np.interp(elapsed, t_arr, T_sample),
        "T_top_FDM_C": np.interp(elapsed, t_arr, T_top_fdm),
        "T_top_lagged_prediction_C": np.interp(elapsed, t_arr, T_top_obs),
    })
    trace.to_csv(out_dir / "sample_temperature_trace.csv", index=False)

    # ---- 周期检测 (复用现有; 若模糊则只报整体样品最大) ----
    # 样品插值到测量时间轴, 保证 t/t_internal/t_sample 等长
    T_sample_meas = np.interp(elapsed, t_arr, T_sample)
    cyc = detect_repeated_cycles(elapsed, tint, T_sample_meas)
    repeated = cyc["repeated_cycles"]
    if repeated:
        sample_peaks = [float(c["sample_high_peak_C"]) for c in repeated]
        internal_peaks = [float(c["internal_high_peak_C"])
                          for c in repeated]
        gaps = [ip - sp for ip, sp in zip(internal_peaks, sample_peaks)]
    else:
        sample_peaks = internal_peaks = gaps = []

    def _stats(x):
        if not x:
            return {"min": np.nan, "max": np.nan, "mean": np.nan,
                    "median": np.nan, "std": np.nan}
        return {"min": float(np.min(x)), "max": float(np.max(x)),
                "mean": float(np.mean(x)), "median": float(np.median(x)),
                "std": float(np.std(x))}

    sp_stats = _stats(sample_peaks)
    ip_stats = _stats(internal_peaks)
    gap_stats = _stats(gaps)

    i_max_idx = int(np.argmax(tint))
    s_max_idx = int(np.argmax(T_sample))
    internal_max = float(np.max(tint))
    sample_max = float(np.max(T_sample))
    t_sample_max = float(t_arr[s_max_idx])
    # 内部温度在样品最大时刻 (内部在 elapsed 轴, 插值到 t_arr 时刻)
    internal_at_sample_max = float(np.interp(t_arr[s_max_idx], elapsed, tint))
    peak_gap = internal_at_sample_max - sample_max

    flags = threshold_flags(T_sample)
    rep_counts = repeated_peak_counts(sample_peaks)

    summary = {
        "protocol": name,
        "k_eff": FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK,
        "cp_eff": FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK,
        "tau_top": FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s,
        "alpha_eff": FROZEN_STRATEGY_G_CANDIDATE.alpha_eff_m2_s,
        "effusivity": FROZEN_STRATEGY_G_CANDIDATE.effusivity,
        "environment_C": env["T_environment_C"],
        "environment_source": env["environment_source"],
        "internal_max_C": internal_max,
        "sample_max_C": sample_max,
        "time_of_sample_max_s": t_sample_max,
        "internal_at_sample_max_C": internal_at_sample_max,
        "internal_minus_sample_peak_gap_C": peak_gap,
        "repeated_cycle_count": len(repeated),
        "repeated_sample_peak_min_C": sp_stats["min"],
        "repeated_sample_peak_max_C": sp_stats["max"],
        "repeated_sample_peak_mean_C": sp_stats["mean"],
        "repeated_sample_peak_median_C": sp_stats["median"],
        "repeated_sample_peak_std_C": sp_stats["std"],
        "repeated_internal_peak_mean_C": ip_stats["mean"],
        "repeated_internal_minus_sample_peak_gap_mean_C": gap_stats["mean"],
        **flags,
        **rep_counts,
    }
    pd.DataFrame([summary]).to_csv(
        out_dir / "sample_peak_summary.csv", index=False)

    # ---- 图: 主图 (样品峰温度) ----
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(elapsed, tint, color="#7f7f7f", lw=1.1, ls=":",
            label="Measured internal temperature")
    ax.plot(elapsed, np.interp(elapsed, t_arr, T_sample),
            color="#2ca02c", lw=2.0, label="Predicted sample temperature")
    ax.plot(elapsed, np.interp(elapsed, t_arr, T_top_fdm),
            color="#1f77b4", lw=1.2, ls="--", alpha=0.7,
            label="Raw Top COC FDM (optional)")
    ax.axhline(90.0, color="#d62728", ls="--", lw=1.2, alpha=0.8,
               label="90 C thermal reference")
    ax.axhline(95.0, color="#8c564b", ls="--", lw=1.2, alpha=0.8,
               label="95 C thermal reference")
    ax.annotate(f"predicted sample max = {sample_max:.2f} C",
                xy=(t_sample_max, sample_max),
                xytext=(t_sample_max - 40, sample_max - 8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                fontsize=9, color="black")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(f"{name} — predicted sample peak (frozen output-side lag)\n"
                 f"(k={FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK}, "
                 f"cp={FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK:.0f}, "
                 f"tau_top={FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s:.1f} s; "
                 "h=10 + nonlinear radiation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_peak_temperature.png", dpi=150)
    fig.savefig(out_dir / "sample_peak_temperature.pdf")
    plt.close(fig)

    # ---- 图: 重复周期样品峰 ----
    if repeated:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        nums = [int(c["cycle_number"]) for c in repeated]
        ax.plot(nums, sample_peaks, "o-", color="#2ca02c", lw=1.5,
                label="Predicted sample peak (per repeated cycle)")
        ax.axhline(90.0, color="#d62728", ls="--", lw=1.2, alpha=0.8,
                   label="90 C thermal reference")
        ax.axhline(95.0, color="#8c564b", ls="--", lw=1.2, alpha=0.8,
                   label="95 C thermal reference")
        ax.set_xlabel("Repeated cycle number")
        ax.set_ylabel("Predicted sample peak [C]")
        ax.set_title(f"{name} — repeated-cycle predicted sample peaks")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "repeated_cycle_sample_peaks.png", dpi=150)
        plt.close(fig)

    return summary, trace, repeated, sample_peaks


# ============================================================
# 主流程
# ============================================================

def main():
    comp_dir = OUTPUT_ROOT / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    s_f, trace_f, rep_f, peaks_f = summarize_protocol(
        "DOE11 Faster PCR", DOE11_PATH, OUTPUT_ROOT / "DOE11_faster")
    s_l, trace_l, rep_l, peaks_l = summarize_protocol(
        "Test_PCR Longer Holding", LONGER_PATH,
        OUTPUT_ROOT / "Test_PCR_longer_holding")

    # ---- 跨协议比较 ----
    comp_rows = []
    for label, s, peaks in (("DOE11_faster", s_f, peaks_f),
                            ("Test_PCR_longer_holding", s_l, peaks_l)):
        arr = np.asarray(peaks, dtype=float) if peaks else np.array([])
        n = arr.size
        comp_rows.append({
            "protocol": label,
            "internal_max_C": s["internal_max_C"],
            "sample_max_C": s["sample_max_C"],
            "repeated_sample_peak_mean_C": s["repeated_sample_peak_mean_C"],
            "repeated_sample_peak_median_C":
                s["repeated_sample_peak_median_C"],
            "repeated_sample_peak_max_C": s["repeated_sample_peak_max_C"],
            "mean_internal_minus_sample_peak_gap_C":
                s["repeated_internal_minus_sample_peak_gap_mean_C"],
            "repeated_cycle_count": s["repeated_cycle_count"],
            "frac_cycle_peaks_ge85":
                float(np.sum(arr >= 85.0) / n) if n else np.nan,
            "frac_cycle_peaks_ge90":
                float(np.sum(arr >= 90.0) / n) if n else np.nan,
            "frac_cycle_peaks_ge92":
                float(np.sum(arr >= 92.0) / n) if n else np.nan,
            "frac_cycle_peaks_ge95":
                float(np.sum(arr >= 95.0) / n) if n else np.nan,
        })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(comp_dir / "sample_peak_comparison.csv", index=False)

    # ---- 主比较图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, peaks, title in ((axes[0], peaks_f, "DOE11 Faster"),
                             (axes[1], peaks_l, "Longer Holding")):
        if peaks:
            ax.plot(range(1, len(peaks) + 1), peaks, "o-", color="#2ca02c",
                    lw=1.5)
            ax.axhline(float(np.mean(peaks)), color="#1f77b4", ls="--",
                       lw=1.2, label=f"mean = {np.mean(peaks):.2f} C")
        ax.axhline(90.0, color="#d62728", ls="--", lw=1.2, alpha=0.8,
                   label="90 C thermal reference")
        ax.axhline(95.0, color="#8c564b", ls="--", lw=1.2, alpha=0.8,
                   label="95 C thermal reference")
        ax.set_title(title)
        ax.set_xlabel("Repeated cycle number")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Predicted sample peak [C]")
    fig.suptitle("Faster vs longer-holding — predicted sample peaks\n"
                 "(frozen output-side lag; 90/95 C = thermal reference)")
    fig.tight_layout()
    fig.savefig(comp_dir / "faster_vs_longer_sample_peak.png", dpi=150)
    plt.close(fig)

    # ---- 元数据 ----
    meta = {
        "purpose": "simplified downstream thermal interpretation: "
                   "predicted sample peak temperature only",
        "working_model": "output-side effective lag",
        "frozen": {
            "k_eff": s_f["k_eff"], "cp_eff": s_f["cp_eff"],
            "tau_top": s_f["tau_top"],
            "alpha_eff": s_f["alpha_eff"],
            "effusivity": s_f["effusivity"],
        },
        "fixed_boundary": {
            "h_conv_W_m2K": cr.H_CONV_STRATEGY_E_W_M2K,
            "emissivity": cr.EMISSIVITY_STRATEGY_E,
            "sigma_SB": cr.SIGMA_SB_W_M2_K4,
            "view_factor": cr.VIEW_FACTOR_STRATEGY_E,
            "radiation": "nonlinear Stefan-Boltzmann",
        },
        "lag_placement": {
            "chosen": "output-side",
            "historical_note": "output-side and input-side lag produced "
                               "similar Top COC fitting, while sample "
                               "reconstruction differed; retained output-side "
                               "as working reduced-order architecture "
                               "(modelling choice, not proof of unique lag)",
            "lag_does_not_affect_sample": True,
        },
        "environment_rule": {
            "calibration": "first valid measured Top COC",
            "internal_only_proxy":
                "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT",
            "no_silent_25C_fallback": True,
        },
        "initial_condition": "first valid internal temperature (uniform)",
        "bottom_dirichlet": "measured internal temperature trace",
        "thresholds": {"values": list(THRESHOLDS),
                       "note": "descriptive thermal plausibility reference "
                               "only; not calibration or pass/fail"},
        "no_dwell_analysis": True,
        "no_new_phase_heuristics": True,
        "no_refit": True,
        "no_optimizer": True,
        "no_lag_placement_rerun": True,
        "historical_outputs_unchanged": True,
        "limitation": "no direct sample-layer temperature measurement exists; "
                      "predicted sample peak is a model estimate",
    }
    (comp_dir / "frozen_sample_peak_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = _summary_text(s_f, s_l, peaks_f, peaks_l)
    (comp_dir / "frozen_sample_peak_summary.txt").write_text(
        txt, encoding="utf-8")
    print(txt)


def _band(mean_peak):
    if not np.isfinite(mean_peak):
        return "N/A"
    if mean_peak < 85.0:
        return "<85 C"
    if mean_peak < 90.0:
        return "85-90 C"
    if mean_peak < 95.0:
        return "90-95 C"
    return ">=95 C"


def _summary_text(s_f, s_l, peaks_f, peaks_l):
    L = []
    A = L.append
    A("=" * 70)
    A("FROZEN OUTPUT-LAG SAMPLE-PEAK ANALYSIS (simplified)")
    A("=" * 70)
    A(f"模型: 输出侧有效滞后; k={s_f['k_eff']}, cp={s_f['cp_eff']}, "
      f"tau_top={s_f['tau_top']} s; h=10 + 非线性辐射")
    A("主下游指标 = 预测样品峰温度; 不做 dwell / 相位启发式。")
    A("")
    for s, peaks, label in ((s_f, peaks_f, "DOE11 FASTER"),
                            (s_l, peaks_l, "LONGER HOLDING")):
        A(f"[{label}]")
        A(f"  internal max = {s['internal_max_C']:.2f} C; "
          f"predicted sample max = {s['sample_max_C']:.2f} C "
          f"@ t={s['time_of_sample_max_s']:.1f} s; "
          f"internal at sample max = {s['internal_at_sample_max_C']:.2f} C; "
          f"gap = {s['internal_minus_sample_peak_gap_C']:.2f} C")
        A(f"  repeated cycles = {s['repeated_cycle_count']}; "
          f"repeated sample peak min/max/mean/median/std = "
          f"{s['repeated_sample_peak_min_C']:.2f}/{s['repeated_sample_peak_max_C']:.2f}/"
          f"{s['repeated_sample_peak_mean_C']:.2f}/"
          f"{s['repeated_sample_peak_median_C']:.2f}/"
          f"{s['repeated_sample_peak_std_C']:.2f} C")
        A(f"  mean repeated internal peak = "
          f"{s['repeated_internal_peak_mean_C']:.2f} C; "
          f"mean internal-sample gap = "
          f"{s['repeated_internal_minus_sample_peak_gap_mean_C']:.2f} C")
        A(f"  overall sample max >=85: "
          f"{'YES' if s['overall_ge85'] else 'NO'}; "
          f">=90: {'YES' if s['overall_ge90'] else 'NO'}; "
          f">=92: {'YES' if s['overall_ge92'] else 'NO'}; "
          f">=95: {'YES' if s['overall_ge95'] else 'NO'}")
        if peaks:
            A(f"  repeated peaks >=85: {s['repeated_peaks_ge85_count']}/"
              f"{s['repeated_cycle_count']}; "
              f">=90: {s['repeated_peaks_ge90_count']}/"
              f"{s['repeated_cycle_count']}; "
              f">=92: {s['repeated_peaks_ge92_count']}/"
              f"{s['repeated_cycle_count']}; "
              f">=95: {s['repeated_peaks_ge95_count']}/"
              f"{s['repeated_cycle_count']}")
        A(f"  descriptive band (mean repeated sample peak): "
          f"{_band(s['repeated_sample_peak_mean_C'])}")
        A("")
    A("跨协议: delta_mean_peak = "
      f"{s_l['repeated_sample_peak_mean_C'] - s_f['repeated_sample_peak_mean_C']:.2f} C; "
      "delta_median_peak = "
      f"{s_l['repeated_sample_peak_median_C'] - s_f['repeated_sample_peak_median_C']:.2f} C")
    A("样品温度为模型预测 (非实测); 无直接样品温度验证。")
    A("阈值比较仅为热学合理性参考, 非通过/失败标准。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
