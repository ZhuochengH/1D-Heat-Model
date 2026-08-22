#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase A — 修正重复周期 dwell 统计 (v2 corrected)
================================================

问题:
    v1 比较表把「全协议 dwell」当作每周期 dwell 的近似 (total/cycles),
    可能混入激活相/非 PCR 时段。

本脚本用同一冻结模型 (k=0.055, cp=1200, tau=8.5) 重新生成:

    calibrated_model_output/
    strategy_G_conservative_cross_protocol_v2_corrected_dwell/

    DOE11_faster/
        phase_specific_thermal_dwell.csv
        per_cycle_thermal_dwell.csv
        sample_temperature_prediction.csv
        sample_temperature_prediction.png/.pdf
        cycle_summary.csv
        thermal_dwell_summary.csv  (旧格式保留诊断)
    Test_PCR_longer_holding/  (同上)
    comparison/
        corrected_repeated_cycle_dwell_comparison.csv
        corrected_repeated_cycle_dwell_comparison.png
        faster_vs_longer_holding_summary.csv  (修正版: 使用重复周期均值 dwell)

不覆盖 v1 输出。冻结模型参数不变。
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
from thermal_model.utilities.predict_sample_from_internal_temperature import (
    load_internal_data, ramp_summary)
from thermal_model.utilities.phase_thermal_dwell import (
    build_phase_intervals,
    phase_specific_dwell_table,
    phase_peak_stats,
    dwell_full_range,
    DWELL_THRESHOLDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"
DOE11_PATH = CALIBRATION_DIR / "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
LONGER_PATH = CALIBRATION_DIR / "Test_PCR longer holding.xlsx"

OUTPUT_ROOT = (PROJECT_ROOT / "calibrated_model_output"
               / "strategy_G_conservative_cross_protocol_v2_corrected_dwell")

SAVE_DT = 0.1
PEAK_THRESHOLD = 88.0
DIP_THRESHOLD = 60.0
MIN_PEAK_SEPARATION_S = 15.0
ACTIVATION_MIN_S = 30.0


def resolve_environment_proxy(t_internal, top_measured=None):
    """下游内部-only 预测环境代理 (显式记录)。"""
    if top_measured is not None:
        arr = np.asarray(top_measured, dtype=float)
        valid = np.flatnonzero(np.isfinite(arr))
        if valid.size:
            return {"T_environment_C": float(arr[valid[0]]),
                    "environment_source": "INITIAL_MEASURED_TOP"}
    tint = np.asarray(t_internal, dtype=float)
    valid = np.flatnonzero(np.isfinite(tint))
    if valid.size == 0:
        raise ValueError("无有效内部温度。")
    return {"T_environment_C": float(tint[valid[0]]),
            "environment_source": "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT"}


def run_frozen_strategy_G(elapsed, t_internal, t_env):
    c = FROZEN_STRATEGY_G_CANDIDATE
    mats = cr.make_convection_radiation_materials(c.k_eff_W_mK,
                                                  c.cp_eff_J_kgK)
    T_initial = float(np.asarray(t_internal, dtype=float)[0])
    result = cr.run_convection_radiation_fdm(
        time_s=elapsed, bottom_temperature_C=t_internal, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=SAVE_DT, T_initial_C=T_initial)
    t_arr = result["t_array"]
    T_sample = result["T_sample_arr"]
    T_top = result["T_top_surface_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top, c.tau_lag_s)
    return t_arr, T_sample, T_top, T_top_obs, T_initial


def sample_at_measurement_times(meas_time, model_time, model_trace):
    return np.interp(meas_time, model_time, model_trace)


def detect_activation_and_repeated_cycles(t, t_internal, t_sample,
                                          peak_threshold=PEAK_THRESHOLD,
                                          dip_threshold=DIP_THRESHOLD,
                                          min_sep=MIN_PEAK_SEPARATION_S,
                                          activation_min_s=ACTIVATION_MIN_S):
    """峰-谷检测 + 激活相分离 (与 v1 相同逻辑, 保持可复用)。"""
    t = np.asarray(t, dtype=float)
    ti = np.asarray(t_internal, dtype=float)
    ts = np.asarray(t_sample, dtype=float)
    n = len(t)

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

    selected = []
    for idx in merged:
        if selected and (t[idx] - t[selected[-1]]) < min_sep:
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
            "sample_trough_time_s": float(t[trough]),
            "has_prior_trough": bool(prior),
        })

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


def process_protocol(name, path, out_dir, sheet=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sheet is None:
        sheet = _resolve_sheet(path)
    data = load_internal_data(path, sheet=sheet, time_col="Time(s)",
                              temp_col="Zone 1 Avg (°C)")
    t = data["source_time_s"]
    elapsed = data["elapsed_time_s"]
    tint = data["T_internal_C"]
    env = resolve_environment_proxy(tint)

    t_arr, T_sample, T_top, T_top_obs, T_init = run_frozen_strategy_G(
        elapsed, tint, env["T_environment_C"])
    T_sample_pred = sample_at_measurement_times(elapsed, t_arr, T_sample)
    T_top_pred = sample_at_measurement_times(elapsed, t_arr, T_top)

    trace = pd.DataFrame({
        "source_time_s": t,
        "elapsed_time_s": elapsed,
        "T_internal_C": tint,
        "T_sample_predicted_C": T_sample_pred,
        "T_top_FDM_C": T_top_pred,
        "T_top_observed_predicted_C": sample_at_measurement_times(
            elapsed, t_arr, T_top_obs),
    })
    trace["delta_sample_minus_internal_C"] = T_sample_pred - tint
    trace.to_csv(out_dir / "sample_temperature_prediction.csv", index=False)

    cyc = detect_activation_and_repeated_cycles(elapsed, tint, T_sample_pred)
    activation = cyc["activation"]
    repeated = cyc["repeated_cycles"]

    # ---- 重复周期指标 (cycle_summary.csv) ----
    cyc_rows = []
    for c in repeated:
        cyc_rows.append({
            "cycle_number": c["cycle_number"],
            "cycle_start_time_s": c["cycle_start_time_s"],
            "internal_peak_time_s": c["internal_peak_time_s"],
            "internal_high_peak_C": c["internal_high_peak_C"],
            "sample_peak_C": c["sample_high_peak_C"],
            "sample_peak_time_s": c["sample_peak_time_s"],
            "internal_low_trough_C": c["internal_low_trough_C"],
            "sample_trough_C": c["sample_low_trough_C"],
            "cycle_duration_s": c.get("cycle_duration_s"),
        })
    pd.DataFrame(cyc_rows).to_csv(out_dir / "cycle_summary.csv", index=False)

    # ---- 相区间 ----
    intervals = build_phase_intervals(activation, repeated, elapsed)
    phase_rows, per_cycle_rows = phase_specific_dwell_table(
        elapsed, T_sample_pred, intervals)
    pd.DataFrame(phase_rows).to_csv(
        out_dir / "phase_specific_thermal_dwell.csv", index=False)
    pd.DataFrame(per_cycle_rows).to_csv(
        out_dir / "per_cycle_thermal_dwell.csv", index=False)

    # ---- 相特定峰值 ----
    peak_stats = phase_peak_stats(
        cyc_rows, activation,
        {"t": elapsed, "T_sample": T_sample_pred})

    # ---- 旧格式诊断 (thermal_dwell_summary.csv, 保留但不作为比较依据) ----
    total_dwell = dwell_full_range(elapsed, T_sample_pred)
    dwell_rows = [{"scope": "TOTAL"} | total_dwell]
    rep_total = {}
    for th in DWELL_THRESHOLDS:
        rep_total[f"sample_ge_{th:.0f}C_s"] = sum(
            row["repeated_cycle_total_dwell_s"]
            for row in phase_rows if row["threshold_C"] == th)
    dwell_rows.append({"scope": "REPEATED_CYCLES_TOTAL"} | rep_total)
    pd.DataFrame(dwell_rows).to_csv(out_dir / "thermal_dwell_summary.csv",
                                    index=False)

    # ---- 重复周期指标 (cycle_summary.csv) ----
    cyc_rows = []
    for c in repeated:
        cyc_rows.append({
            "cycle_number": c["cycle_number"],
            "cycle_start_time_s": c["cycle_start_time_s"],
            "internal_peak_time_s": c["internal_peak_time_s"],
            "internal_high_peak_C": c["internal_high_peak_C"],
            "sample_peak_C": c["sample_high_peak_C"],
            "sample_peak_time_s": c["sample_peak_time_s"],
            "internal_low_trough_C": c["internal_low_trough_C"],
            "sample_trough_C": c["sample_low_trough_C"],
            "cycle_duration_s": c.get("cycle_duration_s"),
        })
    pd.DataFrame(cyc_rows).to_csv(out_dir / "cycle_summary.csv", index=False)

    # ---- 图 ----
    _plot(elapsed, tint, T_sample_pred, T_top_pred, name, out_dir)

    # ---- 汇总 ----
    d = T_sample_pred - tint
    sramp = ramp_summary(elapsed, T_sample_pred)
    peaks = [c["sample_peak_C"] for c in cyc_rows]
    troughs = [c["sample_trough_C"] for c in cyc_rows]
    summary = {
        "protocol": name,
        "source_file": str(path.resolve()),
        "sheet": sheet,
        "valid_points": data["n_valid"],
        "duration_s": data["duration_s"],
        "median_dt_s": data["median_dt"],
        "initial_internal_C": data["T_initial_C"],
        "environment": env,
        "no_fitting": True,
        "activation_interval": intervals["activation_interval"],
        "repeated_cycle_intervals": intervals["repeated_cycle_intervals"],
        "post_cycle_interval": intervals["post_cycle_interval"],
        "activation_sample_max_C": peak_stats["activation_sample_max"],
        "repeated_cycle_peaks": {
            "min_C": peak_stats["repeated_cycle_sample_peak_min_C"],
            "max_C": peak_stats["repeated_cycle_sample_peak_max_C"],
            "mean_C": peak_stats["repeated_cycle_sample_peak_mean_C"],
            "median_C": peak_stats["repeated_cycle_sample_peak_median_C"],
        },
        "repeated_cycle_troughs": {
            "min_C": float(np.min(troughs)) if troughs else None,
            "max_C": float(np.max(troughs)) if troughs else None,
            "mean_C": float(np.mean(troughs)) if troughs else None,
            "median_C": float(np.median(troughs)) if troughs else None,
        },
        "internal": {
            "initial_C": data["T_initial_C"],
            "min_C": data["T_min_C"],
            "max_C": data["T_max_C"],
        },
        "sample_predicted": {
            "min_C": float(np.min(T_sample_pred)),
            "max_C": float(np.max(T_sample_pred)),
            "time_of_max_s": float(elapsed[int(np.argmax(T_sample_pred))]),
            "mean_abs_internal_sample_diff_C": float(np.mean(np.abs(d))),
            "max_heating_rate_C_s": sramp["max_positive_C_per_s"],
            "max_cooling_rate_C_s": sramp["max_negative_C_per_s"],
        },
        "phase_dwell": phase_rows,
        "repeated_cycle_count": len(repeated),
    }
    return summary, trace, phase_rows, per_cycle_rows, cyc_rows


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


def _plot(elapsed, tint, t_sample, t_top, name, out_dir):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(elapsed, tint, color="#7f7f7f", lw=1.2, ls=":",
            label="Internal sensor input (measured)")
    ax.plot(elapsed, t_sample, color="#2ca02c", lw=2.0,
            label="Sample predicted (model estimate)")
    ax.plot(elapsed, t_top, color="#1f77b4", lw=1.4, ls="--",
            label="Top COC FDM (model estimate)")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{name} — Frozen Strategy G Prediction (v2 corrected "
                 "dwell)\n(k=0.055, cp=1200, tau=8.5; h=10 + radiation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_temperature_prediction.png", dpi=150)
    fig.savefig(out_dir / "sample_temperature_prediction.pdf")
    plt.close(fig)


def corrected_dwell_for_protocol(phase_rows):
    """从 phase_rows 提取重复周期均值 dwell dict (阈值 -> mean)。"""
    return {f"sample_ge_{r['threshold_C']:.0f}C_s": r[
        "repeated_cycle_mean_dwell_s"] for r in phase_rows}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comp_dir = OUTPUT_ROOT / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    sum_f, trace_f, phase_f, percyc_f, cyc_f = process_protocol(
        "DOE11 Faster PCR", DOE11_PATH, OUTPUT_ROOT / "DOE11_faster")
    sum_l, trace_l, phase_l, percyc_l, cyc_l = process_protocol(
        "Test_PCR Longer Holding", LONGER_PATH,
        OUTPUT_ROOT / "Test_PCR_longer_holding")

    print(f"[DOE11] 激活区间: {sum_f['activation_interval']}, "
          f"重复周期: {sum_f['repeated_cycle_count']}")
    print(f"[LONG]  激活区间: {sum_l['activation_interval']}, "
          f"重复周期: {sum_l['repeated_cycle_count']}")

    # ---- 修正比较表 (重复周期均值 dwell) ----
    dw_f = corrected_dwell_for_protocol(phase_f)
    dw_l = corrected_dwell_for_protocol(phase_l)
    ths = [75.0, 80.0, 85.0, 90.0, 92.0, 94.0, 95.0]
    comp = pd.DataFrame({
        "threshold_C": ths,
        "DOE11_repeated_mean_dwell_s": [dw_f.get(
            f"sample_ge_{th:.0f}C_s", np.nan) for th in ths],
        "LONGER_repeated_mean_dwell_s": [dw_l.get(
            f"sample_ge_{th:.0f}C_s", np.nan) for th in ths],
        "DOE11_repeated_total_dwell_s": [next(
            r["repeated_cycle_total_dwell_s"] for r in phase_f
            if r["threshold_C"] == th) for th in ths],
        "LONGER_repeated_total_dwell_s": [next(
            r["repeated_cycle_total_dwell_s"] for r in phase_l
            if r["threshold_C"] == th) for th in ths],
        "DOE11_activation_dwell_s": [next(
            r["activation_dwell_s"] for r in phase_f
            if r["threshold_C"] == th) for th in ths],
        "LONGER_activation_dwell_s": [next(
            r["activation_dwell_s"] for r in phase_l
            if r["threshold_C"] == th) for th in ths],
    })
    comp.to_csv(comp_dir / "corrected_repeated_cycle_dwell_comparison.csv",
                index=False)

    # ---- 主呈现比较图 (重复周期均值 dwell) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len([75.0, 80.0, 85.0, 90.0]))
    f_vals = [dw_f.get(f"sample_ge_{th:.0f}C_s", 0.0)
              for th in (75.0, 80.0, 85.0, 90.0)]
    l_vals = [dw_l.get(f"sample_ge_{th:.0f}C_s", 0.0)
              for th in (75.0, 80.0, 85.0, 90.0)]
    w = 0.35
    ax.bar(x - w / 2, f_vals, w, color="#1f77b4", label="DOE11 Faster")
    ax.bar(x + w / 2, l_vals, w, color="#d62728",
           label="Test_PCR Longer Holding")
    ax.set_xticks(x)
    ax.set_xticklabels([">=75 C", ">=80 C", ">=85 C", ">=90 C"])
    ax.set_ylabel("Mean repeated-cycle sample dwell [s/cycle]")
    ax.set_title("Corrected repeated-cycle mean dwell — frozen Strategy G")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(comp_dir / "corrected_repeated_cycle_dwell_comparison.png",
                dpi=150)
    plt.close(fig)

    # ---- 协议级修正汇总 (faster_vs_longer) ----
    rows = []
    for metric in ("repeated_cycle_count",
                   "repeated_cycle_peaks_mean_C",
                   "repeated_cycle_peaks_median_C",
                   "repeated_cycle_peaks_max_C",
                   "activation_sample_max_C",
                   "mean_dwell_ge75", "mean_dwell_ge80",
                   "mean_dwell_ge85", "mean_dwell_ge90",
                   "total_dwell_ge85"):
        if metric.startswith("mean_dwell_ge"):
            th = float(metric.replace("mean_dwell_ge", ""))
            fv = dw_f.get(f"sample_ge_{th:.0f}C_s", np.nan)
            lv = dw_l.get(f"sample_ge_{th:.0f}C_s", np.nan)
        elif metric == "total_dwell_ge85":
            fv = next(r["total_protocol_dwell_s"] for r in phase_f
                      if r["threshold_C"] == 85.0)
            lv = next(r["total_protocol_dwell_s"] for r in phase_l
                      if r["threshold_C"] == 85.0)
        elif metric.startswith("repeated_cycle_peaks"):
            key = metric.replace("repeated_cycle_peaks_", "")
            fv = sum_f["repeated_cycle_peaks"][key]
            lv = sum_l["repeated_cycle_peaks"][key]
        else:
            fv = sum_f[metric]
            lv = sum_l[metric]
        rows.append({"metric": metric, "DOE11_faster": fv,
                     "Test_PCR_longer_holding": lv})
    pd.DataFrame(rows).to_csv(
        comp_dir / "faster_vs_longer_holding_summary.csv", index=False)

    # ---- 元数据 ----
    metadata = {
        "phase": "A",
        "purpose": "corrected repeated-cycle dwell (v2)",
        "frozen_candidate": {
            "k_eff_W_mK": FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK,
            "cp_eff_J_kgK": FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK,
            "tau_lag_s": FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s,
        },
        "dwell_definition": {
            "total_protocol_dwell": "integral over full protocol",
            "activation_dwell": "integral within activation interval only",
            "repeated_cycle_total_dwell": "sum over repeated-cycle intervals",
            "per_cycle_dwell": "per-cycle integrals, mean/median/min/max/std",
            "prohibited": "total_protocol_dwell / cycle_count",
        },
        "DOE11": {"activation_interval": sum_f["activation_interval"],
                  "repeated_cycle_intervals":
                      sum_f["repeated_cycle_intervals"],
                  "post_cycle_interval": sum_f["post_cycle_interval"],
                  "repeated_cycle_count": sum_f["repeated_cycle_count"]},
        "Test_PCR": {"activation_interval": sum_l["activation_interval"],
                     "repeated_cycle_intervals":
                         sum_l["repeated_cycle_intervals"],
                     "post_cycle_interval": sum_l["post_cycle_interval"],
                     "repeated_cycle_count": sum_l["repeated_cycle_count"]},
        "v1_output_unchanged": True,
    }
    with open(OUTPUT_ROOT / "phase_A_metadata.json", "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- 汇总文本 ----
    txt = _summary_text(sum_f, sum_l, phase_f, phase_l)
    (OUTPUT_ROOT / "phase_A_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


def _summary_text(sum_f, sum_l, phase_f, phase_l):
    L = []
    A = L.append
    A("=" * 70)
    A("PHASE A — CORRECTED REPEATED-CYCLE DWELL (v2)")
    A("=" * 70)
    A("")
    for name, s, ph in (("DOE11 FASTER", sum_f, phase_f),
                        ("LONGER HOLDING", sum_l, phase_l)):
        A(f"[{name}]")
        A(f"  激活区间: {s['activation_interval']}")
        A(f"  重复周期: {s['repeated_cycle_count']}; "
          f"区间数: {len(s['repeated_cycle_intervals'])}")
        A(f"  末尾相: {s['post_cycle_interval']}")
        A(f"  激活样品最大: {s['activation_sample_max_C']}")
        rc = s["repeated_cycle_peaks"]
        A(f"  重复周期样品峰: min {rc['min_C']}, max {rc['max_C']}, "
          f"mean {rc['mean_C']}, median {rc['median_C']} C")
        A("  重复周期均值 dwell (s/cycle):")
        for r in ph:
            A(f"    >= {r['threshold_C']:.0f} C: mean {r['repeated_cycle_mean_dwell_s']:.2f}, "
              f"median {r['repeated_cycle_median_dwell_s']:.2f}, "
              f"total {r['repeated_cycle_total_dwell_s']:.1f} s "
              f"(正周期 {r['cycles_with_positive_dwell']}/{r['repeated_cycle_count']})")
        A(f"  激活 dwell: >=75 {next(r['activation_dwell_s'] for r in ph if r['threshold_C']==75):.1f} s, "
          f">=85 {next(r['activation_dwell_s'] for r in ph if r['threshold_C']==85):.1f} s")
        A(f"  全协议 dwell: >=85 {next(r['total_protocol_dwell_s'] for r in ph if r['threshold_C']==85):.1f} s")
        A("")
    A("修正: 不再使用 total/cycles 作为每周期 dwell。")
    A("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    main()
