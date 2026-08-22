#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — 滞后放置架构 PCR 跨协议应用 (run_lag_placement_pcr.py)
=================================================================

把 72C 校准选出的每个架构最佳 tau 应用到两个 PCR 协议
(DOE11 faster + Test_PCR longer holding), 内部-only 环境代理
(INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT), 使用修正后的 Phase A
dwell 定义 (per-cycle 区间, 不用 total/N)。

输入: model_comparison_output/lag_placement_comparison_v1/
    72C_calibration/lag_placement_72C_best.csv (每架构最佳 tau)

输出 (gitignored):
    model_comparison_output/lag_placement_comparison_v1/
        DOE11_faster/        sample_prediction_by_architecture.csv + png
        Test_PCR_longer_holding/ (同上)
        comparison/
            lag_placement_PCR_comparison.csv
            DOE11_sample_by_lag_placement.png
            longer_holding_sample_by_lag_placement.png
            lag_placement_representative_cycles.png
            holding_time_effect_by_lag_placement.png
            repeated_cycle_dwell_by_lag_placement.png
            lag_placement_PCR_metadata.json

约束:
    - 不重拟合 k/cp; 不基于 PCR 样品预测选择滞后放置 (72C 已选);
    - 不引入 tau_input/tau_output 独立参数 (每架构单 tau);
    - 不修改物理; 不覆盖旧输出; 无提交/推送。
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
from thermal_model.utilities import lag_placement_comparison_model as lpm
from thermal_model.utilities.predict_sample_from_internal_temperature import load_internal_data
from thermal_model.utilities.phase_thermal_dwell import (
    build_phase_intervals, phase_specific_dwell_table,
    phase_peak_stats, DWELL_THRESHOLDS)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"
DOE11_PATH = CALIBRATION_DIR / (
    "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx")
LONGER_PATH = CALIBRATION_DIR / "Test_PCR longer holding.xlsx"

OUT_ROOT = (PROJECT_ROOT / "model_comparison_output"
            / "lag_placement_comparison_v1")
OUT72 = OUT_ROOT / "72C_calibration"
OUT_D11 = OUT_ROOT / "DOE11_faster"
OUT_LONG = OUT_ROOT / "Test_PCR_longer_holding"
OUT_COMP = OUT_ROOT / "comparison"

SAVE_DT = 0.1
PEAK_THRESHOLD = 88.0
DIP_THRESHOLD = 60.0
MIN_PEAK_SEPARATION_S = 15.0
ACTIVATION_MIN_S = 30.0


def resolve_environment_proxy(t_internal):
    """内部-only 预测环境代理 (与 Phase A 相同, 显式记录)。"""
    tint = np.asarray(t_internal, dtype=float)
    valid = np.flatnonzero(np.isfinite(tint))
    if valid.size == 0:
        raise ValueError("无有效内部温度。")
    return {"T_environment_C": float(tint[valid[0]]),
            "environment_source": "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT"}


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


def run_architecture_on_protocol(arch, tau, elapsed, tint, t_env):
    """在 PCR 协议上运行指定架构 (固定 k/cp), 返回 t_arr/T_sample/T_top/T_top_obs。"""
    runner = lpm.ARCHITECTURES[arch]["runner"]
    out = runner(elapsed, tint, t_env, tau, save_dt=SAVE_DT)
    return out["t_array"], out["T_sample_fdm"], out["T_top_fdm"], out["T_top_obs"]


def sample_at_measurement_times(meas_time, model_time, model_trace):
    return np.interp(meas_time, model_time, model_trace)


def detect_activation_and_repeated_cycles(t, t_internal, t_sample):
    """与 v1/v2 相同检测 (Phase A 复用)。"""
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
    for k, pidx in enumerate(selected):
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
        if k > 0:
            c["cycle_duration_s"] = float(
                ph["cycle_start_time_s"] - rest[k - 1]["cycle_start_time_s"])
        else:
            c["cycle_duration_s"] = None
        repeated.append(c)

    return {"activation": activation, "repeated_cycles": repeated}


def process_protocol(name, path, best_by_arch, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = _resolve_sheet(path)
    data = load_internal_data(path, sheet=sheet, time_col="Time(s)",
                              temp_col="Zone 1 Avg (°C)")
    elapsed = data["elapsed_time_s"]
    tint = data["T_internal_C"]
    env = resolve_environment_proxy(tint)

    rows = []
    arch_runs = {}
    for arch in ("O", "I", "S"):
        tau = best_by_arch[arch]["best_tau_s"]
        t_arr, T_sample, T_top, T_top_obs = run_architecture_on_protocol(
            arch, tau, elapsed, tint, env["T_environment_C"])
        # 样品温度 (架构 O/I/S 均不滞后样品)
        T_sample_pred = sample_at_measurement_times(elapsed, t_arr, T_sample)
        arch_runs[arch] = {
            "t_arr": t_arr, "T_sample": T_sample, "T_top": T_top,
            "T_top_obs": T_top_obs, "T_sample_pred": T_sample_pred,
            "tau": tau,
        }
        # 检测周期 + 修正 dwell (Phase A 定义)
        cyc = detect_activation_and_repeated_cycles(
            elapsed, tint, T_sample_pred)
        intervals = build_phase_intervals(cyc["activation"],
                                          cyc["repeated_cycles"], elapsed)
        phase_rows, per_cycle_rows = phase_specific_dwell_table(
            elapsed, T_sample_pred, intervals)
        # repeated_metrics 需要 sample_peak_C 键 (与 phase_peak_stats 契约一致)
        cyc_metrics = [dict(c, sample_peak_C=c["sample_high_peak_C"])
                       for c in cyc["repeated_cycles"]]
        peak_stats = phase_peak_stats(
            cyc_metrics, cyc["activation"],
            {"t": elapsed, "T_sample": T_sample_pred})
        arch_runs[arch]["phase_rows"] = phase_rows
        arch_runs[arch]["per_cycle_rows"] = per_cycle_rows
        arch_runs[arch]["intervals"] = intervals
        arch_runs[arch]["peak_stats"] = peak_stats
        arch_runs[arch]["n_repeated"] = len(cyc["repeated_cycles"])
        arch_runs[arch]["sample_max_C"] = float(np.max(T_sample_pred))

        for th in DWELL_THRESHOLDS:
            r = next(pr for pr in phase_rows if pr["threshold_C"] == th)
            rows.append({
                "architecture": arch,
                "tau_s": tau,
                "threshold_C": th,
                "repeated_mean_dwell_s": r["repeated_cycle_mean_dwell_s"],
                "repeated_total_dwell_s": r["repeated_cycle_total_dwell_s"],
                "activation_dwell_s": r["activation_dwell_s"],
                "total_dwell_s": r["total_protocol_dwell_s"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_prediction_by_architecture.csv", index=False)

    # 图: 样品预测迹线 (三架构)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"O": "#1f77b4", "I": "#d62728", "S": "#2ca02c"}
    ax.plot(elapsed, tint, color="#7f7f7f", lw=1.1, ls=":",
            label="Internal sensor input (measured)")
    for arch in ("O", "I", "S"):
        ax.plot(elapsed, arch_runs[arch]["T_sample_pred"], color=colors[arch],
                lw=1.6, label=f"{arch} (tau={arch_runs[arch]['tau']:.1f} s)")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{name} — sample prediction by lag placement (Phase B)\n"
                 "(frozen k=0.055, cp=1200; h=10 + nonlinear radiation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_prediction_by_architecture.png", dpi=150)
    plt.close(fig)
    return arch_runs


def main():
    for d in (OUT_D11, OUT_LONG, OUT_COMP):
        d.mkdir(parents=True, exist_ok=True)

    best_df = pd.read_csv(OUT72 / "lag_placement_72C_best.csv")
    best_by_arch = {}
    for _, row in best_df.iterrows():
        best_by_arch[row["architecture"]] = {
            "best_tau_s": float(row["best_tau_s"]),
            "best_RMSE_C": float(row["best_RMSE_C"]),
            "TAU_BOUNDARY_WARNING": bool(row["TAU_BOUNDARY_WARNING"]),
        }
    print("[best] 72C per-architecture tau:",
          {a: best_by_arch[a]["best_tau_s"] for a in ("O", "I", "S")})

    d11 = process_protocol("DOE11 Faster PCR", DOE11_PATH, best_by_arch,
                           OUT_D11)
    longer = process_protocol("Test_PCR Longer Holding", LONGER_PATH,
                              best_by_arch, OUT_LONG)

    # ---- 比较表 ----
    comp_rows = []
    for arch in ("O", "I", "S"):
        for label, runs, short in (("DOE11", d11, "DOE11"),
                                   ("LONGER", longer, "LONGER")):
            r = runs[arch]
            ps = r["peak_stats"]
            ph85 = next(pr for pr in r["phase_rows"]
                        if pr["threshold_C"] == 85.0)
            ph75 = next(pr for pr in r["phase_rows"]
                        if pr["threshold_C"] == 75.0)
            comp_rows.append({
                "architecture": arch,
                "protocol": short,
                "tau_s": r["tau"],
                "n_repeated_cycles": r["n_repeated"],
                "sample_max_C": r["sample_max_C"],
                "repeated_peak_mean_C": ps["repeated_cycle_sample_peak_mean_C"],
                "repeated_peak_median_C":
                    ps["repeated_cycle_sample_peak_median_C"],
                "ge75_mean_dwell_s_per_cycle":
                    ph75["repeated_cycle_mean_dwell_s"],
                "ge85_mean_dwell_s_per_cycle":
                    ph85["repeated_cycle_mean_dwell_s"],
                "ge85_total_dwell_s": ph85["repeated_cycle_total_dwell_s"],
            })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT_COMP / "lag_placement_PCR_comparison.csv", index=False)

    # ---- 图 1: DOE11 样品峰 vs 架构 ----
    _plot_repeated_peak_by_arch(d11, "DOE11 Faster", OUT_COMP
                                / "DOE11_sample_by_lag_placement.png")
    # ---- 图 2: LONGER ----
    _plot_repeated_peak_by_arch(longer, "Test_PCR Longer Holding", OUT_COMP
                                / "longer_holding_sample_by_lag_placement.png")
    # ---- 图 3: 代表性周期 (重复周期 3-5) ----
    _plot_representative_cycles(d11, longer, OUT_COMP
                                / "lag_placement_representative_cycles.png")
    # ---- 图 4: holding-time 效应 (repeated peak vs protocol) ----
    _plot_holding_time_effect(d11, longer, OUT_COMP
                              / "holding_time_effect_by_lag_placement.png")
    # ---- 图 5: 重复周期 dwell 均值 ----
    _plot_repeated_dwell(d11, longer, OUT_COMP
                         / "repeated_cycle_dwell_by_lag_placement.png")

    # ---- 元数据 ----
    meta = {
        "phase": "B",
        "purpose": "lag placement architecture applied to PCR protocols",
        "environment_rule":
            "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT (per protocol)",
        "dwell_definition": "corrected Phase A per-cycle intervals "
                            "(no total/N)",
        "best_tau_from_72C": best_by_arch,
        "selection_rule": "min 72C RMSE; never based on PCR sample",
        "no_refit": True,
        "no_independent_tau_in_out": True,
        "physics_unchanged": True,
        "no_old_output_overwritten": True,
    }
    (OUT_COMP / "lag_placement_PCR_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 摘要文本 ----
    txt = _summary_text(d11, longer, best_by_arch)
    (OUT_COMP / "lag_placement_PCR_summary.txt").write_text(
        txt, encoding="utf-8")
    print(txt)


def _plot_repeated_peak_by_arch(runs, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    archs = ["O", "I", "S"]
    means = [runs[a]["peak_stats"]["repeated_cycle_sample_peak_mean_C"]
             for a in archs]
    medians = [runs[a]["peak_stats"]["repeated_cycle_sample_peak_median_C"]
               for a in archs]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, means, w, color="#1f77b4", label="mean repeated peak")
    ax.bar(x + w / 2, medians, w, color="#d62728", label="median repeated peak")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(tau={runs[a]['tau']:.1f}s)" for a in archs])
    ax.set_ylabel("Sample repeated-cycle peak [C]")
    ax.set_title(f"{title} — repeated-cycle sample peak by lag placement")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_representative_cycles(runs_d11, runs_long, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"O": "#1f77b4", "I": "#d62728", "S": "#2ca02c"}
    for ax, runs, title in ((axes[0], runs_d11, "DOE11 Faster"),
                            (axes[1], runs_long, "Test_PCR Longer Holding")):
        for arch in ("O", "I", "S"):
            iv = runs[arch]["intervals"]["repeated_cycle_intervals"]
            if len(iv) < 3:
                continue
            s, e = iv[2]  # 第 3 个重复周期
            t = runs[arch]["t_arr"]
            ts = runs[arch]["T_sample"]
            m = (t >= s) & (t <= e + 0.01)
            ax.plot(t[m] - s, ts[m], color=colors[arch], lw=1.6,
                    label=f"{arch} (tau={runs[arch]['tau']:.1f}s)")
        ax.set_title(title)
        ax.set_xlabel("time since cycle start [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Sample temperature [C]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_holding_time_effect(runs_d11, runs_long, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    archs = ["O", "I", "S"]
    x = np.arange(3)
    w = 0.35
    d11_peaks = [runs_d11[a]["peak_stats"]
                 ["repeated_cycle_sample_peak_mean_C"] for a in archs]
    long_peaks = [runs_long[a]["peak_stats"]
                  ["repeated_cycle_sample_peak_mean_C"] for a in archs]
    ax.bar(x - w / 2, d11_peaks, w, color="#1f77b4", label="DOE11 faster")
    ax.bar(x + w / 2, long_peaks, w, color="#d62728",
           label="Test_PCR longer holding")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(tau={runs_d11[a]['tau']:.1f}s)" for a in archs])
    ax.set_ylabel("Mean repeated-cycle sample peak [C]")
    ax.set_title("Holding-time effect by lag placement\n"
                 "(same frozen k/cp, per-architecture best tau)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_repeated_dwell(runs_d11, runs_long, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"O": "#1f77b4", "I": "#d62728", "S": "#2ca02c"}
    for ax, runs, title in ((axes[0], runs_d11, "DOE11 Faster"),
                            (axes[1], runs_long, "Test_PCR Longer Holding")):
        archs = ["O", "I", "S"]
        ths = [75.0, 80.0, 85.0]
        for arch in archs:
            vals = []
            for th in ths:
                ph = next(pr for pr in runs[arch]["phase_rows"]
                          if pr["threshold_C"] == th)
                vals.append(ph["repeated_cycle_mean_dwell_s"])
            ax.plot(ths, vals, "o-", color=colors[arch],
                    label=f"{arch} (tau={runs[arch]['tau']:.1f}s)")
        ax.set_title(title)
        ax.set_xlabel("threshold [C]")
        ax.set_ylabel("Mean repeated-cycle dwell [s/cycle]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summary_text(d11, longer, best_by_arch):
    L = []
    A = L.append
    A("=" * 70)
    A("LAG PLACEMENT — PCR CROSS-PROTOCOL APPLICATION (Phase B)")
    A("=" * 70)
    A("72C 每架构最佳 tau (最小 RMSE, 不基于样品):")
    for arch in ("O", "I", "S"):
        b = best_by_arch[arch]
        A(f"  {arch}: tau={b['best_tau_s']:.1f} s, "
          f"RMSE={b['best_RMSE_C']:.4f} C"
          + ("  [TAU_BOUNDARY_WARNING]" if b["TAU_BOUNDARY_WARNING"] else ""))
    A("")
    A("样品温度为模型预测 (非实测); PCR 数据未参与选参。")
    A("dwell = 修正 Phase A per-cycle 区间均值 (不用 total/N)。")
    for arch in ("O", "I", "S"):
        r = d11[arch]
        ps = r["peak_stats"]
        ph85 = next(pr for pr in r["phase_rows"] if pr["threshold_C"] == 85.0)
        A(f"  [DOE11 {arch}] tau={r['tau']:.1f}s: 周期数={r['n_repeated']}, "
          f"样品峰 mean={ps['repeated_cycle_sample_peak_mean_C']:.2f} C, "
          f"median={ps['repeated_cycle_sample_peak_median_C']:.2f} C, "
          f">=85 mean dwell={ph85['repeated_cycle_mean_dwell_s']:.2f} s/cycle")
    for arch in ("O", "I", "S"):
        r = longer[arch]
        ps = r["peak_stats"]
        ph85 = next(pr for pr in r["phase_rows"] if pr["threshold_C"] == 85.0)
        A(f"  [LONG {arch}] tau={r['tau']:.1f}s: 周期数={r['n_repeated']}, "
          f"样品峰 mean={ps['repeated_cycle_sample_peak_mean_C']:.2f} C, "
          f"median={ps['repeated_cycle_sample_peak_median_C']:.2f} C, "
          f">=85 mean dwell={ph85['repeated_cycle_mean_dwell_s']:.2f} s/cycle")
    A("")
    A("结论按 72C 拟合质量排序; 样品灵敏度仅描述, 不作为选参依据。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
