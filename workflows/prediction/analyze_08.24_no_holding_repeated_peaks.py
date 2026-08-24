#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08.24 15x primers / no-holding — 冻结模型样品预测后处理
=========================================================
只读后处理: 读取 predict_sample_temperature_frozen_model.py 生成的
sample_temperature_bare_vs_insulated.csv, 复用已验证的重复周期峰检测器
thermal_model.utilities.analyze_frozen_sample_peak.detect_repeated_cycles
(与 66C REDO / 3s 扩展分析同一逻辑, 阈值/激活相分离规则完全不变),
计算裸顶/绝缘重复周期样品峰统计、描述性阈值计数、无保持协议特征
(典型峰/谷/振幅)、绝缘样品时间-above 统计 (复用预测工具的
compute_threshold_timing, 时间戳感知)。

周期定义说明 (重要):
    本脚本使用通用的峰-谷检测器 (detect_repeated_cycles), 其 15 s
    最小峰分离阈值会把 08.24 协议的 30 个 Setpoint 循环合并为 16 个
    检测峰, 且首峰 (激活相残留 ~88.98 C) 计入。因此本脚本的
    "16 个重复周期" 是【旧的/通用的峰基分析结果】, 仅作为历史对照。

    更权威的周期定义见:
        workflows/diagnostics/analyze_insulated_sample_sensitivity.py
    其使用 SETPOINT_PROTOCOL_STRUCTURE 窗口 (Setpoint 20/100 C 段,
    激活相 90 C 段排除, 29 个完整循环; 峰窗口=100 C 段, 谷窗口=20 C 段),
    与任何模型参数无关且跨所有敏感性运行冻结。
    本脚本的结果不应被当作 "authoritative repeated-cycle values" 引用。

绝不:
    - 重新拟合 / 优化任何参数
    - 修改 FINAL_FROZEN_THERMAL_MODEL_V1 / 预测脚本 / 检测器
    - 发明新的峰/周期分割算法

输出 (新数据集目录, gitignored):
    repeated_cycle_sample_peaks_bare.csv
    repeated_cycle_sample_peaks_insulated.csv
    repeated_cycle_sample_peaks_comparison.csv
    final_sample_prediction_summary.txt
    final_sample_prediction_figure.png/.pdf      (出版友好: internal + bare + insulated)
    repeated_cycle_sample_peaks.png/.pdf          (峰摘要: bare vs insulated)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.utilities.analyze_frozen_sample_peak import (
    detect_repeated_cycles,
)
from workflows.prediction.predict_sample_temperature_frozen_model import (
    compute_threshold_timing,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 描述性阈值 (与 recalibrate_thermal_model_66C_redo.DESCRIPTIVE_THRESHOLDS
# 逐位一致: 85/86/87/90/92/95)
THRESHOLDS = (85.0, 86.0, 87.0, 90.0, 92.0, 95.0)

# 之前 3s 扩展绝缘样品预测参考 (只读, 不重算; 来源 =
# calibrated_model_output/66C_recalibrated_candidate_v1/
# insulated_3s_sample_prediction/)
PREVIOUS_3S_INSULATED_REF = {
    "overall_sample_max_C": 89.07,
    "repeated_cycle_mean_peak_C": 85.67,
    "repeated_cycle_median_peak_C": 85.52,
}


def repeated_cycle_stats(peaks):
    """重复周期样品峰统计 (与 66C redo repeated_cycle_stats 语义一致)。"""
    arr = np.asarray(peaks, dtype=float)
    if arr.size == 0:
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan,
                "median": np.nan, "std": np.nan}
    return {"n": int(arr.size), "min": float(np.min(arr)),
            "max": float(np.max(arr)), "mean": float(np.mean(arr)),
            "median": float(np.median(arr)), "std": float(np.std(arr))}


def threshold_counts(peaks, thresholds=THRESHOLDS):
    """重复周期样品峰 >= 各阈值的 个数/分数/百分比 (描述性)。"""
    arr = np.asarray(peaks, dtype=float)
    out = {}
    for th in thresholds:
        n_ge = int(np.sum(arr >= th)) if arr.size else 0
        frac = float(n_ge / arr.size) if arr.size else np.nan
        out[int(th)] = {"count": n_ge, "fraction": frac,
                        "percent": float(frac * 100.0) if arr.size else np.nan}
    return out


def main():
    out_dir = (PROJECT_ROOT / "sample_temperature_output"
               / "08.24_15x_primers_no_holding_final_model")
    csv_path = out_dir / "sample_temperature_bare_vs_insulated.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"预测 CSV 不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    t = df["original_time_s"].to_numpy(dtype=float)
    tint = df["measured_internal_C"].to_numpy(dtype=float)
    Tb = df["predicted_sample_bare_C"].to_numpy(dtype=float)
    Ti = df["predicted_sample_insulated_C"].to_numpy(dtype=float)

    # ---- 周期检测 (完全复用已验证检测器, 激活相规则不变) ----
    cyc_b = detect_repeated_cycles(t, tint, Tb)
    cyc_i = detect_repeated_cycles(t, tint, Ti)
    rep_b = cyc_b["repeated_cycles"]
    rep_i = cyc_i["repeated_cycles"]

    def _peaks(rep):
        return [float(c["sample_high_peak_C"]) for c in rep]

    def _troughs(rep):
        return [float(c["sample_low_trough_C"]) for c in rep]

    peaks_b = _peaks(rep_b)
    peaks_i = _peaks(rep_i)
    troughs_b = _troughs(rep_b)
    troughs_i = _troughs(rep_i)
    st_b = repeated_cycle_stats(peaks_b)
    st_i = repeated_cycle_stats(peaks_i)
    tc_b = threshold_counts(peaks_b)
    tc_i = threshold_counts(peaks_i)

    # ---- 无保持协议特征: 典型峰/谷/振幅 (描述性) ----
    def _typical(troughs, st):
        t_mean = float(np.mean(troughs)) if troughs else np.nan
        t_min = float(np.min(troughs)) if troughs else np.nan
        t_max = float(np.max(troughs)) if troughs else np.nan
        amp = float(st["mean"] - t_mean) if troughs else np.nan
        return {"trough_min_C": t_min, "trough_mean_C": t_mean,
                "trough_max_C": t_max,
                "peak_to_trough_amplitude_C": amp}

    typ_b = _typical(troughs_b, st_b)
    typ_i = _typical(troughs_i, st_i)

    # ---- 绝缘样品 dwell (时间戳感知累计 time-above; 复用已验证工具) ----
    dwell = {th: compute_threshold_timing(t, Ti, float(th))
             for th in (85.0, 86.0, 87.0)}

    # ---- 全迹线样品摘要 (插值到实测时间轴, 与预测 summary 一致) ----
    def _full(trace):
        i_max = int(np.argmax(trace))
        return {"min_C": float(np.min(trace)),
                "max_C": float(np.max(trace)),
                "max_time_s": float(t[i_max]),
                "mean_C": float(np.mean(trace)),
                "median_C": float(np.median(trace))}

    full_b = _full(Tb)
    full_i = _full(Ti)

    # ---- 对比 ----
    delta_overall = full_i["max_C"] - full_b["max_C"]
    delta_mean_peak = st_i["mean"] - st_b["mean"]
    delta_median_peak = st_i["median"] - st_b["median"]

    # ---- 3s 参考对比 ----
    ref = PREVIOUS_3S_INSULATED_REF
    comp_3s = {
        "previous_overall_max_C": ref["overall_sample_max_C"],
        "new_overall_max_C": full_i["max_C"],
        "delta_overall_max_C": full_i["max_C"] - ref["overall_sample_max_C"],
        "previous_mean_peak_C": ref["repeated_cycle_mean_peak_C"],
        "new_mean_peak_C": st_i["mean"],
        "delta_mean_peak_C": st_i["mean"] - ref["repeated_cycle_mean_peak_C"],
        "previous_median_peak_C": ref["repeated_cycle_median_peak_C"],
        "new_median_peak_C": st_i["median"],
        "delta_median_peak_C":
            st_i["median"] - ref["repeated_cycle_median_peak_C"],
    }

    # ============================================================
    # 保存 CSVs
    # ============================================================
    def _cycle_frame(rep):
        rows = []
        for c in rep:
            rows.append({
                "cycle_number": int(c["cycle_number"]),
                "cycle_start_time_s": c["cycle_start_time_s"],
                "internal_peak_time_s": c["internal_peak_time_s"],
                "internal_high_peak_C": c["internal_high_peak_C"],
                "sample_peak_time_s": c["sample_peak_time_s"],
                "sample_high_peak_C": c["sample_high_peak_C"],
                "internal_low_trough_C": c["internal_low_trough_C"],
                "sample_low_trough_C": c["sample_low_trough_C"],
            })
        return pd.DataFrame(rows)

    f_b = _cycle_frame(rep_b)
    f_i = _cycle_frame(rep_i)
    f_b.to_csv(out_dir / "repeated_cycle_sample_peaks_bare.csv", index=False)
    f_i.to_csv(out_dir / "repeated_cycle_sample_peaks_insulated.csv",
               index=False)

    comp_df = pd.DataFrame([{
        "metric": "overall_sample_max_C",
        "bare_C": full_b["max_C"], "insulated_C": full_i["max_C"],
        "delta_ins_minus_bare_C": delta_overall,
    }, {
        "metric": "repeated_cycle_mean_peak_C",
        "bare_C": st_b["mean"], "insulated_C": st_i["mean"],
        "delta_ins_minus_bare_C": delta_mean_peak,
    }, {
        "metric": "repeated_cycle_median_peak_C",
        "bare_C": st_b["median"], "insulated_C": st_i["median"],
        "delta_ins_minus_bare_C": delta_median_peak,
    }])
    comp_df.to_csv(out_dir / "repeated_cycle_sample_peaks_comparison.csv",
                   index=False)

    # ============================================================
    # 图 1: 出版友好主图 (internal + bare + insulated)
    # ============================================================
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.plot(t, tint, color="#7f7f7f", lw=1.1, ls=":",
            label="Measured internal temperature (Zone 1)")
    ax.plot(t, Tb, color="#2ca02c", lw=1.8,
            label="Predicted sample temperature — bare")
    ax.plot(t, Ti, color="#d62728", lw=1.8,
            label="Predicted sample temperature — air insulated")
    ax.annotate(f"insulated sample max = {full_i['max_C']:.2f} C",
                xy=(full_i["max_time_s"], full_i["max_C"]),
                xytext=(full_i["max_time_s"] - 60, full_i["max_C"] - 9),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                fontsize=9, color="black")
    ax.axhline(90.0, color="#8c564b", ls="--", lw=1.1, alpha=0.8,
               label="90 C thermal reference")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(
        "08.24 15x primers / no holding — frozen-model predicted sample "
        "temperature\n(FINAL_FROZEN_THERMAL_MODEL_V1; bare vs air "
        "insulated; forward prediction, not measured)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "final_sample_prediction_figure.png", dpi=150)
    fig.savefig(out_dir / "final_sample_prediction_figure.pdf")
    plt.close(fig)

    # ============================================================
    # 图 2: 重复周期峰摘要 (bare vs insulated)
    # ============================================================
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    if rep_b:
        nums_b = [int(c["cycle_number"]) for c in rep_b]
        ax.plot(nums_b, peaks_b, "o-", color="#2ca02c", lw=1.5, ms=5,
                label="Bare repeated-cycle sample peaks")
    if rep_i:
        nums_i = [int(c["cycle_number"]) for c in rep_i]
        ax.plot(nums_i, peaks_i, "s-", color="#d62728", lw=1.5, ms=5,
                label="Insulated repeated-cycle sample peaks")
    ax.axhline(85.0, color="#1f77b4", ls="--", lw=1.0, alpha=0.7,
               label="85 C reference")
    ax.axhline(90.0, color="#d62728", ls="--", lw=1.0, alpha=0.7,
               label="90 C reference")
    ax.set_xlabel("Repeated cycle number")
    ax.set_ylabel("Predicted sample peak [C]")
    ax.set_title("08.24 no-holding — repeated-cycle predicted sample peaks "
                 "(bare vs insulated)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "repeated_cycle_sample_peaks.png", dpi=150)
    fig.savefig(out_dir / "repeated_cycle_sample_peaks.pdf")
    plt.close(fig)

    # ============================================================
    # 文本摘要
    # ============================================================
    L = []
    A = L.append
    A("=" * 72)
    A("08.24 15X PRIMERS / NO-HOLDING — FINAL MODEL SAMPLE PREDICTION SUMMARY")
    A("=" * 72)
    A(f"model id            : FINAL_FROZEN_THERMAL_MODEL_V1")
    A(f"parameters refitted : NO")
    A(f"sample Top-lag      : NO (sample = raw FDM layer temperature)")
    A(f"detector            : detect_repeated_cycles "
      "(analyze_frozen_sample_peak, unchanged thresholds)")
    A(f"activation rule     : unchanged (excluded first phase only if "
      "no prior trough and >= 30 s from start)")
    A("-" * 72)
    A("BARE:")
    A(f"  sample min/mean/median : {full_b['min_C']:.3f} / "
      f"{full_b['mean_C']:.3f} / {full_b['median_C']:.3f} C")
    A(f"  sample max             : {full_b['max_C']:.3f} C @ "
      f"{full_b['max_time_s']:.3f} s")
    A(f"  repeated cycles        : {st_b['n']}")
    A(f"  repeated peak min/max  : {st_b['min']:.3f} / {st_b['max']:.3f} C")
    A(f"  repeated peak mean     : {st_b['mean']:.3f} C")
    A(f"  repeated peak median   : {st_b['median']:.3f} C")
    A(f"  repeated peak std      : {st_b['std']:.3f} C")
    A(f"  typical trough (mean)  : {typ_b['trough_mean_C']:.3f} C")
    A(f"  peak-to-trough amp     : {typ_b['peak_to_trough_amplitude_C']:.3f} C")
    A("INSULATED:")
    A(f"  sample min/mean/median : {full_i['min_C']:.3f} / "
      f"{full_i['mean_C']:.3f} / {full_i['median_C']:.3f} C")
    A(f"  sample max             : {full_i['max_C']:.3f} C @ "
      f"{full_i['max_time_s']:.3f} s")
    A(f"  repeated cycles        : {st_i['n']}")
    A(f"  repeated peak min/max  : {st_i['min']:.3f} / {st_i['max']:.3f} C")
    A(f"  repeated peak mean     : {st_i['mean']:.3f} C")
    A(f"  repeated peak median   : {st_i['median']:.3f} C")
    A(f"  repeated peak std      : {st_i['std']:.3f} C")
    A(f"  typical trough (mean)  : {typ_i['trough_mean_C']:.3f} C")
    A(f"  peak-to-trough amp     : {typ_i['peak_to_trough_amplitude_C']:.3f} C")
    A("INSULATED REPEATED-CYCLE REFERENCES (count / total / percent):")
    for th in THRESHOLDS:
        c = tc_i[int(th)]
        A(f"  >= {th:.0f} C : {c['count']} / {st_i['n']} "
          f"({c['percent']:.1f}%)")
    A("BARE REPEATED-CYCLE REFERENCES (count / total / percent):")
    for th in THRESHOLDS:
        c = tc_b[int(th)]
        A(f"  >= {th:.0f} C : {c['count']} / {st_b['n']} "
          f"({c['percent']:.1f}%)")
    A("BARE VS INSULATED:")
    A(f"  delta overall max          : {delta_overall:+.3f} C")
    A(f"  delta repeated mean peak   : {delta_mean_peak:+.3f} C")
    A(f"  delta repeated median peak : {delta_median_peak:+.3f} C")
    A("COMPARISON WITH PREVIOUS 3S EXTENSION (insulated, read-only ref):")
    for k, label in (("delta_overall_max_C", "  delta overall max"),
                     ("delta_mean_peak_C", "  delta repeated mean peak"),
                     ("delta_median_peak_C", "  delta repeated median peak")):
        A(f"{label} : {comp_3s[k]:+.3f} C")
    A("INSULATED DWELL (time-above, cumulative, timestamp-aware):")
    for th in (85.0, 86.0, 87.0):
        d = dwell[th]
        A(f"  >= {th:.0f} C : {d['time_above_s']:.2f} s "
          f"({d['n_intervals']} intervals)")
    A("-" * 72)
    A("Scientific interpretation:")
    A("  Sample-layer temperature is model-predicted (frozen "
      "FINAL_FROZEN_THERMAL_MODEL_V1), not directly measured.")
    A("  Insulated configuration is a forward extension (3 mm sealed air + "
      "200 um PDMS); not directly sample-validated.")
    A("  No sample-temperature uncertainty attached in this task.")
    A("  Threshold statistics are descriptive thermal-exposure references, "
      "not PCR success/failure criteria.")
    (out_dir / "final_sample_prediction_summary.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8")

    meta = {
        "model_id": "FINAL_FROZEN_THERMAL_MODEL_V1",
        "parameters_refitted": False,
        "sample_top_lag_applied": False,
        "detector": "thermal_model.utilities.analyze_frozen_sample_peak."
                    "detect_repeated_cycles (unchanged)",
        "source_csv": str(csv_path),
        "bare": {"full": full_b, "repeated": st_b,
                 "typical": typ_b, "thresholds": tc_b},
        "insulated": {"full": full_i, "repeated": st_i,
                      "typical": typ_i, "thresholds": tc_i},
        "comparison": {"delta_overall_max_C": delta_overall,
                       "delta_mean_peak_C": delta_mean_peak,
                       "delta_median_peak_C": delta_median_peak},
        "previous_3s_reference": PREVIOUS_3S_INSULATED_REF,
        "comparison_vs_3s": comp_3s,
        "dwell_insulated": {str(int(th)): dwell[th] for th in (85.0, 86.0,
                                                               87.0)},
    }
    (out_dir / "repeated_cycle_analysis_metadata.json").write_text(
        json.dumps(meta, indent=2, default=float), encoding="utf-8")

    print(f"out_dir: {out_dir}")
    print(f"bare repeated cycles: {st_b['n']} | "
          f"insulated repeated cycles: {st_i['n']}")
    print(f"bare peaks: {peaks_b}")
    print(f"insulated peaks: {peaks_i}")
    print(f"insulated dwell: "
          + ", ".join(f">={th:.0f}C {dwell[th]['time_above_s']:.2f}s"
                      for th in (85.0, 86.0, 87.0)))
    print(f"delta vs 3s: overall {comp_3s['delta_overall_max_C']:+.3f} | "
          f"mean peak {comp_3s['delta_mean_peak_C']:+.3f} | "
          f"median peak {comp_3s['delta_median_peak_C']:+.3f}")


if __name__ == "__main__":
    main()
