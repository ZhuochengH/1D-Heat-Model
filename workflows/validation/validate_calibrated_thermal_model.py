#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic transfer validation of the frozen calibrated thermal model
===================================================================

Purpose
-------
Test whether the FROZEN calibrated model (NOMINAL_BARE_TOP_CALIBRATION_V1,
k_eff = 0.0165 W/(m K), cp_eff = 900 J/(kg K), rho = 1020 kg/m3) can predict
the Top COC temperature of an INDEPENDENT protocol when supplied only with
that protocol's measured internal-temperature boundary trace.

This is NOT a calibration. NO refitting:

    k_eff / cp_eff / h_conv / ambient / time offset / temperature offset /
    sensor tau / contact resistance / initial condition / regime weighting
    are all FROZEN.  No optimizer, no parameter scan.

The measured Top COC temperature is used ONLY for comparison / residual /
metrics. It never influences simulation parameters, alignment, initial
condition, material values, or boundary preprocessing.

Reuse (no duplicated physics / alignment):
    - heat_model.run_simulation               : authoritative FDM solver
    - calibrated_model_config                 : frozen nominal configuration
    - align_internal_and_top_temperature      : validated alignment pipeline
    - scan_effective_thermal_parameters.sample_prediction_at_measurement_times
        : corrected time sampling (query axis = measurement TIME, never
          measured temperature values — regression-guarded against the
          historical temperature-as-time bug)
    - classify_temperature_regimes            : descriptive regime diagnostics
      (default thresholds, no tuning)

Usage
-----
    python validate_calibrated_thermal_model.py \
        --aligned-csv temperature_alignment_output/60C_redo/aligned_internal_top_temperature.csv \
        --experiment-name 60C_redo \
        --output-dir calibrated_model_output/60C_redo_transfer_check_v1

    (or omit --aligned-csv and pass --top-file / --internal-file to run the
     alignment pipeline first)
"""
import argparse
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
from thermal_model.config.calibrated_model_config import (
    NOMINAL_BARE_TOP_CALIBRATION_V1,
    make_nominal_calibrated_materials,
    nominal_layer_stack,
)
from thermal_model.utilities.scan_effective_thermal_parameters import (
    sample_prediction_at_measurement_times,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认 60°C 实验 (本次 transfer check)
DEFAULT_EXPERIMENT_NAME = "60C_redo"
DEFAULT_ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "60C_redo"
    / "aligned_internal_top_temperature.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "calibrated_model_output" / "60C_redo_transfer_check_v1"
)
DEFAULT_REGIME_LABELED_CSV = (
    PROJECT_ROOT / "temperature_regime_output" / "60C_redo"
    / "temperature_regime_labeled.csv"
)
DEFAULT_REF_METRICS_JSON = (
    PROJECT_ROOT / "calibrated_model_output" / "72C_corrected_objective_v1"
    / "final_72C_metadata.json"
)

H_CONV = 5.0
T_AMB = 25.0
SAVE_DT = 0.1

# 对齐/分类脚本的默认输入文件 (仅当 --aligned-csv 未给出时用于内联对齐)
DEFAULT_TOP_FILE = (
    PROJECT_ROOT.parent / "Calibration" / "extension 60°C_redo.xls"
)
DEFAULT_INTERNAL_FILE = (
    PROJECT_ROOT.parent / "Calibration"
    / "08.06 COC_top 60°C_zone1_temperature_analysis.xlsx"
)

TIME_COL = "time_s"
TINT_COL = "T_internal_interpolated_C"
TTOP_COL = "T_top_measured_C"


# ============================================================
# Git 元数据
# ============================================================

def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_describe():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ============================================================
# 数据加载
# ============================================================

def load_aligned_data(path=None):
    """加载对齐实验数据 (time_s / T_internal_interpolated_C / T_top_measured_C)。"""
    p = Path(path) if path else DEFAULT_ALIGNED_CSV
    if not p.is_file():
        raise FileNotFoundError(f"对齐数据集不存在: {p}")
    df = pd.read_csv(p)
    for col in (TIME_COL, TINT_COL, TTOP_COL):
        if col not in df.columns:
            raise KeyError(f"对齐数据集缺少列 {col!r}; 实际列: {list(df.columns)}")
    t = df[TIME_COL].to_numpy(dtype=float)
    t_int = df[TINT_COL].to_numpy(dtype=float)
    t_top = df[TTOP_COL].to_numpy(dtype=float)
    return t, t_int, t_top


def run_inline_alignment(top_file, internal_file, max_top_rows,
                         internal_sheet="Extracted_Data",
                         internal_col="Zone 1 Avg (°C)",
                         internal_time_col="Time(s)",
                         top_sheet="Data", top_col="T Avg",
                         top_diag_time_col="RECTime"):
    """内联对齐 (复用 align_internal_and_top_temperature 的同一算法)。"""
    from thermal_model.utilities.align_internal_and_top_temperature import (
        align_to_top_reference,
        load_internal_series_time_col,
        load_top_series_top_reference,
    )

    top = load_top_series_top_reference(
        top_file, column=top_col, sheet=top_sheet,
        diag_time_col=top_diag_time_col,
    )
    internal = load_internal_series_time_col(
        internal_file, column=internal_col, sheet=internal_sheet,
        time_col=internal_time_col,
    )
    aligned = align_to_top_reference(
        internal, top, max_top_rows=max_top_rows
    )
    return aligned, top, internal


# ============================================================
# 冻结配置
# ============================================================

def build_frozen_config():
    """返回冻结的名义标定配置 (仅读取, 永不修改)。"""
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    layers = nominal_layer_stack(cal)
    mats = make_nominal_calibrated_materials(cal)
    return cal, layers, mats, H_CONV, T_AMB


def compute_initial_condition(t_int):
    """初始条件 = 第一个对齐的 T_internal (既有策略, 不拟合)。"""
    return float(np.asarray(t_int, dtype=float)[0])


# ============================================================
# 指标
# ============================================================

def compute_validation_metrics(t, residual):
    """扩展指标: RMSE/MAE/mean/max/median abs/p95 abs (残差 = pred - meas)。"""
    r = np.asarray(residual, dtype=float)
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(r)
    if not finite.any():
        raise ValueError("残差序列全为 NaN。")
    r = r[finite]
    i = int(np.argmax(np.abs(r)))
    return {
        "RMSE_top_C": float(np.sqrt(np.mean(r ** 2))),
        "MAE_top_C": float(np.mean(np.abs(r))),
        "median_absolute_error_C": float(np.median(np.abs(r))),
        "p95_absolute_error_C": float(np.percentile(np.abs(r), 95)),
        "mean_residual_top_C": float(np.mean(r)),
        "max_positive_residual_C": float(np.max(r)),
        "max_negative_residual_C": float(np.min(r)),
        "max_absolute_residual_C": float(np.max(np.abs(r))),
        "time_of_max_abs_residual_s": float(t[finite][i]),
    }


def residual_fractions(residual):
    r = np.asarray(residual, dtype=float)
    n = r.size
    return {
        "fraction_positive_pct": float(100.0 * np.mean(r > 0)),
        "fraction_negative_pct": float(100.0 * np.mean(r < 0)),
        "fraction_abs_le_1C_pct": float(100.0 * np.mean(np.abs(r) <= 1.0)),
        "fraction_abs_le_2C_pct": float(100.0 * np.mean(np.abs(r) <= 2.0)),
        "fraction_abs_le_3C_pct": float(100.0 * np.mean(np.abs(r) <= 3.0)),
    }


def temperature_bin_stats(t_top, residual, bins=(40.0, 50.0, 60.0),
                          labels=("<40 C", "40-50 C", "50-60 C", ">60 C")):
    """残差按 T_top_measured 分箱的均值/点数 (仅描述, 不拟合)。"""
    t_top = np.asarray(t_top, dtype=float)
    r = np.asarray(residual, dtype=float)
    edges = (-np.inf,) + tuple(bins) + (np.inf,)
    out = {}
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        mask = (t_top > lo) & (t_top <= hi) if np.isfinite(lo) else \
               (t_top <= hi)
        if not np.isfinite(hi):
            mask = (t_top > lo)
        sel = r[mask]
        if sel.size:
            out[lab] = {
                "n": int(sel.size),
                "mean_residual_C": float(np.mean(sel)),
                "rmse_C": float(np.sqrt(np.mean(sel ** 2))),
                "mae_C": float(np.mean(np.abs(sel))),
            }
        else:
            out[lab] = {"n": 0}
    return out


# ============================================================
# 温区诊断 (复用分类脚本, 默认阈值)
# ============================================================

def regime_diagnostics(labeled_csv, residual, time_s):
    """从 regime 分类结果计算每温区的 n / RMSE / MAE / mean residual。"""
    labeled_csv = Path(labeled_csv)
    if not labeled_csv.is_file():
        return None
    df = pd.read_csv(labeled_csv)
    if "regime" not in df.columns:
        return None
    labels = df["regime"].to_numpy()
    if len(labels) != len(residual):
        return None
    r = np.asarray(residual, dtype=float)
    out = {}
    for name in ["TRANSIENT_HEATING", "TRANSIENT_COOLING", "SETTLING",
                 "STEADY", "TRANSITION_OTHER"]:
        mask = labels == name
        sel = r[mask]
        if sel.size:
            out[name] = {
                "n": int(sel.size),
                "RMSE_C": float(np.sqrt(np.mean(sel ** 2))),
                "MAE_C": float(np.mean(np.abs(sel))),
                "mean_residual_C": float(np.mean(sel)),
            }
        else:
            out[name] = {"n": int(sel.size)}
    return out


# ============================================================
# 72°C 参考指标
# ============================================================

def load_72c_reference_metrics(ref_json=None):
    ref_json = Path(ref_json) if ref_json else DEFAULT_REF_METRICS_JSON
    m = json.loads(ref_json.read_text(encoding="utf-8"))
    met = m["metrics"]
    return {
        "k_eff_W_mK": float(m["k_eff_W_mK"]),
        "cp_eff_J_kgK": float(m["cp_eff_J_kgK"]),
        "RMSE_top_C": float(met["RMSE_top_C"]),
        "MAE_top_C": float(met["MAE_top_C"]),
        "mean_residual_top_C": float(met["mean_residual_top_C"]),
        "max_absolute_residual_C": float(met["max_absolute_residual_C"]),
    }


# ============================================================
# 运行冻结模型 (单一 FDM 运行, 无任何参数搜索)
# ============================================================

def run_frozen_simulation(t, t_int, cal, layers, mats):
    """一次固定参数的 FDM 运行。返回 (t_arr, T_top_surface_arr, T_sample_arr)。"""
    T_initial = compute_initial_condition(t_int)
    result = heat_model.run_simulation(
        time_s=t,
        bottom_temperature_C=t_int,
        materials=mats,
        layers=layers,
        h_conv=H_CONV,
        T_air_ambient=T_AMB,
        save_dt=SAVE_DT,
        T_initial_C=T_initial,
    )
    return result["t_array"], result["T_top_surface_arr"], \
        result["T_sample_arr"]


def run_transfer_validation(t, t_int, t_top_meas, output_dir,
                            experiment_name=DEFAULT_EXPERIMENT_NAME,
                            regime_labeled_csv=None,
                            ref_metrics_json=None,
                            source_files=None,
                            alignment_summary=None):
    """完整 transfer-check 流水线 (冻结模型, 无拟合)。

    返回 (metrics, fractions, regime, bins, comparison)。
    """
    cal, layers, mats, h_conv, t_amb = build_frozen_config()

    # ---- 单一 FDM 运行 (固定参数) ----
    t_arr, T_top_surface, T_sample = run_frozen_simulation(
        t, t_int, cal, layers, mats
    )

    # ---- 修正时间采样: 查询轴 = 实测时间 (绝不用温度值) ----
    T_top_pred = sample_prediction_at_measurement_times(
        t, t_arr, T_top_surface
    )
    T_sample_pred = sample_prediction_at_measurement_times(
        t, t_arr, T_sample
    )

    # ---- 残差 = pred - meas ----
    residual = T_top_pred - t_top_meas
    metrics = compute_validation_metrics(t, residual)
    fractions = residual_fractions(residual)
    bins = temperature_bin_stats(t_top_meas, residual)
    regime = regime_diagnostics(regime_labeled_csv, residual, t)

    # ---- 72°C 参考 ----
    ref = load_72c_reference_metrics(ref_metrics_json)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 主迹线 CSV ----
    trace = pd.DataFrame({
        "time_s": t,
        "T_internal_C": t_int,
        "T_top_measured_C": t_top_meas,
        "T_top_predicted_C": T_top_pred,
        "T_sample_predicted_C": T_sample_pred,
        "top_residual_C": residual,
    })
    prefix = f"{experiment_name.split('_')[0]}_transfer"
    trace_csv = output_dir / f"{prefix}_trace.csv"
    trace.to_csv(trace_csv, index=False)

    # ---- 主图 (PNG + PDF) ----
    _plot_trace(t, t_int, t_top_meas, T_top_pred, T_sample_pred,
                experiment_name, output_dir, prefix)
    # ---- 残差图 ----
    _plot_residual(t, residual, metrics, output_dir, prefix)
    # ---- 残差 vs 温度 ----
    _plot_residual_vs_temperature(t_top_meas, residual,
                                  output_dir, "top", experiment_name)
    _plot_residual_vs_temperature(t_int, residual,
                                  output_dir, "internal", experiment_name)

    # ---- 72C vs 60C 对比 ----
    comparison = {
        "quantity": ["k_eff_W_mK", "cp_eff_J_kgK", "refitted",
                     "RMSE_top_C", "MAE_top_C", "mean_residual_top_C",
                     "max_absolute_residual_C"],
        "72C_calibration": [
            ref["k_eff_W_mK"], ref["cp_eff_J_kgK"], "YES (original)",
            ref["RMSE_top_C"], ref["MAE_top_C"],
            ref["mean_residual_top_C"], ref["max_absolute_residual_C"],
        ],
        "60C_transfer": [
            cal.k_eff_W_mK, cal.cp_eff_J_kgK, "NO",
            metrics["RMSE_top_C"], metrics["MAE_top_C"],
            metrics["mean_residual_top_C"],
            metrics["max_absolute_residual_C"],
        ],
    }
    cmp_df = pd.DataFrame(comparison)
    ref_label = "72C"
    exp_label = experiment_name.split("_")[0]
    cmp_csv = output_dir / \
        f"comparison_{ref_label}_calibration_vs_{exp_label}_transfer.csv"
    cmp_df.to_csv(cmp_csv, index=False)
    _plot_comparison(ref, metrics, output_dir, ref_label, exp_label)

    # ---- 元数据 ----
    metadata = {
        "analysis_name": f"{experiment_name}_transfer_check_v1",
        "model_source": "bare-top-calibrated-model-v1",
        "source_calibration": f"{cal.source_analysis} / 72C",
        "validation_dataset": {
            "top_file": str(source_files["top_file"]),
            "internal_file": str(source_files["internal_file"]),
        },
        "refitted": False,
        "k_eff_W_mK": cal.k_eff_W_mK,
        "cp_eff_J_kgK": cal.cp_eff_J_kgK,
        "rho_COC_kg_m3": cal.rho_COC_kg_m3,
        "geometry": "bare-top 850 um (BARE_TOP_COC_LAYERS)",
        "h_conv_W_m2K": h_conv,
        "T_air_ambient_C": t_amb,
        "initial_condition_mode": "first aligned internal temperature",
        "measurement_count": int(len(t)),
        "time_range_s": [float(t[0]), float(t[-1])],
        "internal_temperature_range_C": [float(np.min(t_int)),
                                         float(np.max(t_int))],
        "measured_top_temperature_range_C": [float(np.min(t_top_meas)),
                                             float(np.max(t_top_meas))],
        "metrics": metrics,
        "residual_fractions_pct": fractions,
        "temperature_bins": bins,
        "regime_diagnostics": regime,
        "alignment_summary": alignment_summary,
        "git_commit": git_head(),
        "git_tag": git_describe(),
        "note": (
            "Same frozen parameters as the 72C calibration "
            "(k_eff=0.0165 W/(m K), cp_eff=900 J/(kg K), rho=1020 kg/m3); "
            "NO parameter fitting was performed on this 60C dataset. "
            "T_sample_predicted is a model estimate (control-volume spatial "
            "average), not experimentally measured, and is excluded from all "
            "validation decisions. k_eff/cp_eff are system-level effective "
            "parameters, NOT intrinsic COC material constants. Residual "
            "sign convention: residual = predicted - measured."
        ),
    }
    meta_path = output_dir / f"{prefix}_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    # ---- 摘要 txt ----
    _write_summary(experiment_name, prefix, output_dir, cal, metrics,
                   fractions, regime, bins, ref)

    return metrics, fractions, regime, bins, comparison


# ============================================================
# 绘图
# ============================================================

def _plot_trace(t, t_int, t_top_meas, t_top_pred, t_sample_pred,
                experiment_name, output_dir, prefix):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(t, t_int, color="#7f7f7f", lw=1.2, ls=":",
            label="Internal sensor input")
    ax.plot(t, t_top_meas, color="#d62728", lw=1.6,
            label="Top COC measured")
    ax.plot(t, t_top_pred, color="#1f77b4", lw=2.0,
            label="Top COC predicted — frozen model")
    ax.plot(t, t_sample_pred, color="#2ca02c", lw=1.6, ls="--",
            label="Sample predicted (estimated, not measured)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{experiment_name.replace('_', ' ').upper()} Protocol — "
                 "Transfer Check of Calibrated Thermal Model\n"
                 "(frozen k_eff=0.0165, cp_eff=900 — no refit)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_trace.png", dpi=200)
    fig.savefig(output_dir / f"{prefix}_trace.pdf")
    plt.close(fig)


def _plot_residual(t, residual, metrics, output_dir, prefix):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, residual, color="#1f77b4", lw=1.0)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Top residual: predicted - measured [$^\circ$C]")
    ax.set_title("60°C Transfer — Residual vs Time")
    ax.grid(True, ls="--", alpha=0.5)
    ax.text(0.02, 0.98,
            f"RMSE = {metrics['RMSE_top_C']:.3f} °C\n"
            f"MAE  = {metrics['MAE_top_C']:.3f} °C",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_residual.png", dpi=200)
    plt.close(fig)


def _plot_residual_vs_temperature(x, residual, output_dir, which,
                                  experiment_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, residual, s=12, alpha=0.6, color="#1f77b4")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(f"{which.capitalize()} temperature [°C]")
    ax.set_ylabel("Top residual [°C]")
    ax.set_title(f"60°C Transfer — Residual vs {which.capitalize()} "
                 "Temperature\n(descriptive only — no k(T)/cp(T) fit)")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir /
                f"residual_vs_{which}_temperature.png", dpi=200)
    plt.close(fig)


def _plot_comparison(ref, metrics, output_dir, ref_label, exp_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = ["RMSE", "MAE"]
    ref_vals = [ref["RMSE_top_C"], ref["MAE_top_C"]]
    exp_vals = [metrics["RMSE_top_C"], metrics["MAE_top_C"]]
    x = np.arange(len(groups))
    w = 0.35
    ax.bar(x - w / 2, ref_vals, w, label=f"{ref_label} calibration",
           color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, exp_vals, w,
           label=f"{exp_label} transfer (no refit)",
           color="#1f77b4", alpha=0.85)
    for xi, v in zip(x - w / 2, ref_vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, exp_vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Top temperature error [°C]")
    ax.set_title(f"{ref_label} Calibration vs {exp_label} Independent "
                 "Transfer Check\n(same frozen parameters — 60C not refitted)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir /
                f"comparison_{ref_label}_calibration_vs_{exp_label}_"
                f"transfer.png", dpi=200)
    plt.close(fig)


# ============================================================
# 摘要
# ============================================================

def _write_summary(experiment_name, prefix, output_dir, cal, metrics,
                   fractions, regime, bins, ref):
    lines = [
        f"{experiment_name} INDEPENDENT PROTOCOL TRANSFER CHECK SUMMARY",
        "=" * 70,
        f"Refitting performed: NO",
        f"Frozen parameters: k_eff = {cal.k_eff_W_mK} W/(m K), "
        f"cp_eff = {cal.cp_eff_J_kgK} J/(kg K), rho = {cal.rho_COC_kg_m3} "
        f"kg/m3",
        f"Source calibration: {cal.source_analysis} / 72C",
        "",
        "60C TOP PREDICTION METRICS:",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("")
    lines.append("Residual fractions:")
    for k, v in fractions.items():
        lines.append(f"  {k}: {v:.2f}")
    lines.append("")
    lines.append("Regime diagnostics (descriptive, default thresholds):")
    if regime:
        for name, d in regime.items():
            if d.get("n", 0):
                lines.append(
                    f"  {name}: n={d['n']} RMSE={d['RMSE_C']:.4f} "
                    f"MAE={d['MAE_C']:.4f} mean={d['mean_residual_C']:.4f}"
                )
            else:
                lines.append(f"  {name}: n=0")
    else:
        lines.append("  (no regime labeled file available)")
    lines.append("")
    lines.append("Temperature bins (residual mean by T_top_measured):")
    for lab, d in bins.items():
        if d.get("n", 0):
            lines.append(
                f"  {lab}: n={d['n']} mean={d['mean_residual_C']:.4f} "
                f"RMSE={d['rmse_C']:.4f}"
            )
        else:
            lines.append(f"  {lab}: n=0")
    lines.append("")
    lines.append("72C calibration vs 60C transfer:")
    lines.append(
        f"  72C calibration: RMSE={ref['RMSE_top_C']:.4f} "
        f"MAE={ref['MAE_top_C']:.4f} (refitted on dataset: YES)"
    )
    lines.append(
        f"  60C transfer:    RMSE={metrics['RMSE_top_C']:.4f} "
        f"MAE={metrics['MAE_top_C']:.4f} (refitted on dataset: NO)"
    )
    (output_dir / f"{prefix}_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aligned-csv", default=None,
                   help="对齐数据集 CSV (优先使用; 缺省时用 --top-file/"
                        "--internal-file 内联对齐)")
    p.add_argument("--top-file", default=str(DEFAULT_TOP_FILE))
    p.add_argument("--internal-file", default=str(DEFAULT_INTERNAL_FILE))
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--regime-labeled-csv", default=str(DEFAULT_REGIME_LABELED_CSV))
    p.add_argument("--ref-metrics-json", default=str(DEFAULT_REF_METRICS_JSON))
    p.add_argument("--max-top-rows", type=int, default=2000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    alignment_summary = None
    source_files = {
        "top_file": Path(args.top_file),
        "internal_file": Path(args.internal_file),
    }

    if args.aligned_csv:
        t, t_int, t_top = load_aligned_data(args.aligned_csv)
        source_files["top_file"] = "see alignment_metadata.json"
        source_files["internal_file"] = "see alignment_metadata.json"
        alignment_summary = {
            "aligned_csv": str(Path(args.aligned_csv).resolve()),
            "reused_aligned_dataset": True,
            "extrapolation_used": False,
        }
    else:
        aligned, top, internal = run_inline_alignment(
            args.top_file, args.internal_file, args.max_top_rows
        )
        t = aligned["time_s"]
        t_int = aligned["T_internal"]
        t_top = aligned["T_top"]
        alignment_summary = {
            "reused_aligned_dataset": False,
            "aligned_points": int(aligned["n_aligned"]),
            "excluded_early": int(aligned["n_excluded_early"]),
            "excluded_late": int(aligned["n_excluded_late"]),
            "interpolation_method": aligned["interpolation_method"],
            "extrapolation_used": False,
            "top_valid_rows": int(top["n_valid"]),
            "internal_valid_rows": int(internal["n_valid"]),
        }

    print(f"[data] {len(t)} 点, t [{t[0]:.1f}, {t[-1]:.1f}] s")
    cal, layers, mats, h_conv, t_amb = build_frozen_config()
    print(f"[frozen] {cal.name}: k_eff={cal.k_eff_W_mK} "
          f"cp_eff={cal.cp_eff_J_kgK} rho={cal.rho_COC_kg_m3} — NO refit")

    metrics, fractions, regime, bins, comparison = run_transfer_validation(
        t, t_int, t_top, args.output_dir,
        experiment_name=args.experiment_name,
        regime_labeled_csv=args.regime_labeled_csv,
        ref_metrics_json=args.ref_metrics_json,
        source_files=source_files,
        alignment_summary=alignment_summary,
    )
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"[output] {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
