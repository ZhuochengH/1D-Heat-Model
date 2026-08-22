#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy B — fast-PCR-oriented (alpha_eff, cp_eff) calibration analysis
=======================================================================

两条并行标定策略 (见 CALIBRATION_STRATEGIES.md):

  Strategy A  top_rmse_optimal_k_cp_v1   CURRENT ACCEPTED   (k, cp)
  Strategy B  fast_pcr_oriented_alpha_cp_v1  EXPERIMENTAL   (alpha, cp)

本脚本只实现策略 B 的分析管线:

    1. 重参数化换算: alpha = k/(rho*cp), k = alpha*rho*cp (rho=1020);
    2. 等价性回归: (alpha_A, 900) 必须复现策略 A 的 k=0.0165 与 RMSE≈0.7337;
    3. 粗网格扫描: 9 alpha x 7 cp = 63 点, 目标 = 修正测量时间 72C 顶部 RMSE;
    4. RMSE-only 参考最优 (重参数化后应与策略 A 物理等价);
    5. 近最优带 (STRICT/MODERATE/APPLICATION) + 各带最高 alpha 候选;
    6. RMSE<=1C 内部约定候选;
    7. Pareto 前沿 (min RMSE, max alpha);
    8. (可选) 一次局部细扫;
    9. DOE11 样品预测敏感性 (仅图示, 不拟合);
    10. 72C / DOE11 对比图 + 候选表 + 摘要 + 元数据。

重要:
    - 不使用任何复合 RMSE-alpha 分数;
    - 不以 DOE11 样品预测选择/拟合参数;
    - 不修改 NOMINAL_BARE_TOP_CALIBRATION_V1 / 不移动 tag;
    - 全部输出在 parameter_scan_output/72C/fast_pcr_oriented_alpha_cp_v1/。

用法:
    uv run python alpha_cp_calibration_strategy.py --stage equivalence
    uv run python alpha_cp_calibration_strategy.py --stage coarse
    uv run python alpha_cp_calibration_strategy.py --stage analysis
    uv run python alpha_cp_calibration_strategy.py --stage all
"""
import argparse
import json
import subprocess
import sys
import time
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
    evaluate_point,
    evaluate_point_safe,
    load_experiment,
    sample_prediction_at_measurement_times,
)
from thermal_model.utilities.predict_sample_from_internal_temperature import (
    DEFAULT_INPUT as DOE11_INPUT,
    detect_cycles,
    load_internal_data,
    ramp_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "fast_pcr_oriented_alpha_cp_v1"
)

RHO_COC = 1020.0

ALPHA_GRID_COARSE = [
    1.0e-8, 1.4e-8, 1.8e-8, 2.5e-8, 3.5e-8,
    5.0e-8, 7.0e-8, 1.0e-7, 1.4e-7,
]
CP_GRID_COARSE = [600.0, 900.0, 1200.0, 1600.0, 2200.0, 3000.0, 4000.0]

# 一次局部细扫: 粗扫描表明 72C RMSE 在 alpha 方向极陡,
# 细扫聚焦 1.8e-8 ~ 2.6e-8 与中低 cp, 用于量化陡峭度并
# 找出 RMSE<=1C 边界的最高 alpha (若存在)。
ALPHA_GRID_FINE = [
    1.85e-8, 1.9e-8, 2.0e-8, 2.1e-8, 2.2e-8,
    2.3e-8, 2.4e-8, 2.5e-8, 2.6e-8,
]
CP_GRID_FINE = [500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0]

# 近最优带 (绝对 RMSE 增量, 项目级约定, 非置信区间)
BANDS = {
    "STRICT": 0.05,
    "MODERATE": 0.10,
    "APPLICATION": 0.20,
}
# 内部约定: RMSE <= 1 C
RMSE_LE_1C_THRESHOLD = 1.0

L_BOTTOM_UM = 180.0
L_TO_SAMPLE_MID_UM = 190.0


# ============================================================
# 换算
# ============================================================

def alpha_from_k_cp(k_eff, cp_eff, rho=RHO_COC):
    """k/cp -> alpha (m2/s)。"""
    return float(k_eff) / (rho * float(cp_eff))


def k_from_alpha_cp(alpha_eff, cp_eff, rho=RHO_COC):
    """alpha/cp -> k (W/m K)。"""
    return float(alpha_eff) * rho * float(cp_eff)


def strategy_a_alpha():
    """当前策略 A 的有效扩散率 (精确计算)。"""
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    return alpha_from_k_cp(cal.k_eff_W_mK, cal.cp_eff_J_kgK,
                           cal.rho_COC_kg_m3)


def t_diff_um2_s(alpha_eff, length_um):
    """特征 L^2/alpha 时间尺度 (仅标度指标, 非精确延迟/时间常数)。"""
    return (float(length_um) * 1e-6) ** 2 / float(alpha_eff)


# ============================================================
# alpha/cp 评估 (派生 k 后走既有 solver)
# ============================================================

def evaluate_alpha_cp(alpha_eff, cp_eff, t_proto, t_int, t_top_meas,
                      **kw):
    """在 (alpha, cp) 上评估: k = alpha*rho*cp, 复用 evaluate_point。"""
    k = k_from_alpha_cp(alpha_eff, cp_eff, RHO_COC)
    out = evaluate_point(k, cp_eff, t_proto, t_int, t_top_meas, **kw)
    out["alpha_eff_m2_s"] = float(alpha_eff)
    out["derived_k_eff_W_mK"] = float(k)
    out["t_diff_180um_s"] = t_diff_um2_s(alpha_eff, L_BOTTOM_UM)
    out["t_diff_190um_s"] = t_diff_um2_s(alpha_eff, L_TO_SAMPLE_MID_UM)
    return out


def evaluate_alpha_cp_safe(alpha_eff, cp_eff, t_proto, t_int, t_top_meas,
                           **kw):
    try:
        return evaluate_alpha_cp(alpha_eff, cp_eff, t_proto, t_int,
                                 t_top_meas, **kw)
    except Exception as exc:  # noqa: BLE001
        return {
            "alpha_eff_m2_s": float(alpha_eff),
            "cp_eff_J_kgK": float(cp_eff),
            "derived_k_eff_W_mK": k_from_alpha_cp(alpha_eff, cp_eff, RHO_COC),
            "RMSE_C": np.nan, "MAE_C": np.nan,
            "mean_residual_C": np.nan, "max_abs_error_C": np.nan,
            "max_positive_residual_C": np.nan, "max_negative_residual_C": np.nan,
            "t_diff_180um_s": t_diff_um2_s(alpha_eff, L_BOTTOM_UM),
            "t_diff_190um_s": t_diff_um2_s(alpha_eff, L_TO_SAMPLE_MID_UM),
            "runtime_s": np.nan, "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# 等价性回归 (Stage equivalence)
# ============================================================

def equivalence_check(t_proto, t_int, t_top_meas, output_dir):
    """用 alpha_A/cp=900 复现策略 A。派生 k 必须 == 0.0165, RMSE ~ 0.7337。"""
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = strategy_a_alpha()
    row = evaluate_alpha_cp(alpha_A, cal.cp_eff_J_kgK,
                            t_proto, t_int, t_top_meas)
    k_derived = row["derived_k_eff_W_mK"]
    rmse_new = row["RMSE_C"]
    rmse_ref = 0.7337
    k_ok = abs(k_derived - cal.k_eff_W_mK) < 1e-12
    rmse_ok = abs(rmse_new - rmse_ref) < 0.01
    result = {
        "alpha_A_m2_s": alpha_A,
        "cp": cal.cp_eff_J_kgK,
        "derived_k_eff_W_mK": k_derived,
        "expected_k_eff_W_mK": cal.k_eff_W_mK,
        "k_match": bool(k_ok),
        "RMSE_via_alpha_cp_C": rmse_new,
        "RMSE_reference_72C_C": rmse_ref,
        "rmse_match": bool(rmse_ok),
        "pass": bool(k_ok and rmse_ok),
    }
    (output_dir / "equivalence_regression.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"[equivalence] alpha_A={alpha_A:.6e}")
    print(f"  derived k = {k_derived:.10f} (expect {cal.k_eff_W_mK}) "
          f"-> {'PASS' if k_ok else 'FAIL'}")
    print(f"  RMSE = {rmse_new:.6f} (expect ~{rmse_ref}) "
          f"-> {'PASS' if rmse_ok else 'FAIL'}")
    return result


# ============================================================
# 粗扫描 (Stage coarse, 续跑)
# ============================================================

def alpha_cp_points(alpha_values, cp_values):
    return [(a, c) for a in alpha_values for c in cp_values]


def _alpha_cp_key(alpha, cp):
    return f"{float(alpha):.6e}|{float(cp):.1f}"


def run_coarse(t_proto, t_int, t_top_meas, output_dir):
    return _run_grid(t_proto, t_int, t_top_meas, output_dir,
                     ALPHA_GRID_COARSE, CP_GRID_COARSE,
                     "alpha_cp_coarse_scan.csv", label="coarse")


def run_fine(t_proto, t_int, t_top_meas, output_dir):
    """一次局部细扫 (仅当粗扫描不足以刻画权衡时调用)。"""
    return _run_grid(t_proto, t_int, t_top_meas, output_dir,
                     ALPHA_GRID_FINE, CP_GRID_FINE,
                     "alpha_cp_fine_scan.csv", label="fine")


def _run_grid(t_proto, t_int, t_top_meas, output_dir,
              alpha_values, cp_values, csv_name, label):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    points = alpha_cp_points(alpha_values, cp_values)
    done = set()
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        done = set(_alpha_cp_key(a, c) for a, c in
                   zip(df["alpha_eff_m2_s"], df["cp_eff_J_kgK"]))
    new_rows = []
    for alpha, cp in points:
        key = _alpha_cp_key(alpha, cp)
        if key in done:
            continue
        row = evaluate_alpha_cp_safe(alpha, cp, t_proto, t_int, t_top_meas)
        new_rows.append(row)
        print(f"  [{label}] alpha={alpha:.3e} cp={cp:.0f} "
              f"k={row['derived_k_eff_W_mK']:.5f} "
              f"RMSE={row['RMSE_C']:.4f}")
        if csv_path.exists():
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False,
                                       index=False)
        else:
            pd.DataFrame([row]).to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    print(f"[{label}] total {len(df)} points, failures: "
          f"{(df['status'] == 'FAILED').sum()}")
    return df


# ============================================================
# 近最优 / Pareto / 候选
# ============================================================

def rmse_min_from_df(df):
    ok = df["status"] == "OK"
    sub = df[ok]
    return float(sub["RMSE_C"].min())


def highest_alpha_in_band(df, rmse_min, band_C):
    ok = df["status"] == "OK"
    sub = df[ok & (df["RMSE_C"] <= rmse_min + band_C)]
    if sub.empty:
        return None
    i = int(sub["alpha_eff_m2_s"].idxmax())
    return df.loc[i]


def highest_alpha_rmse_le_1c(df):
    ok = df["status"] == "OK"
    sub = df[ok & (df["RMSE_C"] <= RMSE_LE_1C_THRESHOLD)]
    if sub.empty:
        return None
    i = int(sub["alpha_eff_m2_s"].idxmax())
    return df.loc[i]


def pareto_front(df):
    """min RMSE, max alpha。j 支配 i: rmse_j<=rmse_i 且 alpha_j>=alpha_i, 至少一个严格。"""
    ok = df["status"] == "OK"
    sub = df[ok].reset_index(drop=True)
    n = len(sub)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rmse_j_le = sub.loc[j, "RMSE_C"] <= sub.loc[i, "RMSE_C"]
            alpha_j_ge = sub.loc[j, "alpha_eff_m2_s"] >= \
                sub.loc[i, "alpha_eff_m2_s"]
            strict = (sub.loc[j, "RMSE_C"] < sub.loc[i, "RMSE_C"]) or \
                (sub.loc[j, "alpha_eff_m2_s"] > sub.loc[i, "alpha_eff_m2_s"])
            if rmse_j_le and alpha_j_ge and strict:
                dominated[i] = True
                break
    return sub[~dominated].reset_index(drop=True)


def near_optimal_alpha_ranges(df, rmse_min, band_C):
    ok = df["status"] == "OK"
    sub = df[ok & (df["RMSE_C"] <= rmse_min + band_C)]
    if sub.empty:
        return None
    return {"min": float(sub["alpha_eff_m2_s"].min()),
            "max": float(sub["alpha_eff_m2_s"].max()),
            "n": int(len(sub))}


# ============================================================
# DOE11 敏感性 (仅图示/描述, 不拟合)
# ============================================================

_doe11_cache = None


def load_doe11():
    global _doe11_cache
    if _doe11_cache is None:
        data = load_internal_data(DOE11_INPUT)
        _doe11_cache = data
    return _doe11_cache


def make_candidate_materials(k_eff, cp_eff):
    mats = heat_model.copy_default_materials()
    coc = mats["COC"]
    mats["COC"] = heat_model.Material(
        name=coc.name, k_W_mK=float(k_eff),
        rho_kg_m3=coc.rho_kg_m3, cp_J_kgK=float(cp_eff),
    )
    return mats


def doe11_candidate_metrics(k_eff, cp_eff):
    """给定 (k, cp) 运行 DOE11 样品预测, 返回描述性指标 (仅说明)。"""
    data = load_doe11()
    e = data["elapsed_time_s"]
    ti = data["T_internal_C"]
    mats = make_candidate_materials(k_eff, cp_eff)
    layers = heat_model.BARE_TOP_COC_LAYERS
    T_initial = float(ti[0])
    result = heat_model.run_simulation(
        time_s=e, bottom_temperature_C=ti, materials=mats, layers=layers,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1, T_initial_C=T_initial,
    )
    t_arr = result["t_array"]
    T_sample = sample_prediction_at_measurement_times(e, t_arr,
                                                      result["T_sample_arr"])
    rr = ramp_summary(e, T_sample)
    cycles = detect_cycles(e, ti, T_sample)
    cycle_peaks = [c["sample_high_peak_C"] for c in cycles]
    cycle_peaks_cycling = cycle_peaks[1:]  # 排除初始 90C 相 (周期 1)
    return {
        "DOE11_sample_max_C": float(np.max(T_sample)),
        "DOE11_cycle_peak_min_C": float(np.min(cycle_peaks_cycling))
        if cycle_peaks_cycling else np.nan,
        "DOE11_cycle_peak_max_C": float(np.max(cycle_peaks_cycling))
        if cycle_peaks_cycling else np.nan,
        "DOE11_sample_max_heating_rate_C_s": float(
            rr["max_positive_C_per_s"]),
        "DOE11_sample_max_cooling_rate_C_s": float(
            rr["max_negative_C_per_s"]),
        "DOE11_sample_ge_75C_s": _doe11_dwell_ge(e, T_sample, 75.0),
        "DOE11_sample_ge_80C_s": _doe11_dwell_ge(e, T_sample, 80.0),
    }


def _doe11_dwell_ge(t, T, th):
    """>=th 的区间积分停留时间。"""
    total = 0.0
    for i in range(len(t) - 1):
        a, b = t[i], t[i + 1]
        Ta, Tb = T[i], T[i + 1]
        if Ta >= th and Tb >= th:
            total += (b - a)
        elif Ta >= th or Tb >= th:
            if Tb != Ta:
                frac = (th - Ta) / (Tb - Ta)
                total += (b - a) * (1.0 - frac) if Ta >= th else (b - a) * frac
    return float(total)


# ============================================================
# 图形
# ============================================================

def plot_landscape(df, output_dir):
    ok = df["status"] == "OK"
    sub = df[ok]
    alphas = np.sort(sub["alpha_eff_m2_s"].unique())
    cps = np.sort(sub["cp_eff_J_kgK"].unique())
    Z = np.full((len(alphas), len(cps)), np.nan)
    for _, r in sub.iterrows():
        ai = int(np.searchsorted(alphas, r["alpha_eff_m2_s"]))
        ci = int(np.searchsorted(cps, r["cp_eff_J_kgK"]))
        Z[ai, ci] = r["RMSE_C"]
    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolormesh(cps, alphas, Z, shading="auto", cmap="viridis",
                         norm=plt.matplotlib.colors.LogNorm(
                             vmin=np.nanmin(Z), vmax=np.nanmax(Z)))
    ax.set_yscale("log")
    cb = fig.colorbar(mesh, ax=ax, label="RMSE_top [°C]")
    ax.set_xlabel("cp_eff [J/(kg·K)]")
    ax.set_ylabel("alpha_eff [m²/s] (log)")
    ax.set_title("Strategy B — 72C Top RMSE landscape in (alpha, cp)")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_landscape_alpha_cp.png", dpi=200)
    plt.close(fig)


def plot_profiles(df, output_dir):
    ok = df["status"] == "OK"
    sub = df[ok]
    # vs alpha: 固定 cp=900 (最近策略 A) 与总体最小
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cp in sorted(sub["cp_eff_J_kgK"].unique()):
        s = sub[sub["cp_eff_J_kgK"] == cp].sort_values("alpha_eff_m2_s")
        ax.plot(s["alpha_eff_m2_s"], s["RMSE_C"], marker="o", ms=3,
                label=f"cp={cp:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha_eff [m²/s] (log)")
    ax.set_ylabel("RMSE_top [°C]")
    ax.set_title("Strategy B — RMSE profile vs alpha (per cp)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_profile_vs_alpha.png", dpi=200)
    plt.close(fig)

    # vs cp: 固定 alpha (最近策略 A)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    alpha_A = strategy_a_alpha()
    for a in sorted(sub["alpha_eff_m2_s"].unique()):
        s = sub[sub["alpha_eff_m2_s"] == a].sort_values("cp_eff_J_kgK")
        ax.plot(s["cp_eff_J_kgK"], s["RMSE_C"], marker="o", ms=3,
                label=f"alpha={a:.1e}")
    ax.axvline(900, color="grey", ls=":", label="cp=900 (Strategy A)")
    ax.set_xlabel("cp_eff [J/(kg·K)]")
    ax.set_ylabel("RMSE_top [°C]")
    ax.set_title("Strategy B — RMSE profile vs cp (per alpha)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_profile_vs_cp.png", dpi=200)
    plt.close(fig)


def plot_pareto(df, front, candidates, rmse_min, output_dir):
    ok = df["status"] == "OK"
    sub = df[ok]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sub["alpha_eff_m2_s"], sub["RMSE_C"], s=14, alpha=0.55,
               color="#1f77b4", label="coarse scan points")
    ax.scatter(front["alpha_eff_m2_s"], front["RMSE_C"], s=26,
               facecolors="none", edgecolors="#ff7f0e", linewidths=1.2,
               label="Pareto front (min RMSE, max alpha)")
    # 标注候选
    markers = {
        "strategy_A": (alpha_from_k_cp(0.0165, 900.0), 0.7337, "Strategy A",
                       "black", "s"),
        "rmse_only": (rmse_min["alpha"], rmse_min["RMSE"], "RMSE-only",
                      "purple", "D"),
    }
    for name, cand in candidates.items():
        if cand is not None:
            markers[name] = (cand["alpha_eff_m2_s"], cand["RMSE_C"], name,
                             "red", "*")
    for key, (a, r, lab, col, mk) in markers.items():
        ax.scatter([a], [r], s=70 if mk != "*" else 160, marker=mk,
                   color=col, edgecolors="k", zorder=5)
        ax.annotate(lab, (a, r), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=col)
    ax.set_xscale("log")
    ax.set_xlabel("alpha_eff [m²/s] (log)")
    ax.set_ylabel("RMSE_top [°C]")
    ax.set_title("Strategy B — RMSE vs alpha Pareto tradeoff\n"
                 "(72C corrected-time objective; candidates marked)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_vs_alpha_pareto.png", dpi=200)
    plt.close(fig)


def plot_72c_comparison(candidates, t_proto, t_int, t_top_meas, output_dir):
    """72°C 顶部拟合对比: 实测 + 策略 A + 高 alpha 候选。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_proto, t_top_meas, color="black", lw=1.6,
            label="Measured Top COC")
    # 策略 A
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    mats = make_nominal_calibrated_materials(cal)
    result = heat_model.run_simulation(
        time_s=t_proto, bottom_temperature_C=t_int, materials=mats,
        layers=nominal_layer_stack(cal), h_conv=5.0, T_air_ambient=25.0,
        save_dt=0.1, T_initial_C=float(t_int[0]))
    pred_A = sample_prediction_at_measurement_times(
        t_proto, result["t_array"], result["T_top_surface_arr"])
    ax.plot(t_proto, pred_A, color="#1f77b4", lw=1.4,
            label="Strategy A (k=0.0165, cp=900)")
    shown = {"STRICT", "MODERATE", "APPLICATION", "RMSE_LE_1C"}
    colors = {"STRICT": "#2ca02c", "MODERATE": "#ff7f0e",
              "APPLICATION": "#d62728", "RMSE_LE_1C": "#9467bd"}
    for name in ("STRICT", "MODERATE", "APPLICATION", "RMSE_LE_1C"):
        cand = candidates.get(name)
        if cand is None or name not in shown:
            continue
        mats = make_candidate_materials(cand["derived_k_eff_W_mK"],
                                        cand["cp_eff_J_kgK"])
        r2 = heat_model.run_simulation(
            time_s=t_proto, bottom_temperature_C=t_int, materials=mats,
            layers=heat_model.BARE_TOP_COC_LAYERS, h_conv=5.0,
            T_air_ambient=25.0, save_dt=0.1, T_initial_C=float(t_int[0]))
        pred = sample_prediction_at_measurement_times(
            t_proto, r2["t_array"], r2["T_top_surface_arr"])
        ax.plot(t_proto, pred, color=colors[name], lw=1.2, ls="--",
                label=f"{name} (alpha={cand['alpha_eff_m2_s']:.1e})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("72C Top-Fit Comparison — fit quality sacrificed as "
                 "alpha increases")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_comparison_72C.png", dpi=200)
    plt.close(fig)


def plot_doe11_comparison(candidates, output_dir):
    """DOE11 样品预测敏感性 (仅图示; 样品温度是模型估计, 非测量)。"""
    data = load_doe11()
    e = data["elapsed_time_s"]
    ti = data["T_internal_C"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(e, ti, color="#7f7f7f", lw=1.2, ls=":",
            label="Internal sensor input (measured)")
    # 策略 A
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    mats = make_nominal_calibrated_materials(cal)
    r = heat_model.run_simulation(
        time_s=e, bottom_temperature_C=ti, materials=mats,
        layers=nominal_layer_stack(cal), h_conv=5.0, T_air_ambient=25.0,
        save_dt=0.1, T_initial_C=float(ti[0]))
    ax.plot(e, sample_prediction_at_measurement_times(
        e, r["t_array"], r["T_sample_arr"]), color="#1f77b4", lw=1.6,
        label="Strategy A sample (k=0.0165, cp=900)")
    # RMSE-only 参考 + <=1C + APPLICATION (最多 3 条策略 B)
    sel_names = []
    if candidates.get("RMSE_ONLY") is not None:
        sel_names.append("RMSE_ONLY")
    for nm in ("RMSE_LE_1C", "APPLICATION"):
        if candidates.get(nm) is not None and nm not in sel_names:
            sel_names.append(nm)
    colors = {"RMSE_ONLY": "#2ca02c", "RMSE_LE_1C": "#9467bd",
              "APPLICATION": "#d62728"}
    for nm in sel_names[:3]:
        cand = candidates[nm]
        mats = make_candidate_materials(cand["derived_k_eff_W_mK"],
                                        cand["cp_eff_J_kgK"])
        r2 = heat_model.run_simulation(
            time_s=e, bottom_temperature_C=ti, materials=mats,
            layers=heat_model.BARE_TOP_COC_LAYERS, h_conv=5.0,
            T_air_ambient=25.0, save_dt=0.1, T_initial_C=float(ti[0]))
        ax.plot(e, sample_prediction_at_measurement_times(
            e, r2["t_array"], r2["T_sample_arr"]), color=colors[nm],
            lw=1.4, ls="--",
            label=f"{nm} sample (alpha={cand['alpha_eff_m2_s']:.1e})")
    ax.set_xlabel("Time [s] (elapsed)")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("DOE11 Fast-PCR Sample Prediction — Sensitivity to "
                 "Effective Thermal Diffusivity\n"
                 "(sample temperatures are model estimates, NOT "
                 "measurements)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_comparison_DOE11.png", dpi=200)
    plt.close(fig)


# ============================================================
# 分析 (Stage analysis)
# ============================================================

def run_analysis(t_proto, t_int, t_top_meas, output_dir, fine_df=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_csv = output_dir / "alpha_cp_coarse_scan.csv"
    if not coarse_csv.exists():
        raise FileNotFoundError(f"缺少粗扫描结果: {coarse_csv}")
    coarse = pd.read_csv(coarse_csv)

    # combined = coarse (+fine 若有)
    combined = coarse.copy()
    if fine_df is not None and len(fine_df):
        combined = pd.concat([combined, fine_df], ignore_index=True)
    combined.to_csv(output_dir / "alpha_cp_combined_scan.csv", index=False)
    scan_df = combined

    # ---- RMSE-only 参考 ----
    rmse_min_val = rmse_min_from_df(scan_df)
    ok = scan_df["status"] == "OK"
    i_min = int(scan_df[ok]["RMSE_C"].idxmin())
    rmse_only = scan_df.loc[i_min]
    rmse_only_rec = {
        "alpha": float(rmse_only["alpha_eff_m2_s"]),
        "cp": float(rmse_only["cp_eff_J_kgK"]),
        "k": float(rmse_only["derived_k_eff_W_mK"]),
        "RMSE": float(rmse_only["RMSE_C"]),
    }

    # ---- 近最优带 ----
    bands = {}
    for name, band in BANDS.items():
        hi = highest_alpha_in_band(scan_df, rmse_min_val, band)
        ar = near_optimal_alpha_ranges(scan_df, rmse_min_val, band)
        bands[name] = {"band_C": band, "highest_alpha": hi, "alpha_range": ar}

    # ---- RMSE<=1C 候选 ----
    le1c = highest_alpha_rmse_le_1c(scan_df)

    # ---- Pareto ----
    front = pareto_front(scan_df)

    # ---- 候选字典 (供图/表) ----
    candidates = {
        "STRICT": bands["STRICT"]["highest_alpha"],
        "MODERATE": bands["MODERATE"]["highest_alpha"],
        "APPLICATION": bands["APPLICATION"]["highest_alpha"],
        "RMSE_LE_1C": le1c,
        "RMSE_ONLY": rmse_only,
    }

    # ---- 图 ----
    plot_landscape(scan_df, output_dir)
    plot_profiles(scan_df, output_dir)
    plot_pareto(scan_df, front, {k: v for k, v in candidates.items()
                                 if k != "RMSE_ONLY"},
                rmse_only_rec, output_dir)
    plot_72c_comparison(candidates, t_proto, t_int, t_top_meas, output_dir)
    plot_doe11_comparison(candidates, output_dir)

    # ---- 候选表 (含 DOE11 敏感性) ----
    rows = []
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    a_A = strategy_a_alpha()
    rows.append({
        "candidate_name": "STRATEGY_A",
        "alpha_eff_m2_s": a_A,
        "cp_eff_J_kgK": cal.cp_eff_J_kgK,
        "derived_k_eff_W_mK": cal.k_eff_W_mK,
        "RMSE_72C_C": 0.7337,
        "Delta_RMSE_C": 0.0,
        "RMSE_relative_increase_pct": 0.0,
        "MAE_72C_C": 0.5628,
        "mean_residual_72C_C": -0.2727,
        "characteristic_tdiff_180um_s": t_diff_um2_s(a_A, L_BOTTOM_UM),
        "characteristic_tdiff_190um_s": t_diff_um2_s(a_A, L_TO_SAMPLE_MID_UM),
        "status": "CURRENT_ACCEPTED_72C_MODEL",
    })
    name_map = {"STRICT": "HIGH_ALPHA_STRICT_CANDIDATE",
                "MODERATE": "HIGH_ALPHA_MODERATE_CANDIDATE",
                "APPLICATION": "HIGH_ALPHA_APPLICATION_CANDIDATE",
                "RMSE_LE_1C": "HIGH_ALPHA_RMSE_LE_1C_CANDIDATE",
                "RMSE_ONLY": "RMSE_ONLY_REFERENCE"}
    for key, cand in candidates.items():
        if cand is None:
            continue
        if key == "RMSE_ONLY":
            cand = rmse_only  # Series
        d = doe11_candidate_metrics(float(cand["derived_k_eff_W_mK"]),
                                    float(cand["cp_eff_J_kgK"]))
        rows.append({
            "candidate_name": name_map[key],
            "alpha_eff_m2_s": float(cand["alpha_eff_m2_s"]),
            "cp_eff_J_kgK": float(cand["cp_eff_J_kgK"]),
            "derived_k_eff_W_mK": float(cand["derived_k_eff_W_mK"]),
            "RMSE_72C_C": float(cand["RMSE_C"]),
            "Delta_RMSE_C": float(cand["RMSE_C"]) - rmse_min_val,
            "RMSE_relative_increase_pct": 100.0 * (
                float(cand["RMSE_C"]) - rmse_min_val) / rmse_min_val,
            "MAE_72C_C": float(cand["MAE_C"]),
            "mean_residual_72C_C": float(cand["mean_residual_C"]),
            "characteristic_tdiff_180um_s": t_diff_um2_s(
                float(cand["alpha_eff_m2_s"]), L_BOTTOM_UM),
            "characteristic_tdiff_190um_s": t_diff_um2_s(
                float(cand["alpha_eff_m2_s"]), L_TO_SAMPLE_MID_UM),
            "DOE11_sample_max_C": d["DOE11_sample_max_C"],
            "DOE11_cycle_peak_min_C": d["DOE11_cycle_peak_min_C"],
            "DOE11_cycle_peak_max_C": d["DOE11_cycle_peak_max_C"],
            "DOE11_sample_max_heating_rate_C_s": d[
                "DOE11_sample_max_heating_rate_C_s"],
            "DOE11_sample_max_cooling_rate_C_s": d[
                "DOE11_sample_max_cooling_rate_C_s"],
            "status": "EXPERIMENTAL_CANDIDATE",
        })
    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(output_dir / "alpha_cp_near_optimal_candidates.csv",
                   index=False)

    # DOE11 明细 CSV
    doe11_rows = []
    for key, cand in candidates.items():
        if cand is None:
            continue
        if key == "RMSE_ONLY":
            cand = rmse_only
        d = doe11_candidate_metrics(float(cand["derived_k_eff_W_mK"]),
                                    float(cand["cp_eff_J_kgK"]))
        doe11_rows.append({
            "candidate_name": name_map[key],
            "alpha_eff_m2_s": float(cand["alpha_eff_m2_s"]),
            "cp_eff_J_kgK": float(cand["cp_eff_J_kgK"]),
            "derived_k_eff_W_mK": float(cand["derived_k_eff_W_mK"]),
            "DOE11_sample_max_C": d["DOE11_sample_max_C"],
            "DOE11_cycle_peak_min_C": d["DOE11_cycle_peak_min_C"],
            "DOE11_cycle_peak_max_C": d["DOE11_cycle_peak_max_C"],
            "DOE11_sample_max_heating_rate_C_s": d[
                "DOE11_sample_max_heating_rate_C_s"],
            "DOE11_sample_max_cooling_rate_C_s": d[
                "DOE11_sample_max_cooling_rate_C_s"],
            "DOE11_sample_ge_75C_s": d["DOE11_sample_ge_75C_s"],
            "DOE11_sample_ge_80C_s": d["DOE11_sample_ge_80C_s"],
        })
    pd.DataFrame(doe11_rows).to_csv(
        output_dir / "candidate_DOE11_summary.csv", index=False)

    # ---- 摘要 txt ----
    _write_summary(output_dir, rmse_only_rec, bands, le1c, front, cand_df)

    # ---- 元数据 ----
    _write_metadata(output_dir, scan_df, rmse_only_rec, bands, le1c, front)

    return scan_df, rmse_only_rec, bands, le1c, front, cand_df


def _fmt_cand(cand, prefix=""):
    if cand is None:
        return f"{prefix}None"
    if isinstance(cand, dict):
        return (f"{prefix}alpha={cand['alpha']:.3e} cp={cand['cp']:.0f} "
                f"k={cand['k']:.5f} RMSE={cand['RMSE']:.4f}")
    return (f"{prefix}alpha={cand['alpha_eff_m2_s']:.3e} "
            f"cp={cand['cp_eff_J_kgK']:.0f} "
            f"k={cand['derived_k_eff_W_mK']:.5f} "
            f"RMSE={cand['RMSE_C']:.4f}")


def _write_summary(output_dir, rmse_only, bands, le1c, front, cand_df):
    lines = [
        "FAST-PCR-ORIENTED alpha_eff/cp_eff STRATEGY (Strategy B) — SUMMARY",
        "=" * 72,
        "Status: EXPERIMENTAL / PROVISIONAL — NOT accepted as final model.",
        "Strategy A (top_rmse_optimal_k_cp_v1) remains CURRENT ACCEPTED: "
        "k=0.0165, cp=900.",
        "",
        "Reparameterization: alpha = k/(rho*cp), k = alpha*rho*cp, rho=1020.",
        "Parameterization alone does NOT change the RMSE optimum.",
        "",
        f"RMSE-only reference: {_fmt_cand(rmse_only, '')}",
        "",
        "Near-optimal bands (absolute Delta_RMSE):",
    ]
    for name, b in bands.items():
        hi = b["highest_alpha"]
        ar = b["alpha_range"]
        if hi is not None:
            lines.append(
                f"  {name} (<= {b['band_C']:.2f} C): "
                f"n={ar['n'] if ar else 0}, "
                f"alpha range [{ar['min']:.2e}, {ar['max']:.2e}] m2/s; "
                f"highest-alpha: {_fmt_cand(hi, '')}"
            )
        else:
            lines.append(f"  {name} (<= {b['band_C']:.2f} C): none")
    lines.append("")
    if le1c is not None:
        lines.append(f"Highest-alpha RMSE<=1C candidate: "
                     f"{_fmt_cand(le1c, '')}")
        lines.append("  (RMSE<=1C is an INTERNAL PROJECT CONVENTION, "
                     "not a universal acceptance criterion.)")
    else:
        lines.append("Highest-alpha RMSE<=1C candidate: none "
                     "(no coarse point with RMSE<=1C).")
    lines.append("")
    lines.append(f"Pareto front points: {len(front)}")
    if len(front):
        lines.append(f"  alpha range: [{front['alpha_eff_m2_s'].min():.2e}, "
                     f"{front['alpha_eff_m2_s'].max():.2e}] m2/s")
        lines.append(f"  RMSE range: [{front['RMSE_C'].min():.4f}, "
                     f"{front['RMSE_C'].max():.4f}] C")
    lines.append("")
    lines.append("Candidate table (see alpha_cp_near_optimal_candidates.csv):")
    lines.append(cand_df.to_string(index=False))
    lines.append("")
    lines.append("NOTE: DOE11 sample predictions are APPLICATION-SENSITIVITY "
                 "ILLUSTRATIONS ONLY — DOE11 has no measured sample truth and "
                 "was NOT used to select or fit parameters.")
    (output_dir / "summary_fast_pcr_strategy.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata(output_dir, scan_df, rmse_only, bands, le1c, front):
    def _cand_dict(cand):
        if cand is None:
            return None
        if isinstance(cand, dict):
            return cand
        return {k: cand[k] for k in [
            "alpha_eff_m2_s", "cp_eff_J_kgK", "derived_k_eff_W_mK",
            "RMSE_C", "MAE_C", "mean_residual_C", "t_diff_180um_s",
            "t_diff_190um_s"] if k in cand}

    metadata = {
        "strategy_id": "fast_pcr_oriented_alpha_cp_v1",
        "status": "EXPERIMENTAL / PROVISIONAL",
        "accepted_as_final": False,
        "strategy_A": {
            "id": "top_rmse_optimal_k_cp_v1",
            "status": "CURRENT ACCEPTED",
            "k_eff_W_mK": NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK,
            "cp_eff_J_kgK": NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK,
            "alpha_eff_m2_s": strategy_a_alpha(),
            "RMSE_72C_C": 0.7337,
        },
        "reparameterization": {
            "alpha": "k/(rho*cp)", "k": "alpha*rho*cp", "rho_kg_m3": 1020.0,
            "parameterization_alone_changes_optimum": False,
        },
        "coarse_scan": {
            "alpha_values_m2_s": ALPHA_GRID_COARSE,
            "cp_values_J_kgK": CP_GRID_COARSE,
            "combinations": len(ALPHA_GRID_COARSE) * len(CP_GRID_COARSE),
            "evaluated": int(len(scan_df)),
            "failures": int((scan_df["status"] == "FAILED").sum()),
        },
        "rmse_only_reference": rmse_only,
        "bands": {name: {"band_C": b["band_C"],
                         "highest_alpha": _cand_dict(b["highest_alpha"]),
                         "alpha_range": b["alpha_range"]}
                  for name, b in bands.items()},
        "rmse_le_1C_candidate": _cand_dict(le1c),
        "rmse_le_1C_note": (
            "Internal project convention, NOT a universal acceptance "
            "criterion."),
        "pareto_front": {
            "points": len(front),
            "alpha_range": [float(front["alpha_eff_m2_s"].min()),
                            float(front["alpha_eff_m2_s"].max())],
            "RMSE_range": [float(front["RMSE_C"].min()),
                           float(front["RMSE_C"].max())],
        },
        "doe11_note": (
            "DOE11 sample predictions are application-sensitivity "
            "illustrations only; DOE11 has no measured sample truth and was "
            "NOT part of the fitting/selection objective."),
        "planned_future": (
            "J = 0.5*RMSE_slow^2 + 0.5*RMSE_fast^2 (documented only; "
            "requires reliable synchronized fast-transient data)."),
        "git_commit": _git_head(),
        "git_tag": _git_describe(),
    }
    (output_dir / "alpha_cp_strategy_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_describe():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ============================================================
# CLI
# ============================================================

def load_72c():
    return load_experiment()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=["equivalence", "coarse", "fine", "analysis",
                            "all"])
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_proto, t_int, t_top_meas = load_72c()

    if args.stage in ("equivalence", "all"):
        eq = equivalence_check(t_proto, t_int, t_top_meas, output_dir)
        if not eq["pass"]:
            print("[STOP] equivalence regression FAILED — aborting.")
            return 2

    if args.stage in ("coarse", "all"):
        t0 = time.perf_counter()
        run_coarse(t_proto, t_int, t_top_meas, output_dir)
        print(f"[coarse] elapsed {time.perf_counter() - t0:.1f} s")

    if args.stage == "fine":
        t0 = time.perf_counter()
        run_fine(t_proto, t_int, t_top_meas, output_dir)
        print(f"[fine] elapsed {time.perf_counter() - t0:.1f} s")

    if args.stage in ("analysis", "all"):
        # 自动合并细扫 (若存在)
        fine_df = None
        fine_csv = output_dir / "alpha_cp_fine_scan.csv"
        if fine_csv.exists():
            fine_df = pd.read_csv(fine_csv)
            print(f"[analysis] merged fine scan: {len(fine_df)} points")
        scan_df, rmse_only, bands, le1c, front, cand_df = run_analysis(
            t_proto, t_int, t_top_meas, output_dir, fine_df=fine_df)
        print(f"[analysis] RMSE-only: {_fmt_cand(rmse_only, '')}")
        for name, b in bands.items():
            print(f"  {name}: {_fmt_cand(b['highest_alpha'], '')}")
        print(f"  RMSE<=1C: {_fmt_cand(le1c, '')}")
        print(f"  pareto: {len(front)} points")
        print(f"[output] {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
