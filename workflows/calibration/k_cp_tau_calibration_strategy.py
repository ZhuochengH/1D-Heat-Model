#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy D — lag-separated (k_eff, cp_eff, tau_lag) 三参数表征
=================================================================

科学问题:
    当未解析的外部/系统滞后被显式表示 (tau_lag) 时, 是否存在
    更高 k_eff、cp_eff 不过度极端 (cp >= 800 J/(kg K)) 的
    拟合良好解, 能复现 72C 实测 Top COC 动力学?

直接参数 (扫描/拟合):
    k_eff   [W/(m K)]
    cp_eff  [J/(kg K)]
    tau_lag [s]

固定:
    rho_COC = 1020 kg/m3

派生 (不直接拟合):
    alpha_eff   = k/(rho*cp)
    effusivity  = sqrt(k*rho*cp)
    Rth_area_bottom = L_bottom/k   (L_bottom = 180 um)

滞后架构: 复用 lag_augmented_thermal_model.py 的输出侧一阶观察滞后
    tau_lag * dT_top_obs/dt + T_top_obs = T_top_FDM
    (精确分段线性递推; tau=0 恒等; 只作用于顶部观测, 不作用于样品)

选参哲学 (分层, 无任意复合分数):
    1. 拟合质量 (72C 修正测量时间 RMSE);
    2. 物理合理性 (cp >= 800 的 NON_EXTREME_CP_SUBSET);
    3. 在近等价且物理合理的模型中, 倾向更高 k。
    **目标函数不含任何 k 奖励/惩罚。**

计算效率:
    每个 (k, cp) 只跑一次 FDM (63 次), 11 个 tau 复用同一 T_top_FDM
    迹线做滞后剖面 (693 个指标组合)。

本任务不调用任何连续优化器; 不使用 DOE11 / PCR 样品预测参与选择。

输出目录: parameter_scan_output/72C/lag_separated_k_cp_tau_v1/
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
from thermal_model.config import calibrated_model_config as cmc
from thermal_model.core.lag_augmented_thermal_model import (
    RHO_COC,
    apply_first_order_lag,
    make_lag_materials,
)
from thermal_model.utilities.scan_effective_thermal_parameters import load_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "lag_separated_k_cp_tau_v1"
)

# ---------------------------------------------------------------
# 网格 (任务指定)
# ---------------------------------------------------------------
K_GRID = [0.0165, 0.025, 0.040, 0.060, 0.080, 0.100, 0.130, 0.160, 0.200]
CP_GRID = [600.0, 800.0, 900.0, 1100.0, 1400.0, 1800.0, 2200.0]
TAU_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]

NON_EXTREME_CP_MIN = 800.0     # NON_EXTREME_CP_SUBSET: cp >= 800
LARGE_LAG_THRESHOLD = 16.0     # tau >= 16 s -> LARGE_LAG_WARNING
TAU_BOUNDARY = 20.0            # tau == 20 s -> TAU_SCAN_BOUNDARY_WARNING

L_BOTTOM_M = 180e-6

BANDS = {"STRICT": 0.05, "MODERATE": 0.10, "APPLICATION": 0.20}
RMSE_LE_1C = 1.0
BALANCED_RMSE = 1.0
BALANCED_RMSE_RELAXED = 1.2


# ============================================================
# 派生量
# ============================================================

def alpha_from_k_cp(k, cp, rho=RHO_COC):
    return float(k) / (rho * float(cp))


def effusivity_from_k_cp(k, cp, rho=RHO_COC):
    return np.sqrt(float(k) * rho * float(cp))


def rth_area_bottom(k):
    return L_BOTTOM_M / float(k)


# ============================================================
# FDM-once-per-(k,cp) 缓存
# ============================================================

_fdm_cache = {}


def run_fdm_cached(k, cp, t_proto, t_int, h_conv=5.0, t_amb=25.0,
                   save_dt=0.1):
    """每个 (k, cp) 只跑一次 FDM; 返回 (t_arr, T_top_FDM, T_sample_FDM)。"""
    key = (float(k), float(cp))
    if key in _fdm_cache:
        return _fdm_cache[key]
    mats = make_lag_materials(k, cp)
    result = heat_model.run_simulation(
        time_s=t_proto,
        bottom_temperature_C=t_int,
        materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS,
        h_conv=h_conv, T_air_ambient=t_amb, save_dt=save_dt,
        T_initial_C=float(t_int[0]),
    )
    entry = (result["t_array"], result["T_top_surface_arr"],
             result["T_sample_arr"])
    _fdm_cache[key] = entry
    return entry


def clear_fdm_cache():
    _fdm_cache.clear()


# ============================================================
# 指标
# ============================================================

def metrics_for_prediction(t_proto, t_top_meas, pred):
    r = np.asarray(pred, dtype=float) - np.asarray(t_top_meas, dtype=float)
    return {
        "RMSE_72C_C": float(np.sqrt(np.mean(r ** 2))),
        "MAE_72C_C": float(np.mean(np.abs(r))),
        "mean_residual_72C_C": float(np.mean(r)),
        "max_abs_residual_72C_C": float(np.max(np.abs(r))),
    }


def lagged_top_prediction(t_proto, t_arr, t_top_fdm, tau):
    """施加滞后并插值到实测时间 (查询轴 = 实测时间, 绝不用温度值)。"""
    if tau == 0.0:
        t_top_obs = t_top_fdm.copy()
    else:
        t_top_obs = apply_first_order_lag(t_arr, t_top_fdm, tau)
    return np.interp(t_proto, t_arr, t_top_obs)


def evaluate_k_cp_tau(k, cp, tau, t_proto, t_int, t_top_meas, **kw):
    """单点评估: 复用缓存的 (k,cp) FDM 迹线 + 滞后剖面。"""
    t_arr, t_top_fdm, t_sample_fdm = run_fdm_cached(k, cp, t_proto, t_int,
                                                    **kw)
    pred = lagged_top_prediction(t_proto, t_arr, t_top_fdm, tau)
    m = metrics_for_prediction(t_proto, t_top_meas, pred)
    alpha = alpha_from_k_cp(k, cp)
    return {
        "k_eff_W_mK": float(k),
        "cp_eff_J_kgK": float(cp),
        "tau_lag_s": float(tau),
        "alpha_eff_m2_s": alpha,
        "effusivity_J_s05_m2_K": effusivity_from_k_cp(k, cp),
        "Rth_area_bottom_m2K_W": rth_area_bottom(k),
        "RMSE_72C_C": m["RMSE_72C_C"],
        "MAE_72C_C": m["MAE_72C_C"],
        "mean_residual_72C_C": m["mean_residual_72C_C"],
        "max_abs_residual_72C_C": m["max_abs_residual_72C_C"],
        "cp_non_extreme": bool(cp >= NON_EXTREME_CP_MIN),
        "large_lag_warning": bool(tau >= LARGE_LAG_THRESHOLD),
        "tau_boundary_warning": bool(tau >= TAU_BOUNDARY - 1e-12),
        "status": "OK",
    }


def run_full_scan(t_proto, t_int, t_top_meas, output_dir):
    """63 次 FDM + 11 tau 剖面 = 693 行。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "k_cp_tau_full_scan.csv"
    clear_fdm_cache()
    rows = []
    t0 = time.perf_counter()
    n_fdm = 0
    for k in K_GRID:
        for cp in CP_GRID:
            n_fdm += 1
            for tau in TAU_GRID:
                rows.append(evaluate_k_cp_tau(k, cp, tau, t_proto, t_int,
                                              t_top_meas))
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[scan] FDM runs: {n_fdm} (unique (k,cp)); "
          f"rows: {len(df)}; failures: {(df['status'] == 'FAILED').sum()}; "
          f"elapsed {time.perf_counter() - t0:.1f} s")
    return df


# ============================================================
# 剖面 / 候选
# ============================================================

def profile_best_tau(df):
    """每个 (k, cp) 的最佳 tau。"""
    rows = []
    for (k, cp), sub in df.groupby(["k_eff_W_mK", "cp_eff_J_kgK"]):
        i_best = int(sub["RMSE_72C_C"].idxmin())
        best = sub.loc[i_best]
        tau0 = sub[sub["tau_lag_s"] == 0.0]
        rmse_tau0 = float(tau0["RMSE_72C_C"].iloc[0]) if len(tau0) else np.nan
        rows.append({
            "k_eff_W_mK": float(k),
            "cp_eff_J_kgK": float(cp),
            "alpha_eff_m2_s": alpha_from_k_cp(k, cp),
            "effusivity_J_s05_m2_K": effusivity_from_k_cp(k, cp),
            "Rth_area_bottom_m2K_W": rth_area_bottom(k),
            "best_tau_s": float(best["tau_lag_s"]),
            "best_RMSE_C": float(best["RMSE_72C_C"]),
            "RMSE_tau0_C": rmse_tau0,
            "RMSE_improvement_due_to_lag_C": rmse_tau0 - float(
                best["RMSE_72C_C"]),
            "MAE_at_best_tau_C": float(best["MAE_72C_C"]),
            "mean_residual_at_best_tau_C": float(best["mean_residual_72C_C"]),
            "large_lag_warning": bool(best["large_lag_warning"]),
            "tau_boundary_warning": bool(best["tau_boundary_warning"]),
        })
    return pd.DataFrame(rows)


def global_rmse_minimum(df):
    i = int(df["RMSE_72C_C"].idxmin())
    return df.loc[i]


def near_optimal_sets(df, rmse_min):
    """返回 {band_name: DataFrame}。"""
    out = {}
    for name, band in BANDS.items():
        out[name] = df[df["RMSE_72C_C"] <= rmse_min + band].copy()
    out["RMSE_LE_1C"] = df[df["RMSE_72C_C"] <= RMSE_LE_1C].copy()
    return out


def highest_k_candidate(near_set, cp_min=NON_EXTREME_CP_MIN):
    """带内最高 k 候选 (cp >= cp_min)。无则 None。"""
    sub = near_set[near_set["cp_eff_J_kgK"] >= cp_min]
    if sub.empty:
        return None
    i = int(sub["k_eff_W_mK"].idxmax())
    return sub.loc[i]


def balanced_high_k_candidate(df, rmse_threshold=BALANCED_RMSE):
    """规则: RMSE<=th, cp>=800, tau<16, 选最高 k。无则 None。"""
    sub = df[(df["RMSE_72C_C"] <= rmse_threshold) &
             (df["cp_eff_J_kgK"] >= NON_EXTREME_CP_MIN) &
             (df["tau_lag_s"] < LARGE_LAG_THRESHOLD)]
    if sub.empty:
        return None
    i = int(sub["k_eff_W_mK"].idxmax())
    return sub.loc[i]


def summarize_set(near_set):
    if near_set.empty:
        return None
    return {
        "n_total": int(len(near_set)),
        "n_cp_ge_800": int((near_set["cp_eff_J_kgK"] >= NON_EXTREME_CP_MIN).sum()),
        "k_range": [float(near_set["k_eff_W_mK"].min()),
                    float(near_set["k_eff_W_mK"].max())],
        "cp_range": [float(near_set["cp_eff_J_kgK"].min()),
                     float(near_set["cp_eff_J_kgK"].max())],
        "tau_range": [float(near_set["tau_lag_s"].min()),
                      float(near_set["tau_lag_s"].max())],
        "alpha_range": [float(near_set["alpha_eff_m2_s"].min()),
                        float(near_set["alpha_eff_m2_s"].max())],
    }


# ============================================================
# 局部灵敏度 / 可辨识性
# ============================================================

def local_identifiability(k, cp, tau, t_proto, t_int, t_top_meas,
                          rel_pert=0.02, tau_pert=0.5):
    """在 (log k, log cp, tau) 上做局部数值灵敏度。

    对预测的滞后顶部迹线 (插值到实测时间) 计算 Jacobian (N x 3),
    奇异值 / 条件数 / 成对归一化相关性。
    仅诊断用, 不构成完整统计可辨识性证明。
    """
    base = lagged_top_prediction(t_proto, *run_fdm_cached(k, cp, t_proto,
                                                          t_int)[:2], tau)
    r0 = np.asarray(base, dtype=float)
    cols = []
    # log(k) 扰动
    for (pname, pval, delta) in (
        ("log_k", k, k * rel_pert),
        ("log_cp", cp, cp * rel_pert),
        ("tau", tau, tau_pert),
    ):
        if pname == "log_k":
            pred = lagged_top_prediction(
                t_proto, *run_fdm_cached(k + delta, cp, t_proto, t_int)[:2],
                tau)
        elif pname == "log_cp":
            pred = lagged_top_prediction(
                t_proto, *run_fdm_cached(k, cp + delta, t_proto, t_int)[:2],
                tau)
        else:
            pred = lagged_top_prediction(t_proto, *run_fdm_cached(k, cp,
                                                                  t_proto,
                                                                  t_int)[:2],
                                         tau + delta)
        cols.append((np.asarray(pred, dtype=float) - r0) / delta)

    J = np.column_stack(cols)
    # 归一化列 (避免量纲支配)
    Jn = J / np.linalg.norm(J, axis=0, keepdims=True)
    s = np.linalg.svd(Jn, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
    corr = np.corrcoef(J.T)
    # 最强相关对
    pairs = []
    n = 3
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(corr[i, j])))
    strongest = max(pairs, key=lambda x: abs(x[2]))
    names = ["log(k)", "log(cp)", "tau"]
    return {
        "singular_values": [float(v) for v in s],
        "condition_number": cond,
        "correlation_matrix": corr.tolist(),
        "strongest_pair": (names[strongest[0]], names[strongest[1]],
                           strongest[2]),
        "identifiability_warning": bool(cond > 100.0 or
                                        abs(strongest[2]) > 0.999),
    }


# ============================================================
# 图形
# ============================================================

def _grid_landscape(prof, value_col, title, path, cmap="viridis",
                    vmin=None, vmax=None, log_color=False, markers=(),
                    contour=None):
    ks = np.sort(prof["k_eff_W_mK"].unique())
    cps = np.sort(prof["cp_eff_J_kgK"].unique())
    Z = np.full((len(ks), len(cps)), np.nan)
    for _, r in prof.iterrows():
        ki = int(np.searchsorted(ks, r["k_eff_W_mK"]))
        ci = int(np.searchsorted(cps, r["cp_eff_J_kgK"]))
        Z[ki, ci] = r[value_col]
    fig, ax = plt.subplots(figsize=(9, 6))
    kw = dict(shading="auto", cmap=cmap)
    if vmin is not None and vmax is not None:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if log_color:
        from matplotlib.colors import LogNorm
        kw["norm"] = LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    mesh = ax.pcolormesh(cps, ks, Z, **kw)
    cb = fig.colorbar(mesh, ax=ax, label=title.split("—")[-1].strip())
    if contour is not None:
        ax.contour(cps, ks, Z, levels=[contour], colors="white", ls="--",
                   linewidths=1.2)
    ax.set_xlabel("cp_eff [J/(kg·K)]")
    ax.set_ylabel("k_eff [W/(m·K)]")
    ax.set_title(title)
    for (x, y, lab) in markers:
        ax.plot(x, y, "k*", ms=14)
        ax.annotate(lab, (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_profiled_rmse(prof, markers, output_dir):
    _grid_landscape(prof, "best_RMSE_C",
                    "Profiled RMSE (min over tau) — k vs cp",
                    output_dir / "profiled_rmse_k_cp.png",
                    vmin=0.5, vmax=4.0, markers=markers, contour=1.0)


def plot_best_tau(prof, output_dir):
    _grid_landscape(prof, "best_tau_s", "Best tau_lag [s] — k vs cp",
                    output_dir / "best_tau_k_cp.png",
                    cmap="plasma", vmin=0.0, vmax=20.0)


def plot_alpha(prof, output_dir):
    prof2 = prof.copy()
    _grid_landscape(prof2, "alpha_eff_m2_s",
                    "Derived alpha_eff [m²/s] — k vs cp (log color)",
                    output_dir / "alpha_k_cp.png",
                    cmap="inferno", log_color=True)


def plot_best_rmse_vs_k(prof, output_dir):
    rows = []
    for k, sub in prof.groupby("k_eff_W_mK"):
        i = int(sub["best_RMSE_C"].idxmin())
        best = sub.loc[i]
        rows.append(best)
    b = pd.DataFrame(rows).sort_values("k_eff_W_mK")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(b["k_eff_W_mK"], b["best_RMSE_C"], marker="o",
            label="min RMSE (profiled over cp and tau)")
    ax.axhline(0.7337, color="black", ls=":", label="Strategy A RMSE 0.7337")
    ax.axhline(1.0, color="green", ls="-.", alpha=0.6, label="RMSE = 1.0 C")
    ax.set_xlabel("k_eff [W/(m·K)]")
    ax.set_ylabel("Best RMSE_top [°C]")
    ax.set_title("Best achievable 72C RMSE vs k (profiled over cp, tau)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "best_rmse_vs_k.png", dpi=200)
    plt.close(fig)
    return b


def plot_trace_comparison(t_proto, t_int, t_top_meas, candidates, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_proto, t_top_meas, color="black", lw=1.6,
            label="Measured Top COC")
    # 策略 A
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    tA, topA, _ = run_fdm_cached(cal.k_eff_W_mK, cal.cp_eff_J_kgK,
                                 t_proto, t_int)
    ax.plot(t_proto, np.interp(t_proto, tA, topA), color="#1f77b4", lw=1.4,
            label="Strategy A (k=0.0165, cp=900, tau=0)")
    # 全局最小
    g = candidates.get("GLOBAL")
    if g is not None:
        pred = lagged_top_prediction(t_proto, *run_fdm_cached(
            g["k_eff_W_mK"], g["cp_eff_J_kgK"], t_proto, t_int)[:2],
            g["tau_lag_s"])
        ax.plot(t_proto, pred, color="#2ca02c", lw=1.4, ls="--",
                label=f"Global min (k={g['k_eff_W_mK']:.3f}, "
                      f"cp={g['cp_eff_J_kgK']:.0f}, tau={g['tau_lag_s']:.1f})")
    # 平衡候选
    bal = candidates.get("BALANCED")
    if bal is not None:
        pred = lagged_top_prediction(t_proto, *run_fdm_cached(
            bal["k_eff_W_mK"], bal["cp_eff_J_kgK"], t_proto, t_int)[:2],
            bal["tau_lag_s"])
        ax.plot(t_proto, pred, color="#d62728", lw=1.4, ls="--",
                label=f"Balanced (k={bal['k_eff_W_mK']:.3f}, "
                      f"cp={bal['cp_eff_J_kgK']:.0f}, tau={bal['tau_lag_s']:.1f})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("72C Top-Fit — Strategy D candidates vs Strategy A\n"
                 "(lagged observation prediction)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "72C_trace_comparison_k_cp_tau.png", dpi=200)
    plt.close(fig)


def plot_residual_comparison(t_proto, t_int, t_top_meas, candidates,
                             output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    tA, topA, _ = run_fdm_cached(cal.k_eff_W_mK, cal.cp_eff_J_kgK,
                                 t_proto, t_int)
    rA = np.interp(t_proto, tA, topA) - t_top_meas
    ax.plot(t_proto, rA, color="#1f77b4", lw=1.0, label="Strategy A")
    for name, col in (("GLOBAL", "#2ca02c"), ("BALANCED", "#d62728")):
        c = candidates.get(name)
        if c is None:
            continue
        pred = lagged_top_prediction(t_proto, *run_fdm_cached(
            c["k_eff_W_mK"], c["cp_eff_J_kgK"], t_proto, t_int)[:2],
            c["tau_lag_s"])
        ax.plot(t_proto, pred - t_top_meas, color=col, lw=1.2, ls="--",
                label=f"{name}")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual (pred - meas) [°C]")
    ax.set_title("72C Residual vs Time — Strategy A vs Strategy D candidates")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "72C_residual_comparison_k_cp_tau.png", dpi=200)
    plt.close(fig)


# ============================================================
# 输出
# ============================================================

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


def _cand_dict(cand):
    if cand is None:
        return None
    return {
        "k_eff_W_mK": float(cand["k_eff_W_mK"]),
        "cp_eff_J_kgK": float(cand["cp_eff_J_kgK"]),
        "tau_lag_s": float(cand["tau_lag_s"]),
        "alpha_eff_m2_s": float(cand["alpha_eff_m2_s"]),
        "effusivity_J_s05_m2_K": float(cand["effusivity_J_s05_m2_K"]),
        "Rth_area_bottom_m2K_W": float(cand["Rth_area_bottom_m2K_W"]),
        "RMSE_72C_C": float(cand["RMSE_72C_C"]),
        "MAE_72C_C": float(cand["MAE_72C_C"]),
        "mean_residual_72C_C": float(cand["mean_residual_72C_C"]),
    }


def run_analysis(t_proto, t_int, t_top_meas, output_dir, df=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_csv(output_dir / "k_cp_tau_full_scan.csv")

    # ---- 剖面 ----
    prof = profile_best_tau(df)
    prof.to_csv(output_dir / "k_cp_profiled_best_tau.csv", index=False)

    # ---- 全局最小 ----
    g = global_rmse_minimum(df)

    # ---- 近最优集 ----
    sets = near_optimal_sets(df, float(g["RMSE_72C_C"]))
    set_summary = {name: summarize_set(s) for name, s in sets.items()}

    # ---- 高 k 候选 ----
    high_k = {}
    for name in ("STRICT", "MODERATE", "APPLICATION", "RMSE_LE_1C"):
        cand = highest_k_candidate(sets[name])
        if cand is not None:
            cand["Delta_RMSE_C"] = float(cand["RMSE_72C_C"]) - float(
                g["RMSE_72C_C"])
        high_k[name] = cand

    # ---- 平衡候选 ----
    bal = balanced_high_k_candidate(df, BALANCED_RMSE)
    bal_relaxed = balanced_high_k_candidate(df, BALANCED_RMSE_RELAXED)

    # ---- 候选表 ----
    cand_rows = []
    def _row(name, cand, delta=True):
        if cand is None:
            return
        r = dict(cand)
        r["candidate_name"] = name
        if delta and "Delta_RMSE_C" not in r:
            r["Delta_RMSE_C"] = float(r["RMSE_72C_C"]) - float(
                g["RMSE_72C_C"])
        cand_rows.append(r)
    _row("GLOBAL_RMSE_MINIMUM", g, delta=False)
    _row("HIGH_K_STRICT_CANDIDATE", high_k["STRICT"])
    _row("HIGH_K_MODERATE_CANDIDATE", high_k["MODERATE"])
    _row("HIGH_K_APPLICATION_CANDIDATE", high_k["APPLICATION"])
    _row("HIGH_K_RMSE_LE_1C_CANDIDATE", high_k["RMSE_LE_1C"])
    _row("BALANCED_HIGH_K_CANDIDATE", bal, delta=False)
    _row("BALANCED_HIGH_K_CANDIDATE_RELAXED_1_2C", bal_relaxed, delta=False)
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(output_dir / "high_k_candidate_summary.csv", index=False)

    # 近最优候选明细 (全部点在带内 + cp>=800)
    near_df_rows = []
    for name, s in sets.items():
        sub = s[s["cp_eff_J_kgK"] >= NON_EXTREME_CP_MIN]
        tmp = sub.copy()
        tmp["fit_set"] = name
        near_df_rows.append(tmp)
    near_all = pd.concat(near_df_rows, ignore_index=True) if near_df_rows \
        else pd.DataFrame()
    near_all.to_csv(output_dir / "near_optimal_k_cp_tau_candidates.csv",
                    index=False)

    # ---- 图 ----
    markers = [
        (900.0, 0.0165, "A"),
        (g["cp_eff_J_kgK"], g["k_eff_W_mK"], "global"),
    ]
    if bal is not None:
        markers.append((bal["cp_eff_J_kgK"], bal["k_eff_W_mK"], "bal"))
    hk_app = high_k["APPLICATION"]
    if hk_app is not None:
        markers.append((hk_app["cp_eff_J_kgK"], hk_app["k_eff_W_mK"], "app"))
    plot_profiled_rmse(prof, markers, output_dir)
    plot_best_tau(prof, output_dir)
    plot_alpha(prof, output_dir)
    best_vs_k = plot_best_rmse_vs_k(prof, output_dir)
    plot_trace_comparison(t_proto, t_int, t_top_meas,
                          {"GLOBAL": g, "BALANCED": bal}, output_dir)
    plot_residual_comparison(t_proto, t_int, t_top_meas,
                             {"GLOBAL": g, "BALANCED": bal}, output_dir)

    # ---- 可辨识性 ----
    ident = {}
    ident["global_min"] = local_identifiability(
        g["k_eff_W_mK"], g["cp_eff_J_kgK"], g["tau_lag_s"],
        t_proto, t_int, t_top_meas)
    if bal is not None:
        ident["balanced"] = local_identifiability(
            bal["k_eff_W_mK"], bal["cp_eff_J_kgK"], bal["tau_lag_s"],
            t_proto, t_int, t_top_meas)
    (output_dir / "local_identifiability_diagnostics.json").write_text(
        json.dumps(ident, indent=2), encoding="utf-8")

    # ---- 元数据 / 摘要 ----
    metadata = {
        "strategy_id": "lag_separated_k_cp_tau_v1",
        "status": "EXPERIMENTAL / THREE-PARAMETER CHARACTERIZATION",
        "accepted_as_nominal": False,
        "grid": {
            "k": K_GRID, "cp": CP_GRID, "tau": TAU_GRID,
            "unique_fdm_runs": len(K_GRID) * len(CP_GRID),
            "rows": len(df),
        },
        "global_rmse_minimum": _cand_dict(g),
        "near_optimal_sets": set_summary,
        "high_k_candidates": {k: _cand_dict(v) for k, v in high_k.items()},
        "balanced_high_k_candidate": _cand_dict(bal),
        "balanced_1_2C": _cand_dict(bal_relaxed),
        "non_extreme_cp_subset_min": NON_EXTREME_CP_MIN,
        "cp_scan_min": 600.0,
        "large_lag_threshold_s": LARGE_LAG_THRESHOLD,
        "tau_scan_boundary_s": TAU_BOUNDARY,
        "identifiability": ident,
        "git_commit": _git_head(),
        "git_tag": _git_describe(),
        "note": (
            "No k-reward/penalty in objective. cp>=800 is a project-level "
            "reporting subset, not a literature confidence interval. "
            "tau>=16 s flagged LARGE_LAG; tau=20 s flagged boundary. "
            "DOE11 / PCR sample predictions NOT used for selection."
        ),
    }
    (output_dir / "strategy_D_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_summary(output_dir, g, sets, high_k, bal, bal_relaxed, prof,
                   best_vs_k, ident)

    return prof, g, sets, high_k, bal, bal_relaxed, ident


def _write_summary(output_dir, g, sets, high_k, bal, bal_relaxed, prof,
                   best_vs_k, ident):
    lines = [
        "STRATEGY D — lag-separated (k, cp, tau) 3-PARAMETER CHARACTERIZATION",
        "=" * 72,
        f"Grid: k {len(K_GRID)} x cp {len(CP_GRID)} x tau {len(TAU_GRID)}",
        f"Unique FDM runs: {len(K_GRID)*len(CP_GRID)}; rows: "
        f"{len(K_GRID)*len(CP_GRID)*len(TAU_GRID)}",
        "",
        f"GLOBAL RMSE MINIMUM: k={g['k_eff_W_mK']:.4f} "
        f"cp={g['cp_eff_J_kgK']:.0f} tau={g['tau_lag_s']:.2f} "
        f"alpha={g['alpha_eff_m2_s']:.3e} RMSE={g['RMSE_72C_C']:.4f} "
        f"cp>=800: {'YES' if g['cp_eff_J_kgK']>=NON_EXTREME_CP_MIN else 'NO'}",
        "",
        "Near-optimal sets (Delta from global min):",
    ]
    for name, s in sets.items():
        ss = summarize_set(s)
        if ss is None:
            lines.append(f"  {name}: empty")
        else:
            lines.append(
                f"  {name}: n={ss['n_total']} (cp>=800: {ss['n_cp_ge_800']}) "
                f"k[{ss['k_range'][0]:.3f},{ss['k_range'][1]:.3f}] "
                f"cp[{ss['cp_range'][0]:.0f},{ss['cp_range'][1]:.0f}] "
                f"tau[{ss['tau_range'][0]:.1f},{ss['tau_range'][1]:.1f}] "
                f"alpha[{ss['alpha_range'][0]:.2e},{ss['alpha_range'][1]:.2e}]"
            )
    lines.append("")
    lines.append("High-k candidates (cp>=800):")
    for name, c in high_k.items():
        if c is None:
            lines.append(f"  {name}: none")
        else:
            lines.append(
                f"  {name}: k={c['k_eff_W_mK']:.4f} cp={c['cp_eff_J_kgK']:.0f} "
                f"tau={c['tau_lag_s']:.1f} alpha={c['alpha_eff_m2_s']:.2e} "
                f"RMSE={c['RMSE_72C_C']:.4f} "
                f"Delta={c.get('Delta_RMSE_C', float('nan')):.4f}")
    lines.append("")
    lines.append("Balanced high-k candidate (RMSE<=1, cp>=800, tau<16):")
    lines.append(f"  {bal.to_dict() if bal is not None else 'NONE'}")
    lines.append("Relaxed (RMSE<=1.2):")
    lines.append(f"  {bal_relaxed.to_dict() if bal_relaxed is not None else 'NONE'}")
    lines.append("")
    lines.append("Best RMSE vs k (profiled over cp, tau):")
    lines.append(best_vs_k[["k_eff_W_mK", "best_RMSE_C",
                            "best_tau_s"]].to_string(index=False))
    lines.append("")
    lines.append("Identifiability (local sensitivity):")
    for key, d in ident.items():
        lines.append(f"  {key}: cond={d['condition_number']:.1f} "
                     f"s={['%.3f' % v for v in d['singular_values']]} "
                     f"strongest={d['strongest_pair']} "
                     f"warn={d['identifiability_warning']}")
    lines.append("")
    lines.append("NOTE: no k-reward in objective; DOE11/PCR not used; "
                 "candidates are NOT the final model.")
    (output_dir / "strategy_D_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=["regression", "scan", "analysis", "all"])
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return p.parse_args(argv)


def strategy_a_regression(t_proto, t_int, t_top_meas, output_dir):
    """k=0.0165, cp=900, tau=0 必须复现 RMSE ~0.7337。"""
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = alpha_from_k_cp(cal.k_eff_W_mK, cal.cp_eff_J_kgK)
    r = evaluate_k_cp_tau(0.0165, 900.0, 0.0, t_proto, t_int, t_top_meas)
    ok = (abs(alpha_A - 0.0165 / (RHO_COC * 900.0)) < 1e-20 and
          abs(r["RMSE_72C_C"] - 0.7337) < 0.01)
    res = {
        "k": 0.0165, "cp": 900.0, "tau": 0.0,
        "alpha": alpha_A, "RMSE": r["RMSE_72C_C"], "expected": 0.7337,
        "pass": bool(ok),
    }
    (output_dir / "strategy_A_regression.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"[regression] alpha={alpha_A:.6e} RMSE={r['RMSE_72C_C']:.6f} "
          f"(expect ~0.7337) -> {'PASS' if ok else 'FAIL'}")
    return res


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_proto, t_int, t_top_meas = load_experiment()

    if args.stage in ("regression", "all"):
        res = strategy_a_regression(t_proto, t_int, t_top_meas, output_dir)
        if not res["pass"]:
            print("[STOP] Strategy A regression FAILED — aborting.")
            return 2

    if args.stage in ("scan", "all"):
        t0 = time.perf_counter()
        df = run_full_scan(t_proto, t_int, t_top_meas, output_dir)
        print(f"[scan] elapsed {time.perf_counter() - t0:.1f} s")

    if args.stage in ("analysis", "all"):
        prof, g, sets, high_k, bal, bal_relaxed, ident = run_analysis(
            t_proto, t_int, t_top_meas, output_dir)
        print(f"[analysis] global min: k={g['k_eff_W_mK']:.4f} "
              f"cp={g['cp_eff_J_kgK']:.0f} tau={g['tau_lag_s']:.1f} "
              f"RMSE={g['RMSE_72C_C']:.4f}")
        for name, c in high_k.items():
            print(f"  {name}: "
                  f"{'none' if c is None else 'k=%.4f cp=%.0f tau=%.1f RMSE=%.4f' % (c['k_eff_W_mK'], c['cp_eff_J_kgK'], c['tau_lag_s'], c['RMSE_72C_C'])}")
        print(f"  BALANCED: {'none' if bal is None else 'k=%.4f cp=%.0f tau=%.1f RMSE=%.4f' % (bal['k_eff_W_mK'], bal['cp_eff_J_kgK'], bal['tau_lag_s'], bal['RMSE_72C_C'])}")
        print(f"[output] {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
