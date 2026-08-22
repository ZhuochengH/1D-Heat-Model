#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy G — Strategy F 盆地的靶向局部细化
==========================================

科学问题:
    Strategy F 粗网格给出 k=0.055, cp=1200, tau=8.0 s, RMSE=0.9554 C,
    但 tau=8 s 是粗网格上界, RMSE 在 tau 方向仍下降。
    本任务只做一次局部细化:

    1. 验证 k≈0.055 是否为稳定的内部最优 (k 局部网格 0.050-0.060);
    2. 分辨 tau 真实最优 (tau 局部网格 4.0-12.0 s, 步长 0.5);
    3. 刻画 cp/tau 权衡 (cp 1000-1800, 关注 k=0.055 的 cp-tau RMSE 矩阵)。

物理 (不变, 复用 Strategy E/F):
    h_conv = 10.0 W/(m2 K)
    epsilon_surface = 0.90
    sigma_SB = 5.670374419e-8 W/(m2 K4)
    F_view = 1.0
    非线性 Stefan-Boltzmann 辐射 (求解器内不线性化)
    rho_COC = 1020 kg/m3
    环境 = 第一个有效实测 Top COC 温度 (恒定)
    初始场 = 第一个内部温度
    底部边界 = 实测内部温度 Dirichlet
    滞后 = 既有输出侧一阶滞后 (只作用 T_top_observed_predicted)

目标 (无应用偏差):
    72C 修正测量时间 RMSE (等权, 查询轴 = 实测时间)
    无高-k 奖励 / 无高-alpha 奖励 / 无样品温度参与选参。

计算效率:
    25 个唯一 (k,cp) 各跑一次非线性 FDM, 17 个 tau 复用同一 FDM 迹线
    -> 25 FDM + 425 组合。

本任务: 无 DOE11 / 长保持 PCR / 连续优化器 / 自动二次细化。
输出目录: parameter_scan_output/72C/convection_radiation_k_cp_tau_local_refinement_v1/
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "72C"
    / "aligned_internal_top_temperature.csv")
STRATEGY_F_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "convection_radiation_k_cp_tau_calibration_v1")
OUTPUT_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "convection_radiation_k_cp_tau_local_refinement_v1")

# ---------------------------------------------------------------
# 局部网格 (任务指定, 精确)
# ---------------------------------------------------------------
K_GRID = [0.0500, 0.0525, 0.0550, 0.0575, 0.0600]
CP_GRID = [1000.0, 1200.0, 1400.0, 1600.0, 1800.0]
TAU_GRID = [4.0 + 0.5 * i for i in range(17)]   # 4.0 ... 12.0 步长 0.5

K_LOCAL_LO, K_LOCAL_HI = 0.0500, 0.0600
CP_LOCAL_LO, CP_LOCAL_HI = 1000.0, 1800.0
TAU_LOCAL_LO, TAU_LOCAL_HI = 4.0, 12.0

RHO_COC = 1020.0
L_BOTTOM_M = 180e-6
SAVE_DT = 1.0

# Strategy F 锚点
ANCHOR_A = {"k": 0.055, "cp": 1200.0, "tau": 8.0, "RMSE": 0.9554}
ANCHOR_B = {"k": 0.055, "cp": 1800.0, "tau": 5.0, "RMSE": 1.1037}

BANDS = {"STRICT": 0.05, "MODERATE": 0.10, "APPLICATION": 0.20}
RMSE_LE_1C = 1.0
RMSE_LE_1_2C = 1.2
MODERATE_LAG_TAU_MAX = 6.0

# 分类阈值
K_STABLE_LO, K_STABLE_HI = 0.0525, 0.0575


# ============================================================
# 派生量
# ============================================================

def alpha_from_k_cp(k, cp, rho=RHO_COC):
    return float(k) / (rho * float(cp))


def effusivity_from_k_cp(k, cp, rho=RHO_COC):
    return float(np.sqrt(float(k) * rho * float(cp)))


def rth_area_bottom(k):
    return L_BOTTOM_M / float(k)


# ============================================================
# FDM-once-per-(k,cp) 缓存 (Strategy E 非线性求解器)
# ============================================================

_fdm_cache = {}


def run_cr_fdm_cached(k, cp, t_proto, t_int, t_env, save_dt=SAVE_DT):
    """每个 (k,cp) 只跑一次 Strategy E 非线性 FDM。"""
    key = (float(k), float(cp))
    if key in _fdm_cache:
        return _fdm_cache[key]
    mats = cr.make_convection_radiation_materials(k, cp)
    result = cr.run_convection_radiation_fdm(
        time_s=t_proto,
        bottom_temperature_C=t_int,
        materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS,
        T_air_C=t_env,
        T_surroundings_C=t_env,
        save_dt=save_dt,
        T_initial_C=float(t_int[0]),
    )
    entry = (result["t_array"], result["T_top_surface_arr"],
             result["T_sample_arr"])
    _fdm_cache[key] = entry
    return entry


def clear_fdm_cache():
    _fdm_cache.clear()


# ============================================================
# 指标 (查询轴 = 实测时间)
# ============================================================

def lagged_top_prediction(t_proto, t_arr, t_top_fdm, tau):
    if tau == 0.0:
        t_top_obs = t_top_fdm.copy()
    else:
        t_top_obs = apply_first_order_lag(t_arr, t_top_fdm, tau)
    return np.interp(t_proto, t_arr, t_top_obs)


def metrics_for_prediction(pred, t_top_meas):
    r = np.asarray(pred, dtype=float) - np.asarray(t_top_meas, dtype=float)
    return {
        "RMSE_72C_C": float(np.sqrt(np.mean(r ** 2))),
        "MAE_72C_C": float(np.mean(np.abs(r))),
        "median_abs_residual_C": float(np.median(np.abs(r))),
        "mean_residual_C": float(np.mean(r)),
        "max_abs_residual_C": float(np.max(np.abs(r))),
    }


def evaluate_k_cp_tau(k, cp, tau, t_proto, t_int, t_top_meas, t_env,
                      save_dt=SAVE_DT):
    t_arr, t_top_fdm, _ = run_cr_fdm_cached(k, cp, t_proto, t_int, t_env,
                                            save_dt)
    pred = lagged_top_prediction(t_proto, t_arr, t_top_fdm, tau)
    m = metrics_for_prediction(pred, t_top_meas)
    return {
        "k_eff_W_mK": float(k),
        "cp_eff_J_kgK": float(cp),
        "tau_lag_s": float(tau),
        "alpha_eff_m2_s": alpha_from_k_cp(k, cp),
        "effusivity": effusivity_from_k_cp(k, cp),
        "Rth_bottom_area_m2K_W": rth_area_bottom(k),
        "RMSE_72C_C": m["RMSE_72C_C"],
        "MAE_72C_C": m["MAE_72C_C"],
        "median_abs_residual_C": m["median_abs_residual_C"],
        "mean_residual_C": m["mean_residual_C"],
        "max_abs_residual_C": m["max_abs_residual_C"],
        "k_local_boundary": bool(k <= K_LOCAL_LO + 1e-12 or
                                 k >= K_LOCAL_HI - 1e-12),
        "cp_local_boundary": bool(cp <= CP_LOCAL_LO + 1e-12 or
                                  cp >= CP_LOCAL_HI - 1e-12),
        "tau_local_boundary": bool(tau >= TAU_LOCAL_HI - 1e-12),
        "status": "OK",
    }


def run_full_scan(t_proto, t_int, t_top_meas, t_env, output_dir):
    """25 次 Strategy E FDM + 17 tau 剖面 = 425 行。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "local_k_cp_tau_full_scan.csv"
    clear_fdm_cache()
    rows = []
    t0 = time.perf_counter()
    n_fdm = 0
    for k in K_GRID:
        for cp in CP_GRID:
            n_fdm += 1
            for tau in TAU_GRID:
                rows.append(evaluate_k_cp_tau(k, cp, tau, t_proto, t_int,
                                              t_top_meas, t_env))
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[scan] unique FDM runs: {n_fdm} (expected 25); "
          f"rows: {len(df)} (expected 425); "
          f"failures: {(df['status'] == 'FAILED').sum()}; "
          f"elapsed {time.perf_counter() - t0:.1f} s")
    return df


# ============================================================
# Strategy F 锚点回归
# ============================================================

def strategy_f_anchor_regression(t_proto, t_int, t_top_meas, t_env):
    """复现 Strategy F 两个已知点 (Anchor A / Anchor B)。"""
    out = {}
    for name, a in (("A", ANCHOR_A), ("B", ANCHOR_B)):
        t_arr, t_top_fdm, _ = run_cr_fdm_cached(
            a["k"], a["cp"], t_proto, t_int, t_env)
        pred = lagged_top_prediction(t_proto, t_arr, t_top_fdm, a["tau"])
        m = metrics_for_prediction(pred, t_top_meas)
        ok = abs(m["RMSE_72C_C"] - a["RMSE"]) < 0.01
        out[name] = {
            "k": a["k"], "cp": a["cp"], "tau": a["tau"],
            "previous_RMSE": a["RMSE"],
            "reproduced_RMSE": m["RMSE_72C_C"],
            "PASS": bool(ok),
        }
    return out


# ============================================================
# 剖面 / 候选
# ============================================================

def profile_best_tau(df):
    """每个 (k,cp) 的最佳 tau。"""
    rows = []
    for (k, cp), sub in df.groupby(["k_eff_W_mK", "cp_eff_J_kgK"]):
        i_best = int(sub["RMSE_72C_C"].idxmin())
        best = sub.loc[i_best]
        def _rmse_at(tau):
            s = sub[sub["tau_lag_s"] == tau]
            return float(s["RMSE_72C_C"].iloc[0]) if len(s) else np.nan
        rows.append({
            "k_eff_W_mK": float(k),
            "cp_eff_J_kgK": float(cp),
            "alpha_eff_m2_s": alpha_from_k_cp(k, cp),
            "effusivity": effusivity_from_k_cp(k, cp),
            "Rth_bottom_area_m2K_W": rth_area_bottom(k),
            "best_tau_s": float(best["tau_lag_s"]),
            "best_RMSE_C": float(best["RMSE_72C_C"]),
            "best_MAE_C": float(best["MAE_72C_C"]),
            "mean_residual_C": float(best["mean_residual_C"]),
            "RMSE_tau5_C": _rmse_at(5.0),
            "RMSE_tau8_C": _rmse_at(8.0),
            "RMSE_tau12_C": _rmse_at(12.0),
            "tau_local_boundary_warning": bool(
                best["tau_lag_s"] >= TAU_LOCAL_HI - 1e-12),
        })
    return pd.DataFrame(rows)


def local_global_minimum(df):
    i = int(df["RMSE_72C_C"].idxmin())
    return df.loc[i]


def near_optimal_sets(df, rmse_min):
    out = {}
    for name, band in BANDS.items():
        out[name] = df[df["RMSE_72C_C"] <= rmse_min + band].copy()
    out["RMSE_LE_1C"] = df[df["RMSE_72C_C"] <= RMSE_LE_1C].copy()
    out["RMSE_LE_1_2C"] = df[df["RMSE_72C_C"] <= RMSE_LE_1_2C].copy()
    return out


def moderate_lag_near_optimal_candidate(df, rmse_min):
    """规则: Delta<=0.20, tau<=6.0 s, 最低 RMSE。无则 None。"""
    sub = df[(df["RMSE_72C_C"] <= rmse_min + BANDS["APPLICATION"]) &
             (df["tau_lag_s"] <= MODERATE_LAG_TAU_MAX + 1e-12)]
    if sub.empty:
        return None
    i = int(sub["RMSE_72C_C"].idxmin())
    return sub.loc[i]


def summarize_set(near_set):
    if near_set.empty:
        return None
    return {
        "n_total": int(len(near_set)),
        "k_range": [float(near_set["k_eff_W_mK"].min()),
                    float(near_set["k_eff_W_mK"].max())],
        "cp_range": [float(near_set["cp_eff_J_kgK"].min()),
                     float(near_set["cp_eff_J_kgK"].max())],
        "tau_range": [float(near_set["tau_lag_s"].min()),
                      float(near_set["tau_lag_s"].max())],
        "alpha_range": [float(near_set["alpha_eff_m2_s"].min()),
                        float(near_set["alpha_eff_m2_s"].max())],
    }


def best_rmse_vs_k(prof):
    rows = []
    for k, sub in prof.groupby("k_eff_W_mK"):
        i = int(sub["best_RMSE_C"].idxmin())
        best = sub.loc[i]
        rows.append({
            "k_eff_W_mK": float(k),
            "best_cp_eff_J_kgK": float(best["cp_eff_J_kgK"]),
            "best_tau_s": float(best["best_tau_s"]),
            "best_alpha_eff_m2_s": float(best["alpha_eff_m2_s"]),
            "best_RMSE_C": float(best["best_RMSE_C"]),
        })
    return pd.DataFrame(rows).sort_values("k_eff_W_mK")


# ============================================================
# 局部可辨识性 (log k, log cp, tau)
# ============================================================

def local_identifiability(k, cp, tau, t_proto, t_int, t_top_meas, t_env,
                          rel_pert=0.02, tau_pert=0.5, save_dt=SAVE_DT):
    t_arr, t_top_fdm, _ = run_cr_fdm_cached(k, cp, t_proto, t_int, t_env,
                                            save_dt)
    r0 = np.asarray(lagged_top_prediction(t_proto, t_arr, t_top_fdm, tau),
                    dtype=float)
    cols = []
    for pname in ("log_k", "log_cp", "tau"):
        if pname == "log_k":
            kp = k + k * rel_pert
            ta, tf, _ = run_cr_fdm_cached(kp, cp, t_proto, t_int, t_env,
                                          save_dt)
            pred = lagged_top_prediction(t_proto, ta, tf, tau)
            delta = k * rel_pert
        elif pname == "log_cp":
            cp2 = cp + cp * rel_pert
            ta, tf, _ = run_cr_fdm_cached(k, cp2, t_proto, t_int, t_env,
                                          save_dt)
            pred = lagged_top_prediction(t_proto, ta, tf, tau)
            delta = cp * rel_pert
        else:
            ta, tf, _ = run_cr_fdm_cached(k, cp, t_proto, t_int, t_env,
                                          save_dt)
            pred = lagged_top_prediction(t_proto, ta, tf, tau + tau_pert)
            delta = tau_pert
        cols.append((np.asarray(pred, dtype=float) - r0) / delta)
    J = np.column_stack(cols)
    Jn = J / np.linalg.norm(J, axis=0, keepdims=True)
    s = np.linalg.svd(Jn, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
    corr = np.corrcoef(J.T)
    names = ["log(k)", "log(cp)", "tau"]
    return {
        "singular_values": [float(v) for v in s],
        "condition_number": cond,
        "correlation_logk_logcp": float(corr[0, 1]),
        "correlation_logk_tau": float(corr[0, 2]),
        "correlation_logcp_tau": float(corr[1, 2]),
        "correlation_matrix": corr.tolist(),
        "identifiability_warning": bool(cond > 100.0 or
                                        np.max(np.abs(corr[
                                            np.triu_indices(3, 1)])) > 0.999),
    }


# ============================================================
# 图形
# ============================================================

def _grid_landscape(prof, value_col, title, path, cmap="viridis",
                    vmin=None, vmax=None, log_color=False, markers=(),
                    contour=None, x_axis="cp", y_axis="tau"):
    """通用 2D 网格图 (x=c 轴, y=行轴, 由 prof 行索引)。"""
    xs = np.sort(prof["cp_eff_J_kgK"].unique())
    ys = np.sort(prof["k_eff_W_mK"].unique())
    Z = np.full((len(ys), len(xs)), np.nan)
    for _, r in prof.iterrows():
        xi = int(np.searchsorted(xs, r["cp_eff_J_kgK"]))
        yi = int(np.searchsorted(ys, r["k_eff_W_mK"]))
        Z[yi, xi] = r[value_col]
    fig, ax = plt.subplots(figsize=(9, 6))
    kw = dict(shading="auto", cmap=cmap)
    if vmin is not None and vmax is not None:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if log_color:
        from matplotlib.colors import LogNorm
        kw["norm"] = LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    mesh = ax.pcolormesh(xs, ys, Z, **kw)
    cb = fig.colorbar(mesh, ax=ax, label=title.split("—")[-1].strip())
    if contour is not None:
        ax.contour(xs, ys, Z, levels=[contour], colors="white",
                   linestyles="--", linewidths=1.2)
    ax.set_xlabel("cp_eff [J/(kg·K)]")
    ax.set_ylabel("k_eff [W/(m·K)]")
    ax.set_title(title)
    for (x, y, lab) in markers:
        ax.plot(x, y, "k*", ms=15)
        ax.annotate(lab, (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_best_rmse_vs_k(prof, output_dir):
    b = best_rmse_vs_k(prof)
    b.to_csv(output_dir / "local_best_rmse_vs_k.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(b["k_eff_W_mK"], b["best_RMSE_C"], marker="o",
            label="min RMSE (profiled over cp, tau)")
    ax.axhline(1.0, color="green", ls="-.", alpha=0.6, label="RMSE = 1.0 C")
    ax.axhline(0.7337, color="black", ls=":",
               label="Strategy A historical RMSE (old h=5 boundary)")
    g = b.loc[b["best_RMSE_C"].idxmin()]
    ax.plot(g["k_eff_W_mK"], g["best_RMSE_C"], "r*", ms=16,
            label=f"local min k = {g['k_eff_W_mK']:.4f}")
    ax.set_xlabel("k_eff [W/(m·K)]")
    ax.set_ylabel("Best RMSE_top [°C]")
    ax.set_title("Strategy G — best RMSE vs k (local grid)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "local_best_rmse_vs_k.png", dpi=200)
    plt.close(fig)
    return b


def plot_cp_tau_rmse_at_k055(df, output_dir):
    """k=0.055 的 cp-tau RMSE 矩阵 (x=cp, y=tau)。"""
    sub = df[df["k_eff_W_mK"] == 0.055]
    cps = np.sort(sub["cp_eff_J_kgK"].unique())
    taus = np.sort(sub["tau_lag_s"].unique())
    Z = np.full((len(taus), len(cps)), np.nan)
    for _, r in sub.iterrows():
        ci = int(np.searchsorted(cps, r["cp_eff_J_kgK"]))
        ti = int(np.searchsorted(taus, r["tau_lag_s"]))
        Z[ti, ci] = r["RMSE_72C_C"]
    out = sub[["cp_eff_J_kgK", "tau_lag_s", "RMSE_72C_C"]].pivot(
        index="tau_lag_s", columns="cp_eff_J_kgK", values="RMSE_72C_C")
    out.to_csv(output_dir / "cp_tau_rmse_at_k055.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(cps, taus, Z, shading="auto", cmap="viridis")
    cb = fig.colorbar(mesh, ax=ax, label="RMSE [°C]")
    ax.contour(cps, taus, Z, levels=[1.0], colors="white", linestyles="--",
               linewidths=1.2)
    # 标记点
    ax.plot(1200.0, 8.0, "r*", ms=15, label="Strategy F coarse (cp=1200, tau=8)")
    ax.plot(1800.0, 5.0, "b*", ms=15, label="alternative (cp=1800, tau=5)")
    i_g = int(sub["RMSE_72C_C"].idxmin())
    g = sub.loc[i_g]
    ax.plot(g["cp_eff_J_kgK"], g["tau_lag_s"], "k*", ms=15,
            label=f"local min at k=0.055 (cp={g['cp_eff_J_kgK']:.0f}, "
                  f"tau={g['tau_lag_s']:.1f})")
    ax.set_xlabel("cp_eff [J/(kg·K)]")
    ax.set_ylabel("tau_lag [s]")
    ax.set_title("Strategy G — RMSE vs (cp, tau) at k = 0.055")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "cp_tau_rmse_at_k055.png", dpi=200)
    plt.close(fig)
    return g


def plot_best_tau_vs_cp_at_k055(df, output_dir):
    """k=0.055: 每个 cp 的最佳 tau / RMSE / alpha。"""
    sub = df[df["k_eff_W_mK"] == 0.055]
    rows = []
    for cp, s in sub.groupby("cp_eff_J_kgK"):
        i = int(s["RMSE_72C_C"].idxmin())
        best = s.loc[i]
        rows.append({
            "cp_eff_J_kgK": float(cp),
            "best_tau_s": float(best["tau_lag_s"]),
            "best_RMSE_C": float(best["RMSE_72C_C"]),
            "alpha_eff_m2_s": float(best["alpha_eff_m2_s"]),
        })
    b = pd.DataFrame(rows).sort_values("cp_eff_J_kgK")
    b.to_csv(output_dir / "best_tau_vs_cp_at_k055.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(b["cp_eff_J_kgK"], b["best_tau_s"], "o-", color="#1f77b4",
             label="best tau (s)")
    ax1.set_xlabel("cp_eff [J/(kg·K)]")
    ax1.set_ylabel("best tau_lag [s]", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(b["cp_eff_J_kgK"], b["best_RMSE_C"], "s--", color="#d62728",
             label="best RMSE (C)")
    ax2.set_ylabel("best RMSE [°C]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Strategy G — best tau and best RMSE vs cp at k = 0.055")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(output_dir / "best_tau_vs_cp_at_k055.png", dpi=200)
    plt.close(fig)
    return b


def plot_rmse_vs_tau_at_k055(df, output_dir):
    """k=0.055: RMSE vs tau for cp = 1200/1400/1600/1800。"""
    sub = df[df["k_eff_W_mK"] == 0.055]
    fig, ax = plt.subplots(figsize=(10, 6))
    for cp in (1200.0, 1400.0, 1600.0, 1800.0):
        s = sub[sub["cp_eff_J_kgK"] == cp].sort_values("tau_lag_s")
        ax.plot(s["tau_lag_s"], s["RMSE_72C_C"], marker="o",
                label=f"cp = {cp:.0f}")
    ax.axhline(1.0, color="grey", ls="-.", alpha=0.7, label="RMSE = 1.0 C")
    ax.set_xlabel("tau_lag [s]")
    ax.set_ylabel("RMSE [°C]")
    ax.set_title("Strategy G — RMSE vs tau at k = 0.055 (cp = 1200-1800)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_vs_tau_at_k055.png", dpi=200)
    plt.close(fig)


def _pred_for_row(row, t_proto, t_int, t_env):
    ta, tf, _ = run_cr_fdm_cached(row["k_eff_W_mK"], row["cp_eff_J_kgK"],
                                  t_proto, t_int, t_env)
    return lagged_top_prediction(t_proto, ta, tf, row["tau_lag_s"])


def plot_trace_comparison(t_proto, t_int, t_top_meas, t_env, g_f, g_g, mod,
                          output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_proto, t_top_meas, color="black", lw=1.6,
            label="Measured Top COC")
    pred_f = _pred_for_row(g_f, t_proto, t_int, t_env)
    ax.plot(t_proto, pred_f, color="#1f77b4", lw=1.3, ls="--",
            label=f"Strategy F coarse (k={g_f['k_eff_W_mK']:.3f}, "
                  f"cp={g_f['cp_eff_J_kgK']:.0f}, tau={g_f['tau_lag_s']:.1f})")
    pred_g = _pred_for_row(g_g, t_proto, t_int, t_env)
    ax.plot(t_proto, pred_g, color="#2ca02c", lw=1.6,
            label=f"Strategy G local opt (k={g_g['k_eff_W_mK']:.4f}, "
                  f"cp={g_g['cp_eff_J_kgK']:.0f}, tau={g_g['tau_lag_s']:.1f})")
    if mod is not None:
        pred_m = _pred_for_row(mod, t_proto, t_int, t_env)
        ax.plot(t_proto, pred_m, color="#d62728", lw=1.3, ls=":",
                label=f"Moderate-lag (k={mod['k_eff_W_mK']:.4f}, "
                      f"cp={mod['cp_eff_J_kgK']:.0f}, "
                      f"tau={mod['tau_lag_s']:.1f})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("72C Top COC — Strategy G refined vs Strategy F coarse")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "72C_trace_comparison_strategy_G.png", dpi=200)
    plt.close(fig)


def plot_raw_vs_lagged(g, t_proto, t_int, t_top_meas, t_env, output_dir):
    ta, tf, _ = run_cr_fdm_cached(g["k_eff_W_mK"], g["cp_eff_J_kgK"],
                                  t_proto, t_int, t_env)
    pred = lagged_top_prediction(t_proto, ta, tf, g["tau_lag_s"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_proto, np.interp(t_proto, ta, tf), color="#1f77b4", lw=1.3,
            label="Raw T_top_FDM (no lag)")
    ax.plot(t_proto, pred, color="#2ca02c", lw=1.5,
            label=f"Lagged T_top_obs (tau={g['tau_lag_s']:.1f} s)")
    ax.plot(t_proto, np.asarray(t_top_meas, dtype=float), color="black",
            lw=1.2, alpha=0.6, label="Measured Top COC")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Strategy G local optimum — raw vs lagged Top "
                 f"(k={g['k_eff_W_mK']:.4f}, cp={g['cp_eff_J_kgK']:.0f})")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_G_raw_vs_lagged_top.png", dpi=200)
    plt.close(fig)


def plot_residual_vs_time(g, t_proto, t_int, t_top_meas, t_env, output_dir):
    pred = _pred_for_row(g, t_proto, t_int, t_env)
    r = pred - np.asarray(t_top_meas, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_proto, r, color="#2ca02c", lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual (pred - meas) [°C]")
    ax.set_title("Strategy G local optimum — residual vs time")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_G_residual_vs_time.png", dpi=200)
    plt.close(fig)
    return r


def regime_metrics(g, t_proto, t_int, t_top_meas, t_env, output_dir):
    try:
        from thermal_model.utilities import classify_temperature_regimes as ctr
    except Exception:  # noqa: BLE001
        return None
    pred = _pred_for_row(g, t_proto, t_int, t_env)
    r = np.asarray(pred, dtype=float) - np.asarray(t_top_meas, dtype=float)
    df = pd.DataFrame({
        "t": t_proto,
        "T_internal_interpolated_C": t_int,
        "T_top_measured_C": t_top_meas,
    })
    try:
        d_int_raw, d_top_raw = ctr.calculate_derivatives(
            df, "t", "T_internal_interpolated_C", "T_top_measured_C")
        d_int = ctr.smooth_derivatives(d_int_raw)
        d_top = ctr.smooth_derivatives(d_top_raw)
        regimes = ctr.classify_regimes(df, d_int, d_top)
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for reg in np.unique(regimes):
        mask = regimes == reg
        rr = r[mask]
        if len(rr) == 0:
            continue
        out[str(reg)] = {
            "n_points": int(len(rr)),
            "RMSE_C": float(np.sqrt(np.mean(rr ** 2))),
            "mean_residual_C": float(np.mean(rr)),
            "MAE_C": float(np.mean(np.abs(rr))),
        }
    return out


# ============================================================
# 输出辅助
# ============================================================

def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
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
        "effusivity": float(cand["effusivity"]),
        "Rth_bottom_area_m2K_W": float(cand["Rth_bottom_area_m2K_W"]),
        "RMSE_72C_C": float(cand["RMSE_72C_C"]),
        "MAE_72C_C": float(cand["MAE_72C_C"]),
        "mean_residual_C": float(cand["mean_residual_C"]),
    }


def _format_row(r):
    return (f"    k={r['k_eff_W_mK']:.4f}, cp={r['cp_eff_J_kgK']:.0f}, "
            f"tau={r['tau_lag_s']:.2f}, alpha={r['alpha_eff_m2_s']:.3e}, "
            f"effusivity={r['effusivity']:.2f}, "
            f"Rth={r['Rth_bottom_area_m2K_W']:.5f}, "
            f"RMSE={r['RMSE_72C_C']:.4f}, MAE={r['MAE_72C_C']:.4f}, "
            f"mean={r['mean_residual_C']:+.4f}")


# ============================================================
# 主流程
# ============================================================

def load_72c():
    df = pd.read_csv(ALIGNED_CSV)
    t = df["time_s"].to_numpy(float)
    t_int = df["T_internal_interpolated_C"].to_numpy(float)
    t_top = df["T_top_measured_C"].to_numpy(float)
    return t, t_int, t_top


def main():
    t0_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_proto, t_int, t_top_meas = load_72c()

    # ---- 环境 / 初始 (动态解析) ----
    env_info = cr.infer_environment_from_initial_top_measurement(
        t_top_meas, time_s=t_proto)
    T_env = env_info["T_environment_C"]
    init_info = cr.infer_environment_from_initial_top_measurement(
        t_int, time_s=t_proto)
    T_init = init_info["T_environment_C"]
    print(f"[env] T_environment = {T_env} C (row {env_info['source_index']}, "
          f"t={env_info['source_time_s']} s); T_initial = {T_init} C")

    # ---- Strategy F 锚点回归 ----
    anchors = strategy_f_anchor_regression(t_proto, t_int, t_top_meas, T_env)
    for name in ("A", "B"):
        a = anchors[name]
        print(f"[anchor {name}] k={a['k']} cp={a['cp']} tau={a['tau']}: "
              f"previous RMSE={a['previous_RMSE']}, "
              f"reproduced={a['reproduced_RMSE']:.4f} -> "
              f"{'PASS' if a['PASS'] else 'FAIL'}")
    if not (anchors["A"]["PASS"] and anchors["B"]["PASS"]):
        raise RuntimeError(
            "Strategy F 锚点回归失败: 物理/目标不一致, 停止局部细化。")

    # ---- 局部扫描 (25 FDM + 425 组合) ----
    full_csv = OUTPUT_DIR / "local_k_cp_tau_full_scan.csv"
    if full_csv.exists():
        df = pd.read_csv(full_csv)
        print(f"[scan] 复用已有 CSV: {len(df)} 行 (跳过 25 次 FDM)")
    else:
        df = run_full_scan(t_proto, t_int, t_top_meas, T_env, OUTPUT_DIR)

    # ---- 剖面 ----
    prof = profile_best_tau(df)
    prof.to_csv(OUTPUT_DIR / "local_k_cp_profiled_best_tau.csv", index=False)

    # ---- 局部全局最小 ----
    g = local_global_minimum(df)
    g_is_k_interior = not g["k_local_boundary"]
    g_is_cp_interior = not g["cp_local_boundary"]
    g_is_tau_interior = not g["tau_local_boundary"]

    # ---- Strategy F 粗最优 (从存储 CSV 读, 保证一致) ----
    f_csv = STRATEGY_F_DIR / "k_cp_tau_full_scan.csv"
    if f_csv.exists():
        f_df = pd.read_csv(f_csv)
        g_f = f_df.loc[int(f_df["RMSE_72C_C"].idxmin())]
    else:
        g_f = pd.Series({"k_eff_W_mK": 0.055, "cp_eff_J_kgK": 1200.0,
                         "tau_lag_s": 8.0, "RMSE_72C_C": 0.9554,
                         "alpha_eff_m2_s": 4.493e-8})

    # ---- 近最优集 ----
    sets = near_optimal_sets(df, float(g["RMSE_72C_C"]))
    set_summary = {name: summarize_set(s) for name, s in sets.items()}

    # ---- 中度滞后候选 ----
    mod = moderate_lag_near_optimal_candidate(df, float(g["RMSE_72C_C"]))

    # ---- 候选表 ----
    near_rows = []
    for name, s in sets.items():
        tmp = s.copy()
        tmp["fit_set"] = name
        near_rows.append(tmp)
    near_all = pd.concat(near_rows, ignore_index=True) if near_rows \
        else pd.DataFrame()
    near_all.to_csv(OUTPUT_DIR / "near_optimal_local_candidates.csv",
                    index=False)

    cand_rows = []
    def _row(name, cand, delta=False):
        if cand is None:
            return
        r = dict(cand)
        r["candidate_name"] = name
        if delta:
            r["Delta_RMSE_C"] = float(r["RMSE_72C_C"]) - float(
                g["RMSE_72C_C"])
        cand_rows.append(r)
    _row("LOCAL_GLOBAL_MINIMUM", g)
    _row("STRATEGY_F_COARSE_OPTIMUM", g_f, delta=True)
    _row("MODERATE_LAG_NEAR_OPTIMAL_CANDIDATE", mod, delta=True)
    cand_df = pd.DataFrame(cand_rows) if cand_rows else pd.DataFrame()
    cand_df.to_csv(OUTPUT_DIR / "refined_candidate_summary.csv", index=False)

    # ---- 图 ----
    best_vs_k = plot_best_rmse_vs_k(prof, OUTPUT_DIR)
    g_at_055 = plot_cp_tau_rmse_at_k055(df, OUTPUT_DIR)
    plot_best_tau_vs_cp_at_k055(df, OUTPUT_DIR)
    plot_rmse_vs_tau_at_k055(df, OUTPUT_DIR)
    plot_trace_comparison(t_proto, t_int, t_top_meas, T_env, g_f, g, mod,
                          OUTPUT_DIR)
    plot_raw_vs_lagged(g, t_proto, t_int, t_top_meas, T_env, OUTPUT_DIR)
    plot_residual_vs_time(g, t_proto, t_int, t_top_meas, T_env, OUTPUT_DIR)
    regime_out = regime_metrics(g, t_proto, t_int, t_top_meas, T_env,
                                OUTPUT_DIR)

    # ---- 局部可辨识性 ----
    ident = local_identifiability(
        g["k_eff_W_mK"], g["cp_eff_J_kgK"], g["tau_lag_s"],
        t_proto, t_int, t_top_meas, T_env)
    with open(OUTPUT_DIR / "local_identifiability_diagnostics.json",
              "w", encoding="utf-8") as f:
        json.dump(ident, f, indent=2)

    # ---- tau 解释 ----
    g_sub = df[(df["k_eff_W_mK"] == g["k_eff_W_mK"]) &
               (df["cp_eff_J_kgK"] == g["cp_eff_J_kgK"])]
    def _rmse_at(tau):
        s = g_sub[g_sub["tau_lag_s"] == tau]
        return float(s["RMSE_72C_C"].iloc[0]) if len(s) else np.nan
    rmse_tau5 = _rmse_at(5.0)
    rmse_tau8 = _rmse_at(8.0)
    rmse_tau12 = _rmse_at(12.0)
    tau_best = float(g["tau_lag_s"])
    if not g_is_tau_interior and tau_best >= 10.0 and \
            rmse_tau5 - float(g["RMSE_72C_C"]) > 0.5:
        tau_label = "LARGE_LAG_DEPENDENCE"
    elif tau_best <= 8.5 and g_is_tau_interior:
        tau_label = "TAU_OPTIMUM_RESOLVED"
    elif not g_is_tau_interior:
        tau_label = "TAU_NOT_RESOLVED"
    else:
        tau_label = "INTERIOR (see summary)"

    # ---- 参数解释分类 ----
    dk = float(g["k_eff_W_mK"]) - float(g_f["k_eff_W_mK"])
    dcp = float(g["cp_eff_J_kgK"]) - float(g_f["cp_eff_J_kgK"])
    dtau = float(g["tau_lag_s"]) - float(g_f["tau_lag_s"])
    drmse = float(g["RMSE_72C_C"]) - float(g_f["RMSE_72C_C"])
    if (not g_is_k_interior) or (not g_is_cp_interior) or \
            (not g_is_tau_interior):
        interp_label = "BOUNDARY-LIMITED"
    elif K_STABLE_LO <= float(g["k_eff_W_mK"]) <= K_STABLE_HI and \
            abs(drmse) < 0.05:
        interp_label = "STABLE"
    else:
        interp_label = "SHIFTED"

    # ---- cp/tau 权衡趋势 (k=0.055) ----
    bt = pd.read_csv(OUTPUT_DIR / "best_tau_vs_cp_at_k055.csv")
    if len(bt) >= 2:
        taus = bt["best_tau_s"].to_numpy(float)
        cps = bt["cp_eff_J_kgK"].to_numpy(float)
        slope = float(np.polyfit(cps, taus, 1)[0])
        if slope < -0.0005:
            trend = f"higher cp -> lower tau (slope {slope:.2e} s per J/kgK)"
        elif slope > 0.0005:
            trend = f"higher cp -> higher tau (slope {slope:.2e})"
        else:
            trend = f"flat (slope {slope:.2e})"
    else:
        trend = "insufficient data"

    # ---- 元数据 ----
    metadata = {
        "strategy": "convection_radiation_k_cp_tau_local_refinement_v1",
        "status": "EXPERIMENTAL / TARGETED LOCAL REFINEMENT",
        "accepted_as_nominal": False,
        "physics": {
            "h_conv_W_m2K": cr.H_CONV_STRATEGY_E_W_M2K,
            "emissivity": cr.EMISSIVITY_STRATEGY_E,
            "sigma_SB_W_m2K4": cr.SIGMA_SB_W_M2_K4,
            "view_factor": cr.VIEW_FACTOR_STRATEGY_E,
            "radiation": "full nonlinear Stefan-Boltzmann",
            "environment_C": T_env,
            "initial_C": T_init,
        },
        "grid": {
            "k_eff_W_mK": K_GRID,
            "cp_eff_J_kgK": CP_GRID,
            "tau_lag_s": TAU_GRID,
            "unique_fdm_runs": 25,
            "total_combinations": 425,
        },
        "anchor_regression": anchors,
        "local_global_minimum": _cand_dict(g),
        "k_interior": bool(g_is_k_interior),
        "cp_interior": bool(g_is_cp_interior),
        "tau_interior": bool(g_is_tau_interior),
        "k_boundary_warning": bool(g["k_local_boundary"]),
        "cp_boundary_warning": bool(g["cp_local_boundary"]),
        "tau_boundary_warning": bool(g["tau_local_boundary"]),
        "strategy_F_coarse_optimum": _cand_dict(g_f),
        "delta_vs_F": {
            "delta_k": dk, "delta_cp": dcp, "delta_tau": dtau,
            "delta_RMSE": drmse,
            "delta_alpha": float(g["alpha_eff_m2_s"]) - float(
                g_f["alpha_eff_m2_s"]),
        },
        "parameter_interpretation": interp_label,
        "tau_interpretation": tau_label,
        "rmse_at_tau5_for_opt_k_cp": rmse_tau5,
        "rmse_at_tau8_for_opt_k_cp": rmse_tau8,
        "rmse_at_tau12_for_opt_k_cp": rmse_tau12,
        "cp_tau_trend_at_k055": trend,
        "near_optimal": set_summary,
        "moderate_lag_near_optimal_candidate": _cand_dict(mod),
        "fit_penalty_moderate_lag": (None if mod is None else float(
            mod["RMSE_72C_C"]) - float(g["RMSE_72C_C"])),
        "regime_residual_metrics": regime_out,
        "local_identifiability": ident,
        "doe11_used": False,
        "longer_holding_pcr_used": False,
        "sample_used_for_selection": False,
        "continuous_optimizer_used": False,
        "high_k_reward_used": False,
        "high_alpha_reward_used": False,
        "second_refinement_automatic": False,
        "git_head": _git_head(),
        "elapsed_s": time.perf_counter() - t0_total,
    }
    with open(OUTPUT_DIR / "strategy_G_metadata.json", "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- 汇总 ----
    summary = _build_summary(df, prof, best_vs_k, g, g_f, mod, sets, ident,
                             anchors, regime_out, interp_label, tau_label,
                             rmse_tau5, rmse_tau8, rmse_tau12, trend,
                             dk, dcp, dtau, drmse, bt, metadata,
                             g_at_055)
    (OUTPUT_DIR / "strategy_G_summary.txt").write_text(summary,
                                                       encoding="utf-8")
    print(summary)
    print(f"[done] total elapsed {time.perf_counter() - t0_total:.1f} s")
    return metadata


def _build_summary(df, prof, best_vs_k, g, g_f, mod, sets, ident, anchors,
                   regime_out, interp_label, tau_label, rmse_tau5,
                   rmse_tau8, rmse_tau12, trend, dk, dcp, dtau, drmse, bt,
                   metadata, g_at_055):
    lines = []
    A = lines.append
    A("=" * 70)
    A("STRATEGY G — TARGETED LOCAL k/cp/tau REFINEMENT (72C)")
    A("=" * 70)
    A("")
    A(f"环境: {metadata['physics']['environment_C']} C (第一个有效实测 Top COC); "
      f"初始: {metadata['physics']['initial_C']} C")
    A(f"锚点回归: A (0.055/1200/8): {anchors['A']['reproduced_RMSE']:.4f} "
      f"(ref {anchors['A']['previous_RMSE']}) "
      f"{'PASS' if anchors['A']['PASS'] else 'FAIL'} | "
      f"B (0.055/1800/5): {anchors['B']['reproduced_RMSE']:.4f} "
      f"(ref {anchors['B']['previous_RMSE']}) "
      f"{'PASS' if anchors['B']['PASS'] else 'FAIL'}")
    A(f"扫描: {len(df)} 行 (25 FDM + 17 tau), 失败 "
      f"{(df['status'] == 'FAILED').sum()}")
    A("")
    A("LOCAL GLOBAL MINIMUM:")
    A("  " + _format_row(g))
    A(f"  k interior: {'YES' if not g['k_local_boundary'] else 'NO'} | "
      f"cp interior: {'YES' if not g['cp_local_boundary'] else 'NO'} | "
      f"tau interior: {'YES' if not g['tau_local_boundary'] else 'NO'}")
    A(f"  k 边界警告: {'YES' if g['k_local_boundary'] else 'NO'} | "
      f"cp 边界警告: {'YES' if g['cp_local_boundary'] else 'NO'} | "
      f"tau 边界警告: {'YES' if g['tau_local_boundary'] else 'NO'}")
    A("")
    A(f"vs Strategy F 粗最优 (k={g_f['k_eff_W_mK']}, "
      f"cp={g_f['cp_eff_J_kgK']}, tau={g_f['tau_lag_s']}, "
      f"RMSE={g_f['RMSE_72C_C']}):")
    A(f"  Delta k = {dk:+.4f}, Delta cp = {dcp:+.0f}, "
      f"Delta tau = {dtau:+.1f}, Delta RMSE = {drmse:+.4f}")
    A(f"  参数解释: {interp_label}")
    A("")
    A("Best RMSE vs k (profile over cp, tau):")
    for _, r in best_vs_k.iterrows():
        A(f"  k={r['k_eff_W_mK']:.4f}: cp={r['best_cp_eff_J_kgK']:.0f}, "
          f"tau={r['best_tau_s']:.1f}, alpha={r['best_alpha_eff_m2_s']:.3e}, "
          f"RMSE={r['best_RMSE_C']:.4f}")
    A("")
    A("cp/tau trade-off at k=0.055:")
    for _, r in bt.iterrows():
        A(f"  cp={r['cp_eff_J_kgK']:.0f}: best tau={r['best_tau_s']:.1f} s, "
          f"best RMSE={r['best_RMSE_C']:.4f}, "
          f"alpha={r['alpha_eff_m2_s']:.3e}")
    A(f"  趋势: {trend}")
    A("")
    A(f"tau 解释: {tau_label}")
    A(f"  最优 k/cp 下 RMSE: tau=5 -> {rmse_tau5:.4f}, "
      f"tau=8 -> {rmse_tau8:.4f}, tau=12 -> {rmse_tau12:.4f}, "
      f"tau_best={g['tau_lag_s']:.1f} -> {g['RMSE_72C_C']:.4f}")
    A("")
    A("Near-optimal regions:")
    for name, s in sets.items():
        ss = summarize_set(s)
        if ss is None:
            A(f"  {name}: empty")
            continue
        A(f"  {name}: n={ss['n_total']}, "
          f"k [{ss['k_range'][0]:.4f}, {ss['k_range'][1]:.4f}], "
          f"cp [{ss['cp_range'][0]:.0f}, {ss['cp_range'][1]:.0f}], "
          f"tau [{ss['tau_range'][0]:.1f}, {ss['tau_range'][1]:.1f}], "
          f"alpha [{ss['alpha_range'][0]:.2e}, "
          f"{ss['alpha_range'][1]:.2e}]")
    A("")
    A("MODERATE_LAG_NEAR_OPTIMAL_CANDIDATE "
      "(Delta<=0.20, tau<=6, min RMSE):")
    if mod is not None:
        A("  " + _format_row(mod))
        A(f"  Delta RMSE = {float(mod['RMSE_72C_C']) - float(g['RMSE_72C_C']):+.4f}")
    else:
        A("  NONE")
    A("")
    A("Local identifiability:")
    A(f"  singular values: {[f'{v:.4e}' for v in ident['singular_values']]}")
    A(f"  condition number: {ident['condition_number']:.3f}")
    A(f"  corr log(k)-log(cp): {ident['correlation_logk_logcp']:.4f}")
    A(f"  corr log(k)-tau: {ident['correlation_logk_tau']:.4f}")
    A(f"  corr log(cp)-tau: {ident['correlation_logcp_tau']:.4f}")
    A(f"  warning: {'YES' if ident['identifiability_warning'] else 'NO'}")
    if regime_out:
        A("")
        A("Regime residual diagnostics (local opt, diagnostic only):")
        for reg_name, mm in regime_out.items():
            A(f"  {reg_name}: n={mm['n_points']}, RMSE={mm['RMSE_C']:.3f}, "
              f"mean={mm['mean_residual_C']:+.3f}, MAE={mm['MAE_C']:.3f}")
    A("")
    A("Scientific conclusion:")
    A(f"  局部细化全局最小: k={g['k_eff_W_mK']:.4f}, "
      f"cp={g['cp_eff_J_kgK']:.0f}, tau={g['tau_lag_s']:.1f} s, "
      f"RMSE={g['RMSE_72C_C']:.4f} C")
    A(f"  参数解释: {interp_label}; tau 解释: {tau_label}")
    A("")
    A("无 DOE11 / 长保持 PCR / 样品选参 / 连续优化 / 高-k 奖励 / "
      "自动二次细化。物理未修改。Strategy F 输出未覆盖。")
    A("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
