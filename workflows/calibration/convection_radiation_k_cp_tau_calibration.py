#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy F — convection + radiation (k_eff, cp_eff, tau_lag) 首次重标定
=====================================================================

科学问题:
    在显式表示更强的 Top COC 热损失 (h_conv=10 + 非线性辐射) 之后,
    实验偏好的 k_eff 是否自然上升到远高于旧值:
        k_eff = 0.0165 W/(m K)
    同时 cp_eff 保持在合理/非极端范围, 且 Top COC 拟合恢复到
    RMSE <= 1 C 左右?

不强迫答案; 让数据决定。

直接参数 (扫描):
    k_eff   [W/(m K)]    9 值
    cp_eff  [J/(kg K)]   7 值
    tau_lag [s]          8 值

固定 (Strategy E 物理, 本任务不修改):
    h_conv = 10.0 W/(m2 K)
    epsilon_surface = 0.90
    sigma_SB = 5.670374419e-8 W/(m2 K4)
    F_view = 1.0
    非线性 Stefan-Boltzmann 辐射 (求解器内不线性化)
    rho_COC = 1020 kg/m3

环境温度规则:
    T_air = T_surroundings =
        第一个有效实测 Top COC 温度 (动态解析, 不硬编码)
    仿真内恒定; 全迹线不作为时变环境。

初始条件:
    T_initial = 第一个有效内部温度 (允许 != 环境温度)。

底部边界:
    实测内部温度迹线 = 时变底部 Dirichlet 边界 (不变)。

滞后:
    既有输出侧一阶滞后 (复用 lag_augmented_thermal_model.apply_first_order_lag;
    tau=0 严格恒等; 只作用于 T_top_observed_predicted)。

目标 (无任何高-k 奖励):
    主目标 = 72C 修正测量时间 RMSE (等权, 查询轴 = 实测时间)。

计算效率:
    每个唯一 (k,cp) 只跑一次非线性 Strategy E FDM (63 次);
    8 个 tau 复用同一 T_top_FDM 迹线做滞后剖面 (63*8 = 504 组合)。

本任务: 无 DOE11 / 长保持 PCR / 连续优化器 / 自动细扫。
输出目录: parameter_scan_output/72C/convection_radiation_k_cp_tau_calibration_v1/
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
STRATEGY_E_CHECK_DIR = (
    PROJECT_ROOT / "model_comparison_output"
    / "convection_radiation_lag_v1")
OUTPUT_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "convection_radiation_k_cp_tau_calibration_v1")

# ---------------------------------------------------------------
# 网格 (任务指定, 精确)
# ---------------------------------------------------------------
K_GRID = [0.0165, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.090, 0.120]
CP_GRID = [700.0, 800.0, 900.0, 1000.0, 1200.0, 1500.0, 1800.0]
TAU_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

NON_EXTREME_CP_MIN = 800.0     # NON_EXTREME_CP_SUBSET: cp >= 800
TAU_UPPER_BOUND = 8.0          # tau == 8 s -> TAU_UPPER_BOUND_WARNING
BALANCED_TAU_MAX = 5.0         # 平衡候选: tau <= 5 s

# 历史策略 A 参考
K_A = 0.0165
CP_A = 900.0
RHO_COC = 1020.0
ALPHA_A = K_A / (RHO_COC * CP_A)   # ~1.797e-8

BANDS = {"STRICT": 0.05, "MODERATE": 0.10, "APPLICATION": 0.20}
RMSE_LE_1C = 1.0
BALANCED_RMSE = 1.0
BALANCED_RMSE_RELAXED = 1.2

L_BOTTOM_M = 180e-6
SAVE_DT = 1.0

# k 位移分类 (描述性项目分类, 非科学标准)
SUBSTANTIAL_FACTOR = 2.0
MODEST_FACTOR_LOW = 1.25


# ============================================================
# 派生量
# ============================================================

def alpha_from_k_cp(k, cp, rho=RHO_COC):
    return float(k) / (rho * float(cp))


def effusivity_from_k_cp(k, cp, rho=RHO_COC):
    return float(np.sqrt(float(k) * rho * float(cp)))


def rth_area_bottom(k):
    return L_BOTTOM_M / float(k)


def ratio_to_A(value, base):
    return float(value) / float(base)


# ============================================================
# FDM-once-per-(k,cp) 缓存 (Strategy E 非线性求解器)
# ============================================================

_fdm_cache = {}


def run_cr_fdm_cached(k, cp, t_proto, t_int, t_env, save_dt=SAVE_DT):
    """每个 (k,cp) 只跑一次 Strategy E 非线性 FDM。

    返回 (t_arr, T_top_FDM, T_sample_FDM)。
    环境 T_air = T_surroundings = t_env (恒定标量)。
    初始场 = 第一个内部温度。
    """
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
    """施加一阶滞后并插值到实测时间 (绝不用温度值作查询坐标)。"""
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
    """单点评估: 复用缓存的 (k,cp) FDM + tau 剖面。"""
    t_arr, t_top_fdm, _ = run_cr_fdm_cached(k, cp, t_proto, t_int, t_env,
                                            save_dt)
    pred = lagged_top_prediction(t_proto, t_arr, t_top_fdm, tau)
    m = metrics_for_prediction(pred, t_top_meas)
    alpha = alpha_from_k_cp(k, cp)
    return {
        "k_eff_W_mK": float(k),
        "cp_eff_J_kgK": float(cp),
        "tau_lag_s": float(tau),
        "alpha_eff_m2_s": alpha,
        "effusivity": effusivity_from_k_cp(k, cp),
        "Rth_bottom_area_m2K_W": rth_area_bottom(k),
        "k_ratio_to_Strategy_A": ratio_to_A(k, K_A),
        "cp_ratio_to_Strategy_A": ratio_to_A(cp, CP_A),
        "alpha_ratio_to_Strategy_A": ratio_to_A(alpha, ALPHA_A),
        "effusivity_ratio_to_Strategy_A": ratio_to_A(
            effusivity_from_k_cp(k, cp), effusivity_from_k_cp(K_A, CP_A)),
        "RMSE_72C_C": m["RMSE_72C_C"],
        "MAE_72C_C": m["MAE_72C_C"],
        "median_abs_residual_C": m["median_abs_residual_C"],
        "mean_residual_C": m["mean_residual_C"],
        "max_abs_residual_C": m["max_abs_residual_C"],
        "cp_non_extreme": bool(cp >= NON_EXTREME_CP_MIN),
        "tau_upper_bound_warning": bool(tau >= TAU_UPPER_BOUND - 1e-12),
        "status": "OK",
    }


def run_full_scan(t_proto, t_int, t_top_meas, t_env, output_dir):
    """63 次 Strategy E FDM + 8 tau 剖面 = 504 行。"""
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
                                              t_top_meas, t_env))
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[scan] unique FDM runs: {n_fdm} (expected 63); "
          f"rows: {len(df)} (expected 504); "
          f"failures: {(df['status'] == 'FAILED').sum()}; "
          f"elapsed {time.perf_counter() - t0:.1f} s")
    return df


# ============================================================
# Strategy E 回归检查 (扫描前)
# ============================================================

def strategy_e_regression_check(t_proto, t_int, t_top_meas, t_env):
    """k=0.0165, cp=900, tau=0 应复现上一任务 Strategy E 检查结果。

    存储参考 (model_check_metadata.json):
        RMSE ~10.63 C, top max ~59.1 C, sample max ~84.8 C。
    返回 dict (含 PASS/FAIL)。
    """
    t_arr, t_top_fdm, t_sample_fdm = run_cr_fdm_cached(
        K_A, CP_A, t_proto, t_int, t_env)
    pred = lagged_top_prediction(t_proto, t_arr, t_top_fdm, 0.0)
    m = metrics_for_prediction(pred, t_top_meas)
    out = {
        "k": K_A, "cp": CP_A, "tau": 0.0,
        "RMSE_C": m["RMSE_72C_C"],
        "top_max_C": float(np.max(t_top_fdm)),
        "sample_max_C": float(np.max(t_sample_fdm)),
    }
    # 存储参考
    ref = {}
    meta_path = STRATEGY_E_CHECK_DIR / "model_check_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ref = {
                "RMSE_C": meta["reference_case_A"]["new"][
                    "RMSE_lagged_top_C"],
                "top_max_C": meta["reference_case_A"]["new"]["top_max_C"],
                "sample_max_C": meta["reference_case_A"]["new"][
                    "sample_max_C"],
            }
        except Exception:  # noqa: BLE001
            ref = {}
    else:
        ref = {"RMSE_C": 10.63, "top_max_C": 59.1, "sample_max_C": 84.8}

    ok_rmse = abs(out["RMSE_C"] - ref["RMSE_C"]) < 0.05
    ok_top = abs(out["top_max_C"] - ref["top_max_C"]) < 0.1
    ok_sample = abs(out["sample_max_C"] - ref["sample_max_C"]) < 0.1
    out["reference"] = ref
    out["PASS"] = bool(ok_rmse and ok_top and ok_sample)
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
        tau0 = sub[sub["tau_lag_s"] == 0.0]
        rmse_tau0 = float(tau0["RMSE_72C_C"].iloc[0]) if len(tau0) else np.nan
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
            "RMSE_tau0_C": rmse_tau0,
            "RMSE_improvement_due_to_lag_C": rmse_tau0 - float(
                best["RMSE_72C_C"]),
            "cp_non_extreme": bool(cp >= NON_EXTREME_CP_MIN),
            "tau_upper_bound_warning": bool(
                best["tau_lag_s"] >= TAU_UPPER_BOUND - 1e-12),
        })
    return pd.DataFrame(rows)


def global_rmse_minimum(df):
    i = int(df["RMSE_72C_C"].idxmin())
    return df.loc[i]


def near_optimal_sets(df, rmse_min):
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


def balanced_physical_candidate(df, rmse_threshold=BALANCED_RMSE):
    """规则: RMSE<=th, cp>=800, tau<=5 s, 选最高 k。无则 None。"""
    sub = df[(df["RMSE_72C_C"] <= rmse_threshold) &
             (df["cp_eff_J_kgK"] >= NON_EXTREME_CP_MIN) &
             (df["tau_lag_s"] <= BALANCED_TAU_MAX + 1e-12)]
    if sub.empty:
        return None
    i = int(sub["k_eff_W_mK"].idxmax())
    return sub.loc[i]


def summarize_set(near_set):
    if near_set.empty:
        return None
    return {
        "n_total": int(len(near_set)),
        "n_cp_ge_800": int((near_set["cp_eff_J_kgK"]
                            >= NON_EXTREME_CP_MIN).sum()),
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
    """对每个 k 在 cp/tau 上 profile, 返回 best row。"""
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


def classify_k_shift(k_global):
    """描述性分类: SUBSTANTIAL / MODEST / NO MATERIAL。"""
    if k_global >= SUBSTANTIAL_FACTOR * K_A:
        return "SUBSTANTIAL"
    if k_global >= MODEST_FACTOR_LOW * K_A:
        return "MODEST"
    return "NO MATERIAL SHIFT"


# ============================================================
# 局部灵敏度 / 可辨识性 (log k, log cp, tau)
# ============================================================

def local_identifiability(k, cp, tau, t_proto, t_int, t_top_meas, t_env,
                          rel_pert=0.02, tau_pert=0.5, save_dt=SAVE_DT):
    """在 (log k, log cp, tau) 上做局部数值灵敏度 (诊断用)。"""
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
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            pairs.append((i, j, float(corr[i, j])))
    strongest = max(pairs, key=lambda x: abs(x[2]))
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
                    contour=None, annotate_upper_tau=False):
    ks = np.sort(prof["k_eff_W_mK"].unique())
    cps = np.sort(prof["cp_eff_J_kgK"].unique())
    Z = np.full((len(ks), len(cps)), np.nan)
    upper = np.zeros((len(ks), len(cps)), dtype=bool)
    for _, r in prof.iterrows():
        ki = int(np.searchsorted(ks, r["k_eff_W_mK"]))
        ci = int(np.searchsorted(cps, r["cp_eff_J_kgK"]))
        Z[ki, ci] = r[value_col]
        if annotate_upper_tau and "tau_upper_bound_warning" in r:
            upper[ki, ci] = bool(r["tau_upper_bound_warning"])
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
        ax.contour(cps, ks, Z, levels=[contour], colors="white",
                   linestyles="--", linewidths=1.2)
    if annotate_upper_tau:
        yx = np.argwhere(upper)
        for yi, xi in yx:
            ax.plot(cps[xi], ks[yi], "r+", ms=12, mew=2,
                    label="tau_best=8s" if yi == 0 and xi == 0 else None)
    ax.set_xlabel("cp_eff [J/(kg·K)]")
    ax.set_ylabel("k_eff [W/(m·K)]")
    ax.set_title(title)
    for (x, y, lab) in markers:
        ax.plot(x, y, "k*", ms=15)
        ax.annotate(lab, (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)
    if annotate_upper_tau and np.any(upper):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_profiled_rmse(prof, markers, output_dir):
    _grid_landscape(prof, "best_RMSE_C",
                    "Profiled RMSE (min over tau) — k vs cp",
                    output_dir / "profiled_rmse_k_cp.png",
                    vmin=0.5, vmax=4.0, markers=markers, contour=1.0)


def plot_best_tau(prof, output_dir):
    _grid_landscape(prof, "best_tau_s",
                    "Best tau_lag [s] — k vs cp (red +: tau_best = 8 s)",
                    output_dir / "best_tau_k_cp.png",
                    cmap="plasma", vmin=0.0, vmax=8.0,
                    annotate_upper_tau=True)


def plot_alpha(prof, output_dir):
    _grid_landscape(prof, "alpha_eff_m2_s",
                    "Derived alpha_eff [m²/s] — k vs cp (log color)",
                    output_dir / "alpha_k_cp.png",
                    cmap="inferno", log_color=True)


def plot_best_rmse_vs_k(prof, output_dir):
    b = best_rmse_vs_k(prof)
    b.to_csv(output_dir / "best_rmse_vs_k.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(b["k_eff_W_mK"], b["best_RMSE_C"], marker="o",
            label="min RMSE (profiled over cp and tau)")
    ax.axhline(0.7337, color="black", ls=":",
               label="Strategy A historical RMSE 0.7337 (old h=5 boundary)")
    ax.axhline(1.0, color="green", ls="-.", alpha=0.6, label="RMSE = 1.0 C")
    ax.axvline(0.0165, color="grey", ls=":", alpha=0.7,
               label="historical k = 0.0165")
    g = b.loc[b["best_RMSE_C"].idxmin()]
    ax.plot(g["k_eff_W_mK"], g["best_RMSE_C"], "r*", ms=16,
            label=f"global-min k = {g['k_eff_W_mK']:.4f}")
    ax.set_xlabel("k_eff [W/(m·K)]")
    ax.set_ylabel("Best RMSE_top [°C]")
    ax.set_title("Best achievable 72C RMSE vs k (Strategy F, profiled over "
                 "cp, tau)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "best_rmse_vs_k.png", dpi=200)
    plt.close(fig)
    return b


def _pred_for_row(row, t_proto, t_int, t_env):
    ta, tf, _ = run_cr_fdm_cached(row["k_eff_W_mK"], row["cp_eff_J_kgK"],
                                  t_proto, t_int, t_env)
    return lagged_top_prediction(t_proto, ta, tf, row["tau_lag_s"])


def plot_trace_comparison(t_proto, t_int, t_top_meas, t_env, g, bal,
                          output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_proto, t_top_meas, color="black", lw=1.6,
            label="Measured Top COC")
    # 历史策略 A (旧 h=5 边界, 无辐射)
    mats_a = cr.make_convection_radiation_materials(K_A, CP_A)
    old_a = heat_model.run_simulation(
        time_s=t_proto, bottom_temperature_C=t_int, materials=mats_a,
        layers=heat_model.BARE_TOP_COC_LAYERS, h_conv=5.0,
        T_air_ambient=t_env, save_dt=SAVE_DT, T_initial_C=float(t_int[0]))
    ax.plot(t_proto, np.interp(t_proto, old_a["t_array"],
                               old_a["T_top_surface_arr"]),
            color="#1f77b4", lw=1.2, ls=":",
            label=f"Strategy A (h=5 conv only, k={K_A}, cp={CP_A:.0f})")
    # 策略 E 旧参数 (h=10+辐射, k/cp 未重标定)
    pred_e = _pred_for_row(pd.Series({"k_eff_W_mK": K_A, "cp_eff_J_kgK": CP_A,
                                      "tau_lag_s": 0.0}),
                           t_proto, t_int, t_env)
    ax.plot(t_proto, pred_e, color="#ff7f0e", lw=1.2, ls="--",
            label=f"Strategy E old params (h=10+rad, k={K_A}, cp={CP_A:.0f}, "
                  "tau=0)")
    # 全局最小
    pred_g = _pred_for_row(g, t_proto, t_int, t_env)
    ax.plot(t_proto, pred_g, color="#2ca02c", lw=1.6, ls="-",
            label=f"Strategy F global min (k={g['k_eff_W_mK']:.4f}, "
                  f"cp={g['cp_eff_J_kgK']:.0f}, tau={g['tau_lag_s']:.2f})")
    if bal is not None:
        pred_b = _pred_for_row(bal, t_proto, t_int, t_env)
        ax.plot(t_proto, pred_b, color="#d62728", lw=1.3, ls="--",
                label=f"Balanced (k={bal['k_eff_W_mK']:.4f}, "
                      f"cp={bal['cp_eff_J_kgK']:.0f}, "
                      f"tau={bal['tau_lag_s']:.2f})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("72C Top COC — Strategy F candidates vs historical "
                 "traces\n(different Top boundaries, see labels)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "72C_trace_comparison_strategy_F.png", dpi=200)
    plt.close(fig)


def plot_raw_vs_lagged(g, t_proto, t_int, t_top_meas, t_env, output_dir):
    ta, tf, _ = run_cr_fdm_cached(g["k_eff_W_mK"], g["cp_eff_J_kgK"],
                                  t_proto, t_int, t_env)
    pred = lagged_top_prediction(t_proto, ta, tf, g["tau_lag_s"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_proto, np.interp(t_proto, ta, tf), color="#1f77b4", lw=1.3,
            label="Raw T_top_FDM (no lag)")
    ax.plot(t_proto, pred, color="#2ca02c", lw=1.5,
            label=f"Lagged T_top_obs (tau={g['tau_lag_s']:.2f} s)")
    ax.plot(t_proto, np.asarray(t_top_meas, dtype=float), color="black",
            lw=1.2, alpha=0.6, label="Measured Top COC")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Strategy F global minimum — raw vs lagged Top "
                 f"(k={g['k_eff_W_mK']:.4f}, cp={g['cp_eff_J_kgK']:.0f})")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_F_raw_vs_lagged_top.png", dpi=200)
    plt.close(fig)
    return ta, tf


def plot_residual_vs_time(g, t_proto, t_int, t_top_meas, t_env, output_dir):
    pred = _pred_for_row(g, t_proto, t_int, t_env)
    r = pred - np.asarray(t_top_meas, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_proto, r, color="#2ca02c", lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual (pred - meas) [°C]")
    ax.set_title("Strategy F global minimum — residual vs time")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_F_residual_vs_time.png", dpi=200)
    plt.close(fig)
    return r


def regime_metrics(g, t_proto, t_int, t_top_meas, t_env, output_dir):
    """复用既有 regime 分类器做诊断 (不改目标权重)。"""
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

    # ---- Strategy E 回归检查 (扫描前) ----
    reg = strategy_e_regression_check(t_proto, t_int, t_top_meas, T_env)
    print(f"[regression] Strategy E k={reg['k']} cp={reg['cp']} tau=0: "
          f"RMSE={reg['RMSE_C']:.4f} C (ref ~{reg['reference']['RMSE_C']:.2f}), "
          f"top_max={reg['top_max_C']:.3f} (ref ~{reg['reference']['top_max_C']:.1f}), "
          f"sample_max={reg['sample_max_C']:.3f} "
          f"(ref ~{reg['reference']['sample_max_C']:.1f}) -> "
          f"{'PASS' if reg['PASS'] else 'FAIL'}")
    if not reg["PASS"]:
        raise RuntimeError(
            "Strategy E 回归失败: 新边界物理与上一任务不一致, 停止。")

    # ---- 完整扫描 (63 FDM + 504 组合) ----
    # 若 full scan CSV 已存在则直接复用 (避免重新运行 23 分钟 FDM 扫描)
    full_csv = OUTPUT_DIR / "k_cp_tau_full_scan.csv"
    if full_csv.exists():
        df = pd.read_csv(full_csv)
        print(f"[scan] 复用已有 CSV: {len(df)} 行 (跳过 63 次 FDM)")
    else:
        df = run_full_scan(t_proto, t_int, t_top_meas, T_env, OUTPUT_DIR)

    # ---- 剖面 ----
    prof = profile_best_tau(df)
    prof.to_csv(OUTPUT_DIR / "k_cp_profiled_best_tau.csv", index=False)

    # ---- 全局最小 ----
    g = global_rmse_minimum(df)

    # ---- 近最优集 ----
    sets = near_optimal_sets(df, float(g["RMSE_72C_C"]))
    set_summary = {name: summarize_set(s) for name, s in sets.items()}

    # ---- 高 k 解释候选 ----
    high_k = {}
    for name in ("STRICT", "MODERATE", "APPLICATION", "RMSE_LE_1C"):
        cand = highest_k_candidate(sets[name])
        if cand is not None:
            cand["Delta_RMSE_C"] = float(cand["RMSE_72C_C"]) - float(
                g["RMSE_72C_C"])
        high_k[name] = cand

    # ---- 平衡候选 ----
    bal = balanced_physical_candidate(df, BALANCED_RMSE)
    bal_relaxed = balanced_physical_candidate(df, BALANCED_RMSE_RELAXED)

    # ---- 候选表 (near_optimal_candidates.csv + physical_candidate_summary.csv) ----
    near_rows = []
    for name, s in sets.items():
        sub = s[s["cp_eff_J_kgK"] >= NON_EXTREME_CP_MIN]
        tmp = sub.copy()
        tmp["fit_set"] = name
        near_rows.append(tmp)
    near_all = pd.concat(near_rows, ignore_index=True) if near_rows \
        else pd.DataFrame()
    near_all.to_csv(OUTPUT_DIR / "near_optimal_candidates.csv", index=False)

    phys_rows = []
    def _phys_row(name, cand, delta=False):
        if cand is None:
            return
        r = dict(cand)
        r["candidate_name"] = name
        if delta and "Delta_RMSE_C" not in r:
            r["Delta_RMSE_C"] = float(r["RMSE_72C_C"]) - float(
                g["RMSE_72C_C"])
        phys_rows.append(r)
    _phys_row("GLOBAL_RMSE_MINIMUM", g)
    _phys_row("HIGH_K_NEAR_EQUIVALENT_CANDIDATE", high_k["MODERATE"],
              delta=True)
    _phys_row("HIGH_K_APPLICATION_CANDIDATE", high_k["APPLICATION"],
              delta=True)
    _phys_row("BALANCED_PHYSICAL_CANDIDATE", bal)
    _phys_row("BALANCED_PHYSICAL_CANDIDATE_RELAXED_1_2C", bal_relaxed)
    phys_df = pd.DataFrame(phys_rows) if phys_rows else pd.DataFrame()
    phys_df.to_csv(OUTPUT_DIR / "physical_candidate_summary.csv", index=False)

    # ---- 图 ----
    markers = [
        (CP_A, K_A, "A (historical)"),
        (g["cp_eff_J_kgK"], g["k_eff_W_mK"], "global"),
    ]
    if bal is not None:
        markers.append((bal["cp_eff_J_kgK"], bal["k_eff_W_mK"], "bal"))
    if high_k["MODERATE"] is not None:
        markers.append((high_k["MODERATE"]["cp_eff_J_kgK"],
                        high_k["MODERATE"]["k_eff_W_mK"], "hk-ne"))
    plot_profiled_rmse(prof, markers, OUTPUT_DIR)
    plot_best_tau(prof, OUTPUT_DIR)
    plot_alpha(prof, OUTPUT_DIR)
    best_vs_k = plot_best_rmse_vs_k(prof, OUTPUT_DIR)
    plot_trace_comparison(t_proto, t_int, t_top_meas, T_env, g, bal,
                          OUTPUT_DIR)
    plot_raw_vs_lagged(g, t_proto, t_int, t_top_meas, T_env, OUTPUT_DIR)
    r_global = plot_residual_vs_time(g, t_proto, t_int, t_top_meas, T_env,
                                     OUTPUT_DIR)

    # ---- regime 残差诊断 ----
    regime_metrics_out = regime_metrics(g, t_proto, t_int, t_top_meas, T_env,
                                        OUTPUT_DIR)

    # ---- 局部可辨识性 ----
    ident = local_identifiability(
        g["k_eff_W_mK"], g["cp_eff_J_kgK"], g["tau_lag_s"],
        t_proto, t_int, t_top_meas, T_env)
    with open(OUTPUT_DIR / "local_identifiability_diagnostics.json",
              "w", encoding="utf-8") as f:
        json.dump(ident, f, indent=2)

    # ---- 分类 ----
    k_shift = classify_k_shift(float(g["k_eff_W_mK"]))
    alpha_ratio_g = float(g["alpha_eff_m2_s"]) / ALPHA_A
    k_ratio_g = float(g["k_eff_W_mK"]) / K_A
    # 24. 扩散率提升是否主要由 k 驱动 (描述性)
    k_driven = bool(k_ratio_g > alpha_ratio_g * 0.6) or \
        bool(float(g["cp_eff_J_kgK"]) / CP_A > 0.6)
    if k_driven:
        k_driven_label = "YES (k 主导)"
    elif k_ratio_g >= 1.0 and float(g["cp_eff_J_kgK"]) / CP_A < 0.6:
        k_driven_label = "PARTIALLY (k 上升但 cp 也明显下降)"
    else:
        k_driven_label = "NO"

    # ---- 元数据 ----
    metadata = {
        "strategy": "convection_radiation_k_cp_tau_calibration_v1",
        "status": "EXPERIMENTAL / FIRST RECALIBRATION UNDER NEW BOUNDARY",
        "accepted_as_nominal": False,
        "physics": {
            "h_conv_W_m2K": cr.H_CONV_STRATEGY_E_W_M2K,
            "emissivity": cr.EMISSIVITY_STRATEGY_E,
            "sigma_SB_W_m2K4": cr.SIGMA_SB_W_M2_K4,
            "view_factor": cr.VIEW_FACTOR_STRATEGY_E,
            "radiation": "full nonlinear Stefan-Boltzmann (solver, not "
                         "linearized)",
            "environment_C": T_env,
            "environment_source_index": int(env_info["source_index"]),
            "environment_source_time_s": env_info["source_time_s"],
            "initial_C": T_init,
            "lag": "existing output-side first-order lag",
        },
        "grid": {
            "k_eff_W_mK": K_GRID,
            "cp_eff_J_kgK": CP_GRID,
            "tau_lag_s": TAU_GRID,
            "unique_fdm_runs": 63,
            "total_combinations": 504,
        },
        "strategy_E_regression_before_scan": reg,
        "global_rmse_minimum": _cand_dict(g),
        "k_shift_classification": k_shift,
        "k_ratio_to_A": k_ratio_g,
        "alpha_ratio_to_A": alpha_ratio_g,
        "cp_ratio_to_A": float(g["cp_eff_J_kgK"]) / CP_A,
        "diffusivity_increase_k_driven": k_driven_label,
        "near_optimal": set_summary,
        "high_k_candidates": {
            "HIGH_K_NEAR_EQUIVALENT_CANDIDATE": _cand_dict(
                high_k["MODERATE"]),
            "HIGH_K_APPLICATION_CANDIDATE": _cand_dict(
                high_k["APPLICATION"]),
        },
        "balanced_physical_candidate": _cand_dict(bal),
        "balanced_physical_candidate_relaxed_1_2C": _cand_dict(bal_relaxed),
        "regime_residual_metrics": regime_metrics_out,
        "local_identifiability": ident,
        "parameter_grid_boundary_warning": bool(
            g["tau_lag_s"] >= TAU_UPPER_BOUND - 1e-12),
        "doe11_used": False,
        "longer_holding_pcr_used": False,
        "sample_used_for_selection": False,
        "fine_scan_performed": False,
        "continuous_optimizer_used": False,
        "high_k_reward_used": False,
        "git_head": _git_head(),
        "elapsed_s": time.perf_counter() - t0_total,
    }
    with open(OUTPUT_DIR / "strategy_F_metadata.json", "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- 汇总文本 ----
    summary = _build_summary(df, prof, best_vs_k, g, sets, high_k, bal,
                             bal_relaxed, k_shift, k_ratio_g, alpha_ratio_g,
                             k_driven_label, ident, reg, regime_metrics_out,
                             metadata, t_proto, t_int, t_top_meas, T_env)
    (OUTPUT_DIR / "strategy_F_summary.txt").write_text(summary,
                                                       encoding="utf-8")
    print(summary)
    print(f"[done] total elapsed {time.perf_counter() - t0_total:.1f} s")
    return metadata


def _build_summary(df, prof, best_vs_k, g, sets, high_k, bal, bal_relaxed,
                   k_shift, k_ratio_g, alpha_ratio_g, k_driven_label,
                   ident, reg, regime_metrics_out, metadata, t_proto,
                   t_int, t_top_meas, t_env):
    lines = []
    A = lines.append
    A("=" * 70)
    A("STRATEGY F — CONVECTION + RADIATION k/cp/tau RECALIBRATION (72C)")
    A("=" * 70)
    A("")
    A(f"环境: T_environment = {metadata['physics']['environment_C']} C "
      f"(第一个有效实测 Top COC); T_initial = "
      f"{metadata['physics']['initial_C']} C")
    A(f"固定边界: h=10, eps=0.90, sigma=5.670374419e-8, F=1.0, "
      f"非线性辐射 (不线性化)")
    A(f"Strategy E 回归 (k=0.0165, cp=900, tau=0): "
      f"RMSE={reg['RMSE_C']:.4f} C, top_max={reg['top_max_C']:.3f} C, "
      f"sample_max={reg['sample_max_C']:.3f} C -> "
      f"{'PASS' if reg['PASS'] else 'FAIL'}")
    A(f"扫描: {len(df)} 行 (63 FDM + 8 tau), "
      f"失败 {(df['status'] == 'FAILED').sum()}")
    A("")
    A(f"GLOBAL RMSE MINIMUM:")
    A("  " + _format_row(g))
    A(f"  k ratio vs A = {k_ratio_g:.3f}, "
      f"cp ratio vs A = {float(g['cp_eff_J_kgK']) / CP_A:.3f}, "
      f"alpha ratio vs A = {alpha_ratio_g:.3f}")
    A(f"  cp>=800: {'YES' if g['cp_non_extreme'] else 'NO'}; "
      f"tau 上界警告: "
      f"{'YES' if g['tau_upper_bound_warning'] else 'NO'}; "
      f"网格边界限制: "
      f"{'YES' if metadata['parameter_grid_boundary_warning'] else 'NO'}")
    A(f"  k 位移分类: {k_shift}")
    A(f"  扩散率提升主要由 k 驱动: {k_driven_label}")
    A("")
    A("Best RMSE vs k (profile over cp, tau):")
    for _, r in best_vs_k.iterrows():
        A(f"  k={r['k_eff_W_mK']:.4f}: cp={r['best_cp_eff_J_kgK']:.0f}, "
          f"tau={r['best_tau_s']:.2f}, alpha={r['best_alpha_eff_m2_s']:.3e}, "
          f"RMSE={r['best_RMSE_C']:.4f}")
    A("")
    A("Near-optimal regions:")
    for name, s in sets.items():
        ss = summarize_set(s)
        if ss is None:
            A(f"  {name}: empty")
            continue
        A(f"  {name}: n={ss['n_total']} (cp>=800: {ss['n_cp_ge_800']}), "
          f"k [{ss['k_range'][0]:.4f}, {ss['k_range'][1]:.4f}], "
          f"cp [{ss['cp_range'][0]:.0f}, {ss['cp_range'][1]:.0f}], "
          f"tau [{ss['tau_range'][0]:.2f}, {ss['tau_range'][1]:.2f}], "
          f"alpha [{ss['alpha_range'][0]:.2e}, "
          f"{ss['alpha_range'][1]:.2e}]")
    A("")
    A("HIGH_K_NEAR_EQUIVALENT_CANDIDATE (Delta<=0.10, cp>=800):")
    if high_k["MODERATE"] is not None:
        A("  " + _format_row(high_k["MODERATE"]))
    else:
        A("  NONE")
    A("HIGH_K_APPLICATION_CANDIDATE (Delta<=0.20, cp>=800):")
    if high_k["APPLICATION"] is not None:
        A("  " + _format_row(high_k["APPLICATION"]))
    else:
        A("  NONE")
    A("BALANCED_PHYSICAL_CANDIDATE (RMSE<=1.0, cp>=800, tau<=5, max k):")
    if bal is not None:
        A("  " + _format_row(bal))
    else:
        A("  NONE")
    A("BALANCED_PHYSICAL_CANDIDATE relaxed (RMSE<=1.2, cp>=800, tau<=5):")
    if bal_relaxed is not None:
        A("  " + _format_row(bal_relaxed))
    else:
        A("  NONE")
    A("")
    A("Local identifiability at global min:")
    A(f"  singular values: {[f'{v:.4e}' for v in ident['singular_values']]}")
    A(f"  condition number: {ident['condition_number']:.3f}")
    A(f"  strongest correlation: {ident['strongest_pair']}")
    A(f"  warning: {'YES' if ident['identifiability_warning'] else 'NO'}")
    if regime_metrics_out:
        A("")
        A("Regime residual diagnostics (global min, diagnostic only):")
        for reg_name, mm in regime_metrics_out.items():
            A(f"  {reg_name}: n={mm['n_points']}, RMSE={mm['RMSE_C']:.3f}, "
              f"mean={mm['mean_residual_C']:+.3f}, MAE={mm['MAE_C']:.3f}")
    A("")
    A("Scientific conclusion:")
    A(f"  新边界 (h=10+非线性辐射) 下首次 k/cp/tau 标定 (无高-k 奖励, "
      f"纯 RMSE):")
    A(f"  全局最小在 k={g['k_eff_W_mK']:.4f}, cp={g['cp_eff_J_kgK']:.0f}, "
      f"tau={g['tau_lag_s']:.2f} s, RMSE={g['RMSE_72C_C']:.4f} C "
      f"(vs 旧边界 Strategy A 0.7337 C)")
    A(f"  k 相对旧值变化: {k_ratio_g:.3f}x -> {k_shift}")
    A(f"  扩散率提升主要由 k 驱动: {k_driven_label}")
    A("")
    A("无 DOE11 / 长保持 PCR / 样品选择 / 细扫 / 连续优化 / 高-k 奖励。")
    A("Strategy E 物理未修改; 历史 Strategy A 未修改。")
    A("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
