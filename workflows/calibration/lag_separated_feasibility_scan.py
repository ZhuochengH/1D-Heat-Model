#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy C 可行性扫描 — 固定 k, 扫 alpha × tau (lag-separated model)
======================================================================

可行性问题:
    非零一阶外部滞后能否在保持良好 72C 顶部拟合的同时, 允许显著更高的
    芯片有效扩散率 alpha_eff?

本任务固定:
    k_eff = 0.0165 W/(m K)   (约保持当前拟合的芯片热阻)

只扫:
    alpha_eff [m2/s]  ×  tau_ext [s]

不跑完整 3-DOF 优化。不修改名义标定。

输出目录: parameter_scan_output/72C/lag_separated_model_feasibility_v1/
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

from thermal_model.config import calibrated_model_config as cmc
from thermal_model.core.lag_augmented_thermal_model import (
    LagAugmentedParameters,
    evaluate_72c_objective,
    evaluate_72c_objective_safe,
)
from thermal_model.utilities.scan_effective_thermal_parameters import load_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "lag_separated_model_feasibility_v1"
)

K_FIXED = 0.0165
RHO_COC = 1020.0

ALPHA_VALUES = [
    0.0165 / (1020.0 * 900.0),  # 精确策略 A alpha (≈1.797e-8)
    2.0e-8,
    2.5e-8,
    3.5e-8,
    5.0e-8,
    7.0e-8,
    1.0e-7,
]
TAU_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

BANDS = [(1.0, "<=1.0 C"), (1.5, "<=1.5 C"), (2.0, "<=2.0 C")]


def _alpha_key(a):
    return f"{float(a):.9e}"


def _row_key(a, tau):
    return f"{float(a):.9e}|{float(tau):.4f}"


def strategy_a_alpha():
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    return cal.k_eff_W_mK / (cal.rho_COC_kg_m3 * cal.cp_eff_J_kgK)


def run_regression(t_proto, t_int, t_top_meas, output_dir):
    """策略 A tau=0 回归: 复现 RMSE ~0.7337。"""
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = strategy_a_alpha()
    p = LagAugmentedParameters(alpha_eff_m2_s=alpha_A, k_eff_W_mK=0.0165,
                               tau_ext_s=0.0)
    r = evaluate_72c_objective(t_proto, t_int, t_top_meas, p)
    res = {
        "alpha_A_m2_s": alpha_A,
        "k_eff_W_mK": 0.0165,
        "derived_cp_eff_J_kgK": p.cp_eff_J_kgK,
        "tau_ext_s": 0.0,
        "RMSE_72C_C": r["RMSE_72C_C"],
        "expected_RMSE_C": 0.7337,
        "pass": abs(r["RMSE_72C_C"] - 0.7337) < 0.01 and
                abs(p.cp_eff_J_kgK - 900.0) < 1e-9,
    }
    (output_dir / "strategy_A_regression.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"[regression] alpha_A={alpha_A:.6e} cp={p.cp_eff_J_kgK:.6f} "
          f"RMSE={r['RMSE_72C_C']:.6f} (expect ~0.7337) "
          f"-> {'PASS' if res['pass'] else 'FAIL'}")
    return res


def run_scan(t_proto, t_int, t_top_meas, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "alpha_tau_feasibility_scan.csv"
    done = set()
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        done = set(_row_key(a, tau) for a, tau in
                   zip(df["alpha_eff_m2_s"], df["tau_ext_s"]))
    for alpha in ALPHA_VALUES:
        for tau in TAU_VALUES:
            key = _row_key(alpha, tau)
            if key in done:
                continue
            p = LagAugmentedParameters(alpha_eff_m2_s=alpha,
                                       k_eff_W_mK=K_FIXED, tau_ext_s=tau)
            row = evaluate_72c_objective_safe(t_proto, t_int, t_top_meas, p)
            print(f"  alpha={alpha:.3e} tau={tau:5.2f} cp={row['derived_cp_eff_J_kgK']:8.1f} "
                  f"RMSE={row['RMSE_72C_C']:.4f}")
            if csv_path.exists():
                pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False,
                                           index=False)
            else:
                pd.DataFrame([row]).to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    fails = (df.get("status", "") == "FAILED").sum() if "status" in df else 0
    print(f"[scan] total {len(df)} points, failures: {fails}")
    return df


def profile_best_tau(df, output_dir=None):
    """每个 alpha 的最优 tau (最小 RMSE)。"""
    rows = []
    ok = df["status"] == "OK" if "status" in df else pd.Series(True, index=df.index)
    for alpha in sorted(df["alpha_eff_m2_s"].unique()):
        sub = df[(df["alpha_eff_m2_s"] == alpha) & ok]
        if sub.empty:
            continue
        i_best = int(sub["RMSE_72C_C"].idxmin())
        best = sub.loc[i_best]
        tau0 = sub[sub["tau_ext_s"] == 0.0]
        rmse_tau0 = float(tau0["RMSE_72C_C"].iloc[0]) if len(tau0) else np.nan
        rows.append({
            "alpha_eff_m2_s": float(alpha),
            "k_eff_W_mK": K_FIXED,
            "derived_cp_eff_J_kgK": K_FIXED / (RHO_COC * float(alpha)),
            "best_tau_s": float(best["tau_ext_s"]),
            "best_RMSE_C": float(best["RMSE_72C_C"]),
            "RMSE_tau0_C": rmse_tau0,
            "RMSE_improvement_from_lag_C": rmse_tau0 - float(best["RMSE_72C_C"]),
        })
    prof = pd.DataFrame(rows)
    if output_dir is not None:
        prof.to_csv(output_dir / "best_tau_profile_vs_alpha.csv", index=False)
    return prof


def plot_landscape(df, output_dir):
    ok = df["status"] == "OK" if "status" in df else pd.Series(True, index=df.index)
    sub = df[ok]
    alphas = np.sort(sub["alpha_eff_m2_s"].unique())
    taus = np.sort(sub["tau_ext_s"].unique())
    Z = np.full((len(alphas), len(taus)), np.nan)
    for _, r in sub.iterrows():
        ai = int(np.searchsorted(alphas, r["alpha_eff_m2_s"]))
        ti = int(np.searchsorted(taus, r["tau_ext_s"]))
        Z[ai, ti] = r["RMSE_72C_C"]
    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(taus, alphas, Z, shading="auto", cmap="viridis")
    cb = fig.colorbar(mesh, ax=ax, label="RMSE_top [°C]")
    ax.set_yscale("log")
    ax.set_xlabel("tau_ext [s]")
    ax.set_ylabel("alpha_eff [m²/s] (log)")
    ax.set_title("Strategy C feasibility — 72C RMSE landscape "
                 "(fixed k=0.0165)")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_landscape_alpha_tau.png", dpi=200)
    plt.close(fig)


def plot_best_rmse_vs_alpha(prof, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(prof["alpha_eff_m2_s"], prof["best_RMSE_C"], marker="o",
            color="#1f77b4", label="best RMSE (after profiling tau)")
    ax.plot(prof["alpha_eff_m2_s"], prof["RMSE_tau0_C"], marker="s",
            color="#d62728", ls="--", label="RMSE at tau=0")
    alpha_A = strategy_a_alpha()
    ax.axvline(alpha_A, color="grey", ls=":", label=f"Strategy A alpha "
               f"({alpha_A:.2e})")
    ax.axhline(0.7337, color="black", ls=":", label="Strategy A RMSE 0.7337")
    ax.axhline(1.0, color="green", ls="-.", alpha=0.6, label="RMSE = 1.0 C")
    ax.set_xscale("log")
    ax.set_xlabel("alpha_eff [m²/s] (log)")
    ax.set_ylabel("RMSE_top [°C]")
    ax.set_title("Strategy C feasibility — best achievable 72C RMSE vs alpha\n"
                 "(fixed k=0.0165; tau profiled per alpha)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "best_rmse_vs_alpha.png", dpi=200)
    plt.close(fig)


def plot_trace_comparison(t_proto, t_int, t_top_meas, prof, df, output_dir):
    """72C 迹线对比: 实测 + 策略 A + 最佳滞后模型的最高可用 alpha。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_proto, t_top_meas, color="black", lw=1.6,
            label="Measured Top COC")

    # 策略 A
    alpha_A = strategy_a_alpha()
    pA = LagAugmentedParameters(alpha_eff_m2_s=alpha_A, k_eff_W_mK=0.0165,
                                tau_ext_s=0.0)
    from thermal_model.core.lag_augmented_thermal_model import run_lag_augmented_model
    oA = run_lag_augmented_model(t_proto, t_int, pA)
    predA = np.interp(t_proto, oA["t_array"], oA["T_top_observed_predicted_C"])
    ax.plot(t_proto, predA, color="#1f77b4", lw=1.4,
            label="Strategy A (tau=0)")

    # 最高可用 alpha: RMSE<=1.5 且 alpha > alpha_A
    candidates = prof[(prof["best_RMSE_C"] <= 1.5) &
                      (prof["alpha_eff_m2_s"] > alpha_A)]
    if not candidates.empty:
        cand = candidates.loc[int(candidates["alpha_eff_m2_s"].idxmax())]
        pC = LagAugmentedParameters(alpha_eff_m2_s=cand["alpha_eff_m2_s"],
                                    k_eff_W_mK=0.0165,
                                    tau_ext_s=cand["best_tau_s"])
        oC = run_lag_augmented_model(t_proto, t_int, pC)
        predC = np.interp(t_proto, oC["t_array"],
                          oC["T_top_observed_predicted_C"])
        ax.plot(t_proto, predC, color="#2ca02c", lw=1.4, ls="--",
                label=f"Strategy C (alpha={cand['alpha_eff_m2_s']:.2e}, "
                      f"tau={cand['best_tau_s']:.1f})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("72C Top-Fit — Lag-Separated Model vs Strategy A\n"
                 "(highest alpha with best RMSE <= 1.5 C, fixed k)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "72C_trace_comparison_lag_model.png", dpi=200)
    plt.close(fig)


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


def write_summary_metadata(output_dir, prof, df):
    alpha_A = strategy_a_alpha()
    # 分带: 每个 alpha 的 best RMSE 落在哪个带
    band_report = {}
    for th, lab in BANDS:
        sub = prof[prof["best_RMSE_C"] <= th]
        if sub.empty:
            band_report[lab] = None
        else:
            i = int(sub["alpha_eff_m2_s"].idxmax())
            band_report[lab] = {
                "alpha": float(sub.loc[i, "alpha_eff_m2_s"]),
                "tau": float(sub.loc[i, "best_tau_s"]),
                "RMSE": float(sub.loc[i, "best_RMSE_C"]),
            }

    # tau 是否随 alpha 增大
    prof_sorted = prof.sort_values("alpha_eff_m2_s")
    tau_increases = bool(np.all(np.diff(prof_sorted["best_tau_s"].to_numpy())
                                >= -1e-12))

    summary = {
        "strategy_id": "lag_separated_alpha_k_tau_v1",
        "status": "EXPERIMENTAL / FEASIBILITY ONLY",
        "k_fixed_W_mK": K_FIXED,
        "alpha_A_m2_s": alpha_A,
        "band_report": band_report,
        "tau_increases_with_alpha": tau_increases,
        "git_commit": _git_head(),
        "git_tag": _git_describe(),
    }
    (output_dir / "feasibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "STRATEGY C — LAG-SEPARATED 3-DOF MODEL FEASIBILITY SUMMARY",
        "=" * 72,
        f"Fixed k_eff = {K_FIXED} W/(m K); scanned alpha x tau.",
        f"Strategy A alpha = {alpha_A:.6e} m2/s (RMSE ~0.7337 C).",
        "",
        "Best tau profile (one row per alpha):",
        prof.to_string(index=False),
        "",
        "Highest alpha within descriptive bands (best RMSE after tau):",
    ]
    for lab, b in band_report.items():
        if b is None:
            lines.append(f"  {lab}: none")
        else:
            lines.append(f"  {lab}: alpha={b['alpha']:.3e}, tau={b['tau']:.2f} s, "
                         f"RMSE={b['RMSE']:.4f} C")
    lines.append("")
    lines.append(f"Does best tau increase with alpha? "
                 f"{'YES' if tau_increases else 'NO'}")
    lines.append("")
    lines.append("NOTE: tau_ext is applied ONLY to top observation, NOT to "
                 "sample temperature. DOE11 sample sensitivity is omitted "
                 "unless a genuinely useful high-alpha candidate exists.")
    (output_dir / "feasibility_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=["regression", "scan", "analysis", "all"])
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_proto, t_int, t_top_meas = load_experiment()

    if args.stage in ("regression", "all"):
        res = run_regression(t_proto, t_int, t_top_meas, output_dir)
        if not res["pass"]:
            print("[STOP] Strategy A regression FAILED — aborting.")
            return 2

    if args.stage in ("scan", "all"):
        t0 = time.perf_counter()
        df = run_scan(t_proto, t_int, t_top_meas, output_dir)
        print(f"[scan] elapsed {time.perf_counter() - t0:.1f} s")
    elif args.stage == "analysis":
        df = pd.read_csv(output_dir / "alpha_tau_feasibility_scan.csv")
    else:
        df = None

    if args.stage in ("analysis", "all"):
        prof = profile_best_tau(df, output_dir)
        plot_landscape(df, output_dir)
        plot_best_rmse_vs_alpha(prof, output_dir)
        plot_trace_comparison(t_proto, t_int, t_top_meas, prof, df, output_dir)
        write_summary_metadata(output_dir, prof, df)
        print(prof.to_string(index=False))
        print(f"[output] {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
