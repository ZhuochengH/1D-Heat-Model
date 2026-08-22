#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy E — 72C 轻量参考检查 (convection + radiation + lag)
============================================================

本脚本只做「模型构建 + 数值验证」后的轻量方向性比较 (任务 34-37):

    - 读取权威 72C 对齐数据
      (temperature_alignment_output/72C/aligned_internal_top_temperature.csv);
    - T_environment = 第一个有效实测 Top COC 温度 (动态解析, 不硬编码);
    - T_initial     = 第一个有效内部温度;
    - 案例 A: k=0.0165, cp=900, tau=0
        旧: h=5 对流-only (heat_model.run_simulation)
        新: h=10 + epsilon=0.90 非线性辐射 (Strategy E)
    - 案例 D: k=0.0165, cp=800, tau=1.0
        旧: h=5 对流-only + 一阶滞后 (lag_augmented_thermal_model)
        新: h=10 + 非线性辐射 + 一阶滞后 (Strategy E)
    - 全部无重拟合 (k/cp/tau/h/eps 均不调整);
    - 计算辐射诊断 (40/60/72/95 C 的 q_rad 与 h_rad_equiv, 仅解释用)。

输出 (model_comparison_output/convection_radiation_lag_v1/):
    fixed_boundary_constants.json
    radiation_diagnostics.csv
    environment_resolution.json
    strategy_A_old_vs_new_boundary.csv
    strategy_D_old_vs_new_boundary.csv
    72C_old_vs_new_boundary.png
    72C_boundary_effect_on_sample.png
    model_check_summary.txt
    model_check_metadata.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_model.core import heat_model
from thermal_model.core import lag_augmented_thermal_model as lm
from thermal_model.core import convection_radiation_thermal_model as cr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "72C"
    / "aligned_internal_top_temperature.csv")
OUTPUT_DIR = (
    PROJECT_ROOT / "model_comparison_output"
    / "convection_radiation_lag_v1")

# 参考案例参数 (任务 35/36; 不重拟合)
K_A, CP_A, TAU_A = 0.0165, 900.0, 0.0
K_D, CP_D, TAU_D = 0.0165, 800.0, 1.0

SAVE_DT = 1.0


def load_72c():
    df = pd.read_csv(ALIGNED_CSV)
    t = df["time_s"].to_numpy(float)
    t_int = df["T_internal_interpolated_C"].to_numpy(float)
    t_top = df["T_top_measured_C"].to_numpy(float)
    return t, t_int, t_top


def rmse(pred, meas):
    r = np.asarray(pred, dtype=float) - np.asarray(meas, dtype=float)
    return float(np.sqrt(np.mean(r ** 2)))


def interp_to_meas(meas_time, model_time, model_trace):
    return np.interp(meas_time, model_time, model_trace)


def run_old_convection_only(t, t_int, t_env, k, cp):
    """旧 h=5 对流-only 模型 (无滞后; 用于案例 A 对比)。"""
    mats = cr.make_convection_radiation_materials(k, cp)
    return heat_model.run_simulation(
        time_s=t, bottom_temperature_C=t_int, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=t_env, save_dt=SAVE_DT,
        T_initial_C=float(t_int[0]))


def run_old_lag_h5(t, t_int, t_env, k, cp, tau):
    """旧策略 D: h=5 对流-only + 输出侧一阶滞后。"""
    alpha = k / (1020.0 * cp)
    params = lm.LagAugmentedParameters(
        alpha_eff_m2_s=alpha, k_eff_W_mK=k, tau_ext_s=tau)
    return lm.run_lag_augmented_model(
        time_s=t, T_internal_C=t_int, parameters=params,
        h_conv=5.0, T_air_ambient=t_env, save_dt=SAVE_DT)


def run_new_strategy_e(t, t_int, t_env, k, cp, tau):
    """新策略 E: h=10 + 非线性辐射 + 输出侧一阶滞后。"""
    return cr.run_convection_radiation_lag_model(
        time_s=t, bottom_temperature_C=t_int, T_environment_C=t_env,
        k_eff_W_mK=k, cp_eff_J_kgK=cp, tau_lag_s=tau, save_dt=SAVE_DT)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t, t_int, t_top = load_72c()

    # ---------------- 环境温度解析 (分析层, 动态) ----------------
    env_info = cr.infer_environment_from_initial_top_measurement(
        t_top, time_s=t)
    T_env = env_info["T_environment_C"]
    init_info = cr.infer_environment_from_initial_top_measurement(
        t_int, time_s=t)
    T_init = init_info["T_environment_C"]  # 第一个有效内部温度 (标量)

    environment_resolution = {
        "dataset": "72C corrected aligned (temperature_alignment_output/72C)",
        "source_file": str(ALIGNED_CSV.resolve()),
        "rule": "T_environment = first valid measured Top COC temperature",
        "T_environment_C": T_env,
        "source_index": env_info["source_index"],
        "source_time_s": env_info["source_time_s"],
        "constant_during_simulation": True,
        "derived_from_internal_temperature": False,
        "hard_coded_25C": False,
        "T_initial_C": T_init,
        "initial_source_index": init_info["source_index"],
        "initial_source_time_s": init_info["source_time_s"],
        "T_initial_equals_environment": bool(T_init == T_env),
        "T_initial_minus_T_environment_C": float(T_init - T_env),
        "T_air_C": T_env,
        "T_surroundings_C": T_env,
    }

    # ---------------- 模型运行 (无重拟合) ----------------
    old_a = run_old_convection_only(t, t_int, T_env, K_A, CP_A)
    new_a = run_new_strategy_e(t, t_int, T_env, K_A, CP_A, TAU_A)
    old_d = run_old_lag_h5(t, t_int, T_env, K_D, CP_D, TAU_D)
    new_d = run_new_strategy_e(t, t_int, T_env, K_D, CP_D, TAU_D)

    # ---------------- 指标 (插值到实测时间) ----------------
    def metrics(model_time, raw_top, lagged_top, sample_top, tag):
        rmse_raw = rmse(interp_to_meas(t, model_time, raw_top), t_top)
        rmse_lag = rmse(interp_to_meas(t, model_time, lagged_top), t_top)
        return {
            "tag": tag,
            "RMSE_raw_top_C": rmse_raw,
            "RMSE_lagged_top_C": rmse_lag,
            "top_max_C": float(np.max(raw_top)),
            "top_min_C": float(np.min(raw_top)),
            "sample_max_C": float(np.max(sample_top)),
            "sample_min_C": float(np.min(sample_top)),
        }

    m_old_a = metrics(old_a["t_array"], old_a["T_top_surface_arr"],
                      old_a["T_top_surface_arr"], old_a["T_sample_arr"],
                      "A_old_h5_convection_only")
    m_new_a = metrics(new_a["t_array"], new_a["T_top_FDM_C"],
                      new_a["T_top_observed_predicted_C"],
                      new_a["T_sample_FDM_C"], "A_new_h10_radiation")
    m_old_d = metrics(old_d["t_array"], old_d["T_top_FDM_C"],
                      old_d["T_top_observed_predicted_C"],
                      old_d["T_sample_FDM_C"], "D_old_h5_lag")
    m_new_d = metrics(new_d["t_array"], new_d["T_top_FDM_C"],
                      new_d["T_top_observed_predicted_C"],
                      new_d["T_sample_FDM_C"], "D_new_h10_radiation_lag")

    # ---------------- 方向性问题 ----------------
    # 1. 新边界是否降低预测顶部温度?
    delta_A_top_mean = float(np.mean(
        new_a["T_top_FDM_C"] - old_a["T_top_surface_arr"]))
    top_lower_A = float(np.max(new_a["T_top_FDM_C"])
                        - np.max(old_a["T_top_surface_arr"]))
    delta_D_top_mean = float(np.mean(
        new_d["T_top_FDM_C"] - old_d["T_top_FDM_C"]))
    # 2. 样品 FDM 变化 (仅因顶部边界)
    delta_A_sample_mean = float(np.mean(
        new_a["T_sample_FDM_C"] - old_a["T_sample_arr"]))
    delta_A_sample_max = float(np.max(np.abs(
        new_a["T_sample_FDM_C"] - old_a["T_sample_arr"])))
    # 3. RMSE 变化
    rmse_change_A = m_new_a["RMSE_lagged_top_C"] - m_old_a["RMSE_raw_top_C"]
    rmse_change_D = m_new_d["RMSE_lagged_top_C"] - m_old_d["RMSE_lagged_top_C"]

    # ---------------- 辐射诊断 (40/60/72/95 C) ----------------
    rad_rows = []
    for Ts in (40.0, 60.0, 72.0, 95.0):
        q = cr.radiative_heat_flux_W_m2(Ts, T_env)
        h = cr.equivalent_radiative_heat_transfer_coefficient(Ts, T_env)
        rad_rows.append({
            "surface_temperature_C": Ts,
            "T_environment_C": T_env,
            "q_rad_W_m2": q,
            "h_rad_equiv_W_m2K": h,
        })

    # ---------------- 输出文件 ----------------
    # 1. fixed_boundary_constants.json
    constants = {
        "strategy": "convection_radiation_lag_k_cp_tau_v1",
        "status": "EXPERIMENTAL / MODEL-CONSTRUCTION ONLY",
        "accepted_as_nominal": False,
        "h_conv_W_m2K": cr.H_CONV_STRATEGY_E_W_M2K,
        "emissivity": cr.EMISSIVITY_STRATEGY_E,
        "sigma_SB_W_m2K4": cr.SIGMA_SB_W_M2_K4,
        "view_factor": cr.VIEW_FACTOR_STRATEGY_E,
        "rho_COC_kg_m3": cr.RHO_COC_STRATEGY_E,
        "radiation_linearized_in_solver": False,
        "fixed_h_rad_used_as_bc": False,
        "top_boundary_equation": "q_cond = q_conv + q_rad",
        "radiation": "full nonlinear Stefan-Boltzmann at every FDM step",
    }
    _dump_json(OUTPUT_DIR / "fixed_boundary_constants.json", constants)

    # 2. radiation_diagnostics.csv
    rad_df = pd.DataFrame(rad_rows)
    rad_df.to_csv(OUTPUT_DIR / "radiation_diagnostics.csv", index=False)

    # 3. environment_resolution.json
    _dump_json(OUTPUT_DIR / "environment_resolution.json",
               environment_resolution)

    # 4. strategy_A_old_vs_new_boundary.csv
    df_a = pd.DataFrame({
        "time_s": t,
        "T_top_measured_C": t_top,
        "old_A_T_top_raw_C": interp_to_meas(
            t, old_a["t_array"], old_a["T_top_surface_arr"]),
        "new_A_T_top_raw_C": interp_to_meas(
            t, new_a["t_array"], new_a["T_top_FDM_C"]),
        "old_A_T_sample_C": interp_to_meas(
            t, old_a["t_array"], old_a["T_sample_arr"]),
        "new_A_T_sample_C": interp_to_meas(
            t, new_a["t_array"], new_a["T_sample_FDM_C"]),
    })
    df_a["delta_T_top_new_minus_old_C"] = (
        df_a["new_A_T_top_raw_C"] - df_a["old_A_T_top_raw_C"])
    df_a["delta_T_sample_new_minus_old_C"] = (
        df_a["new_A_T_sample_C"] - df_a["old_A_T_sample_C"])
    df_a.to_csv(OUTPUT_DIR / "strategy_A_old_vs_new_boundary.csv",
                index=False)

    # 5. strategy_D_old_vs_new_boundary.csv
    df_d = pd.DataFrame({
        "time_s": t,
        "T_top_measured_C": t_top,
        "old_D_T_top_raw_C": interp_to_meas(
            t, old_d["t_array"], old_d["T_top_FDM_C"]),
        "new_D_T_top_raw_C": interp_to_meas(
            t, new_d["t_array"], new_d["T_top_FDM_C"]),
        "old_D_T_top_lagged_C": interp_to_meas(
            t, old_d["t_array"], old_d["T_top_observed_predicted_C"]),
        "new_D_T_top_lagged_C": interp_to_meas(
            t, new_d["t_array"], new_d["T_top_observed_predicted_C"]),
        "old_D_T_sample_C": interp_to_meas(
            t, old_d["t_array"], old_d["T_sample_FDM_C"]),
        "new_D_T_sample_C": interp_to_meas(
            t, new_d["t_array"], new_d["T_sample_FDM_C"]),
    })
    df_d["delta_T_top_new_minus_old_C"] = (
        df_d["new_D_T_top_raw_C"] - df_d["old_D_T_top_raw_C"])
    df_d["delta_T_sample_new_minus_old_C"] = (
        df_d["new_D_T_sample_C"] - df_d["old_D_T_sample_C"])
    df_d.to_csv(OUTPUT_DIR / "strategy_D_old_vs_new_boundary.csv",
                index=False)

    # 6. 72C_old_vs_new_boundary.png
    _plot_top(t, t_top, old_a, new_a, old_d, new_d,
              OUTPUT_DIR / "72C_old_vs_new_boundary.png")

    # 7. 72C_boundary_effect_on_sample.png
    _plot_sample(t, old_a, new_a, old_d, new_d,
                 OUTPUT_DIR / "72C_boundary_effect_on_sample.png")

    # 8-9. summary / metadata
    metadata = {
        "output_dir": str(OUTPUT_DIR.resolve()),
        "dataset": "72C corrected aligned",
        "aligned_rows_used": int(len(t)),
        "save_dt_s": SAVE_DT,
        "reference_case_A": {
            "k_eff_W_mK": K_A, "cp_eff_J_kgK": CP_A, "tau_lag_s": TAU_A,
            "old": m_old_a, "new": m_new_a,
            "RMSE_change_C": rmse_change_A,
            "top_max_old_C": m_old_a["top_max_C"],
            "top_max_new_C": m_new_a["top_max_C"],
            "sample_max_old_C": m_old_a["sample_max_C"],
            "sample_max_new_C": m_new_a["sample_max_C"],
            "top_mean_delta_new_minus_old_C": delta_A_top_mean,
            "top_max_delta_new_minus_old_C": top_lower_A,
            "sample_mean_delta_new_minus_old_C": delta_A_sample_mean,
            "sample_max_abs_delta_C": delta_A_sample_max,
        },
        "reference_case_D": {
            "k_eff_W_mK": K_D, "cp_eff_J_kgK": CP_D, "tau_lag_s": TAU_D,
            "old": m_old_d, "new": m_new_d,
            "RMSE_change_C": rmse_change_D,
            "top_mean_delta_new_minus_old_C": delta_D_top_mean,
        },
        "radiation_diagnostics": rad_rows,
        "newton": {
            "method": "Newton",
            "analytical_derivative": True,
            "abs_tolerance_C": 1e-10,
            "max_iterations": 20,
            "convergence_failures": 0,
            "case_A_new_max_iterations_per_step": int(
                new_a["result"]["newton_max_iterations_per_step"]),
            "case_D_new_max_iterations_per_step": int(
                new_d["result"]["newton_max_iterations_per_step"]),
            "case_A_new_max_abs_residual_W_m2_excl_t0": float(np.max(np.abs(
                new_a["result"]["boundary_residual_arr"][1:]))),
            "case_D_new_max_abs_residual_W_m2_excl_t0": float(np.max(np.abs(
                new_d["result"]["boundary_residual_arr"][1:]))),
        },
        "parameter_scan_performed": False,
        "parameter_optimization_performed": False,
        "k_refitted": False, "cp_refitted": False, "tau_refitted": False,
        "h_fitted": False, "emissivity_fitted": False,
        "environment_fitted": False,
    }
    _dump_json(OUTPUT_DIR / "model_check_metadata.json", metadata)

    summary = _build_summary(constants, environment_resolution, rad_rows,
                             m_old_a, m_new_a, m_old_d, m_new_d,
                             rmse_change_A, rmse_change_D,
                             delta_A_top_mean, top_lower_A,
                             delta_A_sample_mean, delta_A_sample_max,
                             delta_D_top_mean, metadata)
    (OUTPUT_DIR / "model_check_summary.txt").write_text(
        summary, encoding="utf-8")

    print(f"输出目录: {OUTPUT_DIR}")
    print(summary)


def _build_summary(constants, env, rad, m_old_a, m_new_a, m_old_d, m_new_d,
                   rmse_a, rmse_d, dtop_a_mean, dtop_a_max,
                   dsample_a_mean, dsample_a_max, dtop_d_mean, meta):
    lines = []
    lines.append("=" * 68)
    lines.append("STRATEGY E — CONVECTION + RADIATION + LAG MODEL CHECK (72C)")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"固定边界参数: h_conv={constants['h_conv_W_m2K']} W/(m2 K), "
                 f"epsilon={constants['emissivity']}, "
                 f"sigma={constants['sigma_SB_W_m2K4']} W/(m2 K4), "
                 f"F_view={constants['view_factor']}")
    lines.append("")
    lines.append(f"环境温度: T_environment = {env['T_environment_C']} C "
                 f"(source row index {env['source_index']}, "
                 f"time {env['source_time_s']} s)")
    lines.append(f"初始内部温度: T_initial = {env['T_initial_C']} C")
    lines.append(f"内部初始 - 环境 = {env['T_initial_minus_T_environment_C']} C"
                 f"  (两者允许不同)")
    lines.append("")
    lines.append(f"案例 A (k={K_A}, cp={CP_A}, tau={TAU_A}):")
    lines.append(f"  旧 (h=5 对流):  RMSE = {m_old_a['RMSE_raw_top_C']:.4f} C, "
                 f"top max = {m_old_a['top_max_C']:.3f} C, "
                 f"sample max = {m_old_a['sample_max_C']:.3f} C")
    lines.append(f"  新 (h=10+辐射): RMSE = {m_new_a['RMSE_lagged_top_C']:.4f} C, "
                 f"top max = {m_new_a['top_max_C']:.3f} C, "
                 f"sample max = {m_new_a['sample_max_C']:.3f} C")
    lines.append(f"  RMSE 变化 = {rmse_a:+.4f} C")
    lines.append(f"  顶部均值变化 (新-旧) = {dtop_a_mean:+.4f} C; "
                 f"峰值变化 = {dtop_a_max:+.4f} C")
    lines.append(f"  样品均值变化 = {dsample_a_mean:+.4f} C; "
                 f"样品最大绝对差 = {dsample_a_max:.4f} C")
    lines.append("")
    lines.append(f"案例 D (k={K_D}, cp={CP_D}, tau={TAU_D}):")
    lines.append(f"  旧 (h=5 对流+lag): RMSE = "
                 f"{m_old_d['RMSE_lagged_top_C']:.4f} C")
    lines.append(f"  新 (h=10+辐射+lag): RMSE = "
                 f"{m_new_d['RMSE_lagged_top_C']:.4f} C")
    lines.append(f"  RMSE 变化 = {rmse_d:+.4f} C")
    lines.append(f"  原始顶部均值变化 (新-旧) = {dtop_d_mean:+.4f} C")
    lines.append("")
    lines.append("辐射诊断 (epsilon=0.90, F=1.0, "
                 f"T_env={env['T_environment_C']} C):")
    for row in rad:
        lines.append(f"  {row['surface_temperature_C']:.0f} C: "
                     f"q_rad = {row['q_rad_W_m2']:.2f} W/m2, "
                     f"h_rad_equiv = {row['h_rad_equiv_W_m2K']:.3f} W/(m2 K)")
    lines.append("")
    lines.append("方向性结论:")
    lines.append(f"  1. T_environment = {env['T_environment_C']} C vs "
                 f"T_initial = {env['T_initial_C']} C "
                 f"(差 {env['T_initial_minus_T_environment_C']:+.2f} C)")
    lines.append(f"  2. h=10+辐射 是否降低预测顶部: "
                 f"{'YES' if dtop_a_mean < 0 else 'NO'} "
                 f"(案例 A 均值 {dtop_a_mean:+.4f} C)")
    lines.append(f"  3. 案例 A RMSE 变化 (无重拟合): {rmse_a:+.4f} C")
    lines.append(f"  4. 案例 D RMSE 变化 (无重拟合): {rmse_d:+.4f} C")
    lines.append(f"  5. 样品 FDM 变化 (案例 A, 仅顶部边界): "
                 f"均值 {dsample_a_mean:+.4f} C, "
                 f"最大绝对差 {dsample_a_max:.4f} C")
    lines.append(f"  6. 未来 k/cp/tau 重标定需求: "
                 f"{'YES (边界变更显著改变拟合)' if abs(rmse_a) > 0.1 or abs(rmse_d) > 0.1 else '待评估'}")
    lines.append("")
    lines.append("重要解释: 本任务不声称 Strategy E 更准确或为最终物理模型;")
    lines.append("旧 k/cp 在 h=5 对流-only 边界下标定, 更换顶部边界后")
    lines.append("保持旧参数仅是方向/量级/数值正确性验证。")
    lines.append("")
    lines.append("参数扫描: 无 | 参数优化: 无 | k/cp/tau/h/eps/环境重拟合: 无")
    lines.append("=" * 68)
    return "\n".join(lines)


def _plot_top(t, t_top, old_a, new_a, old_d, new_d, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax = axes[0]
    ax.plot(t, t_top, "k.", markersize=2, label="Measured Top COC")
    ax.plot(old_a["t_array"], old_a["T_top_surface_arr"],
            label="OLD h=5 convection only", color="#1f77b4")
    ax.plot(new_a["t_array"], new_a["T_top_FDM_C"],
            label="NEW h=10 + radiation", color="#d62728")
    ax.set_title("Reference Case A  (k=0.0165, cp=900, tau=0)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Top COC temperature (C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, t_top, "k.", markersize=2, label="Measured Top COC")
    ax.plot(old_d["t_array"], old_d["T_top_observed_predicted_C"],
            label="OLD h=5 conv + lag", color="#1f77b4")
    ax.plot(new_d["t_array"], new_d["T_top_observed_predicted_C"],
            label="NEW h=10 + rad + lag", color="#d62728")
    ax.set_title("Reference Case D  (k=0.0165, cp=800, tau=1.0)")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("72C old (h=5 conv-only) vs new (h=10 + radiation) boundary")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_sample(t, old_a, new_a, old_d, new_d, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax = axes[0]
    ax.plot(old_a["t_array"], old_a["T_sample_arr"],
            label="OLD h=5 convection only", color="#1f77b4")
    ax.plot(new_a["t_array"], new_a["T_sample_FDM_C"],
            label="NEW h=10 + radiation", color="#d62728")
    ax.set_title("Sample FDM — Reference Case A")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Sample temperature (C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(old_d["t_array"], old_d["T_sample_FDM_C"],
            label="OLD h=5 conv + lag", color="#1f77b4")
    ax.plot(new_d["t_array"], new_d["T_sample_FDM_C"],
            label="NEW h=10 + rad + lag", color="#d62728")
    ax.set_title("Sample FDM — Reference Case D")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("72C boundary effect on sample FDM prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
