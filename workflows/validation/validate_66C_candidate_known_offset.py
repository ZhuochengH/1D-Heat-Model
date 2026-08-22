#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
66C 候选 — 已知偏移 (Setpoint=90C 事件 +1 s) 零重拟合验证 V2
============================================================

科学目的
--------
上一轮多数据集验证因 60C/72C internal 工作簿无绝对时间戳而将同步判定为
UNCERTAIN (诊断 ONLY)。实验者现提供关键计时信息:

    Top COC 温度计记录在内部协议进入 Setpoint=90.000 C 后约 1.0 s 开始。

因此同步不是未知, 而是**实验已知的事件锚定规则**:

    INTERNAL SETPOINT = 90.000 C   (协议列, 不是实测温度 90 C)
    Top COC 记录 t=0  <->  internal 时间 = t90 + 1.0 s

    synchronization_rule = SETPOINT_90C_EVENT_PLUS_1S
    status              = KNOWN_PHYSICAL_OFFSET
    experimental_offset = +1.0 s  (硬实验输入, 绝不优化)
    optimized_shift     = 0.0 s

锚点定义 (任务 #2-3):
    - 使用协议 Setpoint 列, 不是 Zone 1 实测温度;
    - 取首个进入 Setpoint=90.000 C 的过渡行 (前一行 != 90);
    - 两个工作簿均只有一次进入 90 C 的过渡 (协议起始), 无歧义。

时间映射 (任务 #9-10):
    t_top_on_model_axis = t90_rel + 1.0 s + t_top_rel
    其中 t_top_rel = RECTime - RECTime_first (实际采样时间戳)。

热历史 (任务 #8):
    FDM 从 internal 首点 (t=0) 连续运行完整热历史, 穿过 90C 事件与
    Top 记录起点, 绝不在 Top 起点重置。

环境 (任务 #12):
    60C/72C Top 首点仍为环境温度 (25.35 / 26.40 C, 芯片表面未加热),
    因此 T_environment = 第一个有效实测 Top (INITIAL_MEASURED_TOP),
    恒定, 不优化。

模型 (完全锁定, 零重拟合):
    k=0.0675, cp=700, rho=1020, tau=8.0, h=10, eps=0.90,
    sigma=5.670374419e-8, F=1.0, BARE_TOP_COC_LAYERS。

与旧假设-t0 结果对比 (任务 #16):
    60C old RMSE 1.7405 / mean -0.6348
    72C old RMSE 1.2447 / mean -0.1217
    仅用于理解同步效应; 新已知偏移规则权威, 无论 RMSE 变好变坏。

权威状态更新 (任务 #19):
    60C: authoritative_validation = YES (锚点明确找到)
    72C: authoritative_validation = YES (锚点明确找到)
    3s : 保留权威参考 1.0643 C (不重跑)
    66C: CALIBRATION 参考 0.6368 C

输出 (不覆盖 v1):
    calibrated_model_output/66C_candidate_known_offset_validation_v2/
        60C/ 72C/ comparison/

用法:
    uv run python validate_66C_candidate_known_offset.py
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
from workflows.validation.validate_66C_candidate_multi_dataset import (
    load_top_series,
    load_internal_series,
    DATASETS,
    K_EFF,
    CP_EFF,
    RHO_COC,
    TAU_TOP,
    H_CONV,
    EPS,
    SIGMA,
    F_VIEW,
    AMBIENT_UPPER_C,
    TEMPERATURE_BANDS,
    quality_label,
)
from thermal_model.utilities.predict_sample_from_internal_temperature import _find_column

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "calibrated_model_output"
               / "66C_candidate_known_offset_validation_v2")
OLD_V1_ROOT = (PROJECT_ROOT / "calibrated_model_output"
               / "66C_candidate_multi_dataset_validation_v1")

SAVE_DT = 0.1
EXPERIMENTAL_OFFSET_S = 1.0     # 实验已知: Top 记录在 90C 事件后 1.0 s 开始
SETPOINT_90_C = 90.0            # 协议锚点
SETPOINT_COL = "Setpoint (°C)"

CANDIDATE_ID = "66C_RECALIBRATED_CANDIDATE_V1"
SYNC_RULE = "SETPOINT_90C_EVENT_PLUS_1S"
SYNC_STATUS = "KNOWN_PHYSICAL_OFFSET"

# 旧 (assumed-t0) 参考值 (只读)
OLD_60C_RMSE = 1.7405
OLD_60C_MEAN = -0.6348
OLD_72C_RMSE = 1.2447
OLD_72C_MEAN = -0.1217
REF_66C_CALIB_RMSE = 0.6368
REF_3S_RMSE = 1.0643

# 66C 校准 mean residual (来自上一任务 best_fit_summary)
REF_66C_MEAN_RESIDUAL = 0.0316
REF_3S_MEAN_RESIDUAL = 0.0359


def assert_locked_parameters():
    """运行时断言: 参数与 66C 最优完全一致。"""
    checks = [("k_eff", K_EFF, 0.0675), ("cp_eff", CP_EFF, 700.0),
              ("rho_COC", RHO_COC, 1020.0), ("tau_top", TAU_TOP, 8.0),
              ("h_conv", H_CONV, 10.0), ("epsilon", EPS, 0.90),
              ("sigma_SB", SIGMA, 5.670374419e-8),
              ("view_factor", F_VIEW, 1.0)]
    for name, val, ref in checks:
        if abs(float(val) - float(ref)) > 1e-12:
            raise RuntimeError(
                f"锁定参数断言失败: {name}={val} != {ref}; 禁止任何参数修改。")


# ============================================================
# Setpoint=90 锚点 (任务 #2-3)
# ============================================================

def resolve_setpoint_column(path):
    """解析 Setpoint 列 (空格折叠容错)。"""
    df = pd.read_excel(path, sheet_name="Extracted_Data", nrows=3)
    col = _find_column(df, SETPOINT_COL)
    if col is None:
        raise KeyError(f"找不到 Setpoint 列 {SETPOINT_COL!r}; "
                       f"可用: {list(df.columns)}")
    return col


def find_setpoint90_transition(path):
    """定位首个进入 Setpoint=90.000 C 的过渡行 (前一行 != 90)。

    使用协议 Setpoint 列, 绝不用实测温度。
    返回 dict: column / index / time_s_raw / t90_rel / prev_setpoint /
                n_transitions。
    """
    df = pd.read_excel(path, sheet_name="Extracted_Data")
    col = resolve_setpoint_column(path)
    t_raw = pd.to_numeric(df["Time(s)"], errors="coerce").to_numpy(float)
    sp = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    ok = np.isfinite(t_raw) & np.isfinite(sp)
    t_raw, sp = t_raw[ok], sp[ok]
    t_rel = t_raw - t_raw[0]

    transitions = np.where((sp == SETPOINT_90_C)
                           & (np.roll(sp, 1) != SETPOINT_90_C))[0]
    if transitions.size == 0:
        raise ValueError(f"{path}: 未找到 Setpoint=90 过渡。")
    idx = int(transitions[0])
    prev = float(sp[idx - 1]) if idx > 0 else None
    return {
        "setpoint_column": col,
        "transition_index": idx,
        "time_s_raw": float(t_raw[idx]),
        "t90_rel_s": float(t_rel[idx]),
        "prev_setpoint_C": prev,
        "n_transitions": int(transitions.size),
    }


# ============================================================
# 已知偏移评估
# ============================================================

def evaluate_known_offset(top, internal, t90_rel, save_dt=SAVE_DT):
    """完整热历史 FDM + 输出侧滞后 + 已知偏移时间映射 + 指标。

    时间映射: t_mapped = t90_rel + 1.0 + t_top_rel (任务 #9-10)。
    只比较映射时间位于 internal 历史内的 Top 点 (任务 #11)。
    """
    assert_locked_parameters()
    t_int = internal["t_rel"]
    T_int = internal["T"]
    t_top_rel = top["t_rel"]
    T_top = top["T"]

    t_top_start = t90_rel + EXPERIMENTAL_OFFSET_S
    t_mapped = t_top_start + t_top_rel

    # 重叠窗口: 映射时间必须在 internal 历史范围内 (不外推)
    m = (t_mapped >= t_int[0] - 1e-9) & (t_mapped <= t_int[-1] + 1e-9)
    t_top_used = t_top_rel[m]
    T_top_used = T_top[m]
    t_mapped_used = t_mapped[m]
    if len(t_top_used) < 2:
        raise ValueError("映射后重叠窗口内 Top 点不足。")

    # 环境: 第一个有效实测 Top (须为环境温度, 芯片表面未加热)
    t_env = float(T_top[0])
    if t_env >= AMBIENT_UPPER_C:
        raise ValueError(
            f"Top 首点 {t_env:.1f} C >= {AMBIENT_UPPER_C} C: 不能用作环境。")
    T_init = float(T_int[0])

    # FDM: 完整 internal 历史 (从 t=0 连续运行; 不重置)
    mats = cr.make_convection_radiation_materials(K_EFF, CP_EFF, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_int, bottom_temperature_C=T_int, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=t_env,
        T_surroundings_C=t_env, save_dt=save_dt, T_initial_C=T_init)
    t_arr = result["t_array"]
    T_top_fdm = result["T_top_surface_arr"]
    T_top_obs = apply_first_order_lag(t_arr, T_top_fdm, TAU_TOP)

    # 插值到映射后的实测 Top 时间 (查询轴 = 映射时间)
    T_pred = np.interp(t_mapped_used, t_arr, T_top_obs)
    T_fdm_at_top = np.interp(t_mapped_used, t_arr, T_top_fdm)
    T_int_at_top = np.interp(t_mapped_used, t_int, T_int)

    resid = T_pred - T_top_used
    n = len(resid)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((T_top_used - np.mean(T_top_used)) ** 2))
    metrics = {
        "n_points": n,
        "RMSE_C": float(np.sqrt(np.mean(resid ** 2))),
        "MAE_C": float(np.mean(np.abs(resid))),
        "mean_residual_C": float(np.mean(resid)),
        "median_abs_residual_C": float(np.median(np.abs(resid))),
        "residual_std_C": float(np.std(resid)),
        "max_abs_residual_C": float(np.max(np.abs(resid))),
        "R_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "measured_top_min_C": float(np.min(T_top_used)),
        "measured_top_max_C": float(np.max(T_top_used)),
        "predicted_top_min_C": float(np.min(T_pred)),
        "predicted_top_max_C": float(np.max(T_pred)),
    }

    # regime 诊断 (复用项目分类器)
    regimes = _regime_labels(t_mapped_used, T_int_at_top, T_top_used)
    regime_metrics = {}
    for short, rg in (("heating", "TRANSIENT_HEATING"),
                      ("cooling", "TRANSIENT_COOLING"),
                      ("settling", "SETTLING")):
        mm = regimes == rg
        if mm.sum():
            regime_metrics[f"{short}_n"] = int(mm.sum())
            regime_metrics[f"{short}_RMSE_C"] = float(
                np.sqrt(np.mean(resid[mm] ** 2)))
            regime_metrics[f"{short}_mean_residual_C"] = float(
                np.mean(resid[mm]))
        else:
            regime_metrics[f"{short}_n"] = 0
            regime_metrics[f"{short}_RMSE_C"] = np.nan
            regime_metrics[f"{short}_mean_residual_C"] = np.nan

    # 温度带诊断
    band_metrics = {}
    for lo, hi, name in TEMPERATURE_BANDS:
        mm = (T_top_used >= lo) & (T_top_used < hi)
        if mm.sum():
            band_metrics[f"{name}_n"] = int(mm.sum())
            band_metrics[f"{name}_RMSE_C"] = float(
                np.sqrt(np.mean(resid[mm] ** 2)))
            band_metrics[f"{name}_mean_residual_C"] = float(
                np.mean(resid[mm]))
        else:
            band_metrics[f"{name}_n"] = 0
            band_metrics[f"{name}_RMSE_C"] = np.nan
            band_metrics[f"{name}_mean_residual_C"] = np.nan

    return {
        "t_mapped": t_mapped_used,
        "t_top_rel": t_top_used,
        "T_top_measured": T_top_used,
        "T_internal_at_top": T_int_at_top,
        "T_top_fdm_raw": T_fdm_at_top,
        "T_top_predicted_lagged": T_pred,
        "residual": resid,
        "regimes": regimes,
        "T_env_C": t_env,
        "T_initial_C": T_init,
        "environment_source": "INITIAL_MEASURED_TOP",
        "t_top_start_model_s": t_top_start,
        "n_top_before_overlap": int(len(t_top_rel)),
        "n_top_after_overlap": int(len(t_top_used)),
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "band_metrics": band_metrics,
    }


def _regime_labels(t, T_int, T_top):
    """复用项目 regime 分类器 (与既有脚本一致)。"""
    from thermal_model.utilities.validate_frozen_model_two_new_bare_top_datasets import _regime_labels as _r
    return _r(t, T_int, T_top)


# ============================================================
# 输出
# ============================================================

def write_dataset_outputs(label, cfg, anchor, ev, out_dir):
    """写入 60C/72C 已知偏移验证文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = ev["metrics"]
    reg = ev["regime_metrics"]

    # validation_trace_known_offset.csv
    trace = pd.DataFrame({
        "internal_model_time_s": ev["t_mapped"],
        "top_relative_time_s": ev["t_top_rel"],
        "top_mapped_model_time_s": ev["t_mapped"],
        "measured_top_C": ev["T_top_measured"],
        "internal_interpolated_C": ev["T_internal_at_top"],
        "top_FDM_raw_C": ev["T_top_fdm_raw"],
        "top_predicted_lagged_C": ev["T_top_predicted_lagged"],
        "residual_C": ev["residual"],
        "setpoint90_event_time_s": anchor["t90_rel_s"],
        "experimental_offset_s": EXPERIMENTAL_OFFSET_S,
    })
    trace.to_csv(out_dir / "validation_trace_known_offset.csv", index=False)

    # validation_summary_known_offset.csv
    summary = {
        "dataset": label,
        "candidate_id": CANDIDATE_ID,
        "synchronization_status": SYNC_STATUS,
        "synchronization_rule": SYNC_RULE,
        "experimental_offset_s": EXPERIMENTAL_OFFSET_S,
        "optimized_time_shift_s": 0.0,
        "time_shift_fitted": False,
        "setpoint_column": anchor["setpoint_column"],
        "setpoint90_transition_index": anchor["transition_index"],
        "setpoint90_time_s_raw": anchor["time_s_raw"],
        "setpoint90_t90_rel_s": anchor["t90_rel_s"],
        "prev_setpoint_C": anchor["prev_setpoint_C"],
        "top_start_model_time_s": ev["t_top_start_model_s"],
        "n_top_before_overlap": ev["n_top_before_overlap"],
        "n_top_after_overlap": ev["n_top_after_overlap"],
        "fitted_parameters": False,
        "k_eff": K_EFF, "cp_eff": CP_EFF, "tau_top": TAU_TOP,
        "h_conv": H_CONV, "epsilon": EPS,
        **m,
        **reg,
        **{f"band_{k}": v for k, v in ev["band_metrics"].items()},
        "environment_C": ev["T_env_C"],
        "environment_source": ev["environment_source"],
        "validation_quality": quality_label(m["RMSE_C"]),
        "authoritative_validation": True,
    }
    pd.DataFrame([summary]).to_csv(
        out_dir / "validation_summary_known_offset.csv", index=False)

    # 图: top_COC_validation_known_offset
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.plot(ev["t_mapped"], ev["T_internal_at_top"], color="#7f7f7f", lw=1.1,
            ls=":", label="Measured internal temperature")
    ax.plot(ev["t_mapped"], ev["T_top_measured"], color="#1f77b4", lw=1.6,
            label="Corrected measured Top COC (mapped)")
    ax.plot(ev["t_mapped"], ev["T_top_predicted_lagged"], color="#d62728",
            lw=1.8, label="Predicted Top COC (66C candidate, no refit)")
    ax.plot(ev["t_mapped"], ev["T_top_fdm_raw"], color="#2ca02c", lw=0.9,
            ls="--", alpha=0.7, label="Raw Top COC FDM (no lag)")
    ax.axvline(anchor["t90_rel_s"], color="#8c564b", ls=":", lw=1.4,
               label=f"Setpoint->90 C event (t={anchor['t90_rel_s']:.2f} s)")
    ax.axvline(ev["t_top_start_model_s"], color="#9467bd", ls="--", lw=1.2,
               label=f"Top recording start = t90 + 1 s "
                     f"({ev['t_top_start_model_s']:.2f} s)")
    ax.set_xlabel("Internal/model time [s] (complete thermal history)")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(f"{label} — known-offset zero-refit Top COC validation\n"
                 f"(k={K_EFF}, cp={CP_EFF:.0f}, tau={TAU_TOP:.1f} s; "
                 f"sync={SYNC_RULE}, offset={EXPERIMENTAL_OFFSET_S:+.1f} s)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "top_COC_validation_known_offset.png", dpi=150)
    fig.savefig(out_dir / "top_COC_validation_known_offset.pdf")
    plt.close(fig)

    # 图: residual_vs_time_known_offset
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.plot(ev["t_mapped"], ev["residual"], color="#d62728", lw=1.0,
            label="Residual (predicted - measured)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(ev["t_top_start_model_s"], color="#9467bd", ls="--", lw=1.2,
               label="Top recording start")
    ax.set_xlabel("Internal/model time [s]")
    ax.set_ylabel("Residual [C]")
    ax.set_title(f"{label} — residual vs time (known offset, zero-refit)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_vs_time_known_offset.png", dpi=150)
    plt.close(fig)

    return summary


def write_comparison_outputs(results, anchors):
    """组合输出: 汇总表 / 新旧对比 / 图 / 元数据 / 文本。"""
    comp = OUTPUT_ROOT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)

    # ---- known_offset_validation_summary.csv ----
    rows = [
        {"dataset": "CALIBRATION_66C", "role": "CALIBRATION",
         "authoritative": True, "RMSE_C": REF_66C_CALIB_RMSE,
         "MAE_C": np.nan, "mean_residual_C": REF_66C_MEAN_RESIDUAL,
         "R_squared": np.nan},
        {"dataset": "VALIDATION_60C", "role": "EXTERNAL_VALIDATION",
         "authoritative": True,
         "RMSE_C": results["60C"]["ev"]["metrics"]["RMSE_C"],
         "MAE_C": results["60C"]["ev"]["metrics"]["MAE_C"],
         "mean_residual_C": results["60C"]["ev"]["metrics"]
         ["mean_residual_C"],
         "R_squared": results["60C"]["ev"]["metrics"]["R_squared"]},
        {"dataset": "VALIDATION_72C", "role": "EXTERNAL_VALIDATION",
         "authoritative": True,
         "RMSE_C": results["72C"]["ev"]["metrics"]["RMSE_C"],
         "MAE_C": results["72C"]["ev"]["metrics"]["MAE_C"],
         "mean_residual_C": results["72C"]["ev"]["metrics"]
         ["mean_residual_C"],
         "R_squared": results["72C"]["ev"]["metrics"]["R_squared"]},
        {"dataset": "VALIDATION_3S_SYNCHRONIZED", "role": "EXTERNAL_VALIDATION",
         "authoritative": True, "RMSE_C": REF_3S_RMSE,
         "MAE_C": np.nan, "mean_residual_C": REF_3S_MEAN_RESIDUAL,
         "R_squared": np.nan,
         "note": "read-only reference (previous task, same sync rule)"},
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(comp / "known_offset_validation_summary.csv",
                      index=False)

    # 外部验证均值 (不含校准)
    ext = summary_df[summary_df["role"] == "EXTERNAL_VALIDATION"]
    mean_rmse = float(ext["RMSE_C"].mean())
    median_rmse = float(ext["RMSE_C"].median())
    worst_rmse = float(ext["RMSE_C"].max())

    # ---- old_vs_new_sync_comparison.csv ----
    pd.DataFrame([
        {"dataset": "60C", "old_sync": "SIMULTANEOUS_START_RELATIVE_T0_ASSUMED",
         "new_sync": SYNC_RULE, "old_RMSE_C": OLD_60C_RMSE,
         "new_RMSE_C": results["60C"]["ev"]["metrics"]["RMSE_C"],
         "delta_RMSE_C": results["60C"]["ev"]["metrics"]["RMSE_C"] - OLD_60C_RMSE,
         "old_mean_residual_C": OLD_60C_MEAN,
         "new_mean_residual_C":
             results["60C"]["ev"]["metrics"]["mean_residual_C"]},
        {"dataset": "72C", "old_sync": "SIMULTANEOUS_START_RELATIVE_T0_ASSUMED",
         "new_sync": SYNC_RULE, "old_RMSE_C": OLD_72C_RMSE,
         "new_RMSE_C": results["72C"]["ev"]["metrics"]["RMSE_C"],
         "delta_RMSE_C": results["72C"]["ev"]["metrics"]["RMSE_C"] - OLD_72C_RMSE,
         "old_mean_residual_C": OLD_72C_MEAN,
         "new_mean_residual_C":
             results["72C"]["ev"]["metrics"]["mean_residual_C"]},
    ]).to_csv(comp / "old_vs_new_sync_comparison.csv", index=False)

    # ---- final_multi_dataset_RMSE_overview.png ----
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    items = [
        ("66C calib", REF_66C_CALIB_RMSE, "#1f77b4"),
        ("60C", results["60C"]["ev"]["metrics"]["RMSE_C"], "#2ca02c"),
        ("72C", results["72C"]["ev"]["metrics"]["RMSE_C"], "#2ca02c"),
        ("3s", REF_3S_RMSE, "#2ca02c"),
    ]
    for name, rmse, color in items:
        ax.bar(name, rmse, color=color, width=0.55)
        ax.text(name, rmse + 0.02, f"{rmse:.3f} C", ha="center", fontsize=9)
    ax.axhline(1.5, color="gray", ls=":", lw=1, label="EXCELLENT (1.5 C)")
    ax.axhline(2.5, color="gray", ls="--", lw=1, label="GOOD (2.5 C)")
    ax.set_ylabel("Top COC RMSE [C]")
    ax.set_title("66C candidate — known-offset multi-dataset zero-refit RMSE\n"
                 "(all external = authoritative; 66C = calibration)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(comp / "final_multi_dataset_RMSE_overview.png", dpi=150)
    plt.close(fig)

    # ---- final_mean_residual_overview.png ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pts = [
        ("60C", results["60C"]["ev"]["metrics"]["mean_residual_C"], "#2ca02c"),
        ("66C", REF_66C_MEAN_RESIDUAL, "#1f77b4"),
        ("72C", results["72C"]["ev"]["metrics"]["mean_residual_C"], "#2ca02c"),
        ("3s", REF_3S_MEAN_RESIDUAL, "#2ca02c"),
    ]
    for x, (name, mr, color) in enumerate(pts):
        ax.plot(x, mr, "o", color=color, ms=9,
                label=f"{name} ({'+' if mr >= 0 else ''}{mr:.3f} C)")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([p[0] for p in pts])
    ax.set_xlabel("Protocol / nominal temperature level")
    ax.set_ylabel("Mean residual [C]")
    ax.set_title("Cross-temperature mean residual — known-offset validation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(comp / "final_mean_residual_overview.png", dpi=150)
    plt.close(fig)

    return summary_df, mean_rmse, median_rmse, worst_rmse


def cross_temperature_bias(results):
    """跨温度偏置判定。"""
    pts = [("60C", results["60C"]["ev"]["metrics"]["mean_residual_C"]),
           ("66C", REF_66C_MEAN_RESIDUAL),
           ("72C", results["72C"]["ev"]["metrics"]["mean_residual_C"]),
           ("3s", REF_3S_MEAN_RESIDUAL)]
    mr = np.array([v for _, v in pts])
    diffs = np.diff(mr)
    max_abs_diff = float(np.max(np.abs(diffs)))
    verdict = "INCONCLUSIVE"
    if max_abs_diff <= 0.4:
        verdict = "NO"
    elif (np.all(diffs >= 0.2) or np.all(diffs <= -0.2)):
        verdict = "YES"
    else:
        verdict = "WEAK"
    evidence = [f"{name}: mean residual {val:+.3f} C" for name, val in pts]
    evidence.append(f"max consecutive |diff| = {max_abs_diff:.3f} C")
    return verdict, evidence


def classify_transfer(results, mean_rmse, worst_rmse, bias):
    """基于 ONLY 热学验证的转移分类。"""
    if worst_rmse <= 2.0:
        cls = "STRONG_MULTI_DATASET_TRANSFER"
        reason = (f"all authoritative external RMSE <= 2.0 C "
                  f"(worst {worst_rmse:.3f} C, mean {mean_rmse:.3f} C); "
                  f"bias verdict: {bias}")
    elif worst_rmse <= 4.0:
        cls = "ACCEPTABLE_MULTI_DATASET_TRANSFER"
        reason = (f"all authoritative external RMSE within a few degrees "
                  f"(worst {worst_rmse:.3f} C); bias verdict: {bias}")
    else:
        cls = "MODEL_TRANSFER_FAILURE"
        reason = f"worst authoritative external RMSE {worst_rmse:.3f} C > 4 C"
    return cls, reason


# ============================================================
# main
# ============================================================

def main():
    assert_locked_parameters()
    results = {}
    anchors = {}
    for ds_key in ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO"):
        cfg = DATASETS[ds_key]
        label = ds_key.split("_")[1]
        print(f"\n=== {label}C — known-offset validation ===")
        anchor = find_setpoint90_transition(cfg["int_path"])
        print(f"  Setpoint col={anchor['setpoint_column']}; "
              f"t90 Time(s)={anchor['time_s_raw']:.3f}; "
              f"t90_rel={anchor['t90_rel_s']:.3f} s; "
              f"transitions={anchor['n_transitions']}; "
              f"prev={anchor['prev_setpoint_C']}")
        top = load_top_series(cfg["top_path"])
        internal = load_internal_series(cfg["int_path"])
        ev = evaluate_known_offset(top, internal, anchor["t90_rel_s"])
        print(f"  Top start on model axis = {ev['t_top_start_model_s']:.3f} s")
        print(f"  Top points: {ev['n_top_before_overlap']} -> "
              f"{ev['n_top_after_overlap']} after overlap")
        m = ev["metrics"]
        print(f"  RMSE={m['RMSE_C']:.4f} | MAE={m['MAE_C']:.4f} | "
              f"mean={m['mean_residual_C']:+.4f} | R2={m['R_squared']:.4f}")
        print(f"  regime: heating={ev['regime_metrics']['heating_RMSE_C']:.3f} "
              f"cooling={ev['regime_metrics']['cooling_RMSE_C']:.3f} "
              f"settling={ev['regime_metrics']['settling_RMSE_C']:.3f}")
        out_dir = OUTPUT_ROOT / label
        summary = write_dataset_outputs(label, cfg, anchor, ev, out_dir)
        results[label] = {"ev": ev, "summary": summary, "anchor": anchor}
        anchors[label] = anchor

    # 组合输出
    comp = OUTPUT_ROOT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    summary_df, mean_rmse, median_rmse, worst_rmse = \
        write_comparison_outputs(results, anchors)

    bias, evidence = cross_temperature_bias(results)
    cls, reason = classify_transfer(results, mean_rmse, worst_rmse, bias)

    # promotion recommendation
    if cls == "STRONG_MULTI_DATASET_TRANSFER" and bias == "NO":
        promo = "YES"
        promo_reason = ("all external validation EXCELLENT/<=2 C with no "
                        "systematic temperature bias")
    elif cls == "STRONG_MULTI_DATASET_TRANSFER":
        promo = "YES_WITH_LIMITATION"
        promo_reason = (f"strong transfer but bias verdict {bias}; "
                        "reviewer confirmation advised")
    elif cls == "ACCEPTABLE_MULTI_DATASET_TRANSFER":
        promo = "YES_WITH_LIMITATION"
        promo_reason = "acceptable transfer; review remaining bias/errors"
    else:
        promo = "NO"
        promo_reason = "transfer failure or strong systematic limitation"

    meta = {
        "candidate_id": CANDIDATE_ID,
        "timing_correction": {
            "new_experimental_information": ("Top COC recording starts "
                                             "1.0 s after internal protocol "
                                             "enters Setpoint=90.000 C"),
            "timing_offset_source": "EXPERIMENTALLY KNOWN",
            "time_offset_fitted": False,
            "synchronization_rule": SYNC_RULE,
            "synchronization_status": SYNC_STATUS,
            "experimental_offset_s": EXPERIMENTAL_OFFSET_S,
            "optimized_time_shift_s": 0.0,
        },
        "parameters": {"k_eff_W_mK": K_EFF, "cp_eff_J_kgK": CP_EFF,
                       "rho_COC": RHO_COC, "tau_top_s": TAU_TOP,
                       "h_conv_W_m2K": H_CONV, "emissivity": EPS,
                       "parameters_refitted": False,
                       "geometry": "BARE_TOP_COC_LAYERS"},
        "anchors": {label: anchors[label] for label in anchors},
        "results": {label: {
            "RMSE_C": results[label]["ev"]["metrics"]["RMSE_C"],
            "MAE_C": results[label]["ev"]["metrics"]["MAE_C"],
            "mean_residual_C":
                results[label]["ev"]["metrics"]["mean_residual_C"],
            "R_squared": results[label]["ev"]["metrics"]["R_squared"],
            "n_points": results[label]["ev"]["metrics"]["n_points"],
            "authoritative": True,
        } for label in results},
        "reference": {"66C_calibration_RMSE_C": REF_66C_CALIB_RMSE,
                      "3s_synchronized_RMSE_C": REF_3S_RMSE},
        "external_validation_stats": {
            "mean_RMSE_C": mean_rmse, "median_RMSE_C": median_rmse,
            "worst_RMSE_C": worst_rmse,
            "calibration_included": False,
        },
        "cross_temperature_bias": {"verdict": bias, "evidence": evidence},
        "classification": {"class": cls, "reason": reason},
        "promotion": {"recommendation": promo, "reason": promo_reason,
                      "frozen_model_edited": False},
        "old_vs_new": {
            "60C": {"old_RMSE_C": OLD_60C_RMSE,
                    "new_RMSE_C": results["60C"]["ev"]["metrics"]["RMSE_C"],
                    "old_mean": OLD_60C_MEAN,
                    "new_mean":
                        results["60C"]["ev"]["metrics"]["mean_residual_C"]},
            "72C": {"old_RMSE_C": OLD_72C_RMSE,
                    "new_RMSE_C": results["72C"]["ev"]["metrics"]["RMSE_C"],
                    "old_mean": OLD_72C_MEAN,
                    "new_mean":
                        results["72C"]["ev"]["metrics"]["mean_residual_C"]},
        },
    }
    (comp / "known_offset_validation_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = _summary_text(results, summary_df, mean_rmse, median_rmse,
                        worst_rmse, bias, evidence, cls, reason,
                        promo, promo_reason)
    (comp / "known_offset_validation_summary.txt").write_text(
        txt, encoding="utf-8")
    print(txt)
    return 0


def _summary_text(results, summary_df, mean_rmse, median_rmse, worst_rmse,
                  bias, evidence, cls, reason, promo, promo_reason):
    L = []
    A = L.append
    A("=" * 74)
    A("66C CANDIDATE — KNOWN-OFFSET MULTI-DATASET VALIDATION V2")
    A("=" * 74)
    A(f"sync rule: {SYNC_RULE}; experimental offset +{EXPERIMENTAL_OFFSET_S} s "
      "(hard input, not fitted); optimized shift 0.0 s")
    A(f"model: k={K_EFF}, cp={CP_EFF:.0f}, rho={RHO_COC}, tau={TAU_TOP} s; "
      "no refit")
    A("")
    for label in ("60C", "72C"):
        m = results[label]["ev"]["metrics"]
        reg = results[label]["ev"]["regime_metrics"]
        anc = results[label]["anchor"]
        A(f"[{label}C]")
        A(f"  Setpoint col={anc['setpoint_column']}; t90 Time(s)="
          f"{anc['time_s_raw']:.3f}; t90_rel={anc['t90_rel_s']:.3f} s; "
          f"transitions={anc['n_transitions']}")
        A(f"  Top start on model axis = {results[label]['ev']['t_top_start_model_s']:.3f} s")
        A(f"  points {results[label]['ev']['n_top_before_overlap']} -> "
          f"{results[label]['ev']['n_top_after_overlap']}")
        A(f"  RMSE {m['RMSE_C']:.4f} | MAE {m['MAE_C']:.4f} | "
          f"mean {m['mean_residual_C']:+.4f} | "
          f"median_abs {m['median_abs_residual_C']:.4f} | "
          f"max_abs {m['max_abs_residual_C']:.4f} | R2 {m['R_squared']:.4f}")
        A(f"  regime: heating {reg['heating_RMSE_C']:.3f} "
          f"(n={reg['heating_n']}), cooling {reg['cooling_RMSE_C']:.3f} "
          f"(n={reg['cooling_n']}), settling {reg['settling_RMSE_C']:.3f} "
          f"(n={reg['settling_n']})")
        A(f"  measured Top range [{m['measured_top_min_C']:.2f}, "
          f"{m['measured_top_max_C']:.2f}] C; quality: "
          f"{results[label]['summary']['validation_quality']}")
        A("")
    A(f"external-validation mean/median/worst RMSE: "
      f"{mean_rmse:.4f} / {median_rmse:.4f} / {worst_rmse:.4f} C "
      "(calibration excluded)")
    A(f"cross-temperature bias: {bias}")
    for e in evidence:
        A(f"  {e}")
    A("")
    A(f"classification: {cls}")
    A(f"  reason: {reason}")
    A(f"promotion recommendation: {promo}")
    A(f"  reason: {promo_reason}")
    A("  frozen_strategy_G_candidate.py NOT edited (awaiting review)")
    A("")
    A("No parameter refitting: CONFIRMED")
    A("No fitted time shift: CONFIRMED (only +1.0 s experimental offset)")
    A("Full internal thermal history preserved: CONFIRMED")
    A("No qPCR/sample information used: CONFIRMED")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    sys.exit(main())
