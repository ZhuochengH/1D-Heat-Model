#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标定裸顶模型最终运行器 (72°C 名义 V1)。

流程:
    aligned 72C 实验 (time_s / T_internal_interpolated_C /
                      T_top_measured_C)
        ↓
    BARE_TOP_COC_LAYERS (850 um, 裸顶, top 观测 x=850 um)
        ↓
    NOMINAL_BARE_TOP_CALIBRATION_V1 (k_eff=0.068, cp_eff=9200, rho=1020)
        ↓
    make_nominal_calibrated_materials() (仅改 COC, 不触碰 DEFAULT_MATERIALS)
        ↓
    直接 T_internal 底部 Dirichlet 边界 (无 a/b/tau/时间平移/偏移)
        ↓
    auto 初始条件 (T_initial = 第一个内部温度)
        ↓
    heat_model.run_simulation(...)  (唯一权威 FDM, 本脚本不含任何 FDM 方程)
        ↓
    预测输出线性插值到实测时间坐标
        ↓
    最终指标 / final_72C_thermal_trace.csv / 图 (PNG+PDF) / 元数据

本脚本不重复网格/FV/稳定性/边界物理; 不修改实测数据;
不拟合任何参数; 生成的输出全部位于 gitignored 的
    calibrated_model_output/72C_nominal_v1/
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "72C"
    / "aligned_internal_top_temperature.csv"
)
# 修正时间目标 (V3) 的最终输出目录。旧目录
# calibrated_model_output/72C_nominal_v1/ 保留为
# "legacy-selected 参数 (0.068/9200) 的诊断迹线" (PROVISIONAL)。
OUTPUT_DIR = (
    PROJECT_ROOT / "calibrated_model_output" / "72C_corrected_objective_v1"
)

H_CONV = 5.0
T_AMB = 25.0
SAVE_DT = 0.1

TIME_COL = "time_s"
TINT_COL = "T_internal_interpolated_C"
TTOP_COL = "T_top_measured_C"


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


def load_experiment(path=None):
    """加载对齐实验 (列名与扫描脚本一致)。"""
    p = Path(path) if path else ALIGNED_CSV
    if not p.is_file():
        raise FileNotFoundError(
            f"对齐数据集不存在: {p}. 请先用 "
            "align_internal_and_top_temperature.py 生成。"
        )
    df = pd.read_csv(p)
    for col in (TIME_COL, TINT_COL, TTOP_COL):
        if col not in df.columns:
            raise KeyError(f"对齐数据集缺少列 {col!r}; 实际列: {list(df.columns)}")
    t = df[TIME_COL].to_numpy(dtype=float)
    t_int = df[TINT_COL].to_numpy(dtype=float)
    t_top = df[TTOP_COL].to_numpy(dtype=float)
    return t, t_int, t_top


def compute_metrics(t, residual):
    """与扫描脚本一致的指标约定 (残差 = pred - meas)。"""
    r = np.asarray(residual, dtype=float)
    return {
        "RMSE_top_C": float(np.sqrt(np.mean(r ** 2))),
        "MAE_top_C": float(np.mean(np.abs(r))),
        "mean_residual_top_C": float(np.mean(r)),
        "max_positive_residual_C": float(np.max(r)),
        "max_negative_residual_C": float(np.min(r)),
        "max_absolute_residual_C": float(np.max(np.abs(r))),
    }


def run_nominal(time_s, t_int, t_top_meas, output_dir,
                calibration=None):
    """运行名义标定模型并导出最终 72°C 热迹线 / 图 / 元数据。

    返回 (metrics, trace_df)。
    """
    cal = calibration if calibration is not None else NOMINAL_BARE_TOP_CALIBRATION_V1
    layers = nominal_layer_stack(cal)
    mats = make_nominal_calibrated_materials(cal)
    T_initial = float(t_int[0])

    result = heat_model.run_simulation(
        time_s=time_s,
        bottom_temperature_C=t_int,
        materials=mats,
        layers=layers,
        h_conv=H_CONV, T_air_ambient=T_AMB, save_dt=SAVE_DT,
        T_initial_C=T_initial,
    )

    t_arr = result["t_array"]
    # 把 FDM 预测线性插值到实测 TIME 坐标 (time_s), 不是温度值:
    # np.interp(实测时间, FDM 时间, 预测温度)  —— 修正版
    T_top_pred = np.interp(time_s, t_arr, result["T_top_surface_arr"])
    T_sample_pred = np.interp(time_s, t_arr, result["T_sample_arr"])
    T_internal_at_meas = t_int  # 输入即实测时间坐标上的内部温度
    residual = T_top_pred - t_top_meas
    metrics = compute_metrics(time_s, residual)

    # 最终迹线 (实测时间坐标)
    trace = pd.DataFrame({
        "time_s": time_s,
        "T_internal_C": T_internal_at_meas,
        "T_top_measured_C": t_top_meas,
        "T_top_predicted_C": T_top_pred,
        "T_sample_predicted_C": T_sample_pred,
        "top_residual_C": residual,
    })
    trace.to_csv(output_dir / "final_72C_thermal_trace.csv", index=False)

    # 最终图 (PNG + PDF, 同一图形)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(time_s, t_int, color="#7f7f7f", lw=1.2, ls=":",
            label="Internal sensor input")
    ax.plot(time_s, t_top_meas, color="#d62728", lw=1.6,
            label="Top COC measured")
    ax.plot(time_s, T_top_pred, color="#1f77b4", lw=2.0,
            label="Top COC predicted")
    ax.plot(time_s, T_sample_pred, color="#2ca02c", lw=1.6, ls="--",
            label="Sample predicted (estimated, not measured)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("72°C Protocol — Calibrated Thermal Model")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "final_72C_thermal_trace.png", dpi=200)
    fig.savefig(output_dir / "final_72C_thermal_trace.pdf")
    plt.close(fig)

    # 元数据
    metadata = {
        "model_version": "bare_top_calibrated_model_v1 (corrected measurement-time objective)",
        "source_calibration": cal.source_analysis,
        "k_eff_W_mK": cal.k_eff_W_mK,
        "cp_eff_J_kgK": cal.cp_eff_J_kgK,
        "rho_COC_kg_m3": cal.rho_COC_kg_m3,
        "interpretation": cal.interpretation,
        "geometry_preset": cal.geometry_preset,
        "total_thickness_um": 850.0,
        "sample_layer_um": [180.0, 200.0],
        "top_observation_um": 850.0,
        "h_conv_W_m2K": H_CONV,
        "T_air_ambient_C": T_AMB,
        "initial_condition_mode": "auto from first T_internal",
        "calibration_objective": (
            "equal-weight top-temperature RMSE (T_top_surface vs measured)"
        ),
        "measurement_points": int(len(t_top_meas)),
        "time_range_s": [float(t_top_meas[0]), float(t_top_meas[-1])],
        "metrics": metrics,
        "git_commit": git_head(),
        "git_tag": git_describe(),
        "note": (
            "Sample temperature is model-estimated (control-volume spatial "
            "average), not directly measured. k_eff/cp_eff are system-level "
            "effective parameters, NOT intrinsic COC properties. Final "
            "output uses the corrected measurement-time objective (V3 "
            "calibration)."
        ),
    }
    (output_dir / "final_72C_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics, trace


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(ALIGNED_CSV))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t, t_int, t_top = load_experiment(args.dataset)
    print(f"[data] {args.dataset}: {len(t)} 点, t [{t[0]:.1f}, {t[-1]:.1f}] s")
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    print(f"[calibration] {cal.name}: k_eff={cal.k_eff_W_mK} "
          f"cp_eff={cal.cp_eff_J_kgK} rho={cal.rho_COC_kg_m3}")
    metrics, trace = run_nominal(t, t_int, t_top, output_dir, cal)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"[output] {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
