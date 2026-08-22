#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — 滞后放置架构比较 (run_lag_placement_comparison.py)
=============================================================

72C 校准数据上比较三种滞后放置架构 (固定 k=0.055, cp=1200):

    O = output-side  (tau 只作用 T_top_observed)
    I = input-side   (tau 过滤底部 Dirichlet 边界, 无输出滤波器)
    S = shared       (同一 tau 输入侧 + 输出侧)

tau 网格 0.0-12.0 s (步长 0.5, 25 值)。每个架构用最小 72C RMSE 选最佳 tau。
目标 = 顶部观察预测 (查询轴=实测时间) vs 实测 Top COC。不基于样品选参。
不重拟合 k/cp; 不引入独立 tau_input/tau_output; 不修改物理; 不覆盖旧输出。

输出 (gitignored):
    model_comparison_output/lag_placement_comparison_v1/
        72C_calibration/
            lag_placement_72C_scan.csv        (3 x 25 = 75 行)
            lag_placement_72C_best.csv        (每架构最佳 tau)
            lag_placement_72C_rmse_vs_tau.png
            lag_placement_72C_metadata.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.utilities import lag_placement_comparison_model as lpm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_CSV = (PROJECT_ROOT / "temperature_alignment_output" / "72C"
               / "aligned_internal_top_temperature.csv")
OUTPUT_ROOT = (PROJECT_ROOT / "model_comparison_output"
               / "lag_placement_comparison_v1")
OUT72 = OUTPUT_ROOT / "72C_calibration"


def load_72c():
    df = pd.read_csv(ALIGNED_CSV)
    t = df["time_s"].to_numpy(float)
    t_int = df["T_internal_interpolated_C"].to_numpy(float)
    t_top = df["T_top_measured_C"].to_numpy(float)
    return t, t_int, t_top


def main():
    t0 = time.perf_counter()
    OUT72.mkdir(parents=True, exist_ok=True)

    t_proto, t_int, t_top_meas = load_72c()
    env_info = cr.infer_environment_from_initial_top_measurement(
        t_top_meas, time_s=t_proto)
    T_env = env_info["T_environment_C"]
    print(f"[72C] rows={len(t_proto)}, t=[{t_proto[0]:.1f}, "
          f"{t_proto[-1]:.1f}] s, T_env={T_env:.2f} C, "
          f"T_initial(internal)={t_int[0]:.2f} C")

    # ---- 扫描 3 架构 x 25 tau ----
    rows, summary = lpm.run_72c_comparison(
        t_proto, t_int, t_top_meas, T_env)
    df = pd.DataFrame(rows)
    df.to_csv(OUT72 / "lag_placement_72C_scan.csv", index=False)

    best_df = pd.DataFrame(summary)
    best_df.to_csv(OUT72 / "lag_placement_72C_best.csv", index=False)

    # ---- 每个架构的 tau 剖面 ----
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"O": "#1f77b4", "I": "#d62728", "S": "#2ca02c"}
    for arch in ("O", "I", "S"):
        sub = df[df["architecture"] == arch].sort_values("tau_lag_s")
        ax.plot(sub["tau_lag_s"], sub["RMSE_72C_C"], "o-", color=colors[arch],
                label=f"{arch} ({lpm.ARCHITECTURES[arch]['label']})")
    ax.axvline(lpm.FROZEN_TAU_S, color="gray", ls="--", lw=1,
               label=f"frozen tau={lpm.FROZEN_TAU_S:.1f} (output-side)")
    ax.set_xlabel("tau [s]")
    ax.set_ylabel("72C RMSE [C]")
    ax.set_title("Lag placement comparison — 72C RMSE vs tau\n"
                 "(frozen k=0.055, cp=1200; h=10 + nonlinear radiation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT72 / "lag_placement_72C_rmse_vs_tau.png", dpi=150)
    plt.close(fig)

    # ---- 元数据 ----
    meta = {
        "phase": "B",
        "purpose": "lag placement architecture comparison (72C calibration)",
        "frozen_k_cp": {"k_eff_W_mK": lpm.FROZEN_K_W_MK,
                        "cp_eff_J_kgK": lpm.FROZEN_CP_J_KGK},
        "tau_grid": {"values": list(lpm.TAU_GRID_S),
                     "n": len(lpm.TAU_GRID_S), "max_s": lpm.TAU_MAX_S},
        "architectures": {
            "O": "output-side lag on T_top_observed only",
            "I": "input-side lag filtering bottom Dirichlet (no output "
                 "filter)",
            "S": "shared single tau (input + output)",
        },
        "selection_rule": "min 72C RMSE per architecture; never based on "
                          "PCR sample prediction",
        "objective": "T_top observed prediction interpolated to measured "
                     "time vs measured Top COC",
        "no_refit": True,
        "no_independent_tau_in_out": True,
        "best_per_architecture": summary,
        "TAU_BOUNDARY_WARNING": [s["architecture"] for s in summary
                                 if s["TAU_BOUNDARY_WARNING"]],
    }
    (OUT72 / "lag_placement_72C_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 文本摘要 ----
    lines = []
    lines.append("=" * 70)
    lines.append("LAG PLACEMENT COMPARISON — 72C CALIBRATION (Phase B)")
    lines.append("=" * 70)
    lines.append(f"环境: {T_env:.2f} C (第一个有效实测 Top COC); "
                 f"初始内部: {t_int[0]:.2f} C")
    lines.append(f"固定: k={lpm.FROZEN_K_W_MK}, cp={lpm.FROZEN_CP_J_KGK}; "
                 f"tau 网格 {lpm.TAU_GRID_S[0]}-{lpm.TAU_GRID_S[-1]} s "
                 f"步长 0.5 ({len(lpm.TAU_GRID_S)} 值)")
    lines.append("目标: 顶部观察预测 (实测时间轴) vs 实测 Top COC; "
                 "每架构最小 RMSE 选 tau")
    lines.append("")
    for s in summary:
        warn = "  <-- TAU_BOUNDARY_WARNING (tau 落网格上界)" \
            if s["TAU_BOUNDARY_WARNING"] else ""
        lines.append(
            f"[{s['architecture']}] best tau={s['best_tau_s']:.1f} s, "
            f"RMSE={s['best_RMSE_C']:.4f} C, MAE={s['best_MAE_C']:.4f}, "
            f"mean={s['best_mean_residual_C']:+.4f}, "
            f"median_abs={s['best_median_abs_residual_C']:.4f}, "
            f"max_abs={s['best_max_abs_residual_C']:.4f}{warn}")
    # 对比冻结候选 (O, tau=8.5)
    row_o85 = df[(df["architecture"] == "O")
                 & (df["tau_lag_s"] == lpm.FROZEN_TAU_S)]
    if len(row_o85):
        r = row_o85.iloc[0]
        lines.append("")
        lines.append(f"参考 (O, tau={lpm.FROZEN_TAU_S:.1f}): "
                     f"RMSE={r['RMSE_72C_C']:.4f} C (冻结候选基线)")
    lines.append("")
    lines.append("tau=0 对所有架构严格相同 (一阶滞后恒等): "
                 f"{df[df['tau_lag_s'] == 0]['RMSE_72C_C'].nunique() == 1}")
    lines.append("不重拟合 k/cp; 不基于 PCR 样品选参; 物理未修改; "
                 "无旧输出被覆盖。")
    txt = "\n".join(lines)
    (OUT72 / "lag_placement_72C_summary.txt").write_text(
        txt, encoding="utf-8")
    print(txt)
    print(f"[done] elapsed {time.perf_counter() - t0:.1f} s; "
          f"rows={len(df)}")


if __name__ == "__main__":
    main()
