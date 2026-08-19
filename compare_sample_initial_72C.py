#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样品空间平均 + 实验初始条件 —— 72C 协议影响分析 (对比/报告脚本, 非产品代码)。

两种效应严格分离:
  A. 样品平均效应  : 同一修正 FDM 温度场、同一初始条件,
                     仅比较 旧算术平均 vs 新控制体积空间平均;
  B. 初始条件效应  : 同一修正 FDM、同一新样品提取,
                     仅比较 T_initial=25  vs  T_initial=第一个内部温度。

另输出:
  - 72C 对齐数据首值诊断;
  - 初始条件差异随时间的衰减 (1.0/0.5/0.1 C 阈值时间);
  - 前几秒模型 vs 实测顶温度表 (仅诊断, 不解释为校准)。

不做任何 k_eff / cp_eff 拟合; 不修改实测 T_top_measured。
"""

from pathlib import Path

import numpy as np
import pandas as pd

import fv_reference
import heat_model
from heat_model import BARE_TOP_COC_LAYERS, DEFAULT_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parent
ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "72C"
    / "aligned_internal_top_temperature.csv"
)

H_CONV = 5.0
T_AMB = 25.0


def metrics(t, d):
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    i = int(np.argmax(np.abs(d)))
    return {
        "max_abs": float(np.max(np.abs(d))),
        "time_of_max": float(t[i]),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
    }


def report(name, t, d):
    m = metrics(t, d)
    print(f"      {name}: max_abs={m['max_abs']:.5f} C at t="
          f"{m['time_of_max']:.1f} s, MAE={m['mae']:.5f}, RMSE={m['rmse']:.5f}")


def first_cross_time(t, d, level):
    """|d| 首次降到 level 以下的时间; 若从未降到该值以下返回 None。"""
    t = np.asarray(t, float)
    a = np.abs(np.asarray(d, float))
    idx = np.where(a < level)[0]
    return float(t[idx[0]]) if idx.size else None


def main():
    print("=" * 78)
    print("样品空间平均 + 实验初始条件 —— 72C 协议影响分析")
    print("=" * 78)

    aligned = pd.read_csv(ALIGNED_CSV)
    t_proto = aligned["time_s"].to_numpy(dtype=float)
    T_int = aligned["T_internal_interpolated_C"].to_numpy(dtype=float)
    T_top_meas = aligned["T_top_measured_C"].to_numpy(dtype=float)

    # ----------------------------------------------------------
    # 0. 首值诊断
    # ----------------------------------------------------------
    print("\n[0] 72C aligned initial values")
    print(f"    first aligned time: {t_proto[0]:.1f} s")
    print(f"    T_internal first:   {T_int[0]:.3f} C")
    print(f"    T_top_measured first: {T_top_meas[0]:.3f} C")
    print(f"    T_internal - T_top_measured: {T_int[0] - T_top_meas[0]:+.3f} C")

    # ----------------------------------------------------------
    # A. 样品平均效应 (同一场, 同一初始=auto)
    # ----------------------------------------------------------
    print("\n[A] 样品平均效应 (同一修正场, T_initial=auto=27.64)")
    r_A = fv_reference.corrected_run(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, t_proto, T_int,
        h_conv=H_CONV, T_air_ambient=T_AMB, save_dt=0.1,
        T_initial_C=float(T_int[0]), return_fields=True,
    )
    ms = r_A["mesh"]
    fields = r_A["T_fields"]
    old_mask = ms.idx_sample  # 旧算术平均掩码 (185-200 um, 不含 180 um)
    old_sample = fields[:, old_mask].mean(axis=1)
    new_sample = fields @ ms.sample_weights
    d_sample = new_sample - old_sample
    report("T_sample old-mean vs new-spatial", r_A["t_array"], d_sample)
    print(f"    weights (bare-top 5 um): "
          f"{np.round(ms.sample_weights[ms.sample_weights > 0], 4).tolist()}")
    print(f"    sum(weights) = {ms.sample_weights.sum():.12f}; "
          f"integral width = 20.0 um")

    # ----------------------------------------------------------
    # B. 初始条件效应 (同一修正场, 同一新样品提取)
    # ----------------------------------------------------------
    print("\n[B] 初始条件效应 (同一修正场, 新样品提取)")
    r_B25 = heat_model.run_simulation(
        t_proto, T_int, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        h_conv=H_CONV, T_air_ambient=T_AMB, save_dt=0.1, T_initial_C=25.0,
    )
    r_Ba = heat_model.run_simulation(
        t_proto, T_int, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        h_conv=H_CONV, T_air_ambient=T_AMB, save_dt=0.1,
        T_initial_C=float(T_int[0]),
    )
    assert np.array_equal(r_B25["t_array"], r_Ba["t_array"])
    tB = r_B25["t_array"]

    for label, span in (("first 10 s", 10.0), ("first 30 s", 30.0),
                        ("full 299 s", 299.0)):
        mask = tB <= span
        print(f"  {label}:")
        report("T_sample     25 vs auto",
               tB[mask], r_Ba["T_sample_arr"][mask] - r_B25["T_sample_arr"][mask])
        report("T_top_surface 25 vs auto",
               tB[mask],
               r_Ba["T_top_surface_arr"][mask] - r_B25["T_top_surface_arr"][mask])

    # 衰减时间
    print("  difference decay (|T_auto - T_25|):")
    for name, a, b in (("T_sample", r_Ba["T_sample_arr"], r_B25["T_sample_arr"]),
                       ("T_top_surface", r_Ba["T_top_surface_arr"],
                        r_B25["T_top_surface_arr"])):
        line = f"    {name}:"
        for level in (1.0, 0.5, 0.1):
            tc = first_cross_time(tB, a - b, level)
            line += f" below {level} C -> {tc if tc is not None else 'NEVER'}"
        print(line)

    # ----------------------------------------------------------
    # C. 前几秒模型 vs 实测 (仅诊断)
    # ----------------------------------------------------------
    print("\n[C] first-few-seconds diagnostic (NOT validation):")
    print("    time  T_internal  T_top_meas  T_top_pred25  T_top_pred_auto")
    for i in range(min(6, len(tB))):
        j = int(np.argmin(np.abs(t_proto - tB[i])))
        print(f"    {tB[i]:5.1f}  {T_int[j]:9.3f}  {T_top_meas[j]:10.3f}  "
              f"{r_B25['T_top_surface_arr'][i]:12.4f}  "
              f"{r_Ba['T_top_surface_arr'][i]:14.4f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
