#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
界面有限体积修正 —— 数值影响分析 (对比/报告脚本, 非产品代码)。

内容:
  1. 裸顶解析稳态基准 (三对角直接求解 + 粗网格显式收敛);
  2. 高对比堆叠基准 (旧方案失败 / 新方案通过);
  3. 均匀材料不变性 (新旧逐位一致);
  4. 瞬态 25 -> 90 °C 阶跃对比 (60 s);
  5. 真实 72 °C 协议: 旧 vs 修正 (T_sample / T_top_surface, 总体 + 分regime);
  6. 性能 (旧 dt / 新 dt / 运行时间)。

不做任何 k_eff / cp_eff 拟合; 不做模型 vs 实测精度解释。
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_model.core import fv_reference
from thermal_model.core import heat_model
from thermal_model.core.heat_model import BARE_TOP_COC_LAYERS, DEFAULT_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "72C"
    / "aligned_internal_top_temperature.csv"
)
REGIME_CSV = (
    PROJECT_ROOT / "temperature_regime_output" / "72C"
    / "temperature_regime_labeled.csv"
)

H_CONV = 5.0
T_AMB = 25.0


def metrics(t, d):
    """对差值序列 d(t) 计算 max_abs/time/mae/rmse。"""
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    i = int(np.argmax(np.abs(d)))
    return {
        "max_abs": float(np.max(np.abs(d))),
        "time_of_max": float(t[i]),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
    }


def interp_onto(new_t, old_t, old_y):
    return np.interp(new_t, old_t, old_y)


def steady_tridiagonal(materials, layers, Tb, h_conv, T_amb, old_scheme=False):
    """稳态 FV 三对角解 (old_scheme=True 时用端点调和平均, 仅对比)。"""
    ms = heat_model.build_layer_stack(materials, layers)
    n = ms.Nx
    if old_scheme:
        k_face = 2 * ms.k[:-1] * ms.k[1:] / (ms.k[:-1] + ms.k[1:])
    else:
        k_face = ms.k_face
    h = ms.h
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = 1.0
    b[0] = Tb
    for j in range(1, n - 1):
        A[j, j - 1] = k_face[j - 1] / h[j - 1]
        A[j, j] = -(k_face[j - 1] / h[j - 1] + k_face[j] / h[j])
        A[j, j + 1] = k_face[j] / h[j]
    A[n - 1, n - 2] = k_face[-1] / h[-1]
    A[n - 1, n - 1] = -(k_face[-1] / h[-1] + h_conv)
    b[n - 1] = -h_conv * T_amb
    return np.linalg.solve(A, b), ms


def main():
    print("=" * 78)
    print("界面有限体积修正 —— 数值影响分析")
    print("=" * 78)

    # ----------------------------------------------------------
    # 1. 裸顶解析稳态基准
    # ----------------------------------------------------------
    Tb = 90.0
    q_ana, pos, T_ana = fv_reference.analytical_steady(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, Tb, H_CONV, T_AMB
    )
    T_new, ms = steady_tridiagonal(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, Tb, H_CONV, T_AMB
    )
    T_old, _ = steady_tridiagonal(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, Tb, H_CONV, T_AMB,
        old_scheme=True,
    )
    print("\n[1] 裸顶解析稳态基准 (Tb=90 C, h_conv=5, Tamb=25)")
    print(f"    analytical q = {q_ana:.6f} W/m^2")
    q_num = -ms.k_face[0] * (T_new[1] - T_new[0]) / ms.h[0]
    print(f"    numerical q  = {q_num:.6f} W/m^2   (err {abs(q_num-q_ana):.3e})")
    for x_target, Ta in zip(pos[1:], T_ana[1:]):
        j = int(np.argmin(np.abs(ms.x - x_target)))
        print(f"    x={x_target*1e6:.0f} um  ana={Ta:.6f}  "
              f"new={T_new[j]:.6f} (err {abs(T_new[j]-Ta):.3e})  "
              f"old={T_old[j]:.6f} (err {abs(T_old[j]-Ta):.3e})")

    # 粗网格显式收敛 (cells=1/层, 界面仍在节点上): 显式 FV 收敛到同一稳态
    coarse = [
        heat_model.Layer("Bottom COC", "COC", 180e-6, cells=1),
        heat_model.Layer("PCR Sample", "Water", 20e-6, cells=1, role="sample"),
        heat_model.Layer("Mineral Oil", "Oil", 50e-6, cells=1),
        heat_model.Layer("Top COC", "COC", 600e-6, cells=1, role="top_surface"),
    ]
    t_long = np.array([0.0, 2000.0])
    Tb_long = np.array([25.0, 90.0])  # interp -> 线性升温; 改用阶跃:
    t_long = np.array([0.0, 0.5, 0.5001, 2000.0])
    Tb_long = np.array([25.0, 25.0, 90.0, 90.0])
    t0 = time.perf_counter()
    res_coarse = heat_model.run_simulation(
        t_long, Tb_long, DEFAULT_MATERIALS, coarse,
        h_conv=H_CONV, T_air_ambient=T_AMB, save_dt=50.0,
    )
    dt_explicit = time.perf_counter() - t0
    Tf = res_coarse["T_final"]
    ms_c = res_coarse["mesh"]
    print(f"    粗网格显式收敛 (cells=1/层, {ms_c.Nx} 节点, "
          f"dt={res_coarse['dt']*1e6:.1f} us, {res_coarse['Nt']:,} 步, "
          f"{dt_explicit:.1f} s):")
    for x_target, Ta in zip(pos[1:], T_ana[1:]):
        j = int(np.argmin(np.abs(ms_c.x - x_target)))
        print(f"      x={x_target*1e6:.0f} um  ana={Ta:.6f}  "
              f"explicit={Tf[j]:.6f} (err {abs(Tf[j]-Ta):.3e})")

    # ----------------------------------------------------------
    # 2. 高对比堆叠基准
    # ----------------------------------------------------------
    hc_mats = {
        "A": heat_model.Material("A", k_W_mK=0.1, rho_kg_m3=1000.0, cp_J_kgK=1000.0),
        "B": heat_model.Material("B", k_W_mK=1000.0, rho_kg_m3=1000.0, cp_J_kgK=1000.0),
        "C": heat_model.Material("C", k_W_mK=0.01, rho_kg_m3=1000.0, cp_J_kgK=1000.0),
    }
    hc_layers = [
        heat_model.Layer("A", "A", 100e-6, cells=4),
        heat_model.Layer("B", "B", 50e-6, cells=2),
        heat_model.Layer("C", "C", 200e-6, cells=2),
    ]
    q_hc, pos_hc, T_hc = fv_reference.analytical_steady(
        hc_mats, hc_layers, 100.0, 10.0, 25.0
    )
    T_hc_new, ms_hc = steady_tridiagonal(hc_mats, hc_layers, 100.0, 10.0, 25.0)
    T_hc_old, _ = steady_tridiagonal(hc_mats, hc_layers, 100.0, 10.0, 25.0,
                                     old_scheme=True)
    print("\n[2] 高对比堆叠基准 (k = 0.1 / 1000 / 0.01)")
    old_max = 0.0
    new_max = 0.0
    for x_target, Ta in zip(pos_hc[1:], T_hc[1:]):
        j = int(np.argmin(np.abs(ms_hc.x - x_target)))
        new_max = max(new_max, abs(T_hc_new[j] - Ta))
        old_max = max(old_max, abs(T_hc_old[j] - Ta))
        print(f"    x={x_target*1e6:.0f} um  ana={Ta:.6f}  "
              f"new_err={abs(T_hc_new[j]-Ta):.3e}  old_err={abs(T_hc_old[j]-Ta):.3f}")
    print(f"    max err: corrected {new_max:.3e}  old {old_max:.3f}")

    # ----------------------------------------------------------
    # 3. 均匀材料不变性
    # ----------------------------------------------------------
    hom_mats = {"COC": heat_model.Material(
        "COC", k_W_mK=0.13, rho_kg_m3=1020.0, cp_J_kgK=1800.0)}
    hom_layers = [heat_model.Layer("Slab", "COC", 500e-6, dx_target_m=5e-6,
                                   role="sample")]
    t_h = np.arange(0.0, 20.0, 0.5)
    Tb_h = 25.0 + 40.0 * (t_h >= 2.0)
    r_hom_new = heat_model.run_simulation(t_h, Tb_h, hom_mats, hom_layers, save_dt=0.5)
    r_hom_old = fv_reference.old_run(hom_mats, hom_layers, t_h, Tb_h, save_dt=0.5)
    hom_max = max(
        np.max(np.abs(r_hom_new["T_sample_arr"] - r_hom_old["T_sample_arr"])),
        np.max(np.abs(r_hom_new["T_outer_surface_arr"]
                      - r_hom_old["T_outer_surface_arr"])),
    )
    print(f"\n[3] 均匀材料不变性: max|new-old| = {hom_max:.3e} (dt same: "
          f"{r_hom_new['dt'] == r_hom_old['dt']})")

    # ----------------------------------------------------------
    # 4. 瞬态 25 -> 90 C 阶跃 (60 s, 细网格)
    # ----------------------------------------------------------
    t_step = np.arange(61.0, dtype=float)
    Tb_step = np.where(t_step < 1.0, 25.0, 90.0)
    t0 = time.perf_counter()
    r_new = heat_model.run_simulation(
        t_step, Tb_step, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, save_dt=1.0,
    )
    t_new_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_old = fv_reference.old_run(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, t_step, Tb_step, save_dt=1.0,
    )
    t_old_wall = time.perf_counter() - t0
    print("\n[4] 瞬态 25->90 C 阶跃 60 s (bare-top 细网格)")
    print(f"    dt: old {r_old['dt']*1e6:.4f} us / new {r_new['dt']*1e6:.4f} us")
    print(f"    wall: old {t_old_wall:.1f} s / new {t_new_wall:.1f} s")
    for name, a, b in (("T_sample", r_new["T_sample_arr"],
                        r_old["T_sample_arr"]),
                       ("T_top_surface", r_new["T_top_surface_arr"],
                        r_old["T_top_surface_arr"])):
        m = metrics(r_new["t_array"], a - b)
        print(f"    {name}: max_abs={m['max_abs']:.5f} C at t="
              f"{m['time_of_max']:.1f} s, MAE={m['mae']:.5f}, "
              f"RMSE={m['rmse']:.5f}")

    # ----------------------------------------------------------
    # 5. 真实 72 C 协议 (aligned T_internal 直接作为底部边界)
    # ----------------------------------------------------------
    print(f"\n[5] 真实 72 C 协议 (aligned_internal_top_temperature.csv)")
    aligned = pd.read_csv(ALIGNED_CSV)
    t_proto = aligned["time_s"].to_numpy(dtype=float)
    T_internal = aligned["T_internal_interpolated_C"].to_numpy(dtype=float)
    regime_df = pd.read_csv(REGIME_CSV)
    regimes = regime_df["regime"].to_numpy(dtype=str)
    if not np.allclose(regime_df["time_s"].to_numpy(dtype=float), t_proto):
        raise ValueError("regime 文件时间轴与 aligned 不一致")

    t0 = time.perf_counter()
    r72_new = heat_model.run_simulation(
        t_proto, T_internal, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        h_conv=H_CONV, T_air_ambient=T_AMB, save_dt=0.1,
    )
    t72_new_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    r72_old = fv_reference.old_run(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, t_proto, T_internal,
        h_conv=H_CONV, T_air_ambient=T_AMB, save_dt=0.1,
    )
    t72_old_wall = time.perf_counter() - t0
    print(f"    points: {len(t_proto)}; dt old {r72_old['dt']*1e6:.4f} us / "
          f"new {r72_new['dt']*1e6:.4f} us")
    print(f"    wall: old {t72_old_wall:.1f} s / new {t72_new_wall:.1f} s")

    # 对齐时间轴 (dt 相同 -> 网格一致, 仍做插值以防万一)
    t_new = r72_new["t_array"]
    for name, a, b in (("T_sample", r72_new["T_sample_arr"],
                        r72_old["T_sample_arr"]),
                       ("T_top_surface", r72_new["T_top_surface_arr"],
                        r72_old["T_top_surface_arr"])):
        b_i = interp_onto(t_new, r72_old["t_array"], b)
        m = metrics(t_new, a - b_i)
        print(f"    OVERALL {name}: max_abs={m['max_abs']:.5f} C at t="
              f"{m['time_of_max']:.1f} s, MAE={m['mae']:.5f}, "
              f"RMSE={m['rmse']:.5f}")

    # regime 分块
    print("    regime-specific (T_sample / T_top_surface):")
    for reg in ("TRANSIENT_HEATING", "TRANSIENT_COOLING", "SETTLING",
                "TRANSITION_OTHER"):
        mask = regimes == reg
        t_reg = t_proto[mask]
        m_s = metrics(t_reg, interp_onto(t_reg, r72_new["t_array"],
                                         r72_new["T_sample_arr"])
                      - interp_onto(t_reg, r72_old["t_array"],
                                    r72_old["T_sample_arr"]))
        m_t = metrics(t_reg, interp_onto(t_reg, r72_new["t_array"],
                                         r72_new["T_top_surface_arr"])
                      - interp_onto(t_reg, r72_old["t_array"],
                                    r72_old["T_top_surface_arr"]))
        print(f"      {reg:18s} n={mask.sum():3d}  "
              f"T_sample max={m_s['max_abs']:.5f} mae={m_s['mae']:.5f} | "
              f"T_top max={m_t['max_abs']:.5f} mae={m_t['mae']:.5f}")

    # 描述性 (不解释为校准质量): 修正版 T_top_surface 与实测 T_top_measured
    T_top_meas = aligned["T_top_measured_C"].to_numpy(dtype=float)
    t_new = r72_new["t_array"]
    T_top_pred = interp_onto(t_new, r72_new["t_array"],
                             r72_new["T_top_surface_arr"])
    m_desc = metrics(t_new, T_top_pred - interp_onto(t_new, t_proto, T_top_meas))
    print(f"    [描述性, 不解释为校准] corrected T_top_surface vs "
          f"T_top_measured: max={m_desc['max_abs']:.2f} C, "
          f"MAE={m_desc['mae']:.2f} C")

    print("\n完成。")


if __name__ == "__main__":
    main()
