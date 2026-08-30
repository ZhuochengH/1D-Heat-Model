#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 CANDIDATE SENSITIVITY — targeted tests
==========================================

覆盖:
  1. V2 sensitivity baseline 正确使用 FC-70 (k=0.070/rho=1940/cp=1050)
  2. V2 insulated geometry 无 PDMS
  3. perturbation 只改变目标参数
  4. sample-layer weighting 不变
  5. tau_top negative control (样品不依赖 tau)
  6. V1 historical workflow 未被破坏 (材料定义保留)
  7. baseline V2 sample metrics 可复现
"""
import numpy as np
import pytest

from thermal_model.core import heat_model
from thermal_model.config import thermal_model_v2_candidate as v2
import workflows.v2_fc70_no_pdms.analyze_v2_sample_sensitivity as sens


# ============================================================
# 1. V2 sensitivity baseline 正确使用 FC-70
# ============================================================

def test_sensitivity_uses_fc70_material():
    mats = sens.make_v2_materials(sens.BASE_K_EFF, sens.BASE_CP_EFF,
                                  sens.BASE_RHO)
    assert "FC70" in mats
    assert "Oil" not in mats
    assert mats["FC70"].k_W_mK == 0.070
    assert mats["FC70"].rho_kg_m3 == 1940.0
    assert mats["FC70"].cp_J_kgK == 1050.0


def test_sensitivity_baseline_constants():
    assert sens.BASE_K_EFF == 0.0700
    assert sens.BASE_CP_EFF == 700.0
    assert sens.BASE_RHO == 1020.0
    assert sens.BASE_H == 10.0
    assert sens.BASE_EPS == 0.90
    assert sens.BASE_TAU == 8.0


# ============================================================
# 2. V2 insulated geometry 无 PDMS
# ============================================================

def test_sensitivity_geometry_no_pdms():
    layers = sens.build_v2_insulated_layers()
    mats = [l.material for l in layers]
    assert "PDMS" not in mats
    assert mats == ["COC", "Water", "FC70", "COC", "Air"]
    names = [l.name for l in layers]
    assert names == ["Bottom COC", "PCR Sample", "FC-70", "Top COC",
                     "Air Gap"]
    total = sum(l.thickness_m for l in layers)
    assert abs(total - 3850e-6) < 1e-12


def test_sensitivity_grid_has_no_pdms_params():
    assert "k_PDMS" not in sens.OAT_GRID_V2
    assert "d_PDMS" not in sens.OAT_GRID_V2
    assert "cp_PDMS" not in sens.OAT_GRID_V2
    assert "k_FC70" in sens.OAT_GRID_V2
    assert "cp_FC70" in sens.OAT_GRID_V2


# ============================================================
# 3. perturbation 只改变目标参数
# ============================================================

def test_case_args_change_only_target():
    # k_eff perturbation
    args = sens._case_args("k_eff", -10, 0.0630)
    k, cp, rho, h, eps, mat_over, lay_over = args
    assert k == 0.0630
    assert cp == sens.BASE_CP_EFF
    assert rho == sens.BASE_RHO
    assert h == sens.BASE_H
    assert eps == sens.BASE_EPS
    assert mat_over == [] and lay_over == []

    # cp_FC70 perturbation
    args = sens._case_args("cp_FC70", 10, 1155.0)
    k, cp, rho, h, eps, mat_over, lay_over = args
    assert mat_over == [("FC70", "cp_J_kgK", 1155.0)]
    assert k == sens.BASE_K_EFF and cp == sens.BASE_CP_EFF
    assert rho == sens.BASE_RHO and h == sens.BASE_H and eps == sens.BASE_EPS

    # d_air perturbation (mm -> m)
    args = sens._case_args("d_air", -10, 2.7)
    k, cp, rho, h, eps, mat_over, lay_over = args
    assert lay_over == [("Air Gap", "thickness", 2.7e-3)]


def test_layer_override_only_changes_target_thickness():
    layers = sens.build_v2_insulated_layers(
        [("Air Gap", "thickness", 2.7e-3)])
    thicknesses = {l.name: l.thickness_m for l in layers}
    assert thicknesses["Air Gap"] == 2.7e-3
    # 其他层不变
    assert thicknesses["Bottom COC"] == 180e-6
    assert thicknesses["PCR Sample"] == 20e-6
    assert thicknesses["FC-70"] == 50e-6
    assert thicknesses["Top COC"] == 600e-6


# ============================================================
# 4. sample-layer weighting 不变
# ============================================================

def test_sample_weights_unchanged():
    mats = sens.make_v2_materials(sens.BASE_K_EFF, sens.BASE_CP_EFF,
                                  sens.BASE_RHO)
    layers = sens.build_v2_insulated_layers()
    mesh = heat_model.build_layer_stack(mats, layers)
    assert float(mesh.sample_weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert mesh.sample_layer_index == 1
    assert abs(mesh.boundaries[1] - 180e-6) < 1e-15
    assert abs(mesh.boundaries[2] - 200e-6) < 1e-15
    # 样品层区间 [180,200] um 内的加权结构不变:
    # 与 V1 相同的样品层 (Water, 20 um, 5 um 网格) -> 每个非界面节点权重
    # 0.25, 界面节点 (样品侧) 0.125, 与 V1 一致。
    si = mesh.sample_layer_index
    xl, xr = mesh.boundaries[si], mesh.boundaries[si + 1]
    in_sample = (mesh.x > xl) & (mesh.x <= xr + 1e-9)
    w = mesh.sample_weights[in_sample]
    # 5 um 网格, 样品区间 (180,200] um: 内部节点权重 0.25,
    # 200 um 界面节点只计样品侧半体积 -> 0.125
    assert np.allclose(np.sort(w), [0.125, 0.25, 0.25, 0.25])


# ============================================================
# 5. tau_top negative control
# ============================================================

def test_tau_top_does_not_affect_sample():
    # 样品温度 = raw FDM 层平均; tau 只作用于 T_top_observed。
    # 用 lag_augmented 模块验证: 对同一条样品迹线施加不同 tau, 样品不变。
    from thermal_model.core.lag_augmented_thermal_model import (
        apply_first_order_lag,
    )
    t = np.linspace(0, 10, 1000)
    x = 25.0 + 60.0 * (t / 10.0)
    y0 = apply_first_order_lag(t, x, 0.0)
    y8 = apply_first_order_lag(t, x, 8.0)
    y16 = apply_first_order_lag(t, x, 16.0)
    # tau 改变会改变滞后输出
    assert not np.allclose(y8, y16)
    # 但样品层从不经过 lag: 即样品 = raw FDM。sensitivity 脚本中
    # run_single_case 从不调用 apply_first_order_lag, 与 tau 无关。
    # 直接断言 sensitivity baseline 的 cycles 与 tau 无关:
    # tau_negative_control.json 写入 sample_identical=True (在 main 中)。
    # 这里验证机制: 网格无 tau 参数, build_all_cases 不含 tau。
    for param, pct, val, rt, args in sens.build_all_cases():
        assert param != "tau_top"


# ============================================================
# 6. V1 historical workflow 未被破坏
# ============================================================

def test_v1_material_definitions_untouched():
    assert heat_model.DEFAULT_MATERIALS["Oil"].k_W_mK == 0.142
    assert heat_model.DEFAULT_MATERIALS["Oil"].rho_kg_m3 == 876.0
    assert heat_model.DEFAULT_MATERIALS["Oil"].cp_J_kgK == 1962.0
    assert heat_model.DEFAULT_MATERIALS["PDMS"].k_W_mK == 0.15
    assert len(heat_model.LEGACY_INSULATED_LAYERS) == 6
    assert "PDMS" in [l.material
                      for l in heat_model.LEGACY_INSULATED_LAYERS]


def test_v1_sensitivity_ranking_csv_exists():
    """V1 权威 sensitivity 结果未被动 (磁盘存在性保护)。"""
    p = (sens.PROJECT_ROOT / "sample_temperature_output"
         / "08.24_15x_no_holding_sensitivity" / "tables"
         / "sensitivity_ranking.csv")
    assert p.is_file()
    import pandas as pd
    df = pd.read_csv(p)
    assert "k_eff" in df["parameter"].values
    assert "k_PDMS" in df["parameter"].values


# ============================================================
# 7. baseline V2 sample metrics 可复现
# ============================================================

def test_v2_baseline_metrics_reproducible():
    """完整运行开销大, 用轻量代理: 直接调用单次 FDM (数据 350 s, ~17k 步)。"""
    from workflows.prediction.predict_sample_temperature_frozen_model import (
        load_internal_data,
    )
    data = load_internal_data(sens.INPUT_XLSX)
    t_sp, sp = sens.load_setpoint(sens.INPUT_XLSX)
    cw = sens.define_cycle_windows(t_sp, sp)
    base = sens.run_single_case(
        sens.BASE_K_EFF, sens.BASE_CP_EFF, sens.BASE_RHO,
        sens.BASE_H, sens.BASE_EPS, t_src=data["source_time_s"],
        T_internal=data["T_internal_C"], windows=cw["windows"])
    assert base["cycles"]["high"]["mean"] == pytest.approx(84.703, abs=0.01)
    assert base["cycles"]["low"]["mean"] == pytest.approx(58.686, abs=0.01)
    assert base["cycles"]["amplitude"]["mean"] == pytest.approx(26.017,
                                                                abs=0.01)
    assert base["overall_sample_max_C"] == pytest.approx(89.097, abs=0.01)
    assert base["Nx"] == 186
    assert base["dt_s"] == pytest.approx(7.8375e-05, rel=1e-6)
    # 周期数 29 (windows 中非排除项)
    n_complete = sum(1 for w in cw["windows"] if not w["excluded"])
    assert n_complete == 29
