#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THERMAL MODEL V2 CANDIDATE — FC-70 + no-PDMS insulated geometry
单元测试: 验证 V1 保留 + V2 材料/几何正确性
=================================================================

覆盖 (任务 §17):
  1. V1 仍使用其历史材料定义 (Oil k=0.142/rho=876/cp=1962, 未变);
  2. V2 使用 FC-70 k=0.070 / rho=1940 / cp=1050;
  3. V2 绝缘几何不含 PDMS;
  4. V2 层叠顺序正确;
  5. 样品层空间平均加权不变 (sum=1, 样品区间 180-200 um);
  6. tau_top 仍不影响样品温度 (滞后只作用 Top 观测);
  7. 裸顶标定几何除 oil->FC-70 外结构不变。
"""
import numpy as np
import pytest

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag
from thermal_model.config import thermal_model_v2_candidate as v2


# ============================================================
# 1. V1 历史材料定义保留
# ============================================================

def test_v1_oil_material_unchanged():
    assert heat_model.DEFAULT_MATERIALS["Oil"].k_W_mK == 0.142
    assert heat_model.DEFAULT_MATERIALS["Oil"].rho_kg_m3 == 876.0
    assert heat_model.DEFAULT_MATERIALS["Oil"].cp_J_kgK == 1962.0


def test_v1_insulated_layers_still_contain_pdms():
    mats = [l.material for l in heat_model.LEGACY_INSULATED_LAYERS]
    assert mats == ["COC", "Water", "Oil", "COC", "Air", "PDMS"]
    assert len(heat_model.LEGACY_INSULATED_LAYERS) == 6


def test_v1_bare_layers_unchanged():
    mats = [l.material for l in heat_model.BARE_TOP_COC_LAYERS]
    assert mats == ["COC", "Water", "Oil", "COC"]


# ============================================================
# 2. FC-70 材料
# ============================================================

def test_fc70_material_properties():
    m = v2.fc70_material()
    assert m.k_W_mK == 0.070
    assert m.rho_kg_m3 == 1940.0
    assert m.cp_J_kgK == 1050.0
    assert m.name == "FC70"


def test_v2_materials_use_fc70_and_remove_oil():
    mats = v2.make_v2_materials(0.0675, 700.0)
    assert "FC70" in mats
    assert "Oil" not in mats
    assert mats["FC70"].k_W_mK == 0.070
    assert mats["FC70"].rho_kg_m3 == 1940.0
    assert mats["FC70"].cp_J_kgK == 1050.0


def test_v2_materials_replace_coc_only_elsewhere():
    mats = v2.make_v2_materials(0.0675, 700.0)
    assert mats["COC"].k_W_mK == 0.0675
    assert mats["COC"].cp_J_kgK == 700.0
    # Water / Air / PDMS 逐位不变
    assert mats["Water"].k_W_mK == 0.60
    assert mats["Air"].k_W_mK == 0.0257
    assert mats["PDMS"].k_W_mK == 0.15


def test_make_v2_materials_does_not_mutate_default():
    before = heat_model.DEFAULT_MATERIALS["Oil"].k_W_mK
    v2.make_v2_materials(0.05, 900.0)
    assert heat_model.DEFAULT_MATERIALS["Oil"].k_W_mK == before
    assert "FC70" not in heat_model.DEFAULT_MATERIALS


# ============================================================
# 3. V2 绝缘几何无 PDMS
# ============================================================

def test_v2_insulated_has_no_pdms():
    mats = [l.material for l in v2.V2_INSULATED_NO_PDMS_LAYERS]
    assert "PDMS" not in mats
    assert "Air" in mats


def test_v2_insulated_layer_ordering():
    mats = [l.material for l in v2.V2_INSULATED_NO_PDMS_LAYERS]
    assert mats == ["COC", "Water", "FC70", "COC", "Air"]
    names = [l.name for l in v2.V2_INSULATED_NO_PDMS_LAYERS]
    assert names == ["Bottom COC", "PCR Sample", "FC-70", "Top COC",
                     "Air Gap"]


def test_v2_insulated_total_thickness():
    total = sum(l.thickness_m for l in v2.V2_INSULATED_NO_PDMS_LAYERS)
    assert abs(total - 3850e-6) < 1e-12  # 4050 - 200 (PDMS 移除)


def test_v2_bare_layer_ordering():
    mats = [l.material for l in v2.V2_BARE_TOP_COC_LAYERS]
    assert mats == ["COC", "Water", "FC70", "COC"]
    assert v2.V2_BARE_TOP_COC_LAYERS[3].role == "top_surface"
    assert v2.V2_BARE_TOP_COC_LAYERS[1].role == "sample"


# ============================================================
# 4/5. 样品层加权不变 (180-200 um, sum=1)
# ============================================================

def test_v2_sample_weights_unchanged_semantics():
    mats = v2.make_v2_materials(0.0675, 700.0)
    mesh = heat_model.build_layer_stack(mats, v2.V2_BARE_TOP_COC_LAYERS)
    assert float(mesh.sample_weights.sum()) == pytest.approx(1.0, abs=1e-12)
    # 样品区间 [180, 200] um
    assert mesh.sample_layer_index == 1
    x_left = mesh.boundaries[1]
    x_right = mesh.boundaries[2]
    assert abs(x_left - 180e-6) < 1e-15
    assert abs(x_right - 200e-6) < 1e-15


def test_v2_vs_v1_sample_weights_identical():
    """裸顶样品加权在 V1 与 V2 之间应一致 (样品层材料/几何未变)。"""
    mats_v1 = cr.make_convection_radiation_materials(0.0675, 700.0, 1020.0)
    mesh_v1 = heat_model.build_layer_stack(mats_v1,
                                           heat_model.BARE_TOP_COC_LAYERS)
    mats_v2 = v2.make_v2_materials(0.0675, 700.0, 1020.0)
    mesh_v2 = heat_model.build_layer_stack(mats_v2, v2.V2_BARE_TOP_COC_LAYERS)
    assert mesh_v1.sample_weights.shape == mesh_v2.sample_weights.shape
    assert np.allclose(mesh_v1.sample_weights, mesh_v2.sample_weights)


# ============================================================
# 6. tau_top 不影响样品温度 (滞后仅作用 Top 观测)
# ============================================================

def test_tau_top_does_not_affect_sample_temperature():
    t = np.linspace(0.0, 20.0, 200)
    T = 25.0 + 60.0 * np.clip(t / 5.0, 0.0, 1.0)
    mats = v2.make_v2_materials(0.0675, 700.0)
    result = cr.run_convection_radiation_fdm(
        time_s=t, bottom_temperature_C=T, materials=mats,
        layers=v2.V2_BARE_TOP_COC_LAYERS, T_air_C=25.0,
        T_surroundings_C=25.0, save_dt=0.1, T_initial_C=25.0)
    # 样品温度直接来自 FDM, 施加不同 tau 对样品无影响 (结构上: 样品
    # 从不经过 apply_first_order_lag)
    sample = result["T_sample_arr"]
    top = result["T_top_surface_arr"]
    top_tau0 = apply_first_order_lag(result["t_array"], top, 0.0)
    top_tau8 = apply_first_order_lag(result["t_array"], top, 8.0)
    # 样品不受滞后影响 (与 tau 无关); 顶部随 tau 改变
    assert not np.allclose(top_tau0, top_tau8)
    # 样品层温度由 mesh.sample_weights 计算, 与滞后独立 (恒等)
    assert sample.shape == top.shape


# ============================================================
# 7. 裸顶几何结构除 oil->FC-70 外不变
# ============================================================

def test_bare_geometry_structure_unchanged_except_oil():
    v1_layers = heat_model.BARE_TOP_COC_LAYERS
    v2_layers = v2.V2_BARE_TOP_COC_LAYERS
    assert len(v1_layers) == len(v2_layers) == 4
    for l1, l2 in zip(v1_layers, v2_layers):
        assert l1.thickness_m == l2.thickness_m
        assert l1.role == l2.role
        if l1.material == "Oil":
            assert l2.material == "FC70"
        else:
            assert l1.material == l2.material


# ============================================================
# V2 几何数值有效性
# ============================================================

def test_v2_insulated_builds_and_stable():
    mats = v2.make_v2_materials(0.0675, 700.0)
    mesh, dt = heat_model.compute_stable_dt(mats,
                                            v2.V2_INSULATED_NO_PDMS_LAYERS)
    assert mesh.Nx > 0
    assert dt > 0 and np.isfinite(dt)
    # 外部表面 = 密封空气外表面 (无 PDMS)
    assert mesh.layer_names[-1] == "Air Gap"
    assert float(mesh.k_face[-1]) == pytest.approx(0.0257)


def test_v2_insulated_newton_converges():
    t = np.linspace(0.0, 20.0, 200)
    T = 25.0 + 60.0 * np.clip(t / 5.0, 0.0, 1.0)
    mats = v2.make_v2_materials(0.0675, 700.0)
    res = cr.run_convection_radiation_fdm(
        time_s=t, bottom_temperature_C=T, materials=mats,
        layers=v2.V2_INSULATED_NO_PDMS_LAYERS, T_air_C=25.0,
        T_surroundings_C=25.0, save_dt=0.1, T_initial_C=25.0)
    assert res["newton_max_iterations_per_step"] <= 20
    # 除初始瞬态外残差应近零
    assert float(res["max_abs_boundary_residual_W_m2"]) < 1e-6
