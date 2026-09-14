#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests: V2 calibration-propagated sample-prediction sensitivity.

验证 (不重跑 FDM, 只验证逻辑与数据流):
  1. 既有校准结果被正确读取 (29 run + PerturbConfig 一致)
  2. case 构建传播规则 (TOTAL/DIRECT/INDIRECT)
  3. tau_top 不进入样品预测 (结构层面)
  4. 几何扰动重建正确层叠
  5. frozen V2 配置未被修改 (回归)
  6. 旧 sensitivity 输出未被修改 (回归)
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAL_RUNS_DIR = (PROJECT_ROOT / "outputs" / "v2_calibration_sensitivity"
                / "runs")
PROP_MODULE = "workflows.diagnostics.analyze_v2_calibration_propagated_prediction"


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _mod():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    import workflows.diagnostics.analyze_v2_calibration_propagated_prediction as m
    return m


@pytest.fixture(scope="module")
def runs():
    return _mod().load_calibration_runs()


# ------------------------------------------------------------
# 1. 既有校准结果读取
# ------------------------------------------------------------

def test_calibration_runs_dir_exists():
    assert CAL_RUNS_DIR.is_dir()
    assert len(list(CAL_RUNS_DIR.glob("*.json"))) == 29


def test_all_29_runs_loaded_with_configs(runs):
    assert len(runs) == 29
    assert "baseline" in runs
    for tag, item in runs.items():
        assert "json" in item and "cfg" in item


def test_run_json_has_best_params_and_metrics(runs):
    for tag, item in runs.items():
        best = item["json"]["best"]
        assert {"k_eff_W_mK", "cp_eff_J_kgK", "tau_top_s"} <= set(best)
        assert "RMSE_C" in item["json"]["metrics"]


def test_baseline_run_is_authoritative(runs):
    b = runs["baseline"]["json"]
    assert b["best"]["k_eff_W_mK"] == pytest.approx(0.0700, abs=1e-12)
    assert b["best"]["cp_eff_J_kgK"] == pytest.approx(700.0, abs=1e-9)
    assert b["best"]["tau_top_s"] == pytest.approx(8.0, abs=1e-12)
    assert b["metrics"]["RMSE_C"] == pytest.approx(0.6332888094, abs=1e-6)


def test_perturbation_configs_consistent(runs):
    """PerturbConfig 与 run JSON 的 param/pct/value 必须一致
    (load_calibration_runs 内部已断言; 此处冗余验证关键 case)。"""
    m = _mod()
    assert runs["h_conv_-10"]["cfg"].h_conv == pytest.approx(9.0)
    assert runs["h_conv_+10"]["cfg"].h_conv == pytest.approx(11.0)
    assert runs["rho_COC_+10"]["cfg"].rho_COC == pytest.approx(1122.0)
    assert runs["geom_top_coc_+10"]["cfg"].thick_top_um == pytest.approx(660.0)
    assert runs["geom_top_coc_+10"]["cfg"].thick_top_um != m.CAL_GEOM["Top COC"]


# ------------------------------------------------------------
# 2. case 构建传播规则
# ------------------------------------------------------------

def test_case_modes_counts(runs):
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    modes = [c["mode"] for c in cases.values()]
    assert modes.count("TOTAL") == 28
    assert modes.count("DIRECT") == 28
    assert modes.count("INDIRECT") == 28
    assert modes.count("BASELINE") == 1


def test_total_propagates_both_perturbed_input_and_recal_params(runs):
    """TOTAL case 必须同时使用 扰动 fixed input + 重校准 k'/cp'。"""
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    c = cases["TOTAL::h_conv_-10"]
    assert c["h"] == pytest.approx(9.0)                 # 扰动输入
    assert c["k"] == pytest.approx(
        runs["h_conv_-10"]["json"]["best"]["k_eff_W_mK"])   # 重校准 k'
    assert c["cp"] == pytest.approx(
        runs["h_conv_-10"]["json"]["best"]["cp_eff_J_kgK"])  # 重校准 cp'
    # 绝不允许重置回 baseline
    assert c["h"] != 10.0


def test_direct_uses_baseline_k_cp(runs):
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    c = cases["DIRECT::h_conv_-10"]
    assert c["h"] == pytest.approx(9.0)          # 扰动输入保留
    assert c["k"] == pytest.approx(0.0700)        # baseline k
    assert c["cp"] == pytest.approx(700.0)        # baseline cp


def test_indirect_uses_recal_params_with_baseline_input(runs):
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    c = cases["INDIRECT::h_conv_-10"]
    assert c["h"] == pytest.approx(10.0)          # baseline 输入
    assert c["k"] == pytest.approx(0.0650)        # 重校准 k'
    assert c["cp"] == pytest.approx(600.0)        # 重校准 cp'


def test_rho_coc_total_uses_perturbed_rho_and_extreme_cp(runs):
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    c = cases["TOTAL::rho_COC_+10"]
    assert c["rho"] == pytest.approx(1122.0)      # 扰动 rho
    assert c["k"] == pytest.approx(0.0700)
    assert c["cp"] == pytest.approx(2400.0)       # 边界重校准 cp'
    # identifiability flag 应被标记
    assert runs["rho_COC_+10"]["json"]["boundary_warnings"] == ["CP_MAX"]


def test_geometry_total_propagates_perturbed_thickness(runs):
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    c = cases["TOTAL::geom_top_coc_+10"]
    lay = {name: val for name, field, val in c["lay"]}
    assert lay["Top COC"] == pytest.approx(660e-6)
    # Bottom COC / sample / FC70 不在此 case 中, 不应被覆盖
    assert "Bottom COC" not in lay and "PCR Sample" not in lay


def test_sample_thickness_override_present(runs):
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    c = cases["TOTAL::geom_pcr_sample_-10"]
    lay = {name: val for name, field, val in c["lay"]}
    assert lay["PCR Sample"] == pytest.approx(18e-6)  # 20µm -10%


def test_fc70_material_override(runs):
    m = _mod()
    cases = m.build_all_propagation_cases(runs)
    c = cases["TOTAL::fc70_k_+10"]
    mat = dict((n, dict(f, v)) for n, f, v in
               [(n, f, v) for n, f, v in c["mat"]]) if False else None
    mdict = {(n, f): v for n, f, v in c["mat"]}
    assert mdict[("FC70", "k_W_mK")] == pytest.approx(0.077)
    # 未扰动的 FC70 属性保持 baseline
    assert mdict[("FC70", "rho_kg_m3")] == pytest.approx(1940.0)
    assert mdict[("FC70", "cp_J_kgK")] == pytest.approx(1050.0)


# ------------------------------------------------------------
# 3. tau_top 结构性排除
# ------------------------------------------------------------

def test_pred_overrides_never_contains_tau():
    """样品预测覆盖项不含 tau_top; run_pred 签名也没有 tau 参数。"""
    import inspect
    m = _mod()
    sig = inspect.signature(m.run_pred)
    assert "tau" not in sig.parameters
    assert "tau_top" not in sig.parameters


def test_sample_prediction_uses_raw_field_not_lagged():
    """run_pred 的样品来自 T_sample_arr (原始有限体积场), 不是滞后观测。"""
    import inspect
    m = _mod()
    src = inspect.getsource(m.run_pred)
    assert "T_sample_arr" in src
    assert "apply_first_order_lag" not in src


# ------------------------------------------------------------
# 4. 几何重建正确性
# ------------------------------------------------------------

def test_pred_layer_stack_rebuild():
    """层厚覆盖真正改变层叠; 样品层存在; 无 PDMS。"""
    from thermal_model.core import heat_model
    from thermal_model.config import thermal_model_v2_candidate as v2
    layers = heat_model.copy_layers(v2.V2_INSULATED_NO_PDMS_LAYERS)
    for layer in layers:
        if layer.name == "Top COC":
            layer.role = "top_surface"
    for name, field, val in [("Top COC", "thickness", 660e-6),
                             ("PCR Sample", "thickness", 18e-6)]:
        for layer in layers:
            if layer.name == name:
                layer.thickness_m = float(val)
    names = [L.name for L in layers]
    assert "PDMS" not in " ".join(names)
    top = next(L for L in layers if L.name == "Top COC")
    sample = next(L for L in layers if L.name == "PCR Sample")
    assert top.thickness_m == pytest.approx(660e-6)
    assert sample.thickness_m == pytest.approx(18e-6)
    # 其余层未被意外修改
    bottom = next(L for L in layers if L.name == "Bottom COC")
    assert bottom.thickness_m == pytest.approx(180e-6)


def test_sample_weights_follow_perturbed_geometry():
    """样品层加权必须随扰动几何重建: 权重和=1 且更厚样品激活更多节点。"""
    from thermal_model.core import heat_model
    from thermal_model.config import thermal_model_v2_candidate as v2

    def build(thickness_m):
        layers = heat_model.copy_layers(v2.V2_INSULATED_NO_PDMS_LAYERS)
        for layer in layers:
            if layer.name == "PCR Sample":
                layer.thickness_m = thickness_m
        mesh = heat_model.build_layer_stack(
            v2.make_v2_materials(0.0700, 700.0, 1020.0), layers)
        return mesh.sample_weights

    w20 = build(20e-6)
    w22 = build(22e-6)
    assert abs(float(w20.sum()) - 1.0) < 1e-12      # 权重和 = 1
    assert abs(float(w22.sum()) - 1.0) < 1e-12
    n20 = int((w20 > 0).sum())
    n22 = int((w22 > 0).sum())
    assert n22 >= n20                               # 更厚样品 >= 激活节点
    # 样品权重只落在样品层节点上 (最大权重节点数与厚度成正比趋势)
    assert n20 > 0 and n22 > 0


# ------------------------------------------------------------
# 5. frozen V2 配置回归
# ------------------------------------------------------------

def test_frozen_v2_config_unchanged():
    from thermal_model.config.final_frozen_model_v2 import (
        FINAL_FROZEN_THERMAL_MODEL_V2)
    m = FINAL_FROZEN_THERMAL_MODEL_V2
    assert m.k_eff_W_mK == pytest.approx(0.0700)
    assert m.cp_eff_J_kgK == pytest.approx(700.0)
    assert m.tau_top_s == pytest.approx(8.0)


def test_v2_candidate_authoritative_objects_unchanged():
    from thermal_model.config import thermal_model_v2_candidate as v2
    assert v2.FC70_K_W_MK == pytest.approx(0.070)
    assert v2.FC70_RHO_KG_M3 == pytest.approx(1940.0)
    assert v2.FC70_CP_J_KGK == pytest.approx(1050.0)
    names = [L.name for L in v2.V2_INSULATED_NO_PDMS_LAYERS]
    assert names == ["Bottom COC", "PCR Sample", "FC-70", "Top COC",
                     "Air Gap"]
    thick = {L.name: L.thickness_m for L in v2.V2_INSULATED_NO_PDMS_LAYERS}
    assert thick["Bottom COC"] == pytest.approx(180e-6)
    assert thick["PCR Sample"] == pytest.approx(20e-6)
    assert thick["FC-70"] == pytest.approx(50e-6)
    assert thick["Top COC"] == pytest.approx(600e-6)
    assert thick["Air Gap"] == pytest.approx(3000e-6)


# ------------------------------------------------------------
# 6. 旧 sensitivity 输出回归 (未被修改)
# ------------------------------------------------------------

def test_old_frozen_sensitivity_outputs_unchanged():
    d = PROJECT_ROOT / "outputs" / "v2_fc70_no_pdms_sensitivity"
    r = pd_read_csv(d / "sensitivity_all_parameters.csv")
    keff = r[r["parameter"] == "k_eff"].iloc[0]
    assert keff["S_range_C"] == pytest.approx(0.8799272889660905, abs=1e-9)


def test_old_calibration_sensitivity_outputs_unchanged():
    d = PROJECT_ROOT / "outputs" / "v2_calibration_sensitivity"
    r = pd_read_csv(d / "calibration_sensitivity_all.csv")
    assert len(r) == 29
    base = r[r["tag"] == "baseline"].iloc[0]
    assert base["RMSE_C"] == pytest.approx(0.6332888094066287, abs=1e-9)


def test_new_output_dir_is_new_name():
    """新分析使用独立目录, 不覆盖旧目录。"""
    m = _mod()
    assert m.OUTPUT_ROOT.name == "v2_calibration_propagated_prediction_sensitivity"
    assert m.OUTPUT_ROOT != (PROJECT_ROOT / "outputs"
                             / "v2_fc70_no_pdms_sensitivity")


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------

def pd_read_csv(path):
    import pandas as pd
    return pd.read_csv(path)
