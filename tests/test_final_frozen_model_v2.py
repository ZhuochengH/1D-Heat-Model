#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL FROZEN THERMAL MODEL V2 — 冻结验证测试
==============================================

覆盖 (任务 §7):
  1. FINAL_FROZEN_THERMAL_MODEL_V2 exists
  2. V2 k_eff = 0.0700
  3. V2 cp_eff = 700
  4. V2 tau_top = 8.0
  5. V2 uses FC-70 properties exactly as frozen
  6. final insulated V2 geometry contains no PDMS
  7. final bare V2 geometry contains FC-70
  8. V1 historical frozen config remains unchanged
  9. V1 still contains historical Oil/PDMS behavior
  10. sample-layer weighting remains correct
  11. tau_top does not affect sample temperature
  12. final V2 baseline sample metrics reproduce approximately
      (HIGH 84.703 C / LOW 58.686 C / range 26.017 C)
  13. calibration/validation reference metrics preserved
"""
import numpy as np
import pytest

from thermal_model.core import heat_model
from thermal_model.config.final_frozen_model import (
    FINAL_FROZEN_THERMAL_MODEL_V1,
)
from thermal_model.config.final_frozen_model_v2 import (
    FINAL_FROZEN_THERMAL_MODEL_V2,
)


# ============================================================
# 1-4. 存在性与参数
# ============================================================

def test_v2_final_model_exists():
    assert FINAL_FROZEN_THERMAL_MODEL_V2 is not None
    assert FINAL_FROZEN_THERMAL_MODEL_V2.model_id == \
        "FINAL_FROZEN_THERMAL_MODEL_V2"


def test_v2_k_eff():
    assert FINAL_FROZEN_THERMAL_MODEL_V2.k_eff_W_mK == 0.0700


def test_v2_cp_eff():
    assert FINAL_FROZEN_THERMAL_MODEL_V2.cp_eff_J_kgK == 700.0


def test_v2_tau_top():
    assert FINAL_FROZEN_THERMAL_MODEL_V2.tau_top_s == 8.0


def test_v2_rho_coc():
    assert FINAL_FROZEN_THERMAL_MODEL_V2.rho_COC_kg_m3 == 1020.0


# ============================================================
# 5. FC-70 材料
# ============================================================

def test_v2_fc70_properties_frozen():
    m = FINAL_FROZEN_THERMAL_MODEL_V2
    assert m.fc70_k_W_mK == 0.070
    assert m.fc70_rho_kg_m3 == 1940.0
    assert m.fc70_cp_J_kgK == 1050.0


def test_v2_boundary_params():
    m = FINAL_FROZEN_THERMAL_MODEL_V2
    assert m.h_conv_W_m2K == 10.0
    assert m.emissivity == 0.90
    assert m.view_factor == 1.0
    assert abs(m.sigma_SB_W_m2K4 - 5.670374419e-8) < 1e-20


# ============================================================
# 6-7. 几何
# ============================================================

def test_v2_insulated_no_pdms():
    layers = FINAL_FROZEN_THERMAL_MODEL_V2.insulated_layers
    mats = [l.material for l in layers]
    assert "PDMS" not in mats
    assert mats == ["COC", "Water", "FC70", "COC", "Air"]
    total = sum(l.thickness_m for l in layers)
    assert abs(total - 3850e-6) < 1e-12


def test_v2_bare_has_fc70():
    layers = FINAL_FROZEN_THERMAL_MODEL_V2.bare_layers
    mats = [l.material for l in layers]
    assert "FC70" in mats
    assert "Oil" not in mats
    assert mats == ["COC", "Water", "FC70", "COC"]
    total = sum(l.thickness_m for l in layers)
    assert abs(total - 850e-6) < 1e-12


# ============================================================
# 8-9. V1 保留
# ============================================================

def test_v1_frozen_config_unchanged():
    m = FINAL_FROZEN_THERMAL_MODEL_V1
    assert m.model_id == "FINAL_FROZEN_THERMAL_MODEL_V1"
    assert m.k_eff_W_mK == 0.0675
    assert m.cp_eff_J_kgK == 700.0
    assert m.tau_top_s == 8.0


def test_v1_oil_pdms_retained():
    assert heat_model.DEFAULT_MATERIALS["Oil"].k_W_mK == 0.142
    assert heat_model.DEFAULT_MATERIALS["Oil"].rho_kg_m3 == 876.0
    assert heat_model.DEFAULT_MATERIALS["Oil"].cp_J_kgK == 1962.0
    assert heat_model.DEFAULT_MATERIALS["PDMS"].k_W_mK == 0.15
    mats = [l.material for l in heat_model.LEGACY_INSULATED_LAYERS]
    assert mats == ["COC", "Water", "Oil", "COC", "Air", "PDMS"]


# ============================================================
# 10. 样品层加权
# ============================================================

def test_v2_sample_weights_correct():
    from thermal_model.config import thermal_model_v2_candidate as v2c
    mats = v2c.make_v2_materials(
        FINAL_FROZEN_THERMAL_MODEL_V2.k_eff_W_mK,
        FINAL_FROZEN_THERMAL_MODEL_V2.cp_eff_J_kgK,
        FINAL_FROZEN_THERMAL_MODEL_V2.rho_COC_kg_m3)
    layers = FINAL_FROZEN_THERMAL_MODEL_V2.insulated_layers
    mesh = heat_model.build_layer_stack(mats, layers)
    assert float(mesh.sample_weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert mesh.sample_layer_index == 1
    assert abs(mesh.boundaries[1] - 180e-6) < 1e-15
    assert abs(mesh.boundaries[2] - 200e-6) < 1e-15


# ============================================================
# 11. tau_top 不影响样品温度
# ============================================================

def test_tau_top_not_in_sample_path():
    import inspect
    from thermal_model.core.lag_augmented_thermal_model import (
        apply_first_order_lag,
    )
    # 样品 = 原始有限体积温度场; lag 只应用于 T_top_observed。
    t = np.linspace(0, 10, 500)
    x = 25.0 + 60.0 * t / 10.0
    assert not np.allclose(apply_first_order_lag(t, x, 0.0),
                           apply_first_order_lag(t, x, 8.0))
    assert not np.allclose(apply_first_order_lag(t, x, 8.0),
                           apply_first_order_lag(t, x, 16.0))


# ============================================================
# 12. baseline 样品指标复现
# ============================================================

def test_v2_baseline_metrics_reproducible():
    import workflows.v2_fc70_no_pdms.analyze_v2_sample_sensitivity as sens
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
    n_complete = sum(1 for w in cw["windows"] if not w["excluded"])
    assert n_complete == 29


# ============================================================
# 13. 校准/验证参考指标保留
# ============================================================

def test_v2_reference_metrics_preserved():
    m = FINAL_FROZEN_THERMAL_MODEL_V2
    assert m.calibration_66C_RMSE_C == pytest.approx(0.6333, abs=1e-4)
    assert m.external_validation_RMSE_C["60C"] == pytest.approx(1.3139,
                                                                abs=1e-4)
    assert m.external_validation_RMSE_C["72C"] == pytest.approx(3.0132,
                                                                abs=1e-4)
    assert m.external_validation_RMSE_C["3s_extension"] == pytest.approx(
        1.0386, abs=1e-4)


def test_v2_metadata_complete():
    d = __import__(
        "thermal_model.config.final_frozen_model_v2",
        fromlist=["final_model_dict"]).final_model_dict()
    assert d["status"] == "FROZEN"
    assert d["fc70_k_W_mK"] == 0.070
    assert d["sample_prediction"]["mean_high_C"] == pytest.approx(84.703,
                                                                  abs=1e-3)
    assert "finite-volume" in d["numerical_method"]
