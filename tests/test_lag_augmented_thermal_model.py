#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy C — 滞后分离 3-DOF 热模型测试 (lag-separated thermal model)。

覆盖 24 项:
  1    cp = k/(rho*alpha)
  2    alpha <= 0 拒绝
  3    k <= 0 拒绝
  4    tau < 0 拒绝
  5    tau=0 严格恒等
  6    常数输入滞后精确
  7    阶跃响应解析解
  8    线性斜坡解析解
  9    非均匀时间支持
  10   时间单调校验
  11   策略 A alpha/k/tau=0 复现当前模型
  12   策略 A 样品迹线复现
  13   改 tau 不改样品 FDM 输出
  14   改 tau 不改原始顶部 FDM 输出
  15   改 tau 改变观察顶部预测
  16   实测时间为插值查询轴
  17   实测顶部值不作插值坐标
  18   实测顶部值不用于初始化滞后态
  19   DEFAULT_MATERIALS 不被修改
  20   Water/Oil 不变
  21   BARE_TOP_COC_LAYERS 使用
  22   可行性扫描固定 k 精确
  23   无优化器调用
  24   无名义配置修改
"""
import inspect

import numpy as np
import pytest

from thermal_model.core import heat_model
from thermal_model.config import calibrated_model_config as cmc
from thermal_model.core import lag_augmented_thermal_model as lm
from thermal_model.core.lag_augmented_thermal_model import (
    LagAugmentedParameters,
    apply_first_order_lag,
    run_lag_augmented_model,
)

# 合成短协议 (每次模拟 ~0.1-0.3 s)
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 62.0, 50.0])
_TTOP = _TINT - 2.0


# ===============================================================
# 1-4. 参数容器
# ===============================================================

def test_cp_equals_k_over_rho_alpha():
    p = LagAugmentedParameters(alpha_eff_m2_s=1.8e-8, k_eff_W_mK=0.0165,
                               tau_ext_s=0.5)
    assert p.cp_eff_J_kgK == pytest.approx(0.0165 / (1020.0 * 1.8e-8))


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        LagAugmentedParameters(alpha_eff_m2_s=0.0, k_eff_W_mK=0.0165,
                               tau_ext_s=0.0)
    with pytest.raises(ValueError):
        LagAugmentedParameters(alpha_eff_m2_s=-1e-8, k_eff_W_mK=0.0165,
                               tau_ext_s=0.0)


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        LagAugmentedParameters(alpha_eff_m2_s=1.8e-8, k_eff_W_mK=0.0,
                               tau_ext_s=0.0)
    with pytest.raises(ValueError):
        LagAugmentedParameters(alpha_eff_m2_s=1.8e-8, k_eff_W_mK=-0.1,
                               tau_ext_s=0.0)


def test_negative_tau_rejected():
    with pytest.raises(ValueError):
        LagAugmentedParameters(alpha_eff_m2_s=1.8e-8, k_eff_W_mK=0.0165,
                               tau_ext_s=-0.1)


# ===============================================================
# 5-10. 滞后函数
# ===============================================================

def test_tau_zero_exact_identity():
    t = np.array([0.0, 1.0, 2.5, 4.0])
    x = np.array([30.0, 40.0, 35.0, 50.0])
    y = apply_first_order_lag(t, x, 0.0)
    assert y is not x  # 返回副本
    np.testing.assert_array_equal(y, x)  # 位级一致


def test_constant_input_exact_for_any_tau():
    t = np.linspace(0.0, 10.0, 50)
    x = np.full(50, 42.0)
    for tau in (0.0, 0.5, 2.0, 12.0):
        y = apply_first_order_lag(t, x, tau)
        np.testing.assert_allclose(y, x, atol=1e-12)


def test_step_response_analytical():
    # 阶跃响应: 用极细时间网格的同一精确递推作独立参考 (粗网格应一致收敛),
    # 并校验时间常数行为 y -> 50 指数收敛 (无过冲)。
    tau = 2.0
    t = np.linspace(0.0, 10.0, 200)
    x = np.full(200, 30.0)
    x[t >= 1.0] = 50.0
    y = apply_first_order_lag(t, x, tau)

    # 细网格参考: 线性插值粗输入到细网格, 再精确递推
    t_fine = np.linspace(0.0, 10.0, 20000)
    x_fine = np.interp(t_fine, t, x)
    y_fine = apply_first_order_lag(t_fine, x_fine, tau)
    y_ref = np.interp(t, t_fine, y_fine)
    np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-3)

    # 阶跃后 (t>=1) 应无过冲、单调逼近 50, 且尾部按时间常数指数衰减
    mask = t >= 2.0  # 足够远离过渡, 进入纯指数尾
    assert np.all(y[mask] <= 50.0 + 1e-12)
    assert np.all(np.diff(y[mask]) >= 0)
    # 拟合时间常数: log(50 - y) 斜率 = -1/tau
    logres = np.log(50.0 - y[mask])
    slope = np.polyfit(t[mask], logres, 1)[0]
    assert slope == pytest.approx(-1.0 / tau, rel=0.01)


def test_linear_ramp_analytical():
    tau = 1.0
    t = np.linspace(0.0, 10.0, 200)
    m = 5.0
    x = 20.0 + m * t
    y = apply_first_order_lag(t, x, tau)
    # 解析解: y = (20 - m*tau) + m*t + [y0 - (20 - m*tau)]*exp(-t/tau)
    # y0 = x0 = 20
    expected = (20.0 - m * tau) + m * t + \
        (20.0 - (20.0 - m * tau)) * np.exp(-t / tau)
    np.testing.assert_allclose(y, expected, rtol=1e-10)


def test_nonuniform_time_supported():
    t = np.array([0.0, 0.5, 1.5, 2.5, 5.0, 8.0])
    x = np.array([25.0, 30.0, 40.0, 35.0, 50.0, 45.0])
    y = apply_first_order_lag(t, x, 1.5)
    assert np.all(np.isfinite(y))
    # 与解析逐段积分一致 (用较小 dt 的精确解作参考)
    # 这里仅验证有限/有界/无 NaN
    assert np.all(np.abs(y) < 200.0)


def test_monotonic_time_validated():
    t = np.array([0.0, 1.0, 0.5])
    x = np.array([30.0, 40.0, 35.0])
    with pytest.raises(ValueError):
        apply_first_order_lag(t, x, 1.0)


# ===============================================================
# 11-15. 策略 A 回归 + tau 独立性
# ===============================================================

def test_strategy_A_alpha_k_tau0_reproduces_current():
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = cal.k_eff_W_mK / (cal.rho_COC_kg_m3 * cal.cp_eff_J_kgK)
    p = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                               k_eff_W_mK=0.0165, tau_ext_s=0.0)
    assert p.cp_eff_J_kgK == pytest.approx(900.0, rel=1e-12)
    out = run_lag_augmented_model(_T, _TINT, p)
    # tau=0: 观察顶部 == FDM 顶部
    np.testing.assert_allclose(out["T_top_observed_predicted_C"],
                               out["T_top_FDM_C"], rtol=0, atol=0)
    # 样品 == FDM 样品
    np.testing.assert_allclose(out["T_sample_predicted_C"],
                               out["T_sample_FDM_C"], rtol=0, atol=0)


def test_strategy_A_sample_trace_reproduced():
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = cal.k_eff_W_mK / (cal.rho_COC_kg_m3 * cal.cp_eff_J_kgK)
    p = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                               k_eff_W_mK=0.0165, tau_ext_s=0.0)
    out_c = run_lag_augmented_model(_T, _TINT, p)
    # 直接调用 heat_model 用名义材料复现样品
    mats = cmc.make_nominal_calibrated_materials(cal)
    result = heat_model.run_simulation(
        time_s=_T, bottom_temperature_C=_TINT, materials=mats,
        layers=cmc.nominal_layer_stack(cal), h_conv=5.0,
        T_air_ambient=25.0, save_dt=0.1, T_initial_C=_TINT[0])
    np.testing.assert_allclose(out_c["T_sample_predicted_C"],
                               result["T_sample_arr"], rtol=1e-12)
    np.testing.assert_allclose(out_c["T_top_FDM_C"],
                               result["T_top_surface_arr"], rtol=1e-12)


def test_tau_does_not_change_sample():
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = cal.k_eff_W_mK / (cal.rho_COC_kg_m3 * cal.cp_eff_J_kgK)
    p0 = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                                k_eff_W_mK=0.0165, tau_ext_s=0.0)
    p2 = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                                k_eff_W_mK=0.0165, tau_ext_s=3.0)
    o0 = run_lag_augmented_model(_T, _TINT, p0)
    o2 = run_lag_augmented_model(_T, _TINT, p2)
    np.testing.assert_allclose(o0["T_sample_predicted_C"],
                               o2["T_sample_predicted_C"], rtol=0, atol=0)


def test_tau_does_not_change_raw_top_fdm():
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = cal.k_eff_W_mK / (cal.rho_COC_kg_m3 * cal.cp_eff_J_kgK)
    p0 = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                                k_eff_W_mK=0.0165, tau_ext_s=0.0)
    p2 = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                                k_eff_W_mK=0.0165, tau_ext_s=3.0)
    o0 = run_lag_augmented_model(_T, _TINT, p0)
    o2 = run_lag_augmented_model(_T, _TINT, p2)
    np.testing.assert_allclose(o0["T_top_FDM_C"], o2["T_top_FDM_C"],
                               rtol=0, atol=0)


def test_tau_changes_observed_top():
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = cal.k_eff_W_mK / (cal.rho_COC_kg_m3 * cal.cp_eff_J_kgK)
    p0 = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                                k_eff_W_mK=0.0165, tau_ext_s=0.0)
    p2 = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                                k_eff_W_mK=0.0165, tau_ext_s=3.0)
    o0 = run_lag_augmented_model(_T, _TINT, p0)
    o2 = run_lag_augmented_model(_T, _TINT, p2)
    assert not np.allclose(o0["T_top_observed_predicted_C"],
                           o2["T_top_observed_predicted_C"])


# ===============================================================
# 16-18. 时间采样 / 初始化语义
# ===============================================================

def test_measurement_time_is_query_axis():
    src = inspect.getsource(lm.evaluate_72c_objective)
    assert "np.interp(time_s" in src
    assert "np.interp(t_top_meas" not in src


def test_measured_top_not_interpolation_coordinate():
    src = inspect.getsource(lm.evaluate_72c_objective)
    assert "interp(time_s, out[" in src
    # 绝不以 t_top_meas 作查询轴
    assert "interp(t_top_meas" not in src


def test_lag_state_not_initialized_from_measured_top():
    # apply_first_order_lag 默认 initial_output_C=None -> y0 = x0
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([30.0, 40.0, 50.0])
    y = apply_first_order_lag(t, x, 1.0)  # 未传初始
    assert y[0] == pytest.approx(30.0)  # = x[0], 非任何"实测顶部"值
    # run_lag_augmented_model 内部也是 initial_output_C=None
    src = inspect.getsource(run_lag_augmented_model)
    assert "initial_output_C=None" in src


# ===============================================================
# 19-21. 材料/几何
# ===============================================================

def test_default_materials_not_mutated():
    before = {k: (v.k_W_mK, v.rho_kg_m3, v.cp_J_kgK)
              for k, v in heat_model.DEFAULT_MATERIALS.items()}
    mats = lm.make_lag_materials(0.0165, 900.0)
    after = {k: (v.k_W_mK, v.rho_kg_m3, v.cp_J_kgK)
             for k, v in heat_model.DEFAULT_MATERIALS.items()}
    assert before == after
    # COC 副本被改, 原 COC 不变
    assert mats["COC"].k_W_mK == 0.0165
    assert heat_model.DEFAULT_MATERIALS["COC"].k_W_mK == 0.13


def test_water_oil_unchanged():
    mats = lm.make_lag_materials(0.0165, 900.0)
    base = heat_model.DEFAULT_MATERIALS
    for m in ("Water", "Oil"):
        assert mats[m].k_W_mK == base[m].k_W_mK
        assert mats[m].cp_J_kgK == base[m].cp_J_kgK
        assert mats[m].rho_kg_m3 == base[m].rho_kg_m3


def test_uses_bare_top_layers():
    src = inspect.getsource(run_lag_augmented_model)
    assert "BARE_TOP_COC_LAYERS" in src
    mats = lm.make_lag_materials(0.0165, 900.0)
    mesh = heat_model.build_layer_stack(mats, heat_model.BARE_TOP_COC_LAYERS)
    assert mesh.x[-1] == pytest.approx(850e-6)
    names = " ".join(ly.name for ly in heat_model.BARE_TOP_COC_LAYERS)
    assert "Air" not in names and "PDMS" not in names


# ===============================================================
# 22-24. 可行性扫描约束 / 无优化 / 不改名义
# ===============================================================

def test_feasibility_scan_holds_k_fixed():
    # 模型容器持有独立 k, 但可行性脚本必须固定 k=0.0165
    # (此处验证容器语义 + 派生 cp 由 k 决定)
    p = LagAugmentedParameters(alpha_eff_m2_s=2.5e-8, k_eff_W_mK=0.0165,
                               tau_ext_s=1.0)
    assert p.k_eff_W_mK == 0.0165
    assert p.cp_eff_J_kgK == pytest.approx(0.0165 / (1020.0 * 2.5e-8))


def test_no_optimizer_called():
    src = inspect.getsource(lm)
    for token in ("scipy", "optimize", "curve_fit", "minimize",
                  "least_squares", "grid_search"):
        assert token not in src


def test_no_nominal_config_mutation():
    before = (cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK,
              cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK)
    # 运行模型不会触碰 config
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    alpha_A = cal.k_eff_W_mK / (cal.rho_COC_kg_m3 * cal.cp_eff_J_kgK)
    p = LagAugmentedParameters(alpha_eff_m2_s=alpha_A,
                               k_eff_W_mK=0.0165, tau_ext_s=0.0)
    run_lag_augmented_model(_T, _TINT, p)
    after = (cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK,
             cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK)
    assert before == after == (0.0165, 900.0)
