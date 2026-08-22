#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy E — 对流 + 非线性辐射 + 一阶滞后模型测试。

覆盖 (任务 23-32 + 参数容器/优化器禁令):
 1   sigma_SB 常数精确 = 5.670374419e-8
 2   生产 h_conv 精确 = 10.0
 3   生产 emissivity 精确 = 0.90
 4   生产 view_factor 精确 = 1.0
 5   Celsius -> Kelvin 换算正确
 6   辐射通量 = 0 当 Ts == Tsur
 7   辐射通量 > 0 当 Ts > Tsur
 8   辐射通量 < 0 当 Ts < Tsur
 9   epsilon=0 -> 辐射通量精确为 0
10   F=0 -> 辐射通量精确为 0
11   等效 h_rad 满足 q_rad = h_rad*(Ts-Tsur) (诊断)
12   参数容器: 默认固定边界参数
13   参数容器: k/cp/rho <= 0 拒绝
14   参数容器: tau < 0 拒绝
15   参数容器: h < 0 拒绝
16   参数容器: eps 越界拒绝
17   参数容器: F 越界拒绝
18   参数容器: alpha/effusivity 派生正确
19   环境解析: 第一个有效实测 Top 值 (NaN 前缀)
20   环境解析: 忽略 inf; 记录来源索引/时间
21   环境解析: 全无效 -> 报错
22   无静默环境回退 (T_environment=None -> 报错)
23   环境 K > 0 校验
24   环境在仿真内恒定 (标量, 不随时变)
25   初始条件 = internal[0] (30 C), 环境 = 25 C, 二者有意不同
26   旧对流回归: eps=0 + h=5 复现 heat_model.run_simulation
27   非线性边界能量平衡 (40/60/72/95 C)
28   辐射增加热损失 (eps 0 vs 0.90, 高温保持)
29   h=10+辐射 冷却顶部多于旧 h=5 模型
30   滞后不改原始 FDM (tau=0 vs tau>0)
31   滞后只改变 T_top_observed_predicted
32   材料完整性: 只替换 COC; water/oil 不变; rho=1020
33   DEFAULT_MATERIALS 不被修改
34   BARE_TOP_COC_LAYERS 不被修改 (同一对象)
35   无优化器调用 (scipy.optimize / minimize / least_squares)
"""
import inspect
from pathlib import Path

import numpy as np
import pytest

from thermal_model.core import heat_model
from thermal_model.core import lag_augmented_thermal_model as lm
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.convection_radiation_thermal_model import (
    ConvectionRadiationParameters,
    SIGMA_SB_W_M2_K4,
    H_CONV_STRATEGY_E_W_M2K,
    EMISSIVITY_STRATEGY_E,
    VIEW_FACTOR_STRATEGY_E,
    RHO_COC_STRATEGY_E,
    KELVIN_OFFSET,
    radiative_heat_flux_W_m2,
    equivalent_radiative_heat_transfer_coefficient,
    infer_environment_from_initial_top_measurement,
    solve_top_surface_temperature,
    boundary_residual_W_m2,
    run_convection_radiation_fdm,
    run_convection_radiation_lag_model,
    make_convection_radiation_materials,
)

MODULE_FILE = Path(cr.__file__)

# 合成短协议 (每次模拟 ~0.1-0.3 s)
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 78.0, 80.0,
                  80.0, 78.0, 70.0, 60.0, 50.0])

K_A = 0.0165   # 策略 A k_eff
CP_A = 900.0   # 策略 A cp_eff


def _materials(k=K_A, cp=CP_A):
    return make_convection_radiation_materials(k, cp)


# ===============================================================
# 1-11. 基本辐射物理
# ===============================================================

def test_sigma_sb_constant_exact():
    assert SIGMA_SB_W_M2_K4 == 5.670374419e-8


def test_production_h_conv_exact():
    assert H_CONV_STRATEGY_E_W_M2K == 10.0


def test_production_emissivity_exact():
    assert EMISSIVITY_STRATEGY_E == 0.90


def test_production_view_factor_exact():
    assert VIEW_FACTOR_STRATEGY_E == 1.0


def test_celsius_to_kelvin_conversion():
    assert 0.0 + KELVIN_OFFSET == pytest.approx(273.15)
    assert 25.0 + KELVIN_OFFSET == pytest.approx(298.15)
    # 辐射辅助内部换算: 通量必须等于直接开尔文四次方公式
    q = radiative_heat_flux_W_m2(72.0, 27.8)
    q_direct = (0.90 * SIGMA_SB_W_M2_K4 * 1.0
                * ((72.0 + 273.15) ** 4 - (27.8 + 273.15) ** 4))
    assert q == pytest.approx(q_direct, rel=1e-14)


def test_radiation_flux_zero_when_equal():
    for T in (25.0, 40.0, 72.0, 95.0):
        assert radiative_heat_flux_W_m2(T, T) == 0.0


def test_radiation_flux_positive_when_hotter():
    q = radiative_heat_flux_W_m2(72.0, 27.8)
    assert q > 0.0


def test_radiation_flux_negative_when_cooler():
    q = radiative_heat_flux_W_m2(20.0, 27.8)
    assert q < 0.0


def test_epsilon_zero_gives_exact_zero_radiation():
    assert radiative_heat_flux_W_m2(95.0, 25.0, emissivity=0.0) == 0.0


def test_view_factor_zero_gives_exact_zero_radiation():
    assert radiative_heat_flux_W_m2(95.0, 25.0, view_factor=0.0) == 0.0


def test_equivalent_h_rad_diagnostic_consistency():
    # h_rad 必须满足 q_rad = h_rad * (Ts_C - Tsur_C)
    for Ts, Tsur in ((40.0, 27.8), (60.0, 27.8), (72.0, 27.8),
                     (95.0, 27.8)):
        q = radiative_heat_flux_W_m2(Ts, Tsur)
        h = equivalent_radiative_heat_transfer_coefficient(Ts, Tsur)
        assert q == pytest.approx(h * (Ts - Tsur), rel=1e-12)


# ===============================================================
# 12-18. 参数容器
# ===============================================================

def test_parameter_defaults():
    p = ConvectionRadiationParameters(k_eff_W_mK=0.0165,
                                      cp_eff_J_kgK=900.0, tau_lag_s=0.0)
    assert p.h_conv_W_m2K == 10.0
    assert p.emissivity == 0.90
    assert p.sigma_SB_W_m2K4 == 5.670374419e-8
    assert p.view_factor == 1.0
    assert p.rho_COC_kg_m3 == 1020.0


def test_parameter_rejects_nonpositive_k_cp_rho():
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0)
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=-0.1, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0)
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=0.0,
                                      tau_lag_s=0.0)
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0, rho_COC_kg_m3=0.0)


def test_parameter_rejects_negative_tau():
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=900.0,
                                      tau_lag_s=-0.1)


def test_parameter_rejects_negative_h():
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0, h_conv_W_m2K=-1.0)


def test_parameter_rejects_out_of_range_emissivity():
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0, emissivity=-0.1)
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0, emissivity=1.1)


def test_parameter_rejects_out_of_range_view_factor():
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0, view_factor=-0.1)
    with pytest.raises(ValueError):
        ConvectionRadiationParameters(k_eff_W_mK=0.0165, cp_eff_J_kgK=900.0,
                                      tau_lag_s=0.0, view_factor=1.1)


def test_parameter_derived_alpha_and_effusivity():
    p = ConvectionRadiationParameters(k_eff_W_mK=0.0165,
                                      cp_eff_J_kgK=900.0, tau_lag_s=0.0)
    assert p.alpha_eff_m2_s == pytest.approx(0.0165 / (1020.0 * 900.0))
    assert p.effusivity == pytest.approx(np.sqrt(0.0165 * 1020.0 * 900.0))


# ===============================================================
# 19-24. 环境温度解析与无回退
# ===============================================================

def test_environment_from_first_valid_top_measurement():
    top = np.array([np.nan, 27.8, 28.0, 28.2, 30.0])
    env = infer_environment_from_initial_top_measurement(top)
    assert env["T_environment_C"] == 27.8
    assert env["source_index"] == 1


def test_environment_ignores_inf_and_records_time():
    top = np.array([np.inf, 27.8, 28.0])
    t = np.array([0.0, 1.0, 2.0])
    env = infer_environment_from_initial_top_measurement(top, time_s=t)
    assert env["T_environment_C"] == 27.8
    assert env["source_index"] == 1
    assert env["source_time_s"] == 1.0


def test_environment_all_invalid_raises():
    with pytest.raises(ValueError):
        infer_environment_from_initial_top_measurement(
            np.array([np.nan, np.nan, np.inf]))


def test_environment_missing_raises_no_silent_fallback():
    # 高层 runner: 缺 T_environment_C -> 报错, 不静默用 25 C
    with pytest.raises(ValueError, match="T_environment_C"):
        run_convection_radiation_lag_model(
            time_s=_T, bottom_temperature_C=_TINT,
            T_environment_C=None,
            k_eff_W_mK=K_A, cp_eff_J_kgK=CP_A, tau_lag_s=0.0)
    # 底层 FDM: 缺 T_air_C -> 报错
    with pytest.raises(ValueError, match="T_air_C"):
        run_convection_radiation_fdm(
            time_s=_T, bottom_temperature_C=_TINT,
            materials=_materials(), layers=heat_model.BARE_TOP_COC_LAYERS,
            T_air_C=None, T_surroundings_C=25.0)


def test_environment_kelvin_positive_required():
    with pytest.raises(ValueError, match="开尔文"):
        run_convection_radiation_fdm(
            time_s=_T, bottom_temperature_C=_TINT,
            materials=_materials(), layers=heat_model.BARE_TOP_COC_LAYERS,
            T_air_C=-300.0, T_surroundings_C=25.0)


def test_environment_constant_scalar_during_run():
    env = 27.8
    out = run_convection_radiation_lag_model(
        time_s=_T, bottom_temperature_C=_TINT,
        T_environment_C=env,
        k_eff_W_mK=K_A, cp_eff_J_kgK=CP_A, tau_lag_s=0.0, save_dt=0.5)
    assert out["T_air_C"] == env
    assert out["T_surroundings_C"] == env
    assert isinstance(out["T_air_C"], float)
    # 模型内部环境是常数标量: 两个环境字段与 T_environment 恒等
    assert out["T_air_C"] == out["T_surroundings_C"]


# ===============================================================
# 25. 初始条件 = internal[0], 环境 = top[0] (有意不同)
# ===============================================================

def test_initial_condition_is_internal_first_value():
    t_int0 = 30.0
    t_env = 25.0
    time_s = np.array([0.0, 0.5, 1.0])
    T_int = np.array([t_int0, 35.0, 40.0])
    out = run_convection_radiation_lag_model(
        time_s=time_s, bottom_temperature_C=T_int,
        T_environment_C=t_env,
        k_eff_W_mK=K_A, cp_eff_J_kgK=CP_A, tau_lag_s=0.0, save_dt=0.5)
    # 首个保存点 = 初始均匀场 internal[0]
    assert out["T_sample_FDM_C"][0] == pytest.approx(t_int0, abs=1e-9)
    assert out["T_top_FDM_C"][0] == pytest.approx(t_int0, abs=1e-9)
    assert out["T_air_C"] == t_env          # 环境不同
    assert out["T_surroundings_C"] == t_env
    assert out["T_air_C"] != t_int0         # 30 vs 25 有意不同


# ===============================================================
# 26. 旧对流回归 (eps=0 + h=5)
# ===============================================================

def test_regression_against_convection_only_solver():
    mats = _materials()
    layers = heat_model.BARE_TOP_COC_LAYERS
    env = 27.8
    t_init = float(_TINT[0])
    save_dt = 0.5

    old = heat_model.run_simulation(
        time_s=_T, bottom_temperature_C=_TINT, materials=mats, layers=layers,
        h_conv=5.0, T_air_ambient=env, save_dt=save_dt,
        T_initial_C=t_init)

    new = run_convection_radiation_fdm(
        time_s=_T, bottom_temperature_C=_TINT, materials=mats, layers=layers,
        T_air_C=env, T_surroundings_C=env,
        h_conv_W_m2K=5.0, emissivity=0.0,
        save_dt=save_dt, T_initial_C=t_init)

    np.testing.assert_allclose(new["T_sample_arr"], old["T_sample_arr"],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new["T_top_surface_arr"],
                               old["T_top_surface_arr"],
                               rtol=1e-6, atol=1e-6)
    # 能量平衡残差: eps=0 时 Newton 一步收敛到解析解, 残差应接近机器精度。
    # 注意: 第一个保存点 (t=0) 是均匀初始场 (表面尚未求解), 残差 = -h*(T0-T_env)
    # 为物理预期的初始瞬态, 不参与该断言。
    resid = new["boundary_residual_arr"]
    assert np.max(np.abs(resid[1:])) < 1e-6


def test_regression_tight_tolerance_sample_trace():
    """样品迹线应远小于顶部: 远离边界, 由底部 Dirichlet 主导。"""
    mats = _materials()
    layers = heat_model.BARE_TOP_COC_LAYERS
    env = 27.8
    t_init = float(_TINT[0])
    save_dt = 0.5
    old = heat_model.run_simulation(
        time_s=_T, bottom_temperature_C=_TINT, materials=mats, layers=layers,
        h_conv=5.0, T_air_ambient=env, save_dt=save_dt, T_initial_C=t_init)
    new = run_convection_radiation_fdm(
        time_s=_T, bottom_temperature_C=_TINT, materials=mats, layers=layers,
        T_air_C=env, T_surroundings_C=env,
        h_conv_W_m2K=5.0, emissivity=0.0,
        save_dt=save_dt, T_initial_C=t_init)
    np.testing.assert_allclose(new["T_sample_arr"], old["T_sample_arr"],
                               rtol=1e-9, atol=1e-9)


# ===============================================================
# 27. 非线性边界能量平衡
# ===============================================================

def test_nonlinear_boundary_energy_balance():
    k_over_dx = 0.0165 / 5e-6   # 策略 A k / 5 um 网格
    env = 27.8
    for T_prev in (40.0, 60.0, 72.0, 95.0):
        Ts = solve_top_surface_temperature(
            T_prev_C=T_prev, T_air_C=env, T_surroundings_C=env,
            k_over_dx=k_over_dx, h_conv_W_m2K=10.0,
            emissivity=0.90, sigma_SB_W_m2K4=SIGMA_SB_W_M2_K4,
            view_factor=1.0)
        r = boundary_residual_W_m2(
            T_prev_C=T_prev, T_surface_C=Ts, T_air_C=env,
            T_surroundings_C=env, k_over_dx=k_over_dx,
            h_conv_W_m2K=10.0, emissivity=0.90,
            sigma_SB_W_m2K4=SIGMA_SB_W_M2_K4, view_factor=1.0)
        assert abs(r) < 1e-8, f"T_prev={T_prev} C 残差 {r:.3e} 过大"


def test_nonlinear_boundary_solution_monotonic_in_prev():
    env = 27.8
    k_over_dx = 0.0165 / 5e-6
    Ts_list = [
        solve_top_surface_temperature(
            T_prev_C=tp, T_air_C=env, T_surroundings_C=env,
            k_over_dx=k_over_dx, h_conv_W_m2K=10.0,
            emissivity=0.90, sigma_SB_W_m2K4=SIGMA_SB_W_M2_K4,
            view_factor=1.0)
        for tp in (40.0, 60.0, 72.0, 95.0)
    ]
    assert Ts_list == sorted(Ts_list)  # 表面温度随内部节点单调上升


# ===============================================================
# 28. 辐射增加热损失
# ===============================================================

def test_radiation_increases_heat_loss():
    # 相同 k/cp/h_conv/底部/环境: eps=0 vs eps=0.90
    env = 27.8
    t_init = float(_TINT[0])
    save_dt = 0.5
    base = dict(time_s=_T, bottom_temperature_C=_TINT,
                materials=_materials(), layers=heat_model.BARE_TOP_COC_LAYERS,
                T_air_C=env, T_surroundings_C=env, h_conv_W_m2K=10.0,
                save_dt=save_dt, T_initial_C=t_init)

    no_rad = run_convection_radiation_fdm(**base, emissivity=0.0)
    with_rad = run_convection_radiation_fdm(**base, emissivity=0.90)

    # 充分演化后 (末段): 有辐射的顶部 <= 无辐射的顶部
    assert np.all(with_rad["T_top_surface_arr"][-3:]
                  <= no_rad["T_top_surface_arr"][-3:] + 1e-9)
    # 末点明确更低
    assert (with_rad["T_top_surface_arr"][-1]
            < no_rad["T_top_surface_arr"][-1])


# ===============================================================
# 29. h=10 + 辐射 冷却顶部多于旧 h=5 模型 (合成保持)
# ===============================================================

def test_h10_radiation_cools_more_than_old_h5():
    # 受控合成保持协议: 底部快速升到 90 C 并保持
    t_hold = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    t_hold_int = np.array([30.0, 60.0, 85.0, 90.0, 90.0, 90.0, 90.0,
                           90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0,
                           90.0, 90.0])
    mats = _materials()
    layers = heat_model.BARE_TOP_COC_LAYERS
    env = 27.8
    save_dt = 0.5

    old = heat_model.run_simulation(
        time_s=t_hold, bottom_temperature_C=t_hold_int, materials=mats,
        layers=layers, h_conv=5.0, T_air_ambient=env, save_dt=save_dt,
        T_initial_C=30.0)

    new = run_convection_radiation_lag_model(
        time_s=t_hold, bottom_temperature_C=t_hold_int,
        T_environment_C=env, k_eff_W_mK=K_A, cp_eff_J_kgK=CP_A,
        tau_lag_s=0.0, save_dt=save_dt)

    # 高温保持末期: 新模型顶部显著更低
    n = min(len(new["T_top_FDM_C"]), len(old["T_top_surface_arr"]))
    assert (new["T_top_FDM_C"][n - 1]
            < old["T_top_surface_arr"][n - 1])
    # 差距应清晰 (不是数值噪声): 末点差 > 0.2 C
    assert (old["T_top_surface_arr"][n - 1]
            - new["T_top_FDM_C"][n - 1]) > 0.2


# ===============================================================
# 30-31. 滞后不改原始 FDM
# ===============================================================

def test_lag_does_not_change_raw_fdm():
    env = 27.8
    save_dt = 0.5
    out0 = run_convection_radiation_lag_model(
        time_s=_T, bottom_temperature_C=_TINT, T_environment_C=env,
        k_eff_W_mK=K_A, cp_eff_J_kgK=CP_A, tau_lag_s=0.0, save_dt=save_dt)
    out1 = run_convection_radiation_lag_model(
        time_s=_T, bottom_temperature_C=_TINT, T_environment_C=env,
        k_eff_W_mK=K_A, cp_eff_J_kgK=CP_A, tau_lag_s=2.0, save_dt=save_dt)

    np.testing.assert_array_equal(out0["T_sample_FDM_C"],
                                  out1["T_sample_FDM_C"])
    np.testing.assert_array_equal(out0["T_top_FDM_C"], out1["T_top_FDM_C"])
    # tau=0 严格恒等: 观察预测 == 原始顶部 FDM
    np.testing.assert_array_equal(out0["T_top_observed_predicted_C"],
                                  out0["T_top_FDM_C"])
    # tau>0: 观察预测与原始顶部不同 (迹线非恒等)
    assert not np.array_equal(out1["T_top_observed_predicted_C"],
                              out1["T_top_FDM_C"])


def test_lag_zero_exact_identity_via_lag_module():
    t = np.array([0.0, 1.0, 2.5, 4.0])
    x = np.array([30.0, 40.0, 35.0, 50.0])
    y = lm.apply_first_order_lag(t, x, 0.0)
    np.testing.assert_array_equal(y, x)  # 位级一致


# ===============================================================
# 32-34. 材料完整性
# ===============================================================

def test_only_coc_replaced():
    mats = _materials(k=0.0165, cp=900.0)
    default = heat_model.DEFAULT_MATERIALS
    for name in ("Water", "Oil", "Air", "PDMS"):
        m_new = mats[name]
        m_def = default[name]
        assert m_new.k_W_mK == m_def.k_W_mK
        assert m_new.rho_kg_m3 == m_def.rho_kg_m3
        assert m_new.cp_J_kgK == m_def.cp_J_kgK
    assert mats["COC"].k_W_mK == 0.0165
    assert mats["COC"].cp_J_kgK == 900.0
    assert mats["COC"].rho_kg_m3 == 1020.0


def test_default_materials_not_mutated():
    before = heat_model.DEFAULT_MATERIALS["COC"]
    _materials(k=0.0165, cp=900.0)
    after = heat_model.DEFAULT_MATERIALS["COC"]
    assert after is before
    assert after.k_W_mK == 0.13
    assert after.cp_J_kgK == 1800.0
    assert after.rho_kg_m3 == 1020.0


def test_bare_top_layers_unchanged_object():
    layers_before = heat_model.BARE_TOP_COC_LAYERS
    _materials(k=0.0165, cp=900.0)
    assert heat_model.BARE_TOP_COC_LAYERS is layers_before
    assert [l.name for l in heat_model.BARE_TOP_COC_LAYERS] == [
        "Bottom COC", "PCR Sample", "Mineral Oil", "Top COC"]


def test_sample_semantics_separated():
    out = run_convection_radiation_lag_model(
        time_s=_T, bottom_temperature_C=_TINT, T_environment_C=27.8,
        k_eff_W_mK=K_A, cp_eff_J_kgK=CP_A, tau_lag_s=1.0, save_dt=0.5)
    np.testing.assert_array_equal(out["T_sample_predicted_C"],
                                  out["T_sample_FDM_C"])


# ===============================================================
# 35. 无优化器调用
# ===============================================================

def test_no_optimizer_usage():
    src = MODULE_FILE.read_text(encoding="utf-8")
    for forbidden in ("scipy.optimize", "minimize", "least_squares",
                      "differential_evolution", "curve_fit"):
        assert forbidden not in src, f"模块中不应出现 {forbidden!r}"


# ===============================================================
# 附加: 求解器不读取实测顶部数据 / CSV (架构约束)
# ===============================================================

def test_solver_does_not_read_csv_or_measured_top():
    src = MODULE_FILE.read_text(encoding="utf-8")
    for forbidden in ("read_csv", "read_excel", "T_top_measured",
                      "aligned_internal_top", "T_top_meas"):
        assert forbidden not in src, f"求解器模块中不应出现 {forbidden!r}"


def test_newton_failure_raises():
    # 强制 Newton 无法收敛: 极小迭代上限 + 恶劣初猜应抛异常
    with pytest.raises(RuntimeError, match="未收敛"):
        solve_top_surface_temperature(
            T_prev_C=72.0, T_air_C=27.8, T_surroundings_C=27.8,
            k_over_dx=0.0165 / 5e-6, h_conv_W_m2K=10.0,
            emissivity=0.90, sigma_SB_W_m2K4=SIGMA_SB_W_M2_K4,
            view_factor=1.0, T_initial_guess_C=1e6,
            abs_tolerance_C=1e-10, max_iterations=1)
