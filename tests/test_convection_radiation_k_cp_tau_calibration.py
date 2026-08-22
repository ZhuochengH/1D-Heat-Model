#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy F — convection + radiation k/cp/tau 重标定测试。

覆盖 (任务 39 的 31 项):
 1  Strategy E 模型被复用 (导入并使用 cr 求解器)
 2  h 保持精确 10
 3  epsilon 保持精确 0.90
 4  非线性辐射保持启用 (求解器内不线性化; 无固定 h_rad)
 5  环境用第一个有效实测 Top COC 值
 6  初始 FDM 场用第一个内部温度
 7  底部边界用动态内部迹线
 8  只有 k/cp/tau 变化 (固定边界不随网格变)
 9  rho 保持 1020
10  Water/Oil 不变
11  alpha = k/(rho*cp)
12  effusivity 公式正确
13  k 网格精确
14  cp 网格精确
15  tau 网格精确
16  唯一 FDM 评估数 = 63
17  总滞后组合 = 504
18  每个唯一 k/cp 对只跑一次 FDM (缓存计数)
19  tau 剖面复用缓存 FDM
20  实测时间是插值查询轴
21  顶部温度值永不作插值查询坐标
22  全局最小选择只用 RMSE
23  目标中无高-k 奖励
24  cp>=800 子集正确
25  tau 上界警告 (8 s) 正确
26  平衡物理候选规则正确
27  无 DOE11
28  无长保持 PCR
29  无连续优化器
30  之前的 Strategy E 输出不被覆盖
31  历史名义 Strategy A 不变
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core import lag_augmented_thermal_model as lm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAL_SCRIPT = (PROJECT_ROOT / "workflows/calibration"
              / "convection_radiation_k_cp_tau_calibration.py")
STRATEGY_E_CHECK_DIR = (
    PROJECT_ROOT / "model_comparison_output"
    / "convection_radiation_lag_v1")


def _load_cal():
    spec = importlib.util.spec_from_file_location("crkt_cal", CAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cal = None


@pytest.fixture(scope="module")
def c():
    global cal
    if cal is None:
        cal = _load_cal()
    return cal


# ===============================================================
# 1-7. 物理复用与边界规则
# ===============================================================

def test_strategy_e_model_reused(c):
    assert c.cr is cr
    assert c.cr.run_convection_radiation_fdm is cr.run_convection_radiation_fdm
    assert c.cr.run_convection_radiation_lag_model is \
        cr.run_convection_radiation_lag_model


def test_h_exactly_10(c):
    assert c.cr.H_CONV_STRATEGY_E_W_M2K == 10.0
    assert c.cr.ConvectionRadiationParameters(k_eff_W_mK=0.0165,
                                              cp_eff_J_kgK=900.0,
                                              tau_lag_s=0.0).h_conv_W_m2K == 10.0


def test_epsilon_exactly_090(c):
    assert c.cr.EMISSIVITY_STRATEGY_E == 0.90


def test_nonlinear_radiation_enabled(c):
    # 非线性边界求解函数体内不使用固定 h_rad (仅诊断辅助含 h_rad_equiv)
    import inspect
    body = inspect.getsource(cr._solve_surface_newton)
    assert "h_rad" not in body
    body2 = inspect.getsource(cr.solve_top_surface_temperature)
    assert "h_rad" not in body2
    # FDM 内部使用非线性 Stefan-Boltzmann (含四次方项)
    text = Path(cr.__file__).read_text(encoding="utf-8")
    assert "** 4" in text or "**4" in text


def test_environment_first_valid_top(c):
    t_top = np.array([np.nan, 27.8, 28.0, 28.2])
    env = cr.infer_environment_from_initial_top_measurement(t_top)
    assert env["T_environment_C"] == 27.8


def test_initial_condition_first_internal(c):
    # 扫描路径: T_initial = 第一个内部温度 (显式传给 FDM)
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert "T_initial_C=float(t_int[0])" in src


def test_bottom_boundary_dynamic_internal(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert "bottom_temperature_C=t_int" in src
    assert "T_surroundings_C=t_env" in src


def test_only_k_cp_tau_vary(c):
    # 网格仅包含 k/cp/tau; 固定边界参数来自 Strategy E 常量
    assert c.K_GRID == [0.0165, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
                        0.090, 0.120]
    assert c.CP_GRID == [700.0, 800.0, 900.0, 1000.0, 1200.0, 1500.0, 1800.0]
    assert c.TAU_GRID == [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    # 固定边界在 evaluate 中不出现可调参数
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert "h_conv_W_m2K=" not in src.replace(
        "cr.H_CONV_STRATEGY_E_W_M2K", "")
    assert "emissivity=" not in src.replace(
        "cr.EMISSIVITY_STRATEGY_E", "")


# ===============================================================
# 9-12. 派生量
# ===============================================================

def test_rho_1020(c):
    assert c.RHO_COC == 1020.0
    mats = cr.make_convection_radiation_materials(0.05, 1000.0)
    assert mats["COC"].rho_kg_m3 == 1020.0


def test_water_oil_unchanged(c):
    mats = cr.make_convection_radiation_materials(0.05, 1000.0)
    default = heat_model.DEFAULT_MATERIALS
    for name in ("Water", "Oil"):
        assert mats[name].k_W_mK == default[name].k_W_mK
        assert mats[name].cp_J_kgK == default[name].cp_J_kgK
        assert mats[name].rho_kg_m3 == default[name].rho_kg_m3


def test_alpha_formula(c):
    k, cp = 0.05, 1000.0
    assert c.alpha_from_k_cp(k, cp) == pytest.approx(k / (1020.0 * cp))


def test_effusivity_formula(c):
    k, cp = 0.05, 1000.0
    assert c.effusivity_from_k_cp(k, cp) == pytest.approx(
        np.sqrt(k * 1020.0 * cp))
    assert c.rth_area_bottom(k) == pytest.approx(180e-6 / k)


# ===============================================================
# 13-19. 网格与缓存
# ===============================================================

def test_k_grid_exact(c):
    assert c.K_GRID == [0.0165, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
                        0.090, 0.120]


def test_cp_grid_exact(c):
    assert c.CP_GRID == [700.0, 800.0, 900.0, 1000.0, 1200.0, 1500.0, 1800.0]


def test_tau_grid_exact(c):
    assert c.TAU_GRID == [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]


def test_unique_fdm_evals_63(c):
    assert len(c.K_GRID) * len(c.CP_GRID) == 63


def test_total_combinations_504(c):
    assert len(c.K_GRID) * len(c.CP_GRID) * len(c.TAU_GRID) == 504


def test_one_fdm_per_k_cp_pair(c):
    """缓存计数: 扫描 (k,cp) 时只调用一次 FDM, tau 剖面不触发新 FDM。"""
    c.clear_fdm_cache()
    t = np.array([0.0, 1.0, 2.0, 3.0])
    tint = np.array([30.0, 40.0, 50.0, 60.0])
    ttop = np.array([28.0, 32.0, 35.0, 38.0])
    env = 27.8
    for tau in c.TAU_GRID:
        c.evaluate_k_cp_tau(0.0165, 900.0, tau, t, tint, ttop, env)
    assert len(c._fdm_cache) == 1  # 只有一个 (k,cp) 键
    c.clear_fdm_cache()


def test_tau_profiling_reuses_cached_fdm(c):
    c.clear_fdm_cache()
    t = np.array([0.0, 1.0, 2.0, 3.0])
    tint = np.array([30.0, 40.0, 50.0, 60.0])
    ttop = np.array([28.0, 32.0, 35.0, 38.0])
    env = 27.8
    for k in c.K_GRID[:2]:
        for cp in c.CP_GRID[:2]:
            for tau in c.TAU_GRID:
                c.evaluate_k_cp_tau(k, cp, tau, t, tint, ttop, env)
    assert len(c._fdm_cache) == 4  # 2x2 (k,cp)
    c.clear_fdm_cache()


# ===============================================================
# 20-21. 插值查询轴
# ===============================================================

def test_measurement_time_is_query_axis(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert "np.interp(t_proto, t_arr, t_top_obs)" in src


def test_top_temperature_never_query_coordinate(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert "np.interp(t_top_meas" not in src
    assert "np.interp(t_top" not in src


# ===============================================================
# 22-26. 选择规则
# ===============================================================

def test_global_minimum_uses_rmse_only(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert '"RMSE_72C_C"].idxmin()' in src


def test_no_high_k_reward(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    # 去掉元数据中的否定报告键 (报告"无高-k 奖励"是合法的)
    src_clean = src.replace('"high_k_reward_used": False', "")
    # 目标函数中没有 lambda 形式的 k 加权项, 没有 RMSE - alpha*k 结构
    for forbidden in ("RMSE - ", "- 0.01 * k"):
        assert forbidden not in src_clean


def test_cp_ge_800_subset(c):
    assert c.NON_EXTREME_CP_MIN == 800.0


def test_tau_upper_bound_warning(c):
    assert c.TAU_UPPER_BOUND == 8.0
    # evaluate: tau=8 触发警告, tau=5 不触发
    c.clear_fdm_cache()
    t = np.array([0.0, 1.0, 2.0])
    tint = np.array([30.0, 40.0, 50.0])
    ttop = np.array([28.0, 33.0, 36.0])
    env = 27.8
    r8 = c.evaluate_k_cp_tau(0.0165, 900.0, 8.0, t, tint, ttop, env)
    r5 = c.evaluate_k_cp_tau(0.0165, 900.0, 5.0, t, tint, ttop, env)
    assert r8["tau_upper_bound_warning"] is True
    assert r5["tau_upper_bound_warning"] is False
    c.clear_fdm_cache()


def test_balanced_physical_candidate_rule(c):
    assert c.BALANCED_RMSE == 1.0
    assert c.BALANCED_TAU_MAX == 5.0
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert 'cp_eff_J_kgK"] >= NON_EXTREME_CP_MIN' in src
    assert 'tau_lag_s"] <= BALANCED_TAU_MAX + 1e-12' in src
    assert '"k_eff_W_mK"].idxmax()' in src


# ===============================================================
# 27-29. 禁止事项
# ===============================================================

def test_no_doe11(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    # 只允许"否定报告"形式的引用: doe11_used=False / 注释说明未使用
    src_clean = src.replace('"doe11_used": False', "")
    src_clean = src_clean.replace("doe11_used", "")
    src_clean = src_clean.replace("DOE11", "")
    src_clean = src_clean.replace("doe11", "")
    # 无任何读取/运行 DOE11 数据的代码路径
    for forbidden in ("DOE11 faster", "08.12_pm_DOE11", "doe11_file",
                      "load_doe11", "doe11_prediction"):
        assert forbidden not in src


def test_no_longer_holding_pcr(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    assert "Test_PCR" not in src


def test_no_continuous_optimizer(c):
    src = CAL_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("scipy.optimize", "differential_evolution",
                      "least_squares", "minimize"):
        assert forbidden not in src


# ===============================================================
# 30-31. 不覆盖 / 不变
# ===============================================================

def test_previous_strategy_e_outputs_not_overwritten(c):
    # 输出目录不同于 Strategy E 检查目录
    assert c.OUTPUT_DIR != STRATEGY_E_CHECK_DIR
    assert "convection_radiation_k_cp_tau_calibration_v1" in str(c.OUTPUT_DIR)
    # 目录名不与旧扫描目录冲突
    assert "lag_separated" not in str(c.OUTPUT_DIR)
    assert "fast_pcr" not in str(c.OUTPUT_DIR)


def test_historical_strategy_a_unchanged(c):
    from thermal_model.config import calibrated_model_config as cmc
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    assert cal.k_eff_W_mK == 0.0165
    assert cal.cp_eff_J_kgK == 900.0
    assert cal.status == "accepted"


def test_sigma_sb_exact(c):
    assert c.cr.SIGMA_SB_W_M2_K4 == 5.670374419e-8


def test_view_factor_exact(c):
    assert c.cr.VIEW_FACTOR_STRATEGY_E == 1.0
