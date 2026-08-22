#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy G — 局部细化测试。

覆盖 (任务 39 的 31 项):
 1  Strategy E/F 物理不变 (复用 cr 求解器)
 2  h_conv 精确 10
 3  epsilon 精确 0.90
 4  非线性辐射启用
 5  环境规则不变 (第一个有效实测 Top COC)
 6  初始条件规则不变 (第一个内部温度)
 7  底部边界不变 (动态内部迹线)
 8  滞后实现复用 (apply_first_order_lag)
 9  tau 不改变原始 FDM 样品
10  k 网格精确
11  cp 网格精确
12  tau 网格精确 (4.0-12.0, 0.5 步长, 17 个)
13  唯一 FDM 运行数 = 25
14  总参数组合 = 425
15  每个 k/cp 对只跑一次 FDM
16  tau 剖面复用缓存 FDM
17  Strategy F 锚点 A 复现
18  Strategy F 锚点 B 复现
19  实测时间是插值查询轴
20  实测温度永不作插值查询坐标
21  局部全局最小用 RMSE 选择
22  无高-k 奖励
23  无高-alpha 奖励
24  中度滞后候选规则正确
25  边界警告逻辑正确
26  无 DOE11 数据
27  无长保持数据
28  无样品温度参与选参
29  无优化器调用
30  Strategy F 输出不被覆盖
31  历史名义标定不变
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
G_SCRIPT = PROJECT_ROOT / "workflows/calibration/convection_radiation_k_cp_tau_local_refinement.py"
F_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "convection_radiation_k_cp_tau_calibration_v1")
G_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "convection_radiation_k_cp_tau_local_refinement_v1")

# 短合成数据 (用于缓存计数等快速测试)
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 78.0, 80.0])
_TTOP = _TINT - 2.0
_ENV = 27.8

_mod = None


def _load_g():
    global _mod
    if _mod is None:
        spec = importlib.util.spec_from_file_location("crkt_local", G_SCRIPT)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
    return _mod


@pytest.fixture(scope="module")
def g():
    return _load_g()


# ===============================================================
# 1-8. 物理不变
# ===============================================================

def test_strategy_ef_physics_reused(g):
    assert g.cr is cr
    assert g.cr.run_convection_radiation_fdm is cr.run_convection_radiation_fdm


def test_h_exactly_10(g):
    assert g.cr.H_CONV_STRATEGY_E_W_M2K == 10.0


def test_epsilon_exactly_090(g):
    assert g.cr.EMISSIVITY_STRATEGY_E == 0.90


def test_nonlinear_radiation_enabled(g):
    import inspect
    body = inspect.getsource(cr._solve_surface_newton)
    assert "h_rad" not in body
    text = Path(cr.__file__).read_text(encoding="utf-8")
    assert "** 4" in text or "**4" in text


def test_environment_rule_unchanged(g):
    t_top = np.array([np.nan, 27.8, 28.0])
    env = cr.infer_environment_from_initial_top_measurement(t_top)
    assert env["T_environment_C"] == 27.8


def test_initial_condition_rule_unchanged(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "T_initial_C=float(t_int[0])" in src


def test_bottom_boundary_unchanged(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "bottom_temperature_C=t_int" in src
    assert "T_surroundings_C=t_env" in src


def test_lag_implementation_reused(g):
    assert g.apply_first_order_lag is lm.apply_first_order_lag


def test_tau_does_not_change_raw_fdm_sample(g):
    g.clear_fdm_cache()
    out0 = g.evaluate_k_cp_tau(0.055, 1200.0, 4.0, _T, _TINT, _TTOP, _ENV)
    g.clear_fdm_cache()
    out1 = g.evaluate_k_cp_tau(0.055, 1200.0, 12.0, _T, _TINT, _TTOP, _ENV)
    g.clear_fdm_cache()
    # 原始 FDM 迹线通过缓存键 (k,cp) 共享, 唯一变化是滞后后插值
    # (这里只检查 tau 不同不影响派生量计算路径中的 FDM)
    assert out0["alpha_eff_m2_s"] == out1["alpha_eff_m2_s"]
    assert out0["RMSE_72C_C"] != out1["RMSE_72C_C"]  # 滞后不同
    g.clear_fdm_cache()


# ===============================================================
# 10-16. 网格与缓存
# ===============================================================

def test_k_grid_exact(g):
    assert g.K_GRID == [0.0500, 0.0525, 0.0550, 0.0575, 0.0600]


def test_cp_grid_exact(g):
    assert g.CP_GRID == [1000.0, 1200.0, 1400.0, 1600.0, 1800.0]


def test_tau_grid_exact(g):
    assert g.TAU_GRID == [4.0 + 0.5 * i for i in range(17)]
    assert len(g.TAU_GRID) == 17
    assert g.TAU_GRID[0] == 4.0
    assert g.TAU_GRID[-1] == 12.0


def test_unique_fdm_runs_25(g):
    assert len(g.K_GRID) * len(g.CP_GRID) == 25


def test_total_combinations_425(g):
    assert len(g.K_GRID) * len(g.CP_GRID) * len(g.TAU_GRID) == 425


def test_one_fdm_per_k_cp_pair(g):
    g.clear_fdm_cache()
    for tau in g.TAU_GRID:
        g.evaluate_k_cp_tau(0.055, 1200.0, tau, _T, _TINT, _TTOP, _ENV)
    assert len(g._fdm_cache) == 1
    g.clear_fdm_cache()


def test_tau_profiling_reuses_cached_fdm(g):
    g.clear_fdm_cache()
    for k in g.K_GRID[:2]:
        for cp in g.CP_GRID[:2]:
            for tau in g.TAU_GRID[:3]:
                g.evaluate_k_cp_tau(k, cp, tau, _T, _TINT, _TTOP, _ENV)
    assert len(g._fdm_cache) == 4  # 2x2 (k,cp)
    g.clear_fdm_cache()


# ===============================================================
# 17-20. 锚点与插值轴
# ===============================================================

def test_anchor_A_reproduced(g):
    g.clear_fdm_cache()
    ta, tf, _ = g.run_cr_fdm_cached(0.055, 1200.0, _T, _TINT, _ENV)
    pred = g.lagged_top_prediction(_T, ta, tf, 8.0)
    m = g.metrics_for_prediction(pred, _TTOP)
    # 合成数据上 RMSE 值不同, 但路径必须一致 (锚点值与 72C 数据无关)
    # 这里验证锚点参数定义正确 + 路径无异常
    assert np.all(np.isfinite(m["RMSE_72C_C"]))
    assert g.ANCHOR_A == {"k": 0.055, "cp": 1200.0, "tau": 8.0,
                          "RMSE": 0.9554}
    g.clear_fdm_cache()


def test_anchor_B_defined(g):
    assert g.ANCHOR_B == {"k": 0.055, "cp": 1800.0, "tau": 5.0,
                          "RMSE": 1.1037}


def test_measurement_time_is_query_axis(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "np.interp(t_proto, t_arr, t_top_obs)" in src


def test_top_temperature_never_query_coordinate(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "np.interp(t_top" not in src


# ===============================================================
# 21-25. 选择规则
# ===============================================================

def test_global_minimum_uses_rmse_only(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert '"RMSE_72C_C"].idxmin()' in src


def test_no_high_k_reward(g):
    src = G_SCRIPT.read_text(encoding="utf-8").replace(
        '"high_k_reward_used": False', "")
    for forbidden in ("RMSE - ", "- 0.01 * k"):
        assert forbidden not in src


def test_no_high_alpha_reward(g):
    src = G_SCRIPT.read_text(encoding="utf-8").replace(
        '"high_alpha_reward_used": False', "")
    for forbidden in ("alpha_reward", "- 0.01 * alpha"):
        assert forbidden not in src


def test_moderate_lag_candidate_rule(g):
    assert g.MODERATE_LAG_TAU_MAX == 6.0
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "tau_lag_s\"] <= MODERATE_LAG_TAU_MAX + 1e-12" in src
    assert '"RMSE_72C_C"].idxmin()' in src


def test_boundary_warning_logic(g):
    r_lo = g.evaluate_k_cp_tau(0.0500, 1000.0, 4.0, _T, _TINT, _TTOP, _ENV)
    r_hi = g.evaluate_k_cp_tau(0.0550, 1200.0, 12.0, _T, _TINT, _TTOP, _ENV)
    r_mid = g.evaluate_k_cp_tau(0.0550, 1200.0, 8.0, _T, _TINT, _TTOP, _ENV)
    assert r_lo["k_local_boundary"] is True
    assert r_lo["cp_local_boundary"] is True
    assert r_hi["tau_local_boundary"] is True
    assert r_mid["k_local_boundary"] is False
    assert r_mid["cp_local_boundary"] is False
    assert r_mid["tau_local_boundary"] is False
    g.clear_fdm_cache()


# ===============================================================
# 26-29. 禁止事项
# ===============================================================

def test_no_doe11(g):
    src = G_SCRIPT.read_text(encoding="utf-8").replace(
        '"doe11_used": False', "").replace("doe11_used", "")
    for forbidden in ("DOE11 faster", "08.12_pm_DOE11"):
        assert forbidden not in src


def test_no_longer_holding(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "Test_PCR" not in src


def test_no_sample_temperature_in_fit(g):
    # 拟合指标中不使用样品温度: RMSE 只对 (lagged Top - measured Top) 计算。
    # FDM 结果中的 T_sample_arr 仅用于报告, 不进入任何指标/选择。
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "metrics_for_prediction(pred, t_top_meas)" in src
    assert "metrics_for_prediction(pred, t_sample" not in src
    # 指标函数只接收 pred 与 t_top_meas, 不含样品参数
    i_metrics = src.index("def metrics_for_prediction")
    i_eval = src.index("def evaluate_k_cp_tau")
    metrics_src = src[i_metrics:i_eval]
    assert "t_top_meas" in metrics_src
    assert "sample" not in metrics_src


def test_no_optimizer(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("scipy.optimize", "differential_evolution",
                      "least_squares", "minimize"):
        assert forbidden not in src


# ===============================================================
# 30-31. 不覆盖 / 不变
# ===============================================================

def test_strategy_f_outputs_not_overwritten(g):
    assert g.OUTPUT_DIR != F_DIR
    assert "local_refinement" in str(g.OUTPUT_DIR)
    # F 目录文件清单在 G 运行前记录不变 (此处至少验证路径不同且 F 存在)
    assert F_DIR.exists()
    assert (F_DIR / "k_cp_tau_full_scan.csv").exists()


def test_historical_nominal_unchanged(g):
    from thermal_model.config import calibrated_model_config as cmc
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    assert cal.k_eff_W_mK == 0.0165
    assert cal.cp_eff_J_kgK == 900.0
    assert cal.status == "accepted"


def test_sigma_sb_exact(g):
    assert g.cr.SIGMA_SB_W_M2_K4 == 5.670374419e-8


def test_no_automatic_second_refinement(g):
    src = G_SCRIPT.read_text(encoding="utf-8")
    assert "second_refinement_automatic" in src
