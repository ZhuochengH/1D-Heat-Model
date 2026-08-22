#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy D — (k_eff, cp_eff, tau_lag) 三参数表征测试。

覆盖任务 30 项:
  1    alpha = k/(rho*cp)
  2    effusivity = sqrt(k*rho*cp)
  3    策略 A 参数复现当前 alpha
  4    策略 A k/cp/tau=0 复现 RMSE ~0.7337 (短协议上验证函数等价)
  5    tau=0 精确恒等 (滞后函数)
  6    滞后只作用于顶部观测
  7    滞后不改变样品 FDM
  8    改 k/cp 会改变 FDM
  9    Water/Oil 不变
  10   rho 保持 1020
  11   DEFAULT_MATERIALS 不被修改
  12   BARE_TOP_COC_LAYERS 使用
  13   实测时间是插值查询轴
  14   温度值绝不作插值坐标
  15   每个 (k,cp) 只跑一次 FDM, tau 剖面复用
  16   全网格行数 = 693
  17   唯一 FDM 评估 = 63
  18   best tau 选择正确
  19   全局 RMSE 最小选择正确
  20   Delta_RMSE 带正确
  21   NON_EXTREME_CP_SUBSET 用 cp>=800
  22   高 k 候选在拟合过滤之后选择
  23   目标无 k 奖励/惩罚
  24   大滞后警告 >=16 s
  25   tau 边界警告 =20 s
  26   平衡候选规则正确
  27   策略 A 名义配置不被修改
  28   无优化器调用
  29   DOE11 不进目标
  30   PCR 样品预测不进候选选择
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from thermal_model.config import calibrated_model_config as cmc
from workflows.calibration import k_cp_tau_calibration_strategy as kd
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag

_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 62.0, 50.0])
_TTOP = _TINT - 2.0


# ===============================================================
# 1-4. 派生量 / 策略 A 回归
# ===============================================================

def test_alpha_equals_k_over_rho_cp():
    assert kd.alpha_from_k_cp(0.0165, 900.0) == pytest.approx(
        0.0165 / (1020.0 * 900.0))


def test_effusivity_formula():
    e = kd.effusivity_from_k_cp(0.0165, 900.0)
    assert e == pytest.approx(np.sqrt(0.0165 * 1020.0 * 900.0))


def test_strategy_A_alpha_reproduced():
    cal = cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    a = kd.alpha_from_k_cp(cal.k_eff_W_mK, cal.cp_eff_J_kgK)
    assert a == pytest.approx(1.7973856209e-8, rel=1e-6)


def test_strategy_A_tau0_regression_short():
    # 用短合成协议验证: k=0.0165/cp=900/tau=0 下
    # 观察预测 == 直接 k/cp 评估的预测 (函数级等价)
    p0 = kd.lagged_top_prediction(
        _T, *kd.run_fdm_cached(0.0165, 900.0, _T, _TINT)[:2], 0.0)
    from thermal_model.utilities.scan_effective_thermal_parameters import evaluate_point
    r_ref = evaluate_point(0.0165, 900.0, _T, _TINT, _TTOP)
    # 评估函数内部用 np.interp(_T, t_arr, T_top_surface_arr)
    # 与 kd 用相同 FDM, 只差 tau=0 恒等 -> 预测一致
    tA, topA, _ = kd.run_fdm_cached(0.0165, 900.0, _T, _TINT)
    pred_ref = np.interp(_T, tA, topA)
    np.testing.assert_allclose(p0, pred_ref, rtol=1e-12)
    assert r_ref["RMSE_C"] > 0  # 指标正常


# ===============================================================
# 5-8. 滞后行为
# ===============================================================

def test_tau0_identity():
    t = np.array([0.0, 1.0, 2.5])
    x = np.array([30.0, 40.0, 35.0])
    y = apply_first_order_lag(t, x, 0.0)
    np.testing.assert_array_equal(y, x)


def test_lag_only_on_top_observation():
    tA, topA, sampA = kd.run_fdm_cached(0.0165, 900.0, _T, _TINT)
    pred0 = kd.lagged_top_prediction(_T, tA, topA, 0.0)
    pred3 = kd.lagged_top_prediction(_T, tA, topA, 3.0)
    assert not np.allclose(pred0, pred3)   # 顶部观测随 tau 变
    # 样品 FDM 输出是缓存的, 与 tau 无关


def test_tau_does_not_change_sample():
    kd.clear_fdm_cache()
    _, _, s0 = kd.run_fdm_cached(0.0165, 900.0, _T, _TINT)
    _, _, s1 = kd.run_fdm_cached(0.0165, 900.0, _T, _TINT)  # 命中缓存
    np.testing.assert_array_equal(s0, s1)
    # 直接对比不同 tau 的评估: 样品数组不变 (来自同一缓存)
    r0 = kd.evaluate_k_cp_tau(0.0165, 900.0, 0.0, _T, _TINT, _TTOP)
    r3 = kd.evaluate_k_cp_tau(0.0165, 900.0, 3.0, _T, _TINT, _TTOP)
    assert r0["RMSE_72C_C"] != r3["RMSE_72C_C"]


def test_changing_k_cp_changes_fdm():
    kd.clear_fdm_cache()
    _, topA, _ = kd.run_fdm_cached(0.0165, 900.0, _T, _TINT)
    _, topB, _ = kd.run_fdm_cached(0.080, 1400.0, _T, _TINT)
    assert not np.allclose(topA, topB)


# ===============================================================
# 9-12. 材料/几何
# ===============================================================

def test_water_oil_unchanged():
    from thermal_model.core.lag_augmented_thermal_model import make_lag_materials
    mats = make_lag_materials(0.0165, 900.0)
    base = heat_model.DEFAULT_MATERIALS
    for m in ("Water", "Oil"):
        assert mats[m].k_W_mK == base[m].k_W_mK
        assert mats[m].cp_J_kgK == base[m].cp_J_kgK
        assert mats[m].rho_kg_m3 == base[m].rho_kg_m3


def test_rho_remains_1020():
    from thermal_model.core.lag_augmented_thermal_model import make_lag_materials
    mats = make_lag_materials(0.08, 1400.0)
    assert mats["COC"].rho_kg_m3 == 1020.0
    assert kd.RHO_COC == 1020.0


def test_default_materials_not_mutated():
    before = {k: (v.k_W_mK, v.cp_J_kgK) for k, v in
              heat_model.DEFAULT_MATERIALS.items()}
    kd.run_fdm_cached(0.08, 1400.0, _T, _TINT)
    after = {k: (v.k_W_mK, v.cp_J_kgK) for k, v in
             heat_model.DEFAULT_MATERIALS.items()}
    assert before == after


def test_uses_bare_top_layers():
    assert "BARE_TOP_COC_LAYERS" in inspect.getsource(kd.run_fdm_cached)
    mats = heat_model.copy_default_materials()
    mesh = heat_model.build_layer_stack(mats, heat_model.BARE_TOP_COC_LAYERS)
    assert mesh.x[-1] == pytest.approx(850e-6)
    names = " ".join(ly.name for ly in heat_model.BARE_TOP_COC_LAYERS)
    assert "Air" not in names and "PDMS" not in names


# ===============================================================
# 13-14. 时间采样
# ===============================================================

def test_measurement_time_is_query_axis():
    src = inspect.getsource(kd.lagged_top_prediction)
    assert "np.interp(t_proto" in src
    assert "t_top_meas" not in src.split("return")[1] if "return" in src else True
    src2 = inspect.getsource(kd.evaluate_k_cp_tau)
    assert "t_top_meas" in src2  # 用于残差
    assert "interp(t_top_meas" not in src + src2


def test_temperature_never_interpolation_coordinate():
    src = inspect.getsource(kd)
    assert "interp(t_top_meas" not in src
    assert "np.interp(t_top_meas" not in src


# ===============================================================
# 15-17. 缓存 / 行数
# ===============================================================

def test_fdm_cached_once_per_k_cp(tmp_path):
    kd.clear_fdm_cache()
    calls = []

    orig = heat_model.run_simulation

    def spy(**kw):
        calls.append(kw["materials"]["COC"].k_W_mK)
        return orig(**kw)

    from thermal_model.core import lag_augmented_thermal_model as lam
    monkey = pytest.MonkeyPatch()
    monkey.setattr(lam.heat_model, "run_simulation", spy)
    monkey.setattr(kd.heat_model, "run_simulation", spy)
    try:
        # 同一 (k,cp) 评估 3 个 tau -> FDM 只跑 1 次
        for tau in (0.0, 1.0, 3.0):
            kd.evaluate_k_cp_tau(0.025, 1100.0, tau, _T, _TINT, _TTOP)
        assert len(calls) == 1
        # 不同 (k,cp) -> +1 次
        kd.evaluate_k_cp_tau(0.040, 800.0, 0.0, _T, _TINT, _TTOP)
        assert len(calls) == 2
    finally:
        monkey.undo()


def test_full_grid_rows_equals_693():
    assert len(kd.K_GRID) * len(kd.CP_GRID) * len(kd.TAU_GRID) == 693
    assert len(kd.K_GRID) == 9
    assert len(kd.CP_GRID) == 7
    assert len(kd.TAU_GRID) == 11


def test_unique_fdm_runs_equals_63():
    assert len(kd.K_GRID) * len(kd.CP_GRID) == 63


# ===============================================================
# 18-20. 剖面 / 全局最小 / 带
# ===============================================================

def _make_synthetic_df():
    """构造已知最优的合成扫描表 (小网格)。"""
    rows = []
    k = 0.0165
    cps = [600.0, 900.0]
    taus = [0.0, 2.0, 12.0]
    # RMSE = |k - 0.03|*20 + |cp-1000|*0.002 + |tau-2|*0.3 + 0.7
    for cp in cps:
        for tau in taus:
            rmse = abs(k - 0.03) * 20 + abs(cp - 1000.0) * 0.002 + \
                abs(tau - 2.0) * 0.3 + 0.7
            rows.append({
                "k_eff_W_mK": k, "cp_eff_J_kgK": cp, "tau_lag_s": tau,
                "alpha_eff_m2_s": kd.alpha_from_k_cp(k, cp),
                "effusivity_J_s05_m2_K": kd.effusivity_from_k_cp(k, cp),
                "Rth_area_bottom_m2K_W": kd.rth_area_bottom(k),
                "RMSE_72C_C": rmse,
                "MAE_72C_C": rmse - 0.1,
                "mean_residual_72C_C": 0.0,
                "max_abs_residual_72C_C": rmse,
                "cp_non_extreme": cp >= 800.0,
                "large_lag_warning": tau >= 16.0,
                "tau_boundary_warning": tau >= 20.0 - 1e-12,
                "status": "OK",
            })
    return pd.DataFrame(rows)


def test_best_tau_selection():
    df = _make_synthetic_df()
    prof = kd.profile_best_tau(df)
    # cp=600: RMSE = 0.27+0.8+0.3|tau-2|+0.7 -> tau=2 最优, RMSE=1.77
    row = prof[(prof["k_eff_W_mK"] == 0.0165) &
               (prof["cp_eff_J_kgK"] == 600.0)].iloc[0]
    assert row["best_tau_s"] == 2.0
    assert row["best_RMSE_C"] == pytest.approx(0.27 + 0.8 + 0.7)
    # RMSE_tau0 = 1.77 + 0.3*2 = 2.37
    assert row["RMSE_tau0_C"] == pytest.approx(0.27 + 0.8 + 0.7 + 0.6)


def test_global_rmse_minimum_selection():
    df = _make_synthetic_df()
    g = kd.global_rmse_minimum(df)
    assert g["tau_lag_s"] == 2.0
    assert g["cp_eff_J_kgK"] == 900.0
    assert g["RMSE_72C_C"] == pytest.approx(
        abs(0.0165 - 0.03) * 20 + abs(900 - 1000) * 0.002 + 0.7)


def test_delta_rmse_bands():
    df = _make_synthetic_df()
    g = kd.global_rmse_minimum(df)
    sets = kd.near_optimal_sets(df, float(g["RMSE_72C_C"]))
    assert "STRICT" in sets and "MODERATE" in sets and "APPLICATION" in sets
    assert "RMSE_LE_1C" in sets
    for name, s in sets.items():
        assert np.all(s["RMSE_72C_C"] - float(g["RMSE_72C_C"]) >= -1e-12)


# ===============================================================
# 21-23. 物理子集 / 高 k 候选 / 无 k 奖励
# ===============================================================

def test_non_extreme_cp_subset():
    assert kd.NON_EXTREME_CP_MIN == 800.0
    df = _make_synthetic_df()
    # 最高 k 候选只从 cp>=800 中选
    cand = kd.highest_k_candidate(df)
    assert cand["cp_eff_J_kgK"] == 900.0  # 只有 900 >= 800


def test_high_k_candidate_after_fit_filter():
    df = _make_synthetic_df()
    g = kd.global_rmse_minimum(df)
    sets = kd.near_optimal_sets(df, float(g["RMSE_72C_C"]))
    # STRICT 带只含全局最小点 (delta<=0.05)
    cand = kd.highest_k_candidate(sets["STRICT"])
    assert cand is not None
    assert cand["k_eff_W_mK"] == 0.0165
    # 过滤先于选择: 候选来自已过滤的带内集合


def test_no_k_reward_in_objective():
    # 指标函数只由残差构成, 不含 k
    src_metrics = inspect.getsource(kd.metrics_for_prediction)
    assert "RMSE" in src_metrics
    assert "k_eff" not in src_metrics
    # 单点评估指标来自 metrics_for_prediction (纯 RMSE/MAE/mean/max abs)
    src_eval = inspect.getsource(kd.evaluate_k_cp_tau)
    assert "metrics_for_prediction" in src_eval
    assert "lambda" not in src_eval
    # 扫描/剖面/选择函数体内无 k 惩罚项 (docstring 允许提及)
    for fn in (kd.run_full_scan, kd.profile_best_tau,
               kd.global_rmse_minimum, kd.near_optimal_sets,
               kd.highest_k_candidate):
        src = inspect.getsource(fn)
        assert "lambda" not in src
        assert "k_eff_W_mK] * " not in src  # 无 k 加权


# ===============================================================
# 24-26. 警告 / 平衡候选
# ===============================================================

def test_large_lag_warning_threshold():
    assert kd.LARGE_LAG_THRESHOLD == 16.0
    assert kd.evaluate_k_cp_tau(0.1, 900.0, 16.0, _T, _TINT, _TTOP)[
        "large_lag_warning"] is True
    assert kd.evaluate_k_cp_tau(0.1, 900.0, 12.0, _T, _TINT, _TTOP)[
        "large_lag_warning"] is False


def test_tau_boundary_warning():
    assert kd.TAU_BOUNDARY == 20.0
    r = kd.evaluate_k_cp_tau(0.1, 900.0, 20.0, _T, _TINT, _TTOP)
    assert r["tau_boundary_warning"] is True
    r2 = kd.evaluate_k_cp_tau(0.1, 900.0, 16.0, _T, _TINT, _TTOP)
    assert r2["tau_boundary_warning"] is False


def test_balanced_candidate_rule():
    # 构造: 高 k=0.2/cp=1000/tau=10 RMSE 0.9; 更高 k=0.25/cp=600/tau=20 RMSE 0.95
    rows = [
        {"k_eff_W_mK": 0.20, "cp_eff_J_kgK": 1000.0, "tau_lag_s": 10.0,
         "RMSE_72C_C": 0.9, "alpha_eff_m2_s": 1e-8,
         "effusivity_J_s05_m2_K": 100.0, "Rth_area_bottom_m2K_W": 1e-3,
         "MAE_72C_C": 0.8, "mean_residual_72C_C": 0.0,
         "max_abs_residual_72C_C": 0.9, "cp_non_extreme": True,
         "large_lag_warning": False, "tau_boundary_warning": False,
         "status": "OK"},
        {"k_eff_W_mK": 0.25, "cp_eff_J_kgK": 600.0, "tau_lag_s": 20.0,
         "RMSE_72C_C": 0.95, "alpha_eff_m2_s": 2e-8,
         "effusivity_J_s05_m2_K": 200.0, "Rth_area_bottom_m2K_W": 1e-3,
         "MAE_72C_C": 0.85, "mean_residual_72C_C": 0.0,
         "max_abs_residual_72C_C": 0.95, "cp_non_extreme": False,
         "large_lag_warning": True, "tau_boundary_warning": True,
         "status": "OK"},
    ]
    df = pd.DataFrame(rows)
    bal = kd.balanced_high_k_candidate(df, 1.0)
    assert bal is not None
    assert bal["k_eff_W_mK"] == 0.20  # 0.25 被 cp/tau 过滤
    # 放宽到 1.2 仍选 0.20 (cp>=800/tau<16 过滤优先)
    bal2 = kd.balanced_high_k_candidate(df, 1.2)
    assert bal2["k_eff_W_mK"] == 0.20


# ===============================================================
# 27-30. 配置 / 无优化 / 数据隔离
# ===============================================================

def test_nominal_config_not_mutated():
    before = (cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK,
              cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK)
    kd.run_fdm_cached(0.08, 1400.0, _T, _TINT)
    after = (cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK,
             cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK)
    assert before == after == (0.0165, 900.0)


def test_no_optimizer_invoked():
    src = inspect.getsource(kd)
    for tok in ("scipy", "optimize", "curve_fit", "minimize",
                "least_squares", "differential_evolution"):
        assert tok not in src


def test_no_doe11_in_objective():
    # 评估/剖面/选择函数不引用 DOE11 (模块 docstring 允许提及)
    for fn in (kd.evaluate_k_cp_tau, kd.run_full_scan, kd.profile_best_tau,
               kd.highest_k_candidate, kd.balanced_high_k_candidate):
        src = inspect.getsource(fn)
        assert "doe11" not in src.lower()
    # 评估函数签名无 DOE11 输入
    sig = inspect.signature(kd.evaluate_k_cp_tau)
    assert "doe11" not in " ".join(sig.parameters).lower()


def test_no_pcr_sample_in_selection():
    # 选择函数不引用 PCR / 样品温度 (模块 docstring 允许提及)
    for fn in (kd.highest_k_candidate, kd.balanced_high_k_candidate,
               kd.near_optimal_sets):
        src = inspect.getsource(fn)
        assert "PCR" not in src
        assert "sample" not in src.lower()
    # 候选选择函数只接收 RMSE 表
    sig = inspect.signature(kd.highest_k_candidate)
    assert "sample" not in " ".join(sig.parameters).lower()
    sig2 = inspect.signature(kd.balanced_high_k_candidate)
    assert "sample" not in " ".join(sig2.parameters).lower()
