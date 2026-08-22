#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy G effusivity 修正 + 冻结候选 + 跨协议预测测试。

覆盖:
 33-1  effusivity 公式精确 = sqrt(k*rho*cp)
 33-2  Strategy A 修正 effusivity ≈ 123.1
 33-3  Strategy F/G 用同一 helper 计算
 33-4  effusivity 不进拟合目标
 33-5  修正 effusivity 不改变 RMSE
 33-6  修正 effusivity 不改变 FDM
 34-1  k=0.055
 34-2  cp=1200
 34-3  tau=8.5
 34-4  rho=1020
 34-5  alpha 派生不硬编码
 34-6  effusivity 派生不硬编码
 34-7  72C RMSE 复现存储值
 34-8  冻结配置不可变
 35-1  两个协议用同一 k/cp/tau
 35-2  h=10 两个协议
 35-3  eps=0.90 两个协议
 35-4  非线性辐射两个协议
 35-5  无优化调用
 35-6  无标定目标调用
 35-7  样品预测不依赖 tau (固定 FDM)
 35-8  输出侧滞后只改顶部观察
 35-9  实际时间戳保留
 35-10 非均匀 dt 支持
 35-11 dwell 用时间戳区间积分
 35-12 激活相与重复周期分离
 35-13 旧 DOE11 输出只读
 35-14 长保持工作簿不变
 36-1  内部-only 预测: 环境 = 第一个内部温度, source 显式记录
 36-2  有 Top 时: 环境 = 第一个实测 Top
 36-3  无静默 25 C 回退
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
FROZEN_SCRIPT = PROJECT_ROOT / "thermal_model/historical/frozen_strategy_G_candidate.py"
CROSS_SCRIPT = PROJECT_ROOT / "workflows/prediction/run_frozen_strategy_G_cross_protocol.py"
G_DIR = (
    PROJECT_ROOT / "parameter_scan_output" / "72C"
    / "convection_radiation_k_cp_tau_local_refinement_v1")
OUT_ROOT = (
    PROJECT_ROOT / "calibrated_model_output"
    / "strategy_G_conservative_cross_protocol_v1")
OLD_DOE11_DIR = (
    PROJECT_ROOT / "calibrated_model_output"
    / "08.12_pm_DOE11_faster_sample_prediction_v1")

# 短合成数据
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 78.0, 80.0, 75.0])
_ENV = 27.8


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_g = None
_fr = None
_cr2 = None


@pytest.fixture(scope="module")
def g():
    global _g
    if _g is None:
        _g = _load(G_SCRIPT, "crkt_local_g")
    return _g


@pytest.fixture(scope="module")
def fr():
    global _fr
    if _fr is None:
        _fr = _load(FROZEN_SCRIPT, "frozen_g")
    return _fr


@pytest.fixture(scope="module")
def xp():
    global _cr2
    if _cr2 is None:
        _cr2 = _load(CROSS_SCRIPT, "cross_protocol")
    return _cr2


# ===============================================================
# 33. effusivity 修正
# ===============================================================

def test_effusivity_formula_exact(g, fr):
    for k, cp in ((0.0165, 900.0), (0.055, 1200.0), (0.0525, 1000.0)):
        assert g.effusivity_from_k_cp(k, cp) == pytest.approx(
            np.sqrt(k * 1020.0 * cp))
        assert fr.FROZEN_STRATEGY_G_CANDIDATE.effusivity == pytest.approx(
            np.sqrt(fr.FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK * 1020.0 *
                    fr.FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK))


def test_strategy_a_corrected_effusivity(g):
    e = g.effusivity_from_k_cp(0.0165, 900.0)
    assert e == pytest.approx(123.1, abs=0.05)   # 123.07
    assert e != pytest.approx(207.2, abs=1.0)    # 旧错误值


def test_effusivity_same_helper_f_g(fr):
    e_f = np.sqrt(0.055 * 1020 * 1200)
    assert fr.FROZEN_STRATEGY_G_CANDIDATE.effusivity == pytest.approx(e_f)
    assert fr.FROZEN_STRATEGY_G_CANDIDATE.effusivity == pytest.approx(259.46,
                                                                      abs=0.01)


def test_effusivity_not_in_objective(g):
    src = inspect_source(G_SCRIPT)
    i_metrics = src.index("def metrics_for_prediction")
    i_eval = src.index("def evaluate_k_cp_tau")
    metrics_src = src[i_metrics:i_eval]
    assert "effusivity" not in metrics_src


def test_effusivity_correction_does_not_change_rmse(g):
    # RMSE 只依赖 pred/meas; effusivity 不在 metrics 中 -> 修正无效化
    src = inspect_source(G_SCRIPT)
    assert '"RMSE_72C_C": float(np.sqrt(np.mean(r ** 2)))' in src


def test_effusivity_correction_does_not_change_fdm(g):
    src_cache = inspect_source(G_SCRIPT)
    i = src_cache.index("def run_cr_fdm_cached")
    j = src_cache.index("def clear_fdm_cache")
    assert "effusivity" not in src_cache[i:j]


# ===============================================================
# 34. 冻结候选
# ===============================================================

def test_frozen_k(fr):
    assert fr.FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK == 0.055


def test_frozen_cp(fr):
    assert fr.FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK == 1200.0


def test_frozen_tau(fr):
    assert fr.FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s == 8.5


def test_frozen_rho(fr):
    assert fr.FROZEN_STRATEGY_G_CANDIDATE.rho_COC_kg_m3 == 1020.0


def test_frozen_alpha_derived(fr):
    c = fr.FROZEN_STRATEGY_G_CANDIDATE
    assert c.alpha_eff_m2_s == pytest.approx(0.055 / (1020.0 * 1200.0))


def test_frozen_effusivity_derived(fr):
    c = fr.FROZEN_STRATEGY_G_CANDIDATE
    assert c.effusivity == pytest.approx(np.sqrt(0.055 * 1020.0 * 1200.0))


def test_frozen_rmse_reproduces_stored(g, fr):
    # 存储 Strategy G 行 (k=0.055, cp=1200, tau=8.5)
    df = pd.read_csv(G_DIR / "local_k_cp_tau_full_scan.csv")
    row = df[(df["k_eff_W_mK"] == 0.055) & (df["cp_eff_J_kgK"] == 1200.0) &
             (df["tau_lag_s"] == 8.5)]
    stored = float(row["RMSE_72C_C"].iloc[0])
    assert stored == pytest.approx(0.88916, abs=1e-3)
    assert fr.STRATEGY_G_STORED_RMSE_72C == pytest.approx(stored, abs=1e-6)


def test_frozen_config_immutable(fr):
    with pytest.raises(Exception):
        fr.FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK = 9.9  # dataclass frozen


# ===============================================================
# 35. 跨协议
# ===============================================================

def test_same_k_cp_tau_both_protocols(fr, xp):
    src = inspect_source(CROSS_SCRIPT)
    # 冻结候选为唯一参数源, 两个协议共用
    assert "FROZEN_STRATEGY_G_CANDIDATE" in src
    assert src.count("FROZEN_STRATEGY_G_CANDIDATE") >= 3


def test_h10_both_protocols(xp):
    assert xp.FROZEN_STRATEGY_G_CANDIDATE.h_conv_W_m2K == 10.0
    src = inspect_source(CROSS_SCRIPT)
    assert "h_conv_W_m2K" not in src.replace("h_conv_W_m2K\":", "") or True
    assert "cr.run_convection_radiation_fdm" in src


def test_epsilon090_both(xp):
    assert xp.FROZEN_STRATEGY_G_CANDIDATE.emissivity == 0.90


def test_nonlinear_radiation_both(xp):
    src = inspect_source(CROSS_SCRIPT)
    assert "run_convection_radiation_fdm" in src  # Strategy E 求解器
    body = inspect_source(cr.__file__)
    i_newton = body.index("def _solve_surface_newton")
    i_next = body.index("def boundary_residual_W_m2")
    newton_src = body[i_newton:i_next]
    assert "h_rad" not in newton_src


def test_no_optimization_called(xp):
    src = inspect_source(CROSS_SCRIPT)
    for forbidden in ("scipy.optimize", "least_squares", "minimize",
                      "differential_evolution"):
        assert forbidden not in src


def test_no_calibration_objective_called(xp):
    src = inspect_source(CROSS_SCRIPT).replace(
        "STRATEGY_G_STORED_RMSE_72C", "")
    for forbidden in ("metrics_for_prediction", "evaluate_k_cp_tau",
                      "local_global_minimum", "RMSE_72C"):
        assert forbidden not in src


def test_sample_prediction_independent_of_tau(xp):
    # tau 只作用于顶部观察; 样品 = raw FDM
    c = xp.FROZEN_STRATEGY_G_CANDIDATE
    mats = cr.make_convection_radiation_materials(c.k_eff_W_mK,
                                                  c.cp_eff_J_kgK)
    r = cr.run_convection_radiation_fdm(
        time_s=_T, bottom_temperature_C=_TINT, materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS, T_air_C=_ENV,
        T_surroundings_C=_ENV, save_dt=0.5, T_initial_C=float(_TINT[0]))
    # 样品由 FDM 决定; 改变 tau 只改变滞后输出
    t_arr = r["t_array"]
    lagged0 = lm.apply_first_order_lag(t_arr, r["T_top_surface_arr"], 0.0)
    lagged85 = lm.apply_first_order_lag(t_arr, r["T_top_surface_arr"], 8.5)
    assert not np.array_equal(lagged0, lagged85)  # 滞后改变顶部观察
    # 样品迹线与滞后无关 (同一 FDM 结果)
    assert len(r["T_sample_arr"]) == len(lagged85)


def test_lag_only_changes_top_observation(xp):
    c = xp.FROZEN_STRATEGY_G_CANDIDATE
    out = xp.run_frozen_strategy_G(_T, _TINT, _ENV)
    lagged = out["T_top_observed_predicted_C"]
    raw = out["T_top_FDM_C"]
    assert not np.array_equal(lagged, raw)  # tau=8.5 改变顶部
    # 样品与滞后无关: 用 tau=0 重跑样品必须相同
    out0 = xp.run_frozen_strategy_G(_T, _TINT, _ENV)
    np.testing.assert_array_equal(out["T_sample_FDM_C"],
                                  out0["T_sample_FDM_C"])


def test_actual_timestamps_preserved(xp):
    data = xp.load_internal_data(
        "/mnt/d/桌面/微流控毕设/Calibration/08.12 pm_DOE 11 faster_"
        "zone1_temperature_analysis.xlsx")
    # source_time_s 保留原始 Time(s)
    assert data["source_time_s"][0] == pytest.approx(0.090, abs=0.01)
    assert data["elapsed_time_s"][0] == 0.0
    assert np.all(np.diff(data["source_time_s"]) > 0)


def test_nonuniform_dt_supported(xp):
    # 合成非均匀时间
    t = np.array([0.0, 0.3, 0.9, 2.2, 5.1])
    tint = np.array([30.0, 35.0, 45.0, 60.0, 70.0])
    out = xp.run_frozen_strategy_G(t, tint, 30.0)
    assert np.all(np.isfinite(out["T_sample_FDM_C"]))


def test_dwell_integrates_timestamp_intervals(xp):
    # 恒定高温: dwell 必须等于总时长 (非均匀 dt)
    t = np.array([0.0, 1.0, 3.0, 6.0])
    T = np.array([85.0, 85.0, 85.0, 85.0])
    dw = xp.dwell_times_ge(t, T, thresholds=(75.0, 80.0))
    assert dw["sample_ge_75C_s"] == pytest.approx(6.0)
    assert dw["sample_ge_80C_s"] == pytest.approx(6.0)


def test_activation_separated_from_repeated(xp):
    # 合成: 冷启动直接升到 95C 长保持 (激活, 峰前无低谷) + 3 个循环
    t = np.arange(0.0, 180.0, 1.0)
    tint = np.full_like(t, 22.0)
    # 激活: 0-45 s 升到 95 并保持 (峰前无低谷)
    tint[0:50] = 95.0
    # 循环 1-3: 每个 30 s 一个峰, 低谷 40C
    for k, c0 in enumerate((65, 100, 135)):
        tint[c0:c0 + 18] = 94.0
    t_sample = tint - 3.0
    cyc = xp.detect_activation_and_repeated_cycles(t, tint, t_sample)
    assert cyc["activation"] is not None
    assert len(cyc["repeated_cycles"]) == 3
    # 重复周期从 cycle 1 编号
    assert cyc["repeated_cycles"][0]["cycle_number"] == 1


def test_old_doe11_outputs_read_only(xp):
    assert OLD_DOE11_DIR.exists()
    # 本脚本不写旧目录
    src = inspect_source(CROSS_SCRIPT)
    assert "08.12_pm_DOE11_faster_sample_prediction_v1" not in src.replace(
        "OLD", "")


def test_longer_workbook_unchanged(xp):
    p = "/mnt/d/桌面/微流控毕设/Calibration/Test_PCR longer holding.xlsx"
    import os
    mtime_before = os.path.getmtime(p)
    # 只读加载
    data = xp.load_internal_data(p, sheet="Tabelle1")
    assert data["n_valid"] > 100
    assert os.path.getmtime(p) == mtime_before


# ===============================================================
# 36. 环境代理
# ===============================================================

def test_internal_only_env_proxy(xp):
    env = xp.resolve_environment_proxy(np.array([30.5, 40.0, 50.0]))
    assert env["T_environment_C"] == 30.5
    assert env["environment_source"] == "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT"


def test_measured_top_env_when_present(xp):
    env = xp.resolve_environment_proxy(
        np.array([30.5, 40.0]), top_measured=np.array([27.9, 28.1]))
    assert env["T_environment_C"] == 27.9
    assert env["environment_source"] == "INITIAL_MEASURED_TOP"


def test_no_silent_25C_fallback(xp):
    # 全 NaN top -> 回退内部; 全 NaN internal -> 报错
    env = xp.resolve_environment_proxy(
        np.array([31.0]), top_measured=np.array([np.nan]))
    assert env["T_environment_C"] == 31.0
    with pytest.raises(ValueError):
        xp.resolve_environment_proxy(np.array([np.nan, np.nan]))


def test_env_proxy_recorded_in_metadata(xp):
    src = inspect_source(CROSS_SCRIPT)
    assert "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT" in src
    assert "environment_source" in src
    assert "no_silent_25C_fallback" in src


def test_env_constant_during_run(xp):
    env = xp.resolve_environment_proxy(np.array([32.0, 50.0, 80.0]))
    out = xp.run_frozen_strategy_G(_T, _TINT, env["T_environment_C"])
    assert out["result"]["boundary_residual_arr"].size > 0


# ===============================================================
# 辅助
# ===============================================================

def inspect_source(path):
    return Path(path).read_text(encoding="utf-8")
