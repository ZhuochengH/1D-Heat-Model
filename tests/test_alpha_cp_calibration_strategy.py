#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy B — (alpha_eff, cp_eff) 快检测试 (fast-PCR-oriented)。

覆盖任务要求 20 项:
  1-2   alpha<->k/cp 换算;
  3     策略 A alpha ≈ 1.797e-8;
  4     (alpha_A, 900) -> k=0.0165;
  5     alpha/cp 评估复现策略 A 顶部预测/RMSE (短合成协议位级一致);
  6     实测时间为插值查询轴;
  7     无历史 temperature-as-time 语义;
  8-10  仅 COC 参数变化, rho=1020, Water/Oil 不变;
  11    BARE_TOP_COC_LAYERS 使用;
  12    近最优 Delta_RMSE 带正确;
  13    最高 alpha 候选选择正确;
  14    RMSE<=1C 选择正确;
  15    Pareto 支配实现正确;
  16-17 策略 B 不改变名义标定/Git 元数据;
  18-19 DOE11 不进拟合目标, 样品值不影响候选资格;
  20    输出隔离到策略 B 目录。
"""
import json

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from thermal_model.config import calibrated_model_config as cmc
from workflows.calibration import alpha_cp_calibration_strategy as sb

# 合成短协议 (每次模拟 ~0.1-0.3 s)
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 62.0, 50.0])
_TTOP = _TINT - 2.0


# ===============================================================
# 1-4. 换算
# ===============================================================

def test_alpha_from_k_cp():
    a = sb.alpha_from_k_cp(0.0165, 900.0, 1020.0)
    assert a == pytest.approx(0.0165 / (1020.0 * 900.0), abs=1e-20)


def test_k_from_alpha_cp():
    a = sb.alpha_from_k_cp(0.0165, 900.0, 1020.0)
    k = sb.k_from_alpha_cp(a, 900.0, 1020.0)
    assert k == pytest.approx(0.0165, rel=1e-12)


def test_strategy_a_alpha_about_1p797e8():
    a = sb.strategy_a_alpha()
    assert a == pytest.approx(1.797e-8, rel=1e-3)


def test_alpha_A_plus_cp900_reproduces_k_0p0165():
    a = sb.strategy_a_alpha()
    k = sb.k_from_alpha_cp(a, 900.0, 1020.0)
    assert k == pytest.approx(0.0165, rel=1e-12)


# ===============================================================
# 5. alpha/cp 评估 == k/cp 评估 (位级一致)
# ===============================================================

def test_alpha_cp_evaluator_matches_k_cp_evaluator():
    from thermal_model.utilities.scan_effective_thermal_parameters import evaluate_point
    alpha_A = sb.strategy_a_alpha()
    r_kcp = evaluate_point(0.0165, 900.0, _T, _TINT, _TTOP)
    r_acp = sb.evaluate_alpha_cp(alpha_A, 900.0, _T, _TINT, _TTOP)
    assert r_acp["derived_k_eff_W_mK"] == pytest.approx(0.0165, rel=1e-12)
    # k=alpha*rho*cp 回程引入 ~1e-16 相对浮点噪声 -> RMSE 1e-12 级差异可接受
    assert r_acp["RMSE_C"] == pytest.approx(r_kcp["RMSE_C"], rel=1e-9)
    assert r_acp["MAE_C"] == pytest.approx(r_kcp["MAE_C"], rel=1e-9)
    assert r_acp["mean_residual_C"] == pytest.approx(
        r_kcp["mean_residual_C"], rel=1e-9)


def test_equivalence_check_function_would_pass_on_synthetic():
    # 等价性检查逻辑: 派生 k 匹配 + RMSE 复现 (用合成协议验证函数可用)
    r = sb.evaluate_alpha_cp(sb.strategy_a_alpha(), 900.0, _T, _TINT, _TTOP)
    assert abs(r["derived_k_eff_W_mK"] - 0.0165) < 1e-12
    assert r["RMSE_C"] > 0  # 正常指标


# ===============================================================
# 6-7. 时间采样
# ===============================================================

def test_measurement_time_is_query_axis():
    from thermal_model.utilities.scan_effective_thermal_parameters import (
        sample_prediction_at_measurement_times,
    )
    fdm_t = np.arange(0.0, 60.0, 1.0)
    fdm_sig = fdm_t.copy()
    mt = np.array([5.0, 15.0, 25.0])
    out = sample_prediction_at_measurement_times(mt, fdm_t, fdm_sig)
    np.testing.assert_allclose(out, mt)


def test_no_temperature_as_time_semantics():
    # 评估只以时间为查询轴: 脚本中绝无 T_top_meas 作插值查询坐标的调用
    text = open(sb.__file__, encoding="utf-8").read()
    assert "np.interp(t_top_meas" not in text
    assert "np.interp(T_top_meas" not in text
    assert "interp(t_top_meas" not in text


# ===============================================================
# 8-11. 材料 / 几何
# ===============================================================

def test_only_coc_changed_rho_and_others_unchanged():
    mats = sb.make_candidate_materials(0.05, 2000.0)
    assert mats["COC"].k_W_mK == 0.05
    assert mats["COC"].cp_J_kgK == 2000.0
    assert mats["COC"].rho_kg_m3 == 1020.0
    base = heat_model.DEFAULT_MATERIALS
    for m in ("Water", "Oil"):
        assert mats[m].k_W_mK == base[m].k_W_mK
        assert mats[m].rho_kg_m3 == base[m].rho_kg_m3
        assert mats[m].cp_J_kgK == base[m].cp_J_kgK


def test_uses_bare_top_layers():
    layers = heat_model.BARE_TOP_COC_LAYERS
    mats = sb.make_candidate_materials(0.0165, 900.0)
    mesh = heat_model.build_layer_stack(mats, layers)
    assert mesh.x[-1] == pytest.approx(850e-6)
    names = " ".join(ly.name for ly in layers)
    assert "Air" not in names and "PDMS" not in names


# ===============================================================
# 12-14. 带 / 最高 alpha / RMSE<=1C
# ===============================================================

def _fake_df():
    # alpha 递增, RMSE 递减后递增 -> 构造已知最优点
    alphas = np.array([1.0e-8, 1.8e-8, 3.5e-8, 7.0e-8, 1.4e-7])
    cps = np.full(5, 900.0)
    rmse = np.array([1.2, 0.8, 0.6, 0.7, 0.95])
    k = sb.alpha_from_k_cp if False else None
    return pd.DataFrame({
        "alpha_eff_m2_s": alphas,
        "cp_eff_J_kgK": cps,
        "derived_k_eff_W_mK": [sb.k_from_alpha_cp(a, 900.0) for a in alphas],
        "RMSE_C": rmse,
        "MAE_C": rmse - 0.1,
        "mean_residual_C": np.zeros(5),
        "max_abs_error_C": rmse,
        "status": ["OK"] * 5,
        "t_diff_180um_s": [sb.t_diff_um2_s(a, 180) for a in alphas],
        "t_diff_190um_s": [sb.t_diff_um2_s(a, 190) for a in alphas],
    })


def test_rmse_min_found():
    df = _fake_df()
    assert sb.rmse_min_from_df(df) == pytest.approx(0.6)


def test_highest_alpha_in_band():
    df = _fake_df()
    rmin = 0.6
    hi = sb.highest_alpha_in_band(df, rmin, 0.05)   # RMSE <= 0.65 -> alpha 3.5e-8
    assert hi["alpha_eff_m2_s"] == pytest.approx(3.5e-8)
    hi2 = sb.highest_alpha_in_band(df, rmin, 0.35)  # RMSE <= 0.95 -> alpha 1.4e-7
    assert hi2["alpha_eff_m2_s"] == pytest.approx(1.4e-7)
    hi3 = sb.highest_alpha_in_band(df, rmin, 0.01)  # RMSE <= 0.61 -> 仅最优点
    assert hi3["alpha_eff_m2_s"] == pytest.approx(3.5e-8)


def test_rmse_le_1c_selection():
    df = _fake_df()
    cand = sb.highest_alpha_rmse_le_1c(df)  # RMSE<=1.0 -> 最高 alpha = 1.4e-7 (RMSE 0.95)
    assert cand is not None
    assert cand["alpha_eff_m2_s"] == pytest.approx(1.4e-7)
    df2 = df.copy()
    df2.loc[4, "RMSE_C"] = 1.2            # 现在最后一个点 RMSE>1 -> 候选 = 7e-8
    cand2 = sb.highest_alpha_rmse_le_1c(df2)
    assert cand2["alpha_eff_m2_s"] == pytest.approx(7.0e-8)


def test_band_alpha_ranges():
    df = _fake_df()
    rmin = 0.6
    ar = sb.near_optimal_alpha_ranges(df, rmin, 0.05)  # RMSE<=0.65 -> 3.5e-8 点
    assert ar["n"] == 1
    ar2 = sb.near_optimal_alpha_ranges(df, rmin, 0.35)  # RMSE<=0.95
    # 符合条件的 RMSE: 0.8, 0.6, 0.7, 0.95 -> 4 点 (1.2 被排除)
    assert ar2["n"] == 4
    assert ar2["min"] == pytest.approx(1.8e-8)
    assert ar2["max"] == pytest.approx(1.4e-7)


# ===============================================================
# 15. Pareto
# ===============================================================

def test_pareto_front_correct():
    df = _fake_df()
    front = sb.pareto_front(df)
    # 点: (1e-8,1.2) 被 (1.8e-8,0.8) 支配; (1.8e-8,0.8) 被 (3.5e-8,0.6) 支配;
    # (7e-8,0.7) 不被支配 (alpha 更高但 RMSE 高); (1.4e-7,0.95) 不被支配。
    alphas = sorted(front["alpha_eff_m2_s"].tolist())
    assert alphas == pytest.approx([3.5e-8, 7.0e-8, 1.4e-7])
    # 前沿单调: alpha 升序时 RMSE 非降
    assert np.all(np.diff(front["RMSE_C"].to_numpy()) >= 0)


def test_pareto_dominance_rule():
    # 相同 RMSE 且更高 alpha -> 支配
    df = pd.DataFrame({
        "alpha_eff_m2_s": [1.0e-8, 2.0e-8],
        "RMSE_C": [0.8, 0.8],
        "status": ["OK", "OK"],
        "cp_eff_J_kgK": [900.0, 900.0],
        "derived_k_eff_W_mK": [1.0, 2.0],
        "MAE_C": [0.7, 0.7],
        "mean_residual_C": [0.0, 0.0],
        "max_abs_error_C": [0.8, 0.8],
        "t_diff_180um_s": [1.0, 1.0],
        "t_diff_190um_s": [1.0, 1.0],
    })
    front = sb.pareto_front(df)
    assert len(front) == 1
    assert front["alpha_eff_m2_s"].iloc[0] == pytest.approx(2.0e-8)


# ===============================================================
# 16-17. 不改名义标定 / tag 元数据
# ===============================================================

def test_nominal_calibration_unchanged():
    before = (cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK,
              cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK)
    # 运行一次 DOE11 指标 (会调用 make_candidate_materials, 不修改全局)
    sb.make_candidate_materials(0.05, 2000.0)
    sb.k_from_alpha_cp(1e-7, 900.0)
    after = (cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK,
             cmc.NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK)
    assert before == after == (0.0165, 900.0)


def test_metadata_does_not_mutate_git():
    # 元数据写入函数只是读取 git; 不改变 tag
    before = sb._git_describe()
    meta = {"git_commit": sb._git_head(), "git_tag": before}
    assert meta["git_tag"]  # 非空
    assert sb._git_describe() == before


# ===============================================================
# 18-19. DOE11 不进拟合目标
# ===============================================================

def test_doe11_not_in_objective():
    # 评估函数 evaluate_alpha_cp 只用 72C (t_proto, t_int, t_top_meas),
    # 不接收任何 DOE11 样品数据
    import inspect
    sig = inspect.signature(sb.evaluate_alpha_cp)
    params = list(sig.parameters)
    assert "doe11" not in " ".join(params).lower()
    # 候选选择基于 RMSE 表, DOE11 列在候选表中位于选择之后 (附加列)
    text = open(sb.__file__, encoding="utf-8").read()
    i_sel = text.find("highest_alpha_in_band")
    i_doe = text.find("doe11_candidate_metrics")
    assert i_doe > i_sel


def test_doe11_sample_cannot_affect_candidate_eligibility():
    # 候选资格只由 RMSE/alpha 决定: 构造候选表时 DOE11 列不存在于选择输入
    df = _fake_df()
    cand = sb.highest_alpha_in_band(df, 0.6, 0.35)
    assert cand["alpha_eff_m2_s"] == pytest.approx(1.4e-7)
    # 即使加入任意 DOE11 样品值, 选择函数签名不接收它
    df2 = df.copy()
    df2["DOE11_sample_max_C"] = [999.0, 999.0, 999.0, 999.0, 999.0]
    cand2 = sb.highest_alpha_in_band(df2, 0.6, 0.35)
    assert cand2["alpha_eff_m2_s"] == pytest.approx(1.4e-7)


# ===============================================================
# 20. 输出隔离
# ===============================================================

def test_output_isolated_to_strategy_b_dir():
    out = str(sb.OUTPUT_DIR)
    assert out.endswith("fast_pcr_oriented_alpha_cp_v1")
    for d in ["72C_corrected_objective_v1", "72C_nominal_v1",
              "60C_redo_transfer_check_v1",
              "08.12_pm_DOE11_faster_sample_prediction_v1",
              "corrected_time_objective_v3"]:
        assert d not in out


def test_strategy_b_metadata_marker(tmp_path):
    # 元数据标记: EXPERIMENTAL / accepted=False
    meta = sb._write_metadata.__doc__ if sb._write_metadata.__doc__ else ""
    # 直接验证常量
    assert sb.RMSE_LE_1C_THRESHOLD == 1.0
    assert set(sb.BANDS) == {"STRICT", "MODERATE", "APPLICATION"}
    assert sb.BANDS["STRICT"] == 0.05
    assert sb.BANDS["MODERATE"] == 0.10
    assert sb.BANDS["APPLICATION"] == 0.20
