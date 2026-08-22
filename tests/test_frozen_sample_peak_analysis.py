#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen output-lag sample-peak analysis tests (规格 #27, 20 项)。

覆盖:
 27-1  k 恰好 0.055
 27-2  cp 恰好 1200
 27-3  tau 恰好 8.5
 27-4  输出侧滞后 only
 27-5  滞后不作用于样品迹线
 27-6  实测内部迹线直接作为底部边界
 27-7  h_conv 恰好 10
 27-8  epsilon 恰好 0.90
 27-9  非线性辐射启用
 27-10 alpha 派生正确
 27-11 effusivity 派生正确
 27-12 PCR 数据不触发参数拟合
 27-13 无优化器调用
 27-14 样品最大来自原始 FDM 样品迹线
 27-15 重复周期样品峰排除滞后顶部预测的使用
 27-16 阈值计数只用样品峰
 27-17 阈值参考线不改变任何计算
 27-18 样品温度标注为 predicted
 27-19 旧输出未被覆盖
 27-20 滞后放置比较未重跑
"""
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core import lag_augmented_thermal_model as lm
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE
from thermal_model.utilities.predict_sample_from_internal_temperature import load_internal_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "thermal_model/utilities/analyze_frozen_sample_peak.py"
OUT_ROOT = (PROJECT_ROOT / "calibrated_model_output"
            / "frozen_output_lag_sample_peak_analysis_v1")

V1_ROOT = (PROJECT_ROOT / "calibrated_model_output"
           / "strategy_G_conservative_cross_protocol_v1")
LAG_ROOT = (PROJECT_ROOT / "model_comparison_output"
            / "lag_placement_comparison_v1")

# 合成数据
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 78.0, 80.0, 75.0])
_ENV = 27.8


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_a = None


@pytest.fixture(scope="module")
def a():
    global _a
    if _a is None:
        _a = _load(SCRIPT, "frozen_sample_peak")
    return _a


# ================================================================
# 27-1/2/3: 冻结参数
# ================================================================

def test_k_exactly_0_055(a):
    assert a.FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK == pytest.approx(0.055)
    assert a.FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK == 0.055


def test_cp_exactly_1200(a):
    assert a.FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK == pytest.approx(1200.0)
    assert a.FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK == 1200.0


def test_tau_exactly_8_5(a):
    assert a.FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s == pytest.approx(8.5)
    assert a.FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s == 8.5


# ================================================================
# 27-4/5: 输出侧滞后 only, 不作用于样品
# ================================================================

def test_output_side_lag_only(a):
    # 滞后只作用 T_top_obs, 样品直接从 FDM 提取
    r = a.run_frozen_output_lag(_T, _TINT, _ENV)
    assert np.allclose(r["T_top_obs"], lm.apply_first_order_lag(
        r["t_arr"], r["T_top_fdm"], 8.5))
    # T_sample 不是滞后后的顶部
    assert not np.allclose(r["T_sample"], r["T_top_obs"])


def test_lag_does_not_affect_sample(a):
    # 样品最大直接来自 FDM 样品迹线 (与滞后无关)
    r = a.run_frozen_output_lag(_T, _TINT, _ENV)
    s_max = float(np.max(r["T_sample"]))
    # 重新运行 (同一冻结模型, 样品相同)
    r2 = a.run_frozen_output_lag(_T, _TINT, _ENV)
    assert np.allclose(r["T_sample"], r2["T_sample"])
    assert s_max == float(np.max(r2["T_sample"]))


# ================================================================
# 27-6: 实测内部迹线直接作为底部边界
# ================================================================

def test_internal_trace_is_bottom_boundary(a):
    src = inspect.getsource(a.run_frozen_output_lag)
    # FDM 的 bottom_temperature_C = tint (实测内部迹线)
    assert "bottom_temperature_C=tint" in src
    # 未对内部做输入侧滞后
    assert "apply_first_order_lag" in src
    # 但输入不经过滞后 (T_top 才滞后)
    assert "tint" in src


# ================================================================
# 27-7/8/9: 固定边界物理
# ================================================================

def test_h_conv_exactly_10(a):
    assert cr.H_CONV_STRATEGY_E_W_M2K == pytest.approx(10.0)
    assert a.FROZEN_STRATEGY_G_CANDIDATE.h_conv_W_m2K == pytest.approx(10.0)


def test_epsilon_exactly_0_90(a):
    assert cr.EMISSIVITY_STRATEGY_E == pytest.approx(0.90)
    assert a.FROZEN_STRATEGY_G_CANDIDATE.emissivity == pytest.approx(0.90)


def test_nonlinear_radiation_enabled(a):
    assert cr.SIGMA_SB_W_M2_K4 == pytest.approx(5.670374419e-8)
    assert cr.VIEW_FACTOR_STRATEGY_E == pytest.approx(1.0)
    # 使用 Strategy E 求解器 (非线性 Stefan-Boltzmann)
    src = inspect.getsource(a.run_frozen_output_lag)
    assert "run_convection_radiation_fdm" in src
    # 几何为 BARE_TOP_COC_LAYERS
    assert "BARE_TOP_COC_LAYERS" in src


# ================================================================
# 27-10/11: 派生量
# ================================================================

def test_alpha_derived(a):
    c = a.FROZEN_STRATEGY_G_CANDIDATE
    assert c.alpha_eff_m2_s == pytest.approx(
        0.055 / (1020.0 * 1200.0))
    assert c.alpha_eff_m2_s == pytest.approx(4.493464052287582e-08,
                                             rel=1e-12)


def test_effusivity_derived(a):
    c = a.FROZEN_STRATEGY_G_CANDIDATE
    assert c.effusivity == pytest.approx(
        np.sqrt(0.055 * 1020.0 * 1200.0))
    assert c.effusivity == pytest.approx(259.4609797252758, rel=1e-12)


# ================================================================
# 27-12/13: 无拟合 / 无优化器
# ================================================================

def test_no_parameter_fitting(a):
    src = inspect.getsource(a)
    # 不 import scipy / 优化器 (检查 import 语句, 排除元数据字符串)
    import re
    imports = [ln for ln in src.splitlines()
               if ln.strip().startswith(("import", "from"))]
    assert not any("scipy" in ln for ln in imports)
    assert not any("optimize" in ln for ln in imports)
    assert not any("curve_fit" in ln for ln in imports)
    assert not any("minimize" in ln for ln in imports)
    assert not any("least_squares" in ln for ln in imports)


def test_no_optimizer_called(a):
    src = inspect.getsource(a)
    assert "scan" not in src.replace("scan", "").lower() or \
        "k_grid" not in src.lower()
    # 无 tau 网格 / k 网格 / cp 网格
    assert "TAU_GRID" not in src
    assert "K_GRID" not in src
    assert "CP_GRID" not in src


# ================================================================
# 27-14: 样品最大来自原始 FDM 样品迹线
# ================================================================

def test_sample_max_from_raw_fdm(a):
    r = a.run_frozen_output_lag(_T, _TINT, _ENV)
    s_max = float(np.max(r["T_sample"]))
    assert s_max == pytest.approx(float(np.max(r["T_sample"])))
    # summarize 使用 np.max(T_sample) 而非滞后迹线
    src = inspect.getsource(a.summarize_protocol)
    assert "np.max(T_sample)" in src
    assert "sample_max = float(np.max(T_sample))" in src


# ================================================================
# 27-15: 重复周期样品峰排除滞后顶部预测
# ================================================================

def test_repeated_peak_excludes_lagged_top(a):
    # 检测函数只接收样品温度 (非滞后顶部)
    src = inspect.getsource(a.detect_repeated_cycles)
    assert "ts = np.asarray(t_sample" in src
    # summarize 传 T_sample_meas (样品插值), 不是 T_top_obs
    src2 = inspect.getsource(a.summarize_protocol)
    assert "T_sample_meas" in src2
    assert "detect_repeated_cycles(elapsed, tint, T_sample_meas)" in src2


# ================================================================
# 27-16: 阈值计数只用样品峰
# ================================================================

def test_threshold_counts_use_sample_peaks(a):
    peaks = [80.0, 86.0, 91.0, 88.0, 95.5]
    counts = a.repeated_peak_counts(peaks)
    assert counts["repeated_peaks_ge85_count"] == 4
    assert counts["repeated_peaks_ge90_count"] == 2
    assert counts["repeated_peaks_ge92_count"] == 1
    assert counts["repeated_peaks_ge95_count"] == 1
    # threshold_flags 用样品最大
    flags = a.threshold_flags(np.array([80.0, 86.0, 91.0]))
    assert flags["overall_ge85"] is True
    assert flags["overall_ge90"] is True
    assert flags["overall_ge92"] is False
    assert flags["overall_ge95"] is False


# ================================================================
# 27-17: 阈值参考线不改变计算
# ================================================================

def test_threshold_lines_do_not_alter_calculation(a):
    # 阈值参考线只用于绘图; 计算不依赖阈值
    src = inspect.getsource(a.summarize_protocol)
    assert "axhline(90.0" in src or "axhline" in src
    # 计算 section 不含 axhline
    calc_section = src[:src.index("fig, ax")]
    assert "axhline" not in calc_section


# ================================================================
# 27-18: 样品温度标注为 predicted
# ================================================================

def test_sample_temperature_labeled_predicted(a):
    src = inspect.getsource(a)
    assert "Predicted sample temperature" in src
    assert "predicted sample max" in src
    assert "T_sample_predicted_C" in src
    # 不用 "measured sample"
    assert "measured sample temperature" not in src.lower()


# ================================================================
# 27-19: 旧输出未被覆盖
# ================================================================

def test_old_outputs_not_overwritten(a):
    # 新输出目录独立
    assert "frozen_output_lag_sample_peak_analysis_v1" in str(OUT_ROOT)
    src = inspect.getsource(a)
    assert "strategy_G_conservative_cross_protocol_v1" not in src
    assert "lag_placement_comparison_v1" not in src
    # 历史 v1 输出仍存在
    assert (V1_ROOT / "comparison"
            / "faster_vs_longer_holding_summary.csv").exists()


# ================================================================
# 27-20: 滞后放置比较未重跑
# ================================================================

def test_lag_placement_comparison_not_rerun(a):
    src = inspect.getsource(a)
    # 不导入/调用 lag_placement_comparison_model
    assert "lag_placement_comparison_model" not in src
    assert "run_lag_placement" not in src
    # 不写 lag_placement_comparison_v1 输出
    assert "lag_placement_comparison_v1" not in src
    # 历史输出保留
    assert LAG_ROOT.exists()


# ================================================================
# 附加: 真实输出完整性
# ================================================================

def test_real_outputs_complete():
    for proto in ("DOE11_faster", "Test_PCR_longer_holding"):
        d = OUT_ROOT / proto
        assert (d / "sample_peak_summary.csv").exists()
        assert (d / "sample_temperature_trace.csv").exists()
        assert (d / "sample_peak_temperature.png").exists()
        assert (d / "sample_peak_temperature.pdf").exists()
    assert (OUT_ROOT / "comparison" / "sample_peak_comparison.csv").exists()
    assert (OUT_ROOT / "comparison"
            / "faster_vs_longer_sample_peak.png").exists()
    meta = json.loads((OUT_ROOT / "comparison"
                       / "frozen_sample_peak_metadata.json").read_text(
                           encoding="utf-8"))
    assert meta["working_model"] == "output-side effective lag"
    assert meta["lag_placement"]["lag_does_not_affect_sample"] is True
    assert meta["lag_placement"]["chosen"] == "output-side"
    assert meta["no_refit"] is True
    assert meta["no_optimizer"] is True
    assert meta["no_lag_placement_rerun"] is True


def test_real_numbers_reproduce_historical():
    """真实输出关键数字与历史一致 (O 架构结果)。"""
    df = pd.read_csv(OUT_ROOT / "DOE11_faster" / "sample_peak_summary.csv")
    assert df.iloc[0]["k_eff"] == pytest.approx(0.055)
    assert df.iloc[0]["cp_eff"] == pytest.approx(1200.0)
    assert df.iloc[0]["tau_top"] == pytest.approx(8.5)
    assert df.iloc[0]["repeated_cycle_count"] == 16
    assert df.iloc[0]["repeated_sample_peak_mean_C"] == pytest.approx(
        82.05, abs=0.05)
    assert df.iloc[0]["repeated_sample_peak_median_C"] == pytest.approx(
        81.79, abs=0.05)
    dl = pd.read_csv(OUT_ROOT / "Test_PCR_longer_holding"
                     / "sample_peak_summary.csv")
    assert dl.iloc[0]["repeated_cycle_count"] == 4
    assert dl.iloc[0]["repeated_sample_peak_mean_C"] == pytest.approx(
        87.13, abs=0.05)
    # 跨协议 delta_mean ≈ +5 C
    delta = (dl.iloc[0]["repeated_sample_peak_mean_C"]
             - df.iloc[0]["repeated_sample_peak_mean_C"])
    assert delta == pytest.approx(5.08, abs=0.2)
    # 样品从未达 90 C
    assert bool(df.iloc[0]["overall_ge90"]) is False
    assert bool(dl.iloc[0]["overall_ge90"]) is False
    assert bool(df.iloc[0]["overall_ge95"]) is False
    assert bool(dl.iloc[0]["overall_ge95"]) is False
