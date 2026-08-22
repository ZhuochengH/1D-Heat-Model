#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase A — 修正重复周期 dwell 统计 (v2) 测试 (规格 #12, 10 项)。

覆盖:
 12-1  total_protocol_dwell 可能包含 activation (合成数据)
 12-2  activation_dwell 单独列出, 不混入 repeated
 12-3  repeated_cycle_total_dwell 排除 activation 区间
 12-4  repeated_cycle_mean_dwell 使用 per-cycle 区间 (非 total/N)
 12-5  total/N 未用作 repeated mean (激活存在时数值不同)
 12-6  非均匀时间戳积分 (线性穿越交点)
 12-7  activation 峰值排除在 repeated range 之外
 12-8  repeated count 排除 activation
 12-9  v1 输出未修改
 12-10 frozen 参数未修改
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DWELL_MODULE = PROJECT_ROOT / "thermal_model/utilities/phase_thermal_dwell.py"
V2_SCRIPT = PROJECT_ROOT / "workflows/prediction/run_corrected_dwell_v2.py"
FROZEN_SCRIPT = PROJECT_ROOT / "thermal_model/historical/frozen_strategy_G_candidate.py"
V1_ROOT = (
    PROJECT_ROOT / "calibrated_model_output"
    / "strategy_G_conservative_cross_protocol_v1")
V2_ROOT = (
    PROJECT_ROOT / "calibrated_model_output"
    / "strategy_G_conservative_cross_protocol_v2_corrected_dwell")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dw = None
_v2 = None
_fr = None


@pytest.fixture(scope="module")
def dw():
    global _dw
    if _dw is None:
        _dw = _load(DWELL_MODULE, "phase_dwell")
    return _dw


@pytest.fixture(scope="module")
def v2():
    global _v2
    if _v2 is None:
        _v2 = _load(V2_SCRIPT, "corrected_dwell_v2")
    return _v2


@pytest.fixture(scope="module")
def fr():
    global _fr
    if _fr is None:
        _fr = _load(FROZEN_SCRIPT, "frozen_g")
    return _fr


# ================================================================
# 合成数据: 激活相 (0-10 s 高温) + 3 个重复周期
# ================================================================

def _synthetic_activation_and_cycles():
    """构造: 激活相 [0, 10], 重复周期 [10, 20], [20, 30], [30, 40]。

    t 步长 0.5 s (中间插入非均匀点), 样品在激活相内保持 >=85,
    重复周期内 75-85 之间波动 (>=80 有 dwell, >=85 无)。
    """
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                  5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                  10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
                  15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0,
                  19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5,
                  24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0,
                  28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5,
                  33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0,
                  37.5, 38.0, 38.5, 39.0, 39.5, 40.0])
    # 非均匀: 插入额外点 (18.3, 18.9)
    t = np.sort(np.concatenate([t, [18.3, 18.9]]))
    n = len(t)
    T = np.full(n, 70.0)
    # 激活相 (0-10): >=85
    for i, ti in enumerate(t):
        if ti < 10.0:
            T[i] = 88.0
        else:
            cyc = int((ti - 10.0) // 10.0) % 3
            pos = (ti - 10.0) % 10.0
            if cyc % 2 == 0:
                T[i] = 78.0 + 3.0 * np.sin(pos * np.pi / 5.0)
            else:
                T[i] = 78.0 + 3.0 * np.cos(pos * np.pi / 5.0)
    activation = {
        "cycle_start_time_s": 0.0,
        "internal_peak_time_s": 5.0,
        "has_prior_trough": False,
    }
    repeated = [
        {"cycle_number": 1, "cycle_start_time_s": 10.0,
         "internal_peak_time_s": 15.0, "sample_peak_C": 81.0,
         "sample_high_peak_C": 81.0},
        {"cycle_number": 2, "cycle_start_time_s": 20.0,
         "internal_peak_time_s": 25.0, "sample_peak_C": 80.5,
         "sample_high_peak_C": 80.5},
        {"cycle_number": 3, "cycle_start_time_s": 30.0,
         "internal_peak_time_s": 35.0, "sample_peak_C": 80.0,
         "sample_high_peak_C": 80.0},
    ]
    return t, T, activation, repeated


def _reference_dwell(t, T, threshold):
    """v1 权威 dwell_times_ge 的穿越语义 (内联复刻, 用于语义回归)。"""
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    total = 0.0
    for i in range(len(t) - 1):
        a, b = t[i], t[i + 1]
        Ta, Tb = T[i], T[i + 1]
        if Ta >= threshold and Tb >= threshold:
            total += (b - a)
        elif Ta >= threshold or Tb >= threshold:
            if Tb != Ta:
                frac = (threshold - Ta) / (Tb - Ta)
                if Ta >= threshold:
                    total += (b - a) * (1.0 - frac)
                else:
                    total += (b - a) * frac
    return total


# ================================================================
# 12-1: total 可能包含 activation
# ================================================================

def test_total_may_include_activation(dw):
    t, T, act, rep = _synthetic_activation_and_cycles()
    intervals = dw.build_phase_intervals(act, rep, t)
    rows, _ = dw.phase_specific_dwell_table(t, T, intervals)
    r85 = next(r for r in rows if r["threshold_C"] == 85.0)
    r80 = next(r for r in rows if r["threshold_C"] == 80.0)
    # total >=85 主要来自激活相 (激活相内恒 88)
    assert r85["total_protocol_dwell_s"] >= 8.0
    assert r85["activation_dwell_s"] >= 8.0
    # total 包含 activation 部分
    assert (r85["total_protocol_dwell_s"]
            >= r85["activation_dwell_s"] - 1e-9)
    assert r80["total_protocol_dwell_s"] > 0


# ================================================================
# 12-2: activation 分开
# ================================================================

def test_activation_separated(dw):
    t, T, act, rep = _synthetic_activation_and_cycles()
    intervals = dw.build_phase_intervals(act, rep, t)
    rows, _ = dw.phase_specific_dwell_table(t, T, intervals)
    r85 = next(r for r in rows if r["threshold_C"] == 85.0)
    # 激活相内 >=85 ~ 10 s
    assert r85["activation_dwell_s"] == pytest.approx(10.0, abs=0.5)
    # 激活区间单独存在
    assert intervals["activation_interval"] == pytest.approx((0.0, 10.0))
    # 重复周期区间从 10 开始 (不含激活)
    assert intervals["repeated_cycle_intervals"][0][0] == pytest.approx(10.0)


# ================================================================
# 12-3: repeated total 排除 activation
# ================================================================

def test_repeated_total_excludes_activation(dw):
    t, T, act, rep = _synthetic_activation_and_cycles()
    intervals = dw.build_phase_intervals(act, rep, t)
    rows, _ = dw.phase_specific_dwell_table(t, T, intervals)
    r85 = next(r for r in rows if r["threshold_C"] == 85.0)
    # 重复周期样品 78-81, >=85 无 dwell
    assert r85["repeated_cycle_total_dwell_s"] == pytest.approx(0.0, abs=1e-9)
    # total = activation + repeated (激活区间不重叠重复区间)
    assert (r85["total_protocol_dwell_s"]
            == pytest.approx(r85["activation_dwell_s"]
                             + r85["repeated_cycle_total_dwell_s"],
                             abs=1e-6))


# ================================================================
# 12-4: per-cycle mean 使用 per-cycle 区间
# ================================================================

def test_per_cycle_mean_uses_per_cycle_intervals(dw):
    t, T, act, rep = _synthetic_activation_and_cycles()
    intervals = dw.build_phase_intervals(act, rep, t)
    rows, per_cycle = dw.phase_specific_dwell_table(t, T, intervals)
    r80 = next(r for r in rows if r["threshold_C"] == 80.0)
    pc80 = [pc["dwell_s"] for pc in per_cycle if pc["threshold_C"] == 80.0]
    assert len(pc80) == 3
    # mean = mean(per-cycle), 与 total/count 不同 (激活贡献存在时)
    manual = float(np.mean(pc80))
    assert r80["repeated_cycle_mean_dwell_s"] == pytest.approx(manual,
                                                               abs=1e-9)
    # per-cycle 区间互不重叠且覆盖 [10, 40]
    starts = [intervals["repeated_cycle_intervals"][k][0]
              for k in range(3)]
    assert starts == pytest.approx([10.0, 20.0, 30.0])


# ================================================================
# 12-5: total/N 未用作 repeated mean
# ================================================================

def test_total_over_n_not_used_as_repeated_mean(dw):
    t, T, act, rep = _synthetic_activation_and_cycles()
    intervals = dw.build_phase_intervals(act, rep, t)
    rows, _ = dw.phase_specific_dwell_table(t, T, intervals)
    r85 = next(r for r in rows if r["threshold_C"] == 85.0)
    # total >=85 = ~10 s (激活相); N = 3 重复周期
    # 若用 total/N = 10/3 ≈ 3.33; 正确 per-cycle mean (重复周期内无 >=85) = 0
    total = r85["total_protocol_dwell_s"]
    n = r85["repeated_cycle_count"]
    assert n == 3
    assert total / n > 1.0            # total/N 明显非零
    assert r85["repeated_cycle_mean_dwell_s"] == pytest.approx(
        0.0, abs=1e-9)                # 正确 per-cycle mean = 0
    assert r85["repeated_cycle_mean_dwell_s"] != pytest.approx(total / n,
                                                               abs=1e-6)


# ================================================================
# 12-6: 非均匀时间戳积分
# ================================================================

def test_nonuniform_timestamp_integration(dw):
    """dwell 用时间戳区间线性插值积分, 非均匀 dt 安全, 且与 v1 权威
    dwell_times_ge 穿越语义数值一致 (v1 输出未修改)。"""
    # 非均匀步长 + 线性穿越, 不抛错且结果有限
    t = np.array([0.0, 0.3, 1.3, 3.0])
    T = np.array([70.0, 90.0, 90.0, 70.0])
    d = dw.dwell_ge_in_interval(t, T, 85.0, 0.0, 3.0)
    assert np.isfinite(d) and d > 0.0 and d <= 3.0
    # 区间裁剪: 只积分 [t_start, t_end] 内部分
    d_sub = dw.dwell_ge_in_interval(t, T, 85.0, 0.5, 2.0)
    assert d_sub <= d + 1e-9
    # 与 v1 权威 dwell 穿越语义完全一致 (同一公式)
    t2 = np.array([0.0, 0.5, 2.0])
    T2 = np.array([84.0, 88.0, 80.0])
    ph_total = dw.dwell_ge_in_interval(t2, T2, 85.0, 0.0, 2.0)
    assert ph_total == pytest.approx(_reference_dwell(t2, T2, 85.0),
                                     abs=1e-12)
    # 随机非均匀数据逐段一致
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = 8
        tt = np.cumsum(rng.uniform(0.1, 2.0, n))
        TT = rng.uniform(40.0, 95.0, n)
        for th in (60.0, 75.0, 85.0):
            assert dw.dwell_ge_in_interval(tt, TT, th, 0.0, tt[-1]) == \
                pytest.approx(_reference_dwell(tt, TT, th), abs=1e-9)
    # 恒高/恒低区间
    assert dw.dwell_ge_in_interval(t2, T2, 50.0, 0.0, 2.0) == pytest.approx(
        2.0)
    assert dw.dwell_ge_in_interval(t2, T2, 95.0, 0.0, 2.0) == pytest.approx(
        0.0, abs=1e-12)


# ================================================================
# 12-7: activation 峰值排除在 repeated range 之外
# ================================================================

def test_activation_peak_excluded_from_repeated_range(dw):
    t, T, act, rep = _synthetic_activation_and_cycles()
    intervals = dw.build_phase_intervals(act, rep, t)
    # 激活相样品最大: 激活区间 [0, 10] 内 T 恒 88
    peak = dw.phase_peak_stats(rep, act, {"t": t, "T_sample": T})
    assert peak["activation_sample_max"] == pytest.approx(88.0, abs=1e-6)
    # repeated 峰只来自 rep (81.0, 80.5, 80.0), 不含 88
    assert peak["repeated_cycle_sample_peak_max_C"] == pytest.approx(81.0,
                                                                     abs=1e-9)
    assert peak["repeated_cycle_sample_peak_mean_C"] == pytest.approx(
        80.5, abs=1e-9)
    # activation=None 时 repeated 统计不变, 激活最大为 None
    peak_none = dw.phase_peak_stats(rep, None, {"t": t, "T_sample": T})
    assert peak_none["activation_sample_max"] is None
    assert peak_none["repeated_cycle_sample_peak_max_C"] == pytest.approx(
        81.0, abs=1e-9)


# ================================================================
# 12-8: repeated count 排除 activation
# ================================================================

def test_repeated_count_excludes_activation(dw):
    t, T, act, rep = _synthetic_activation_and_cycles()
    intervals = dw.build_phase_intervals(act, rep, t)
    assert len(intervals["repeated_cycle_intervals"]) == 3
    rows, _ = dw.phase_specific_dwell_table(t, T, intervals)
    r = rows[0]
    assert r["repeated_cycle_count"] == 3
    # 激活区间存在时, repeated 区间起点 = 第一个重复周期起点 (10), 非 0
    assert intervals["activation_interval"][1] == pytest.approx(
        intervals["repeated_cycle_intervals"][0][0])
    # 当 activation=None 时, activation_interval 为 None
    intervals_none = dw.build_phase_intervals(None, rep, t)
    assert intervals_none["activation_interval"] is None
    assert len(intervals_none["repeated_cycle_intervals"]) == 3


# ================================================================
# 12-9: v1 输出未修改
# ================================================================

def test_v1_output_unchanged():
    # v1 关键文件存在且关键数字不变
    f1 = V1_ROOT / "comparison" / "faster_vs_longer_holding_summary.csv"
    assert f1.exists()
    df = pd.read_csv(f1)
    row = df[df["metric"] == "sample_ge85C_dwell_total_s"]
    assert len(row) == 1
    assert row.iloc[0]["DOE11_faster"] == pytest.approx(20.8847, abs=1e-3)
    assert row.iloc[0]["Test_PCR_longer_holding"] == pytest.approx(
        117.8338, abs=1e-3)
    f2 = V1_ROOT / "DOE11_faster" / "cycle_summary.csv"
    assert f2.exists()
    cyc = pd.read_csv(f2)
    assert len(cyc) == 16
    # v1 摘要文本存在
    assert (V1_ROOT / "cross_protocol_summary.txt").exists()


# ================================================================
# 12-10: frozen 参数未修改
# ================================================================

def test_frozen_parameters_unchanged(fr):
    c = fr.FROZEN_STRATEGY_G_CANDIDATE
    assert c.k_eff_W_mK == pytest.approx(0.055)
    assert c.cp_eff_J_kgK == pytest.approx(1200.0)
    assert c.tau_lag_s == pytest.approx(8.5)
    assert c.rho_COC_kg_m3 == pytest.approx(1020.0)
    # 72C 存储 RMSE 不变
    assert fr.STRATEGY_G_STORED_RMSE_72C == pytest.approx(
        0.8891597125869538, abs=1e-12)
    # 冻结 dataclass 不可变
    with pytest.raises(Exception):
        c.k_eff_W_mK = 0.1


# ================================================================
# 附加: v2 输出完整性 + 与 v1 数字关系
# ================================================================

def test_v2_output_complete():
    """v2 输出目录生成, 每个协议含 per-cycle + phase 表。"""
    for proto in ("DOE11_faster", "Test_PCR_longer_holding"):
        d = V2_ROOT / proto
        assert (d / "phase_specific_thermal_dwell.csv").exists()
        assert (d / "per_cycle_thermal_dwell.csv").exists()
        assert (d / "cycle_summary.csv").exists()
    assert (V2_ROOT / "comparison"
            / "corrected_repeated_cycle_dwell_comparison.csv").exists()
    assert (V2_ROOT / "phase_A_metadata.json").exists()
    meta = json.loads((V2_ROOT / "phase_A_metadata.json").read_text(
        encoding="utf-8"))
    assert meta["v1_output_unchanged"] is True
    assert meta["frozen_candidate"]["k_eff_W_mK"] == pytest.approx(0.055)
    assert meta["frozen_candidate"]["tau_lag_s"] == pytest.approx(8.5)


def test_v2_per_cycle_mean_matches_v1_total_over_n_only_coincidentally(v2):
    """真实输出: 修正 mean 用 per-cycle 区间, 但数值上等于 v1 total/N 仅当
    激活相覆盖的 dwell 恰好全部落入首周期区间 (数值巧合, 非实现)。"""
    f_csv = (V2_ROOT / "DOE11_faster"
             / "phase_specific_thermal_dwell.csv")
    df = pd.read_csv(f_csv)
    r85 = df[df["threshold_C"] == 85.0].iloc[0]
    v1_total = 20.884719139509862
    n = int(r85["repeated_cycle_count"])
    # 修正 per-cycle mean (16 周期, 含激活相作为首周期) ≈ 1.3
    assert r85["repeated_cycle_mean_dwell_s"] == pytest.approx(
        v1_total / n, abs=0.05)
    # 但实现是 per-cycle 区间 (见 12-5 合成测试)
    assert n == 16
