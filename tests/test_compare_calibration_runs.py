"""
交叉验证计算的聚焦单元测试 (compare_calibration_runs)。

覆盖:
- final_calibration_equation.txt 解析;
- Dataset 1 稳态预测;
- 校准范围内/外点识别;
- 主插值验证指标 (排除外推点);
- Bias / MAE / RMSE / Max Absolute Error;
- 对称相对差异;
- 合成重叠范围内的方程比较;
- 加热/冷却重复稳态点的保留;
- 当 Dataset 1 范围为 35-95 °C 时, 30 °C 必须被排除在主指标之外。
"""

import numpy as np
import pandas as pd
import pytest

from compare_calibration_runs import (
    CalibrationEquation,
    DATASET1_RANGE,
    DATASET1_SETPOINTS,
    DATASET2_SETPOINTS,
    SYNTHETIC_DT,
    SYNTHETIC_HOLD_S,
    SYNTHETIC_SETPOINTS,
    build_synthetic_profile,
    build_validation_frame,
    classify_by_range,
    compute_metrics,
    dynamic_trajectory_comparison,
    equation_grid_comparison,
    load_mean_steady_points,
    parse_equation_file,
    per_transition_max_differences,
    predict_dynamic_profile,
    steady_predict,
    symmetric_relative_difference,
    verify_steady_points,
)

# ---------------------------------------------------------------
# 1. 方程解析
# ---------------------------------------------------------------

EQ_FILE_TEXT = """FINAL CALIBRATION MODEL
=======================

Steady-state calibration:
T_inf = 0.950490 * T_set +1.811586

Dynamic calibration:
T_s(t) = (0.950490 * T_set,j +1.811586) + [T_s(t_j) - (0.950490 * T_set,j +1.811586)] * exp(-(t - t_j) / 7.3072)

Model quality:
Steady-state R² = 0.999987
Dynamic RMSE = 0.2151 °C
"""


def test_parse_equation_file_extracts_all_parameters(tmp_path):
    p = tmp_path / "final_calibration_equation.txt"
    p.write_text(EQ_FILE_TEXT, encoding="utf-8")

    eq = parse_equation_file(p)

    assert eq.a == pytest.approx(0.950490, abs=1e-9)
    assert eq.b == pytest.approx(1.811586, abs=1e-9)
    assert eq.tau_eff == pytest.approx(7.3072, abs=1e-9)
    assert eq.r2_steady == pytest.approx(0.999987, abs=1e-9)
    assert eq.rmse_dynamic == pytest.approx(0.2151, abs=1e-9)


def test_parse_equation_file_negative_intercept(tmp_path):
    text = EQ_FILE_TEXT.replace("+1.811586", "-1.234500").replace(
        "(0.950490 * T_set,j +1.811586)", "(0.950490 * T_set,j -1.234500)"
    )
    p = tmp_path / "eq.txt"
    p.write_text(text, encoding="utf-8")

    eq = parse_equation_file(p)
    assert eq.b == pytest.approx(-1.234500, abs=1e-9)


def test_parse_equation_file_missing_steady_line_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("no steady line here", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_equation_file(p)


# ---------------------------------------------------------------
# 2. Dataset 1 稳态预测
# ---------------------------------------------------------------

def test_steady_prediction_matches_linear_equation():
    eq = CalibrationEquation(
        a=0.95, b=1.8, tau_eff=7.3,
        r2_steady=0.9999, rmse_dynamic=0.2, source="synthetic",
    )
    assert eq.predict(40.0) == pytest.approx(39.8, abs=1e-9)
    assert eq.predict(90.0) == pytest.approx(87.3, abs=1e-9)

    preds = eq.predict(np.array([40.0, 50.0, 60.0]))
    np.testing.assert_allclose(preds, [39.8, 49.3, 58.8], atol=1e-9)


def test_steady_predict_helper():
    eq = CalibrationEquation(a=0.95, b=1.8, tau_eff=1.0,
                             r2_steady=None, rmse_dynamic=None, source="s")
    assert steady_predict(eq, [40.0])[0] == pytest.approx(39.8)


# ---------------------------------------------------------------
# 3. 范围内/外点识别
# ---------------------------------------------------------------

def test_classify_by_range():
    assert classify_by_range(40.0, DATASET1_RANGE) == "interpolation"
    assert classify_by_range(35.0, DATASET1_RANGE) == "interpolation"
    assert classify_by_range(95.0, DATASET1_RANGE) == "interpolation"
    assert classify_by_range(30.0, DATASET1_RANGE) == "extrapolation"
    assert classify_by_range(96.0, DATASET1_RANGE) == "extrapolation"


# ---------------------------------------------------------------
# 4+5. 指标计算 (Bias / MAE / RMSE / Max)
# ---------------------------------------------------------------

def test_compute_metrics_known_values():
    m = compute_metrics([1.0, 2.0, 3.0])
    assert m["n"] == 3
    assert m["bias"] == pytest.approx(2.0)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert m["max_abs"] == pytest.approx(3.0)


def test_compute_metrics_handles_nan():
    m = compute_metrics([np.nan, 1.0, -1.0])
    assert m["n"] == 2
    assert m["bias"] == pytest.approx(0.0)


def test_compute_metrics_empty_returns_nan():
    m = compute_metrics([])
    assert m["n"] == 0
    assert np.isnan(m["bias"])


# ---------------------------------------------------------------
# 6. 对称相对差异
# ---------------------------------------------------------------

def test_symmetric_relative_difference():
    assert symmetric_relative_difference(10.0, 12.0) == pytest.approx(
        2.0 / 11.0 * 100.0
    )
    assert symmetric_relative_difference(0.950490, 0.952093) == pytest.approx(
        abs(0.950490 - 0.952093)
        / ((0.950490 + 0.952093) / 2.0) * 100.0
    )


def test_symmetric_relative_difference_near_zero_denominator():
    assert np.isnan(symmetric_relative_difference(0.0, 0.0))
    assert np.isnan(symmetric_relative_difference(0.0, 1e-14))


# ---------------------------------------------------------------
# 辅助: 合成 Dataset 2 稳态点 (13 段, 与真实实验结构一致)
# ---------------------------------------------------------------

def _synthetic_steady_df():
    seq = DATASET2_SETPOINTS
    temps = [30.05, 39.6, 49.2, 58.75, 68.275, 77.85, 87.425,
             77.9, 68.35, 58.9, 49.45, 39.95, 30.5]
    starts = [0, 42, 121, 198, 260, 372, 479, 571, 649, 735, 806, 868, 951]
    ends = starts[1:] + [1029]
    rows = []
    for i, (sp, temp) in enumerate(zip(seq, temps)):
        rows.append({
            "signal": "Mean",
            "segment": i,
            "setpoint": sp,
            "steady_temp": temp,
            "steady_slope_C_per_s": 0.0,
            "accepted": True,
            "start_index": starts[i],
            "end_index": ends[i],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 8. 加热/冷却重复稳态点保留
# ---------------------------------------------------------------

def test_duplicate_setpoints_preserved_heating_and_cooling():
    df = _synthetic_steady_df()
    eq = CalibrationEquation(a=0.950490, b=1.811586, tau_eff=7.3072,
                             r2_steady=None, rmse_dynamic=None, source="s")
    v = build_validation_frame(df, eq, DATASET1_RANGE)

    primary = v[v["classification"] == "interpolation"]
    assert len(primary) == 11                       # 40-90 °C 全部 11 段
    assert len(primary[primary["direction"] == "heating"]) == 6
    assert len(primary[primary["direction"] == "cooling"]) == 5

    # 重复设定点 (40-80 °C) 加热与冷却观测分别保留
    for sp in [40.0, 50.0, 60.0, 70.0, 80.0]:
        rows = primary[primary["T_set"] == sp]
        assert len(rows) == 2
        assert set(rows["direction"]) == {"heating", "cooling"}

    # 90 °C 只有加热观测, 30 °C 只有外推
    assert len(primary[primary["T_set"] == 90.0]) == 1
    assert len(v[v["T_set"] == 30.0]) == 2


# ---------------------------------------------------------------
# 9. 30 °C 从主插值指标中排除
# ---------------------------------------------------------------

def test_30C_excluded_from_primary_metrics_when_range_35_95():
    df = _synthetic_steady_df()
    eq = CalibrationEquation(a=0.950490, b=1.811586, tau_eff=7.3072,
                             r2_steady=None, rmse_dynamic=None, source="s")
    v = build_validation_frame(df, eq, DATASET1_RANGE)

    primary = v[v["classification"] == "interpolation"]
    extrap = v[v["classification"] == "extrapolation"]

    assert set(extrap["T_set"]) == {30.0}
    assert (primary["T_set"] >= DATASET1_RANGE[0]).all()
    assert (primary["T_set"] <= DATASET1_RANGE[1]).all()

    errs = primary["error"]
    assert len(errs) == 11

    # 手工对照: 用全部 13 段 (含 30 °C) 算出的 RMSE 必须与仅主段的 RMSE 不同
    all_errs = v["error"]
    assert compute_metrics(all_errs)["rmse"] != pytest.approx(
        compute_metrics(errs)["rmse"]
    )


def test_build_validation_frame_error_sign_convention():
    df = _synthetic_steady_df()
    eq = CalibrationEquation(a=0.950490, b=1.811586, tau_eff=7.3072,
                             r2_steady=None, rmse_dynamic=None, source="s")
    v = build_validation_frame(df, eq, DATASET1_RANGE)
    row = v.iloc[0]  # seg 0: 30 °C
    assert row["error"] == pytest.approx(
        row["T_measured_B"] - row["T_predicted_A"]
    )
    assert row["absolute_error"] == pytest.approx(abs(row["error"]))


# ---------------------------------------------------------------
# 7. 合成重叠范围内的方程比较
# ---------------------------------------------------------------

def test_equation_grid_comparison_synthetic():
    eqA = CalibrationEquation(a=1.0, b=0.0, tau_eff=1.0,
                              r2_steady=None, rmse_dynamic=None, source="A")
    eqB = CalibrationEquation(a=1.0, b=2.0, tau_eff=1.0,
                              r2_steady=None, rmse_dynamic=None, source="B")

    cmp = equation_grid_comparison(eqA, eqB, overlap=(0.0, 10.0), n=1001)
    assert cmp["mean_signed"] == pytest.approx(-2.0, abs=1e-9)
    assert cmp["mae"] == pytest.approx(2.0, abs=1e-9)
    assert cmp["rmse"] == pytest.approx(2.0, abs=1e-9)
    assert cmp["max_abs"] == pytest.approx(2.0, abs=1e-9)
    assert 0.0 <= cmp["max_abs_setpoint"] <= 10.0


def test_equation_grid_comparison_max_difference_at_edge():
    # 两直线差随温度单调变化 -> 最大 |差| 出现在范围端点
    eqA = CalibrationEquation(a=1.0, b=0.0, tau_eff=1.0,
                              r2_steady=None, rmse_dynamic=None, source="A")
    eqB = CalibrationEquation(a=0.99, b=0.5, tau_eff=1.0,
                              r2_steady=None, rmse_dynamic=None, source="B")
    cmp = equation_grid_comparison(eqA, eqB, overlap=(35.0, 90.0), n=1001)
    # 端点处 |diff| = |0.01*35 - 0.5| = 0.15; |0.01*90 - 0.5| = 0.40
    assert cmp["max_abs"] == pytest.approx(0.40, abs=1e-6)
    assert cmp["max_abs_setpoint"] == pytest.approx(90.0, abs=1e-3)


# ---------------------------------------------------------------
# 补充: 数据完整性验证
# ---------------------------------------------------------------

def test_verify_steady_points_accepts_clean_dataset():
    df = _synthetic_steady_df()
    assert verify_steady_points(df) == []


def test_verify_steady_points_detects_missing_stage():
    df = _synthetic_steady_df()
    df = df[df["segment"] != 6].reset_index(drop=True)   # 删掉 90 °C 段
    problems = verify_steady_points(df)
    assert any("13" in p or "12" in p for p in problems)
    assert any("序列" in p or "sequence" in p.lower() for p in problems)


def test_verify_steady_points_detects_unaccepted_point():
    df = _synthetic_steady_df()
    df.loc[df["segment"] == 12, "accepted"] = False
    problems = verify_steady_points(df)
    assert any("accepted" in p.lower() for p in problems)


def test_verify_steady_points_detects_discontinuity():
    df = _synthetic_steady_df()
    df.loc[df["segment"] == 5, "end_index"] = 500   # 制造不连续
    problems = verify_steady_points(df)
    assert any("连续" in p for p in problems)


# ---------------------------------------------------------------
# 加载 CSV 的函数
# ---------------------------------------------------------------

def test_load_mean_steady_points(tmp_path):
    p = tmp_path / "steady_points.csv"
    _synthetic_steady_df().to_csv(p, index=False)
    df = load_mean_steady_points(p)
    assert len(df) == 13
    assert (df["signal"] == "Mean").all()
    assert list(df["segment"]) == list(range(13))


# ---------------------------------------------------------------
# 完整动态模型: 合成 profile 构建
# ---------------------------------------------------------------

def test_build_synthetic_profile_length_and_sequence():
    time, sps = build_synthetic_profile(
        setpoints=[30.0, 40.0, 50.0], hold_s=60.0, dt=1.0
    )
    assert len(time) == 180
    assert len(sps) == 180
    np.testing.assert_allclose(time, np.arange(180.0))
    # 每 60 个样本为一段
    np.testing.assert_allclose(sps[:60], np.full(60, 30.0))
    np.testing.assert_allclose(sps[60:120], np.full(60, 40.0))
    np.testing.assert_allclose(sps[120:], np.full(60, 50.0))


def test_default_synthetic_profile_is_5_stages_60s():
    time, sps = build_synthetic_profile()
    assert len(time) == len(SYNTHETIC_SETPOINTS) * int(SYNTHETIC_HOLD_S / SYNTHETIC_DT)
    assert list(sps[::60]) == list(SYNTHETIC_SETPOINTS)


# ---------------------------------------------------------------
# 完整动态模型: 一阶递推预测
# ---------------------------------------------------------------

def test_predict_dynamic_profile_converges_to_equilibrium():
    eq = CalibrationEquation(a=1.0, b=0.0, tau_eff=2.0,
                             r2_steady=None, rmse_dynamic=None, source="s")
    # 30 恒定 60 s: 从 T_eq(30)=30 开始, 应保持 30
    time, sps = build_synthetic_profile([30.0], hold_s=60.0, dt=1.0)
    traj, teq = predict_dynamic_profile(eq, sps, dt=1.0)
    np.testing.assert_allclose(traj, 30.0, atol=1e-12)
    np.testing.assert_allclose(teq, 30.0, atol=1e-12)


def test_predict_dynamic_profile_exponential_response():
    # T_eq 阶跃 0 -> 10, tau=1, dt=1:
    # alpha = e^-1
    # T_s[0]=0
    # T_s[1]=10+(0-10)*alpha        = 10*(1-alpha)
    # T_s[2]=10+(T_s[1]-10)*alpha   = 10 - 10*alpha^2
    eq = CalibrationEquation(a=1.0, b=0.0, tau_eff=1.0,
                             r2_steady=None, rmse_dynamic=None, source="s")
    sps = np.concatenate([np.zeros(1), np.full(5, 10.0)])
    traj, _ = predict_dynamic_profile(eq, sps, dt=1.0,
                                      initial_temperature=0.0)
    alpha = np.exp(-1.0)
    assert traj[0] == pytest.approx(0.0)
    assert traj[1] == pytest.approx(10.0 * (1 - alpha), abs=1e-9)
    assert traj[2] == pytest.approx(10.0 - 10.0 * alpha ** 2, abs=1e-9)


def test_predict_dynamic_profile_default_initial_is_equilibrium():
    eq = CalibrationEquation(a=0.95, b=1.8, tau_eff=2.0,
                             r2_steady=None, rmse_dynamic=None, source="s")
    time, sps = build_synthetic_profile([30.0], hold_s=10.0, dt=1.0)
    traj, teq = predict_dynamic_profile(eq, sps, dt=1.0)
    assert traj[0] == pytest.approx(teq[0], abs=1e-9)
    assert traj[0] == pytest.approx(0.95 * 30.0 + 1.8, abs=1e-9)


# ---------------------------------------------------------------
# 完整动态模型: 轨迹对比指标
# ---------------------------------------------------------------

def test_dynamic_trajectory_comparison_same_tau_small_steady_offset():
    # 两模型 tau 相同但 b 不同 -> 全程恒定偏移 (稳态差异)
    eqA = CalibrationEquation(a=1.0, b=0.0, tau_eff=5.0,
                              r2_steady=None, rmse_dynamic=None, source="A")
    eqB = CalibrationEquation(a=1.0, b=0.5, tau_eff=5.0,
                              r2_steady=None, rmse_dynamic=None, source="B")
    time, sps = build_synthetic_profile()
    cmp = dynamic_trajectory_comparison(eqA, eqB, sps, dt=1.0)
    # A - B = -0.5 恒定
    assert cmp["mean_signed"] == pytest.approx(-0.5, abs=1e-9)
    assert cmp["mae"] == pytest.approx(0.5, abs=1e-9)
    assert cmp["rmse"] == pytest.approx(0.5, abs=1e-9)
    assert cmp["max_abs"] == pytest.approx(0.5, abs=1e-9)


def test_dynamic_trajectory_comparison_different_tau_max_during_transient():
    eqA = CalibrationEquation(a=1.0, b=0.0, tau_eff=10.0,
                              r2_steady=None, rmse_dynamic=None, source="A")
    eqB = CalibrationEquation(a=1.0, b=0.0, tau_eff=1.0,
                              r2_steady=None, rmse_dynamic=None, source="B")
    time, sps = build_synthetic_profile()
    cmp = dynamic_trajectory_comparison(eqA, eqB, sps, dt=1.0)
    # 两模型 a, b 相同 -> 平台稳态处 diff -> 0; 差异只出现在瞬态窗口内
    # 加热时 A(tau=10) 落后于 B(tau=1) => A-B<0; 冷却时 A 更高 => A-B>0
    # mean_signed 的正负取决于升降贡献, 不做方向断言; 只验证 max|diff| 存在
    # 且出现在某个瞬态时刻 (不在 profile 首尾恒稳态处)。
    assert cmp["max_abs"] > 0.1
    assert 0.0 <= cmp["max_abs_time_s"] < len(sps)


# ---------------------------------------------------------------
# 完整动态模型: 每段切换最大差异
# ---------------------------------------------------------------

def test_per_transition_max_differences_finds_four_transitions():
    eqA = CalibrationEquation(a=1.0, b=0.0, tau_eff=10.0,
                              r2_steady=None, rmse_dynamic=None, source="A")
    eqB = CalibrationEquation(a=1.0, b=0.0, tau_eff=1.0,
                              r2_steady=None, rmse_dynamic=None, source="B")
    time, sps = build_synthetic_profile()   # 5 段 -> 4 次切换
    trajA, _ = predict_dynamic_profile(eqA, sps, dt=1.0)
    trajB, _ = predict_dynamic_profile(eqB, sps, dt=1.0)
    per = per_transition_max_differences(trajA, trajB, sps, dt=1.0, hold_s=60.0)
    assert len(per) == 4
    assert per["transition"].tolist() == [
        "30 -> 40", "40 -> 50", "50 -> 40", "40 -> 30"
    ]
    assert per["direction"].tolist() == [
        "heating", "heating", "cooling", "cooling"
    ]
    # 加热切换: B(tau=1) 先到, A(tau=10) 落后 -> A-B<0; |diff| 出现在窗口内
    for _, r in per.iterrows():
        assert r["max_abs_difference"] > 0.0
        assert 0.0 <= r["time_of_max_s"] <= 300.0


# ---------------------------------------------------------------
# 合成动态对比中稳态差异远小于瞬态差异 (科学期望)
# ---------------------------------------------------------------

def test_transient_difference_exceeds_steady_difference():
    # 真实两方程: 稳态最大差异 ~0.05 °C, tau 差异大 -> 瞬态差异应更大
    eqA = CalibrationEquation(a=0.950490, b=1.811586, tau_eff=7.3072,
                              r2_steady=None, rmse_dynamic=None, source="A")
    eqB = CalibrationEquation(a=0.952093, b=1.702326, tau_eff=4.4520,
                              r2_steady=None, rmse_dynamic=None, source="B")
    time, sps = build_synthetic_profile()
    cmp = dynamic_trajectory_comparison(eqA, eqB, sps, dt=1.0)

    # 稳态偏移 (方程差异, 见 section 6): 35-90 °C 内 max ~0.053 °C
    steady_max = 0.0532
    assert cmp["max_abs"] > steady_max
    # 而全程 MAE/RMSE 应显著大于稳态方程差异 MAE (0.023 °C)
    assert cmp["rmse"] > 0.023
