"""
第一层验证：分析截止（analysis cutoff）。

科学问题
--------
在最后一段受控 35 °C 保温之后，Peltier 控制器关闭，
表面温度会被动冷却回室温。这段被动冷却数据必须被排除在
标定流程（分段 / 稳态提取 / 稳态回归 / 动态 tau 拟合 /
RMSE / 绘图 / 导出结果）之外，否则会污染最终标定。

这些测试只使用小型合成 NumPy 数组，不依赖真实 Excel 数据。
被测试的行为是 apply_analysis_cutoff（纯工具函数，不改动数学模型）。
"""

import numpy as np
import pytest

from peltier_surface_calibration_v2 import apply_analysis_cutoff


# ---------------------------------------------------------------
# 1. 无 cutoff 时保留全部样本
# ---------------------------------------------------------------
def test_no_cutoff_preserves_every_sample():
    n = 50
    time_s = np.arange(n, dtype=float)
    T1 = 35.0 + 0.1 * np.arange(n, dtype=float)
    T2 = 35.0 - 0.05 * np.arange(n, dtype=float)
    Tmean = (T1 + T2) / 2.0

    t_out, t1_out, t2_out, tm_out = apply_analysis_cutoff(time_s, T1, T2, Tmean)

    assert len(t_out) == n
    np.testing.assert_array_equal(t_out, time_s)
    np.testing.assert_array_equal(t1_out, T1)
    np.testing.assert_array_equal(t2_out, T2)
    np.testing.assert_array_equal(tm_out, Tmean)


# ---------------------------------------------------------------
# 2. 有效 cutoff 只保留 time_s <= analysis_end_s 的样本
# ---------------------------------------------------------------
def test_valid_cutoff_keeps_only_samples_within_limit():
    time_s = np.arange(10.0)          # 0..9
    T1 = np.linspace(30.0, 40.0, 10)

    t_out, t1_out = apply_analysis_cutoff(time_s, T1, analysis_end_s=6.0)

    assert len(t_out) == 7            # 样本 0..6
    assert np.all(t_out <= 6.0)
    np.testing.assert_array_equal(t_out, time_s[:7])
    np.testing.assert_array_equal(t1_out, T1[:7])


# ---------------------------------------------------------------
# 3. 截断后 time/T1/T2/Tmean 保持对齐
# ---------------------------------------------------------------
def test_arrays_remain_aligned_after_truncation():
    rng = np.random.default_rng(42)
    n = 120
    time_s = np.arange(n, dtype=float)
    T1 = 30.0 + rng.normal(0.0, 0.1, n)
    T2 = T1 + 0.5 + rng.normal(0.0, 0.05, n)
    Tmean = (T1 + T2) / 2.0

    t_out, t1_out, t2_out, tm_out = apply_analysis_cutoff(
        time_s, T1, T2, Tmean, analysis_end_s=77.5
    )

    assert len(t1_out) == len(t2_out) == len(tm_out) == len(t_out)
    assert len(t_out) == 78           # 样本 0..77
    np.testing.assert_allclose(tm_out, (t1_out + t2_out) / 2.0, atol=1e-12)
    np.testing.assert_allclose(t1_out, T1[: len(t1_out)])
    np.testing.assert_allclose(t2_out, T2[: len(t2_out)])
    assert np.all(np.diff(t_out) > 0)


# ---------------------------------------------------------------
# 4. 科学失败模式：被动冷却尾段不得残留进下游数组
#    受控 35 °C plateau -> 控制器关闭 -> 被动降到 32 °C
#    cutoff 位于受控 plateau 末尾，冷却数据必须被完全排除
# ---------------------------------------------------------------
def test_passive_cooling_tail_excluded_from_final_plateau():
    n_plateau = 80
    n_cool = 20

    time_s = np.arange(n_plateau + n_cool, dtype=float)   # 0..99
    plateau_temp = np.full(n_plateau, 35.0)               # 受控 35 °C 保温
    cool_temp = np.linspace(35.0, 32.0, n_cool)           # 被动冷却 35 -> 32
    Tmean = np.concatenate([plateau_temp, cool_temp])

    # cutoff 恰好位于受控 plateau 末尾（最后一个受控样本 t=79）
    t_out, tm_out = apply_analysis_cutoff(
        time_s, Tmean, analysis_end_s=79.0
    )

    assert len(t_out) == n_plateau
    assert t_out[-1] == 79.0
    # 被动冷却样本不得出现在传给下游的数组中
    np.testing.assert_array_equal(tm_out, plateau_temp)
    assert not np.any(tm_out < 35.0 - 1e-12)
    assert np.all(t_out <= 79.0)


# ---------------------------------------------------------------
# 5. 负数 / 非有限值 / 非数值 cutoff 必须抛出清晰异常
# ---------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_cutoff",
    [
        -5.0,
        np.nan,
        np.inf,
        -np.inf,
        "not-a-number",
    ],
)
def test_invalid_cutoff_raises_clear_exception(bad_cutoff):
    time_s = np.arange(10.0)
    T1 = np.arange(10.0)

    with pytest.raises(ValueError):
        apply_analysis_cutoff(time_s, T1, analysis_end_s=bad_cutoff)


# ---------------------------------------------------------------
# 6. cutoff 超出最后样本时保留全部可用样本
# ---------------------------------------------------------------
def test_cutoff_beyond_final_sample_preserves_all():
    time_s = np.arange(25.0)
    T1 = np.linspace(35.0, 60.0, 25)
    T2 = np.linspace(34.0, 59.0, 25)
    Tmean = (T1 + T2) / 2.0

    t_out, t1_out, t2_out, tm_out = apply_analysis_cutoff(
        time_s, T1, T2, Tmean, analysis_end_s=1e6
    )

    assert len(t_out) == 25
    np.testing.assert_array_equal(t_out, time_s)
    np.testing.assert_array_equal(t1_out, T1)
    np.testing.assert_array_equal(t2_out, T2)
    np.testing.assert_array_equal(tm_out, Tmean)


# ---------------------------------------------------------------
# 补充：cutoff 早于首个样本时（结果为空）也必须抛出清晰异常
# ---------------------------------------------------------------
def test_cutoff_before_first_sample_raises():
    time_s = np.arange(1.0, 6.0)      # 1..5，首个样本在 t=1
    T1 = np.arange(1.0, 6.0)

    with pytest.raises(ValueError):
        apply_analysis_cutoff(time_s, T1, analysis_end_s=0.5)


# ===============================================================
# 集成级回归：cutoff 必须在分段与模型拟合之前生效
# ===============================================================
def test_cutoff_applies_before_segmentation_in_pipeline(
    monkeypatch, tmp_path
):
    """真实标定管线的集成回归测试（最小化）。

    科学场景：
        受控保温 45 -> 55 -> 35 °C（每段 30 样本，共 90 个受控样本；
        含加热 45->55 与冷却 55->35 两个方向），
        随后控制器关闭，表面温度被动冷却 35 -> 32 °C（20 个样本）。
        cutoff 恰好位于最后受控 35 °C 保温的末尾（t=89）。

    保护的科学回归：
        被动冷却尾段绝不能进入分段 / 稳态提取。
        若开发者误将 apply_analysis_cutoff 的调用移到分段检测之后，
        传入 segments_from_set_column / extract_steady_points 的数组
        将包含冷却尾段（长度 110 而非 90），本测试的 spy 断言必然失败。
        若冷却尾段并入最后 35 °C 段，该段 end 会延伸到 110（而非 90），
        同样被断言捕获。
    """
    import sys

    import pandas as pd

    import peltier_surface_calibration_v2 as mod

    n_seg, n_cool, dt = 30, 20, 1.0
    setpoints = [45.0, 55.0, 35.0]      # 加热与冷却方向都有
    n_controlled = len(setpoints) * n_seg   # 90

    set_series = np.concatenate(
        [np.full(n_seg, sp) for sp in setpoints]
        + [np.full(n_cool, 35.0)]           # 冷却期 Set 仍报 35（控制器关闭）
    )
    temps = np.concatenate(
        [np.full(n_seg, sp) for sp in setpoints]
        + [np.linspace(35.0, 32.0, n_cool)]  # 被动冷却 35 -> 32
    )

    df = pd.DataFrame({
        "Set": set_series,
        "T1": temps,
        "T2": temps + 0.1,
    })
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: df)

    captured = {}
    orig_seg = mod.segments_from_set_column
    orig_extract = mod.extract_steady_points

    def spy_seg(set_values, tol=1e-9):
        arr = np.asarray(set_values, dtype=float)
        captured["seg_set_len"] = len(arr)
        segs = orig_seg(arr, tol=tol)
        captured["last_segment_end"] = segs[-1].end
        captured["last_segment_setpoint"] = segs[-1].setpoint
        return segs

    def spy_extract(values, segments, **kwargs):
        arr = np.asarray(values, dtype=float)
        captured["extract_values_len"] = len(arr)
        return orig_extract(arr, segments, **kwargs)

    monkeypatch.setattr(mod, "segments_from_set_column", spy_seg)
    monkeypatch.setattr(mod, "extract_steady_points", spy_extract)

    cutoff_s = n_controlled - 1             # t = 89，最后受控样本
    monkeypatch.setattr(sys, "argv", [
        "peltier_surface_calibration_v2.py",
        "synthetic.xlsx",
        "--dt", str(dt),
        "--analysis-end-s", str(cutoff_s),
        "--output-dir", str(tmp_path),
    ])

    mod.main()

    # 1) 进入分段函数的 Set 序列只有受控样本
    assert captured["seg_set_len"] == n_controlled
    # 2) 进入稳态提取的温度数组只有受控样本
    assert captured["extract_values_len"] == n_controlled
    # 3) 最后一段是受控 35 °C，end 恰好在受控段末尾
    assert captured["last_segment_setpoint"] == 35.0
    assert captured["last_segment_end"] == n_controlled


# ===============================================================
# 集成级回归（真实执行路径）：--set-col NONE 自动分段
# ===============================================================
def test_cutoff_precedes_auto_segmentation_when_set_col_none(
    monkeypatch, tmp_path
):
    """真实 --set-col NONE 执行路径的集成回归测试。

    真实标定命令使用 --set-col NONE，程序走
        segments_from_detected_changes() -> detect_change_points_from_temperature()
    自动变化点检测分支，而不是 Set 列分段分支。

    科学场景：
        受控保温 45 -> 55 -> 65 -> 55 -> 45 -> 35 °C
        （每段 30 样本，共 180 个受控样本，加热与冷却方向都有），
        随后控制器关闭，表面温度被动冷却 35 -> 32 °C（20 个样本）。
        cutoff 恰好位于最后受控 35 °C 保温的末尾（t=179）。

    证明要点：
        1) 被动冷却样本在自动变化点检测之前已被移除
           （detect_change_points_from_temperature 收到的数组长度 = 180）；
        2) 自动分段从不超过 cutoff（segments_from_detected_changes 收到
           的数组长度 = 180，最后一个变化点 < 180）；
        3) 最后受控 35 °C 段未被冷却尾段污染
           （最后一段 end = 180，setpoint = 35.0，段数 = 6）。

    若开发者误将 cutoff 调用移到自动分段之后，传入
    detect_change_points_from_temperature 的 Tmean 将包含冷却尾段
    （长度 200 而非 180），本测试断言必然失败。
    """
    import sys

    import pandas as pd

    import peltier_surface_calibration_v2 as mod

    n_seg, n_cool, dt = 30, 20, 1.0
    setpoints = [45.0, 55.0, 65.0, 55.0, 45.0, 35.0]
    n_controlled = len(setpoints) * n_seg      # 180

    temps = np.concatenate(
        [np.full(n_seg, sp) for sp in setpoints]
        + [np.linspace(35.0, 32.0, n_cool)]     # 被动冷却 35 -> 32
    )

    # 无 Set 列 —— 模拟真实数据的 --set-col NONE 情形
    df = pd.DataFrame({
        "T1": temps,
        "T2": temps + 0.1,
    })
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: df)

    captured = {}
    orig_detect = mod.detect_change_points_from_temperature
    orig_build = mod.segments_from_detected_changes

    def spy_detect(mean_temp, dt, change_threshold, smooth_window, min_gap_s):
        arr = np.asarray(mean_temp, dtype=float)
        captured["detect_len"] = len(arr)
        cps = orig_detect(
            arr,
            dt=dt,
            change_threshold=change_threshold,
            smooth_window=smooth_window,
            min_gap_s=min_gap_s,
        )
        captured["change_points"] = list(cps)
        return cps

    def spy_build(
        mean_temp,
        setpoint_sequence,
        dt,
        change_threshold,
        smooth_window,
        min_gap_s,
    ):
        arr = np.asarray(mean_temp, dtype=float)
        captured["build_len"] = len(arr)
        segs = orig_build(
            arr,
            setpoint_sequence=setpoint_sequence,
            dt=dt,
            change_threshold=change_threshold,
            smooth_window=smooth_window,
            min_gap_s=min_gap_s,
        )
        captured["n_segments"] = len(segs)
        captured["last_segment_end"] = segs[-1].end
        captured["last_segment_setpoint"] = segs[-1].setpoint
        return segs

    monkeypatch.setattr(mod, "detect_change_points_from_temperature", spy_detect)
    monkeypatch.setattr(mod, "segments_from_detected_changes", spy_build)

    cutoff_s = n_controlled - 1                 # t = 179，最后受控样本
    setpoints_arg = ",".join(str(v) for v in setpoints)
    monkeypatch.setattr(sys, "argv", [
        "peltier_surface_calibration_v2.py",
        "synthetic.xlsx",
        "--dt", str(dt),
        "--set-col", "NONE",
        "--setpoints", setpoints_arg,
        "--analysis-end-s", str(cutoff_s),
        "--output-dir", str(tmp_path),
    ])

    mod.main()

    # 1) 变化点检测只收到受控样本（无冷却尾段）
    assert captured["detect_len"] == n_controlled
    # 2) 自动分段只收到受控样本，且最后一个变化点不超过 cutoff
    assert captured["build_len"] == n_controlled
    assert max(captured["change_points"], default=-1) < n_controlled
    # 3) 最后受控段是 35 °C，end 恰在受控段末尾，未被冷却尾段并入
    assert captured["n_segments"] == len(setpoints)
    assert captured["last_segment_setpoint"] == 35.0
    assert captured["last_segment_end"] == n_controlled
