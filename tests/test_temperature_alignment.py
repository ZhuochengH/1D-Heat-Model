"""
内部/顶部温度对齐脚本测试 (align_internal_and_top_temperature.py) — v3 顶部参考对齐。

覆盖 (任务 18 的 12 项):
1.  顶部参考时间轴生成 0,1,2,... (从有效样本);
2.  内部使用 Time(s) 而非行索引;
3.  非均匀内部时间间隔可接受 (单调即可);
4.  线性插值数学正确 (t=[0.5,1.5], T=[10,20] -> T(1.0)=15);
5.  目标时间等于内部源时间时精确重现;
6.  内部首时间前不外推;
7.  内部末时间后不外推;
8.  排除点正确计数;
9.  对齐输出 T_internal 与 T_top 对应相同 time_s;
10. 不使用行索引对齐;
11. 无信号驱动时间移位;
12. 无 FDM / k / cp 拟合。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "thermal_model/utilities/align_internal_and_top_temperature.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "temperature_alignment", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = None


@pytest.fixture(scope="module")
def m():
    global mod
    if mod is None:
        mod = load_module()
    return mod


# ---------------------------------------------------------------
# 辅助: 合成数据文件
# ---------------------------------------------------------------

def _make_top_xlsx(path, n=350, values=None):
    """顶部文件: T Avg 有效样本 (RECTime 诊断可选)。"""
    if values is None:
        values = [27.8 + 0.2 * i for i in range(n)]
    df = pd.DataFrame({
        "RECTime": pd.to_datetime("2026-08-06 15:20:58") +
                   pd.to_timedelta(np.arange(n), unit="s"),
        "T Avg": values,
    })
    df.to_excel(path, index=False)
    return path


def _make_internal_xlsx(path, t_values, T_values):
    """内部文件: Time(s) + Zone 1 Avg (°C)。"""
    df = pd.DataFrame({
        "Time(s)": t_values,
        "Relative time(s)": [0.0] + list(np.diff(t_values)),
        "Zone 1 Avg (°C)": T_values,
    })
    df.to_excel(path, index=False)
    return path


def _default_internal(m, tmp_path, n=400, dt=1.08):
    """默认内部数据: Time(s) 非均匀 (median ~1.08 s)。"""
    t = np.cumsum([0.0] + [dt] * (n - 1))
    T = [29.0 + 0.5 * i for i in range(n)]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    return m.load_internal_series_time_col(p, column="Zone 1 Avg (°C)",
                                           sheet=0, time_col="Time(s)")


def _default_top(m, tmp_path, n=350):
    p = _make_top_xlsx(tmp_path / "top.xls", n=n)
    return m.load_top_series_top_reference(p, column="T Avg", sheet=0,
                                           diag_time_col="RECTime")


# ===============================================================
# 1. 顶部参考时间轴 0,1,2,...
# ===============================================================

def test_top_reference_time_axis(m, tmp_path):
    top = _default_top(m, tmp_path, n=300)
    assert top["n_valid"] == 300
    np.testing.assert_allclose(top["t_top"][:5], [0, 1, 2, 3, 4])
    assert top["t_top"][-1] == pytest.approx(299.0)
    assert top["median_dt"] == pytest.approx(1.0, abs=0.02)  # RECTime 诊断


def test_top_reference_first_sample_zero(m, tmp_path):
    top = _default_top(m, tmp_path, n=50)
    assert top["t_top"][0] == pytest.approx(0.0)


# ===============================================================
# 2. 内部使用 Time(s) 而非行索引
# ===============================================================

def test_internal_uses_time_col_not_index(m, tmp_path):
    """内部 Time(s) 从 0.5 开始 (非 0) -> t_internal 必须反映 Time(s)。"""
    t = [0.5, 1.5, 2.5, 3.5]
    T = [10.0, 20.0, 30.0, 40.0]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    internal = m.load_internal_series_time_col(
        p, column="Zone 1 Avg (°C)", sheet=0, time_col="Time(s)"
    )
    np.testing.assert_allclose(internal["t_internal"], [0.5, 1.5, 2.5, 3.5])
    assert internal["first_time"] == pytest.approx(0.5)


# ===============================================================
# 3. 非均匀内部时间间隔可接受
# ===============================================================

def test_non_uniform_internal_time_accepted(m, tmp_path):
    t = [0.0, 1.2, 2.4, 3.9, 5.5]   # 非均匀
    T = [10.0, 11.0, 12.0, 13.0, 14.0]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    internal = m.load_internal_series_time_col(
        p, column="Zone 1 Avg (°C)", sheet=0, time_col="Time(s)"
    )
    assert internal["n_valid"] == 5
    assert internal["median_dt"] == pytest.approx(np.median(np.diff(t)))


def test_non_monotonic_internal_time_raises(m, tmp_path):
    t = [0.0, 2.0, 1.5, 4.0]
    T = [10.0, 20.0, 15.0, 30.0]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    with pytest.raises(ValueError, match="严格递增"):
        m.load_internal_series_time_col(
            p, column="Zone 1 Avg (°C)", sheet=0, time_col="Time(s)"
        )


# ===============================================================
# 4. 线性插值数学正确
# ===============================================================

def test_linear_interpolation_math(m, tmp_path):
    """t=[0.5,1.5], T=[10,20] -> T(1.0)=15 (任务指定合成测试)。"""
    t = [0.5, 1.5]
    T = [10.0, 20.0]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    internal = m.load_internal_series_time_col(
        p, column="Zone 1 Avg (°C)", sheet=0, time_col="Time(s)"
    )
    # 顶部时间 = 0,1,2,...; 只有 t=1.0 落在内部 [0.5,1.5] 内
    top = m.load_top_series_top_reference(
        _make_top_xlsx(tmp_path / "top.xls", n=5),
        column="T Avg", sheet=0, diag_time_col=None,
    )
    aligned = m.align_to_top_reference(internal, top, max_top_rows=5)
    # 对齐点 = 1.0 s (0 和 2..4 在内部范围外)
    assert aligned["n_aligned"] == 1
    assert aligned["time_s"][0] == pytest.approx(1.0)
    assert aligned["T_internal"][0] == pytest.approx(15.0)   # 10 + (20-10)*0.5
    assert aligned["n_excluded_early"] == 1
    assert aligned["n_excluded_late"] == 3


# ===============================================================
# 5. 目标时间等于内部源时间时精确重现
# ===============================================================

def test_interpolation_exact_at_source_times(m, tmp_path):
    t = [0.0, 1.0, 2.0, 3.0]
    T = [10.0, 20.0, 30.0, 40.0]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    internal = m.load_internal_series_time_col(
        p, column="Zone 1 Avg (°C)", sheet=0, time_col="Time(s)"
    )
    top = m.load_top_series_top_reference(
        _make_top_xlsx(tmp_path / "top.xls", n=4),
        column="T Avg", sheet=0, diag_time_col=None,
    )
    aligned = m.align_to_top_reference(internal, top, max_top_rows=4)
    assert aligned["n_aligned"] == 4
    np.testing.assert_allclose(aligned["T_internal"], [10, 20, 30, 40])


# ===============================================================
# 6+7+8. 无外推 + 排除计数
# ===============================================================

def test_no_extrapolation_before_internal_start(m, tmp_path):
    """内部从 2.0 s 开始 -> 顶部 0,1 s 被排除。"""
    t = [2.0, 3.0, 4.0, 5.0]
    T = [30.0, 40.0, 50.0, 60.0]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    internal = m.load_internal_series_time_col(
        p, column="Zone 1 Avg (°C)", sheet=0, time_col="Time(s)"
    )
    top = m.load_top_series_top_reference(
        _make_top_xlsx(tmp_path / "top.xls", n=6),
        column="T Avg", sheet=0, diag_time_col=None,
    )
    aligned = m.align_to_top_reference(internal, top, max_top_rows=6)
    assert aligned["n_excluded_early"] == 2     # t=0,1
    assert aligned["n_excluded_late"] == 0
    np.testing.assert_allclose(aligned["time_s"], [2, 3, 4, 5])
    np.testing.assert_allclose(aligned["T_internal"], [30, 40, 50, 60])


def test_no_extrapolation_after_internal_end(m, tmp_path):
    """内部到 3.0 s 结束 -> 顶部 4,5 s 被排除。"""
    t = [0.0, 1.0, 2.0, 3.0]
    T = [10.0, 20.0, 30.0, 40.0]
    p = _make_internal_xlsx(tmp_path / "int.xlsx", t, T)
    internal = m.load_internal_series_time_col(
        p, column="Zone 1 Avg (°C)", sheet=0, time_col="Time(s)"
    )
    top = m.load_top_series_top_reference(
        _make_top_xlsx(tmp_path / "top.xls", n=6),
        column="T Avg", sheet=0, diag_time_col=None,
    )
    aligned = m.align_to_top_reference(internal, top, max_top_rows=6)
    assert aligned["n_excluded_late"] == 2      # t=4,5
    assert aligned["n_excluded_early"] == 0
    np.testing.assert_allclose(aligned["time_s"], [0, 1, 2, 3])


def test_excluded_counts_sum(m, tmp_path):
    internal = _default_internal(m, tmp_path, n=200)   # 0..~215 s
    top = _default_top(m, tmp_path, n=300)             # 0..299 s
    aligned = m.align_to_top_reference(internal, top, max_top_rows=300)
    assert aligned["n_excluded_late"] == 300 - aligned["n_aligned"]
    assert aligned["n_excluded_early"] == 0


# ===============================================================
# 9. 对齐输出对应相同 time_s
# ===============================================================

def test_aligned_columns_same_time(m, tmp_path):
    internal = _default_internal(m, tmp_path, n=400)
    top = _default_top(m, tmp_path, n=300)
    aligned = m.align_to_top_reference(internal, top, max_top_rows=300)
    assert len(aligned["time_s"]) == len(aligned["T_internal"])
    assert len(aligned["T_internal"]) == len(aligned["T_top"])
    assert len(aligned["delta"]) == len(aligned["T_top"])
    # 内部插值源点区间包围目标时间
    for i in range(0, len(aligned["time_s"]), 50):
        assert (aligned["internal_t_before"][i]
                <= aligned["time_s"][i]
                <= aligned["internal_t_after"][i])


# ===============================================================
# 10+11. 无行索引对齐 / 无信号移位
# ===============================================================

def test_no_row_index_alignment(m):
    assert not hasattr(m, "align_traces")
    assert not hasattr(m, "establish_physical_alignment")
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "row_index_alignment_used" in src       # 元数据标记 false
    assert "cross" not in src.lower() or "correlate" not in src.lower()


def test_no_signal_based_shift(m):
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "signal_based_time_shift_used" in src
    for bad in ["cross_correlate", "np.correlate", "fftshift", "find_peaks"]:
        assert bad not in src


# ===============================================================
# 12. 无 FDM / k / cp 拟合
# ===============================================================

def test_no_fdm_or_fitting_functions(m):
    assert not hasattr(m, "run_fdm")
    assert not hasattr(m, "optimize")
    assert not hasattr(m, "fit_k_eff")
    assert not hasattr(m, "fit_cp_eff")
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "scipy.optimize" not in src
    assert "grid_search" not in src


def test_delta_metrics_correct(m):
    met = m.compute_delta_metrics(np.array([5.0, 4.0, 3.0]))
    assert met["mean"] == pytest.approx(4.0)
    assert met["mean_abs"] == pytest.approx(4.0)
    assert met["min"] == pytest.approx(3.0)
    assert met["max"] == pytest.approx(5.0)
    assert met["max_abs"] == pytest.approx(5.0)


def test_describe_series(m):
    s = m.describe_series(np.array([30.0, 32.0, 28.0, 35.0]))
    assert s["min"] == pytest.approx(28.0)
    assert s["max"] == pytest.approx(35.0)
    assert s["initial"] == pytest.approx(30.0)
    assert s["final"] == pytest.approx(35.0)


def test_cli_defaults_72C_top_reference(m):
    args = m.parse_args([])
    assert "72" in args.internal_xlsx and "72" in args.top_xlsx
    assert "72C" in args.output_dir
    assert args.internal_time_col == "Time(s)"
    assert args.internal_col == "Zone 1 Avg (°C)"
    assert args.top_col == "T Avg"
    assert args.max_top_rows == 300
