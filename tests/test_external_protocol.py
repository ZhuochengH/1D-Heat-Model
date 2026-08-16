"""
外部实测协议 -> 动态表面模型 -> FDM 管道测试 (sample and heater T in one plot.py)。

覆盖 (任务 15 的 18 项):
1.  --protocol-mode 必须显式 (无静默回退);
2.  excel 模式无 --protocol-xlsx 报错;
3.  builtin 模式仅显式选择才生效;
4.  Excel 精确列 "Zone 1   Avg (°C)" 加载为 T_internal (含空格折叠容错);
5.  协议截断保留恰好前 300 有效行;
6.  截断保留对应时间戳与温度;
7.  校准文件是 a/b 的默认来源;
8.  两个 CLI override 替换文件值;
9.  只提供 calibration-a / calibration-b 之一报错;
10. tau_eff 动态模型必需;
11. tau_eff <= 0 / NaN / inf 报错;
12. 恒定 T_internal 从平衡初值出发 -> T_surface 保持恒定;
13. 温度阶跃 -> 一阶响应方向正确;
14. 更小 tau -> 更快响应;
15. T_surface 收敛到 T_surface_eq;
16. 数组长度对齐;
17. FDM Section 5 接收 T_surface_dynamic (而非 T_internal / T_surface_eq);
18. 改变 tau 改变动态边界但不修改原始 T_internal。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "sample and heater T in one plot.py"

# 真实校准文件 (任务 4: 默认来源)
REAL_CALIBRATION_FILE = (
    PROJECT_ROOT / "calibration_output" / "final_calibration_equation.txt"
)


def load_module():
    """加载主脚本模块 (文件名含空格, 用 importlib 按路径加载)。"""
    spec = importlib.util.spec_from_file_location("sample_heater_fdm", SCRIPT_PATH)
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
# 辅助: 合成协议 Excel
# ---------------------------------------------------------------

def _make_protocol_xlsx(path, column="Zone 1   Avg (°C)", values=None,
                        with_time=False):
    if values is None:
        values = [30.0, 40.0, 50.0, 40.0, 30.0]
    df = pd.DataFrame({column: values})
    if with_time:
        df.insert(0, "RECTime", pd.to_datetime("2026-08-16 10:00:00") +
                  pd.to_timedelta(np.arange(len(values)), unit="s"))
    df.to_excel(path, index=False)
    return path


# ===============================================================
# 1+3. 显式协议模式 — 无静默回退
# ===============================================================

def test_protocol_mode_is_required(m):
    """不带 --protocol-mode 必须报错 (argparse required)。"""
    with pytest.raises(SystemExit):
        m.parse_args(["--tau-eff", "7.3"])


def test_protocol_mode_must_be_valid_choice(m):
    with pytest.raises(SystemExit):
        m.parse_args(["--protocol-mode", "auto"])


def test_resolve_protocol_mode_excel_without_xlsx_raises(m):
    """excel 模式无 --protocol-xlsx -> 明确报错, 不静默回退 builtin。"""
    with pytest.raises(ValueError, match="--protocol-xlsx"):
        m.resolve_protocol_mode("excel", None)


def test_resolve_protocol_mode_excel_with_xlsx_ok(m):
    assert m.resolve_protocol_mode("excel", "some.xlsx") == "excel"


def test_resolve_protocol_mode_builtin_explicit_only(m):
    """builtin 模式只有显式选择才生效; 无参数时不自动激活。"""
    assert m.resolve_protocol_mode("builtin", None) == "builtin"
    with pytest.raises(ValueError):
        m.resolve_protocol_mode("", None)


# ===============================================================
# 4. Excel 列加载为 T_internal (含空格折叠容错)
# ===============================================================

def test_load_protocol_exact_column_three_spaces(m, tmp_path):
    """任务指定列名 'Zone 1   Avg (°C)' (三空格) 直接匹配。"""
    p = _make_protocol_xlsx(tmp_path / "p.xlsx", column="Zone 1   Avg (°C)")
    t, T_int = m.load_protocol_from_excel(p, column="Zone 1   Avg (°C)")
    np.testing.assert_allclose(T_int, [30, 40, 50, 40, 30])
    np.testing.assert_allclose(t, [0, 1, 2, 3, 4])


def test_load_protocol_column_single_space_fold_matches(m, tmp_path):
    """真实文件列 'Zone 1 Avg (°C)' (单空格) 经空格折叠容错匹配默认三空格列名。"""
    p = _make_protocol_xlsx(tmp_path / "p.xlsx", column="Zone 1 Avg (°C)")
    t, T_int = m.load_protocol_from_excel(p, column="Zone 1   Avg (°C)")
    np.testing.assert_allclose(T_int, [30, 40, 50, 40, 30])


def test_load_protocol_missing_column_raises(m, tmp_path):
    p = _make_protocol_xlsx(tmp_path / "p.xlsx")
    with pytest.raises(KeyError):
        m.load_protocol_from_excel(p, column="No Such Column")


def test_load_protocol_time_column(m, tmp_path):
    p = _make_protocol_xlsx(tmp_path / "p.xlsx", with_time=True)
    t, _ = m.load_protocol_from_excel(p, column="Zone 1   Avg (°C)",
                                      time_col="RECTime")
    np.testing.assert_allclose(t, [0, 1, 2, 3, 4], atol=1e-6)


def test_load_protocol_forward_fills_nan(m, tmp_path):
    p = _make_protocol_xlsx(tmp_path / "p.xlsx",
                            values=[30.0, np.nan, 50.0])
    _, T_int = m.load_protocol_from_excel(p, column="Zone 1   Avg (°C)")
    assert T_int[1] == pytest.approx(30.0)


# ===============================================================
# 5+6. 截断保留前 max_rows 有效行及对应时间戳/温度
# ===============================================================

def test_truncate_keeps_exactly_first_300_rows(m):
    rng = np.random.default_rng(0)
    t = np.arange(400, dtype=float)
    T = 25.0 + rng.normal(0, 0.1, 400)
    t_cut, T_cut, n_used = m.truncate_protocol(t, T, max_rows=300)
    assert n_used == 300
    assert len(t_cut) == 300
    assert len(T_cut) == 300
    np.testing.assert_allclose(t_cut, t[:300])
    np.testing.assert_allclose(T_cut, T[:300])


def test_truncate_preserves_timestamp_temperature_pairs(m):
    t = np.array([0.0, 1.1, 2.2, 3.3, 4.4])     # 非均匀时间戳
    T = np.array([30.0, 31.5, 33.0, 34.5, 36.0])
    t_cut, T_cut, n_used = m.truncate_protocol(t, T, max_rows=3)
    assert n_used == 3
    np.testing.assert_allclose(t_cut, [0.0, 1.1, 2.2])
    np.testing.assert_allclose(T_cut, [30.0, 31.5, 33.0])


def test_truncate_none_keeps_all(m):
    t = np.arange(10.0)
    T = np.arange(10.0)
    t_cut, T_cut, n_used = m.truncate_protocol(t, T, None)
    assert n_used == 10
    assert len(t_cut) == 10


def test_truncate_maxrows_beyond_length_keeps_all(m):
    t = np.arange(5.0)
    T = np.arange(5.0)
    t_cut, T_cut, n_used = m.truncate_protocol(t, T, max_rows=300)
    assert n_used == 5
    assert len(t_cut) == 5


# ===============================================================
# 7+8+9. 校准参数来源
# ===============================================================

def test_calibration_file_is_default_source(m):
    """校准文件是 a/b 的默认来源 (任务 4 指定文件)。"""
    a, b, src = m.resolve_calibration_parameters(str(REAL_CALIBRATION_FILE))
    assert a == pytest.approx(0.950490, abs=1e-6)
    assert b == pytest.approx(1.811586, abs=1e-6)
    assert "file:" in src
    assert "calibration_output" in src


def test_load_surface_calibration_parses_real_file(m):
    a, b = m.load_surface_calibration(REAL_CALIBRATION_FILE)
    assert a == pytest.approx(0.950490, abs=1e-6)
    assert b == pytest.approx(1.811586, abs=1e-6)


def test_both_cli_overrides_replace_file_values(m):
    a, b, src = m.resolve_calibration_parameters(
        str(REAL_CALIBRATION_FILE), a_override=1.0, b_override=0.5
    )
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(0.5)
    assert src == "CLI override"


def test_single_override_raises(m):
    with pytest.raises(ValueError, match="同时提供"):
        m.resolve_calibration_parameters(str(REAL_CALIBRATION_FILE),
                                         a_override=1.0)
    with pytest.raises(ValueError, match="同时提供"):
        m.resolve_calibration_parameters(str(REAL_CALIBRATION_FILE),
                                         b_override=1.0)


def test_calibration_file_missing_raises(m, tmp_path):
    with pytest.raises(FileNotFoundError):
        m.load_surface_calibration(tmp_path / "no_such.txt")


# ===============================================================
# 10+11. tau_eff 校验
# ===============================================================

def test_tau_eff_required(m):
    with pytest.raises(ValueError, match="tau_eff"):
        m.validate_tau_eff(None)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf, "abc"])
def test_tau_eff_invalid_values_raise(m, bad):
    with pytest.raises(ValueError):
        m.validate_tau_eff(bad)


def test_tau_eff_valid_positive(m):
    assert m.validate_tau_eff("7.3072") == pytest.approx(7.3072)


# ===============================================================
# 12-15. 动态表面模型物理行为
# ===============================================================

def test_constant_t_internal_keeps_surface_constant(m):
    """恒定 T_internal 从平衡初值出发 -> T_surface 恒定等于 T_surface_eq。"""
    T_eq = np.full(100, 30.0)          # 恒定 30 °C 平衡目标
    t = np.arange(100, dtype=float) * 1.0
    T_surf = m.apply_dynamic_surface_model(T_eq, t, tau_eff=5.0)
    np.testing.assert_allclose(T_surf, 30.0, atol=1e-12)
    assert T_surf[0] == pytest.approx(T_eq[0])


def test_step_response_direction(m):
    """温度阶跃 -> 一阶响应方向正确 (升温时表面温度向新平衡逼近)。"""
    T_eq = np.concatenate([np.full(10, 30.0), np.full(90, 50.0)])
    t = np.arange(100, dtype=float) * 1.0
    T_surf = m.apply_dynamic_surface_model(T_eq, t, tau_eff=5.0)
    assert T_surf[0] == pytest.approx(30.0)
    assert T_surf[10] > 30.0                     # 阶跃后开始升温
    assert T_surf[10] < 50.0                     # 尚未到达新平衡
    assert T_surf[99] > T_surf[10]               # 持续收敛
    assert T_surf[99] < 50.0                     # 一阶响应不过冲


def test_smaller_tau_faster_response(m):
    """更小 tau -> 更快响应 (同输入, 更早到达平衡)。"""
    T_eq = np.concatenate([np.full(10, 30.0), np.full(90, 50.0)])
    t = np.arange(100, dtype=float) * 1.0
    T_fast = m.apply_dynamic_surface_model(T_eq, t, tau_eff=1.0)
    T_slow = m.apply_dynamic_surface_model(T_eq, t, tau_eff=10.0)
    # 在阶跃后的同一时刻, 快 tau 应比慢 tau 更接近 50
    for i in range(20, 100):
        assert T_fast[i] > T_slow[i]
    assert abs(T_fast[99] - 50.0) < abs(T_slow[99] - 50.0)


def test_surface_converges_to_equilibrium(m):
    """T_surface 收敛到 T_surface_eq (长时间后)。"""
    T_eq = np.concatenate([np.full(5, 30.0), np.full(2000, 50.0)])
    t = np.arange(len(T_eq), dtype=float) * 1.0
    T_surf = m.apply_dynamic_surface_model(T_eq, t, tau_eff=7.3072)
    assert abs(T_surf[-1] - 50.0) < 1e-6


# ===============================================================
# 16. 数组长度对齐
# ===============================================================

def test_prepare_fdm_boundary_array_alignment(m):
    t_proto = np.arange(11, dtype=float)
    T_int = 25.0 + 2.0 * np.sin(t_proto)
    time_fdm = np.linspace(0, 10, 1001)
    Ti, Teq, Ts = m.prepare_fdm_boundary(T_int, t_proto, 0.95, 1.8,
                                          tau_eff=7.3, time_fdm=time_fdm)
    assert len(time_fdm) == len(Ti) == len(Teq) == len(Ts)
    # 插值端点一致性
    assert Ti[0] == pytest.approx(T_int[0])
    assert Ti[-1] == pytest.approx(T_int[-1])


# ===============================================================
# 17. FDM 边界变量 = T_surface_dynamic (非 T_internal / T_surface_eq)
# ===============================================================

def test_fdm_boundary_is_dynamic_surface(m):
    """
    验证 prepare_fdm_boundary 返回的边界数组是动态模型输出:
      - T_surface_fdm 与 T_internal_fdm 不同 (瞬态);
      - T_surface_fdm 与 T_surface_eq_fdm 不同 (瞬态);
      - 动态模型基于 T_surface_eq 且被 tau 平滑。
    """
    # 10 s 30 °C 后阶跃到 80 °C 并保持 490 s (≈67 tau, 足够完全收敛)
    t_proto = np.arange(501, dtype=float)
    T_int = np.concatenate([np.full(10, 30.0), np.full(491, 80.0)])
    time_fdm = np.linspace(0, 500, 50001)
    Ti, Teq, Ts = m.prepare_fdm_boundary(T_int, t_proto, 0.95, 1.8,
                                          tau_eff=7.3, time_fdm=time_fdm)

    # 阶跃后: Teq 立即跳到新值, Ts 平滑逼近 -> 两者在瞬态明显不同
    step_idx = 1000  # t=10 s (0.01 s 间隔)
    assert abs(Teq[step_idx + 100] - Teq[step_idx]) < 1e-9
    assert abs(Ts[step_idx + 100] - Teq[step_idx]) > 1.0
    # Ti 是插值后的 T_internal (阶跃后 = 80.0); Teq 是校准后平衡目标
    assert Ti[step_idx] == pytest.approx(80.0)
    assert Teq[step_idx] == pytest.approx(0.95 * 80.0 + 1.8)
    # Ts 在瞬态介于初值与目标之间 (不过冲)
    assert Ts[step_idx] < Teq[step_idx] - 1.0
    # 长时间后完全收敛到 Teq
    assert abs(Ts[-1] - Teq[-1]) < 1e-3


def test_main_uses_dynamic_surface_as_boundary(m):
    """源码级验证: FDM 主循环边界使用 T_surface_fdm (而非 Ti/Teq/理想)。"""
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    # Section 5 的边界赋值必须用动态表面温度
    assert "T[0]    = T_surface_fdm[n]" in src
    # 边界数组由 prepare_fdm_boundary 返回
    assert "T_internal_fdm, T_surface_eq_fdm, T_surface_fdm = prepare_fdm_boundary(" in src
    # FDM 内部节点更新保持原样 (未修改求解器)
    assert "T[1:-1] = c_c * T[1:-1]" in src


# ===============================================================
# 18. 改变 tau 改变动态边界但不修改 T_internal
# ===============================================================

def test_changing_tau_changes_surface_but_not_internal(m):
    t_proto = np.arange(31, dtype=float)
    T_int = np.concatenate([np.full(10, 30.0), np.full(21, 80.0)])
    time_fdm = np.linspace(0, 30, 3001)

    Ti1, _, Ts1 = m.prepare_fdm_boundary(T_int, t_proto, 0.95, 1.8,
                                          tau_eff=2.0, time_fdm=time_fdm)
    Ti2, _, Ts2 = m.prepare_fdm_boundary(T_int, t_proto, 0.95, 1.8,
                                          tau_eff=10.0, time_fdm=time_fdm)

    # T_internal 不变 (同一插值)
    np.testing.assert_allclose(Ti1, Ti2)
    # 动态边界不同
    assert not np.allclose(Ts1, Ts2)
    assert np.max(np.abs(Ts1 - Ts2)) > 0.5


# ===============================================================
# 完整数据流 (集成): Excel -> T_internal -> 校准 -> 动态 -> 边界
# ===============================================================

def test_end_to_end_pipeline(m, tmp_path):
    p = _make_protocol_xlsx(tmp_path / "p.xlsx", with_time=True)
    t, T_int = m.load_protocol_from_excel(
        p, column="Zone 1   Avg (°C)", time_col="RECTime"
    )
    a, b = m.load_surface_calibration(REAL_CALIBRATION_FILE)
    T_eq = m.apply_surface_calibration(T_int, a, b)
    T_surf = m.apply_dynamic_surface_model(T_eq, t, tau_eff=7.3072)
    assert len(t) == len(T_int) == len(T_eq) == len(T_surf)
    assert T_eq[0] == pytest.approx(a * 30.0 + b)
    assert T_surf[0] == pytest.approx(T_eq[0])


def test_cli_defaults(m):
    args = m.parse_args(["--protocol-mode", "excel",
                         "--protocol-xlsx", "dummy.xlsx"])
    assert args.protocol_col == "Zone 1   Avg (°C)"
    assert args.protocol_sheet == 0
    assert args.protocol_time_col is None
    assert args.max_protocol_rows is None
    assert args.calibration_a is None
    assert args.calibration_b is None
    assert args.tau_eff is None
    assert "calibration_output" in args.calibration_file
    assert "dynamic_first300" in args.output_dir
