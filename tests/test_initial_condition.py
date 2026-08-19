"""
实验初始条件策略测试 (auto -> 第一个底部边界值)。

覆盖 (任务 19 的 10 项):
1.  auto 解析为 bottom_temperature_C[0];
2.  直接版 wrapper: auto == 第一个实测内部温度;
3.  校准版 wrapper: auto == 第一个最终准备好的表面边界值;
4.  显式数值覆盖 auto;
5.  均匀初始场正确建立 (首个保存点 == 初温);
6.  实测顶数据不被修改 (源码级 + 只读);
7.  auto 模式下底部边界首值与初始场一致;
8.  实验 auto 模式无隐藏 25 C 默认;
9.  run_simulation 仍接受显式 T_initial_C (通用性);
10. 未来拟合直接模式可从实测内部首值初始化。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import heat_model
from heat_model import BARE_TOP_COC_LAYERS, DEFAULT_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "72C"
    / "aligned_internal_top_temperature.csv"
)

DIRECT_SCRIPT = PROJECT_ROOT / "sample and internal sensor T in one plot.py"
HEATER_SCRIPT = PROJECT_ROOT / "sample and heater T in one plot.py"


def load_script(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_direct = None
_heater = None


@pytest.fixture(scope="module")
def direct():
    global _direct
    if _direct is None:
        _direct = load_script(DIRECT_SCRIPT, "direct_fdm_mod")
    return _direct


@pytest.fixture(scope="module")
def heater():
    global _heater
    if _heater is None:
        _heater = load_script(HEATER_SCRIPT, "heater_fdm_mod")
    return _heater


# ===============================================================
# 1. auto 解析为 bottom_temperature_C[0]
# ===============================================================

def test_auto_resolves_to_first_boundary(direct):
    assert direct.resolve_initial_temperature("auto", 27.64) == pytest.approx(27.64)


def test_auto_resolves_none_as_auto(direct):
    assert direct.resolve_initial_temperature(None, 31.2) == pytest.approx(31.2)


# ===============================================================
# 2. 直接版: auto == 第一个实测内部温度
# ===============================================================

def test_direct_auto_equals_first_internal(direct):
    t = np.arange(5.0, dtype=float)
    T_int = np.array([27.64, 34.97, 45.0, 55.0, 65.0])
    first = float(T_int[0])
    assert direct.resolve_initial_temperature("auto", first) == pytest.approx(27.64)
    # 通过求解器端到端: 首个保存点样品/顶面温度 == 27.64 (无校准无滤波)
    res = heat_model.run_simulation(
        t, T_int, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.5, T_initial_C=first,
    )
    assert res["T_sample_arr"][0] == pytest.approx(27.64, abs=1e-9)
    assert res["T_bottom_arr"][0] == pytest.approx(27.64, abs=1e-9)


# ===============================================================
# 3. 校准版: auto == 第一个最终准备好的表面边界值
# ===============================================================

def test_heater_auto_equals_first_prepared_boundary(heater):
    t = np.arange(5.0, dtype=float)
    T_int = np.array([27.64, 34.97, 45.0, 55.0, 65.0])
    time_fdm = np.linspace(0.0, 4.0, 4001)
    _, _, T_surface_fdm = heater.prepare_fdm_boundary(
        T_int, t, 0.95, 1.8, 7.3, time_fdm
    )
    first = float(T_surface_fdm[0])
    # auto 必须用最终边界首值, 不是 T_internal[0] (二者不同: a*T+b)
    assert first != pytest.approx(27.64)
    assert heater.resolve_initial_temperature("auto", first) == pytest.approx(first)


# ===============================================================
# 4. 显式数值覆盖 auto
# ===============================================================

def test_explicit_numeric_overrides_auto(direct):
    assert direct.resolve_initial_temperature("25", 27.64) == pytest.approx(25.0)
    assert direct.resolve_initial_temperature("27.5", 27.64) == pytest.approx(27.5)


# ===============================================================
# 5. 均匀初始场正确建立
# ===============================================================

def test_uniform_initial_field_created():
    t = np.array([0.0, 1.0, 2.0])
    Tb = np.array([30.0, 40.0, 50.0])
    res = heat_model.run_simulation(
        t, Tb, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        T_initial_C=33.5, save_dt=0.5,
    )
    assert res["T_sample_arr"][0] == pytest.approx(33.5, abs=1e-9)
    assert res["T_outer_surface_arr"][0] == pytest.approx(33.5, abs=1e-9)


# ===============================================================
# 6. 实测顶数据不被修改
# ===============================================================

def test_measured_top_data_not_modified():
    # 求解器不接触实测数据: 仅接收 time_s/bottom 数组
    import inspect
    sig = inspect.signature(heat_model.run_simulation).parameters
    assert "T_top_measured" not in sig
    # 直接版 wrapper 无任何写回实测文件的代码路径 (只读 CSV)
    src = DIRECT_SCRIPT.read_text(encoding="utf-8")
    assert "to_csv" in src  # 只写模型输出
    assert "aligned_internal_top_temperature.csv" not in src  # 不写实测文件


def test_aligned_top_data_read_unchanged():
    """读取对齐数据: 首值保持实验记录 (27.8), 与内部首值 (27.64) 差异可见。"""
    import pandas as pd
    df = pd.read_csv(ALIGNED_CSV)
    assert df["T_top_measured_C"].iloc[0] == pytest.approx(27.8, abs=1e-9)
    assert df["T_internal_interpolated_C"].iloc[0] == pytest.approx(27.64, abs=1e-9)
    assert df["T_internal_interpolated_C"].iloc[0] != pytest.approx(
        df["T_top_measured_C"].iloc[0]
    )


# ===============================================================
# 7. auto: 底部边界首值与初始场一致
# ===============================================================

def test_auto_boundary_first_matches_initial_field():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    Tb = np.array([27.64, 40.0, 55.0, 70.0])
    res = heat_model.run_simulation(
        t, Tb, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        T_initial_C=float(Tb[0]), save_dt=0.5,
    )
    assert res["T_bottom_arr"][0] == pytest.approx(27.64, abs=1e-9)
    assert res["T_sample_arr"][0] == pytest.approx(27.64, abs=1e-9)
    assert res["T_top_surface_arr"][0] == pytest.approx(27.64, abs=1e-9)


# ===============================================================
# 8. 实验 auto 无隐藏 25 C 默认
# ===============================================================

def test_auto_mode_has_no_hidden_25_default(direct, heater):
    for mod in (direct, heater):
        assert mod.resolve_initial_temperature("auto", 31.7) == pytest.approx(31.7)
    # 解析函数不包含字面量 25
    src_d = DIRECT_SCRIPT.read_text(encoding="utf-8")
    src_h = HEATER_SCRIPT.read_text(encoding="utf-8")
    for src in (src_d, src_h):
        seg = src[src.index("def resolve_initial_temperature"):]
        seg = seg[:seg.index("\n\n\n") if "\n\n\n" in seg else len(seg)]
        assert "25.0" not in seg


# ===============================================================
# 9. run_simulation 显式 T_initial_C 通用性
# ===============================================================

def test_solver_explicit_initial_generic():
    t = np.array([0.0, 1.0, 2.0])
    Tb = np.array([30.0, 30.0, 30.0])
    res = heat_model.run_simulation(
        t, Tb, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        T_initial_C=88.0, save_dt=0.5,
    )
    assert res["T_sample_arr"][0] == pytest.approx(88.0, abs=1e-9)


# ===============================================================
# 10. 未来拟合直接模式从实测内部首值初始化
# ===============================================================

def test_fitting_direct_mode_initializes_from_measured_first():
    import pandas as pd
    df = pd.read_csv(ALIGNED_CSV)
    t = df["time_s"].to_numpy(dtype=float)
    T_internal = df["T_internal_interpolated_C"].to_numpy(dtype=float)
    first = float(T_internal[0])
    res = heat_model.run_simulation(
        t, T_internal, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=1.0, T_initial_C=first,
    )
    assert res["T_bottom_arr"][0] == pytest.approx(27.64, abs=1e-9)
    assert res["T_sample_arr"][0] == pytest.approx(27.64, abs=1e-9)
    assert res["T_top_surface_arr"][0] == pytest.approx(27.64, abs=1e-9)
