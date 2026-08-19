"""
样品层空间平均测试 (节点控制体积重叠加权)。

覆盖 (任务 9 的 10 项):
1.  均匀 20 um 样品 (180/185/190/195/200) 权重 == 梯形公式
    [0.5,1,1,1,0.5]/4;
2.  恒定温度场 -> 精确等于该常数;
3.  线性温度场 -> 精确解析空间均值;
4.  界面节点 (180/200 um) 半控制体积贡献正确;
5.  权重和 == 1;
6.  权重非负;
7.  任意非均匀样品网格 -> 正确空间平均;
8.  网格密度变化不改变线性场的样品平均;
9.  sample-role 检测保持通用 (任意层位置);
10. 裸顶样品保持 180-200 um。
"""

import numpy as np
import pytest

import heat_model
from heat_model import (
    BARE_TOP_COC_LAYERS,
    DEFAULT_MATERIALS,
    Layer,
    Material,
    build_layer_stack,
    compute_sample_weights,
)


def sample_mean(ms, T_field):
    """用 LayerStack.sample_weights 计算样品空间平均。"""
    return float(np.dot(ms.sample_weights, np.asarray(T_field, dtype=float)))


# ===============================================================
# 1. 均匀 5 um 样品 -> 梯形权重
# ===============================================================

def test_uniform_sample_trapezoidal_weights():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    x = ms.x
    nodes_um = (x * 1e6).round(3)
    sample_nodes = np.where(
        (x > 180e-6 - 1e-9) & (x < 200e-6 + 1e-9)
    )[0]
    # 有重量的节点 = 180..200 um 全部 5 个 (含界面边界节点)
    assert len(sample_nodes) == 5
    assert (nodes_um[sample_nodes] == np.array([180.0, 185.0, 190.0, 195.0,
                                                200.0])).all()
    w = ms.sample_weights[sample_nodes]
    np.testing.assert_allclose(w, np.array([0.5, 1.0, 1.0, 1.0, 0.5]) / 4,
                               rtol=0, atol=1e-12)


# ===============================================================
# 2-3. 恒定 / 线性场
# ===============================================================

def test_constant_field_mean_exact():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    T = np.full(ms.Nx, 73.25)
    assert sample_mean(ms, T) == pytest.approx(73.25, abs=1e-12)


def test_linear_field_mean_exact():
    """T(x) = a + b*x -> 空间均值 = a + b*(200+180)/2 um。"""
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    a, b = 20.0, 1e5  # °C, °C/m
    T = a + b * ms.x
    mean_ana = a + b * 190e-6
    assert sample_mean(ms, T) == pytest.approx(mean_ana, rel=1e-10, abs=1e-9)


# ===============================================================
# 4. 界面半控制体积贡献
# ===============================================================

def test_interface_half_cv_contributions():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    # 180 um (节点 36) 与 200 um (节点 40) 的权重应为内部节点的一半
    j180 = int(np.argmin(np.abs(ms.x - 180e-6)))
    j200 = int(np.argmin(np.abs(ms.x - 200e-6)))
    interior = int(np.argmin(np.abs(ms.x - 190e-6)))
    assert ms.sample_weights[j180] == pytest.approx(ms.sample_weights[interior] / 2)
    assert ms.sample_weights[j200] == pytest.approx(ms.sample_weights[interior] / 2)
    # 180 um 左侧 (Bottom COC 半区间) 不计入样品
    assert ms.sample_weights[j180 - 1] == 0.0
    # 200 um 右侧 (Oil 半区间) 不计入样品
    assert ms.sample_weights[j200 + 1] == 0.0


# ===============================================================
# 5-6. 权重和 / 非负
# ===============================================================

def test_weights_sum_to_one_and_nonneg():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert np.sum(ms.sample_weights) == pytest.approx(1.0, abs=1e-12)
    assert np.all(ms.sample_weights >= 0.0)


# ===============================================================
# 7-8. 任意非均匀网格 / 网格密度不变性
# ===============================================================

def _synthetic_sample_stack(dx_left_um, dx_right_um):
    """构造样品层 40-60 um、两侧网格分辨率不同的非均匀堆叠。"""
    mats = {
        "A": Material("A", k_W_mK=0.5, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "B": Material("B", k_W_mK=2.0, rho_kg_m3=200.0, cp_J_kgK=2000.0),
    }
    layers = [
        Layer("L", "A", 40e-6, dx_target_m=dx_left_um * 1e-6),
        Layer("S", "B", 20e-6, dx_target_m=5e-6, role="sample"),
        Layer("R", "A", 40e-6, dx_target_m=dx_right_um * 1e-6),
    ]
    return build_layer_stack(mats, layers)


def test_nonuniform_sample_mesh_mean():
    ms = _synthetic_sample_stack(7.0, 3.0)
    # 样品区间 [40,60] um; 线性场精确均值
    a, b = 10.0, 2e5
    T = a + b * ms.x
    mean_ana = a + b * 50e-6
    assert sample_mean(ms, T) == pytest.approx(mean_ana, rel=1e-10, abs=1e-9)
    assert np.sum(ms.sample_weights) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dx_left,dx_right", [(5, 5), (7, 3), (2, 9), (11, 4)])
def test_mesh_density_invariance_linear_field(dx_left, dx_right):
    """线性场的样品空间平均与网格无关 (梯形/重叠积分精确)。"""
    ms = _synthetic_sample_stack(dx_left, dx_right)
    a, b = 30.0, 1.5e5
    T = a + b * ms.x
    assert sample_mean(ms, T) == pytest.approx(a + b * 50e-6, rel=1e-9, abs=1e-8)


# ===============================================================
# 9. sample-role 通用性
# ===============================================================

def test_sample_role_generic_position():
    mats = {
        "A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0),
    }
    layers = [
        Layer("L1", "A", 30e-6, cells=3),
        Layer("S", "A", 10e-6, cells=2, role="sample"),
        Layer("L2", "A", 60e-6, cells=3),
    ]
    ms = build_layer_stack(mats, layers)
    assert ms.sample_layer_index == 1
    assert ms.boundaries[1] == pytest.approx(30e-6)
    assert ms.boundaries[2] == pytest.approx(40e-6)
    # 样品层 [30,40] um; 权重覆盖恰好该区间
    assert ms.sample_weights[ms.x < 30e-6 - 1e-9].sum() == 0.0
    assert ms.sample_weights[ms.x > 40e-6 + 1e-9].sum() == 0.0
    assert np.sum(ms.sample_weights) == pytest.approx(1.0, abs=1e-12)
    # 线性场精确
    T = 5.0 + 3e5 * ms.x
    assert sample_mean(ms, T) == pytest.approx(5.0 + 3e5 * 35e-6, rel=1e-10)


# ===============================================================
# 10. 裸顶样品几何保持
# ===============================================================

def test_bare_top_sample_interval_unchanged():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.boundaries[1] == pytest.approx(180e-6)
    assert ms.boundaries[2] == pytest.approx(200e-6)
    assert ms.sample_layer_index == 1
    # 积分宽度 = 样品厚度
    raw = compute_sample_weights(ms.x, 180e-6, 200e-6)
    assert np.sum(raw) == pytest.approx(20e-6, rel=1e-10, abs=1e-12)


# ===============================================================
# 求解器层面: 初始均匀场 -> 首个保存点样品温度 == 初温
# ===============================================================

def test_run_simulation_sample_mean_initial_uniform():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    Tb = 27.64 + np.array([0.0, 5.0, 10.0, 15.0])
    res = heat_model.run_simulation(
        t, Tb, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.5, T_initial_C=27.64,
    )
    # 首个保存点 (t=0) 捕获更新前的初始均匀场
    assert res["T_sample_arr"][0] == pytest.approx(27.64, abs=1e-9)
    assert res["T_top_surface_arr"][0] == pytest.approx(27.64, abs=1e-9)
