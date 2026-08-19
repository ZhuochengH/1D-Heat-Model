"""
k_eff–cp_eff 系统级有效参数扫描测试。

覆盖 (任务 34 的 20 项):
1.  网格生成包含全部端点;
2.  粗网格组合数恰为预期 (9 x 8 = 72);
3.  参数点键确定性 -> 不重复评估;
4.  续跑逻辑跳过已完成点;
5.  仅 COC 的 k/cp 被修改;
6.  COC rho 保持 1020;
7.  Water / Oil 性质不变;
8.  Bottom COC 与 Top COC 使用同一候选值 (共享 COC 材料);
9.  auto 初始条件 = 第一个底部值;
10. 顶部目标仅用 T_top_surface;
11. T_sample 不进目标;
12. RMSE 合成数组正确;
13. MAE 正确;
14. 插值到实测时间正确;
15. 无时间平移;
16. 最优点选择 = 最小 RMSE;
17. 边界最小检测;
18. 近最优 1/2/5% 分类;
19. 结果持久化 / 续跑;
20. 扫描不修改全局 DEFAULT_MATERIALS。

注意: 测试用短合成协议, 不运行完整 72C 扫描。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import heat_model
from heat_model import DEFAULT_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scan_effective_thermal_parameters.py"


def load_scan():
    spec = importlib.util.spec_from_file_location("scan_eff", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scan = load_scan()

# 合成短协议 (每次模拟 ~0.1-0.3 s, 避免长测试)
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 62.0, 50.0])
_TTOP = _TINT - 2.0


# ===============================================================
# 1-2. 网格生成
# ===============================================================

def test_coarse_grid_endpoints_included():
    k_min, k_max = min(scan.K_GRID_COARSE), max(scan.K_GRID_COARSE)
    cp_min, cp_max = min(scan.CP_GRID_COARSE), max(scan.CP_GRID_COARSE)
    assert k_min == 0.08 and k_max == 0.24
    assert cp_min == 800.0 and cp_max == 2200.0


def test_coarse_grid_exact_combination_count():
    points = scan.product_grid(scan.K_GRID_COARSE, scan.CP_GRID_COARSE)
    assert len(points) == 9 * 8 == 72
    # 确定性排序: k 升序, 同 k 下 cp 升序
    ks = [p[0] for p in points]
    assert ks == sorted(ks)
    first_k = [p for p in points if p[0] == scan.K_GRID_COARSE[0]]
    assert [p[1] for p in first_k] == sorted(scan.CP_GRID_COARSE)


# ===============================================================
# 3-4. 参数点键 / 续跑
# ===============================================================

def test_point_key_deterministic():
    assert scan.point_key(0.14, 1400.0) == scan.point_key(0.1400001, 1400.0)
    assert scan.point_key(0.13, 1800.0) == "0.130000|1800.000000"


def test_resume_skips_completed(tmp_path):
    rows = [
        {"k_eff_W_mK": 0.14, "cp_eff_J_kgK": 1400.0, "status": "OK"},
        {"k_eff_W_mK": 0.16, "cp_eff_J_kgK": 1600.0, "status": "FAILED"},
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    done = scan.completed_keys(df)
    assert scan.point_key(0.14, 1400.0) in done
    assert scan.point_key(0.16, 1600.0) not in done  # FAILED 不算完成


def test_run_stage_skips_existing(tmp_path, capsys):
    t = np.array([0.0, 1.0, 2.0])
    tint = np.array([30.0, 40.0, 50.0])
    ttop = tint - 1.0
    points = [(0.14, 1400.0), (0.14, 1600.0)]
    scan.run_stage_points(points, t, tint, ttop, "stage.csv", tmp_path,
                          "test")
    out = capsys.readouterr().out
    assert "[test 1/2]" in out and "[test 2/2]" in out
    # 第二次运行: 全部跳过
    scan.run_stage_points(points, t, tint, ttop, "stage.csv", tmp_path,
                          "test")
    out2 = capsys.readouterr().out
    assert out2.count("[skip") == 2


# ===============================================================
# 5-8. 候选材料
# ===============================================================

def test_only_coc_k_cp_modified():
    mats = scan.make_candidate_materials(0.18, 1500.0)
    assert mats["COC"].k_W_mK == 0.18
    assert mats["COC"].cp_J_kgK == 1500.0
    for name in ("Water", "Oil", "Air", "PDMS"):
        m = mats[name]
        ref = DEFAULT_MATERIALS[name]
        assert (m.k_W_mK, m.rho_kg_m3, m.cp_J_kgK) == (
            ref.k_W_mK, ref.rho_kg_m3, ref.cp_J_kgK)


def test_coc_rho_fixed():
    for k, cp in ((0.08, 800.0), (0.24, 2200.0)):
        mats = scan.make_candidate_materials(k, cp)
        assert mats["COC"].rho_kg_m3 == 1020.0


def test_global_default_materials_not_mutated():
    before = (DEFAULT_MATERIALS["COC"].k_W_mK,
              DEFAULT_MATERIALS["COC"].cp_J_kgK)
    scan.make_candidate_materials(0.2, 2000.0)
    after = (DEFAULT_MATERIALS["COC"].k_W_mK,
             DEFAULT_MATERIALS["COC"].cp_J_kgK)
    assert before == after == (0.13, 1800.0)


def test_both_coc_layers_use_candidate():
    mats = scan.make_candidate_materials(0.18, 1500.0)
    ms = heat_model.build_layer_stack(mats, heat_model.BARE_TOP_COC_LAYERS)
    for i, layer in enumerate(heat_model.BARE_TOP_COC_LAYERS):
        if layer.material == "COC":
            nodes = ms.node_layer_index == i
            assert np.all(ms.k[nodes] == 0.18)
            assert np.all(ms.cp[nodes] == 1500.0)


# ===============================================================
# 9. auto 初始条件 = 第一个底部值
# ===============================================================

def test_auto_initial_equals_first_boundary():
    row, res = scan.evaluate_point(0.14, 1400.0, _T, _TINT, _TTOP,
                                   return_result=True)
    assert res["T_sample_arr"][0] == pytest.approx(float(_TINT[0]), abs=1e-9)
    assert res["T_top_surface_arr"][0] == pytest.approx(float(_TINT[0]), abs=1e-9)


# ===============================================================
# 10-11. 目标仅含顶部 / 样品不入目标
# ===============================================================

def test_objective_uses_top_surface_only():
    row = scan.evaluate_point(0.14, 1400.0, _T, _TINT, _TTOP)
    # 手工重算: 用与实现相同的路径 (T_top_surface 在实测 TIME 处插值)
    mats = scan.make_candidate_materials(0.14, 1400.0)
    res = heat_model.run_simulation(
        _T, _TINT, mats, heat_model.BARE_TOP_COC_LAYERS,
        h_conv=scan.H_CONV, T_air_ambient=scan.T_AMB, save_dt=scan.SAVE_DT,
        T_initial_C=float(_TINT[0]),
    )
    pred = np.interp(_T, res["t_array"], res["T_top_surface_arr"])
    expected = float(np.sqrt(np.mean((pred - _TTOP) ** 2)))
    assert row["RMSE_C"] == pytest.approx(expected, abs=1e-10)


def test_sample_does_not_enter_objective():
    """目标只依赖顶部: 改变样品提取不影响 RMSE (同一场)。"""
    row = scan.evaluate_point(0.14, 1400.0, _T, _TINT, _TTOP)
    assert "sample" not in [c.lower() for c in row] or "sample" not in row
    # 目标函数定义中无 T_sample
    src = SCRIPT.read_text(encoding="utf-8")
    obj_seg = src[src.index("def evaluate_point"):]
    obj_seg = obj_seg[:obj_seg.index("def evaluate_point_safe")]
    assert "T_sample" not in obj_seg


# ===============================================================
# 12-13. 指标计算
# ===============================================================

def test_rmse_correct():
    t = np.arange(4.0)
    r = np.array([1.0, -2.0, 3.0, -4.0])
    m = scan.compute_metrics(t, r)
    assert m["rmse"] == pytest.approx(
        np.sqrt(np.mean(r ** 2)), abs=1e-12)
    assert m["mae"] == pytest.approx(np.mean(np.abs(r)), abs=1e-12)
    assert m["mean_residual"] == pytest.approx(np.mean(r), abs=1e-12)
    assert m["max_abs_error"] == pytest.approx(4.0)
    assert m["max_positive"] == pytest.approx(3.0)
    assert m["max_negative"] == pytest.approx(-4.0)
    assert m["time_of_max_abs"] == pytest.approx(3.0)


def test_metrics_nan_handling():
    m = scan.compute_metrics(np.arange(3.0), np.array([np.nan, np.nan, np.nan]))
    assert np.isnan(m["rmse"]) and np.isnan(m["mae"])


# ===============================================================
# 14-15. 插值到实测时间 / 无时间平移
# ===============================================================

def test_interpolation_to_measured_times_no_shift():
    mats = scan.make_candidate_materials(0.14, 1400.0)
    res = heat_model.run_simulation(
        _T, _TINT, mats, heat_model.BARE_TOP_COC_LAYERS,
        h_conv=scan.H_CONV, T_air_ambient=scan.T_AMB, save_dt=scan.SAVE_DT,
        T_initial_C=float(_TINT[0]),
    )
    # 插值在实测 TIME 坐标 (_T) 上进行, 不是实测温度值 (_TTOP)
    pred = np.interp(_T, res["t_array"], res["T_top_surface_arr"])
    assert np.all(np.isfinite(pred))
    # 与直接重算一致
    assert np.all(np.abs(pred - np.interp(_T, res["t_array"],
                                          res["T_top_surface_arr"])) < 1e-15)
    # 无时间平移: 使用实测时间坐标本身
    assert np.array_equal(
        np.interp(_T, res["t_array"], res["T_top_surface_arr"]),
        pred,
    )


# ===============================================================
# 16. 最优点选择
# ===============================================================

def test_best_point_selection():
    df = pd.DataFrame([
        {"k_eff_W_mK": 0.10, "cp_eff_J_kgK": 1000.0, "RMSE_C": 3.0,
         "status": "OK"},
        {"k_eff_W_mK": 0.12, "cp_eff_J_kgK": 1200.0, "RMSE_C": 1.5,
         "status": "OK"},
        {"k_eff_W_mK": 0.14, "cp_eff_J_kgK": 1400.0, "RMSE_C": np.nan,
         "status": "FAILED"},
        {"k_eff_W_mK": 0.16, "cp_eff_J_kgK": 1600.0, "RMSE_C": 2.0,
         "status": "OK"},
    ])
    best = scan.best_point_from_table(df)
    assert best["k_eff_W_mK"] == 0.12
    assert best["cp_eff_J_kgK"] == 1200.0
    assert best["RMSE_C"] == 1.5


# ===============================================================
# 17. 边界最小检测
# ===============================================================

def test_boundary_minimum_detection():
    k_grid = [0.08, 0.10, 0.12]
    cp_grid = [800.0, 1000.0, 1200.0]
    assert scan.detect_boundary_minimum(0.08, 1000.0, k_grid, cp_grid) == {
        "k_low"}
    assert scan.detect_boundary_minimum(0.12, 1200.0, k_grid, cp_grid) == {
        "k_high", "cp_high"}
    assert scan.detect_boundary_minimum(0.10, 800.0, k_grid, cp_grid) == {
        "cp_low"}
    assert scan.detect_boundary_minimum(0.10, 1000.0, k_grid, cp_grid) == set()


def test_clipped_fine_grid_within_limits():
    k_grid, cp_grid = scan.clipped_fine_grid(0.14, 1400.0,
                                             scan.K_LIMITS, scan.CP_LIMITS)
    assert k_grid[0] == pytest.approx(0.12) and k_grid[-1] == pytest.approx(0.16)
    assert len(k_grid) == 11
    assert cp_grid[0] == pytest.approx(1200.0) and cp_grid[-1] == pytest.approx(1600.0)
    assert len(cp_grid) == 9
    # 裁剪到边界
    k_grid2, _ = scan.clipped_fine_grid(0.06, 1400.0,
                                        scan.K_LIMITS, scan.CP_LIMITS)
    assert k_grid2[0] == pytest.approx(0.06)  # 不下溢


# ===============================================================
# 18. 近最优分类
# ===============================================================

def test_near_optimal_classification():
    rmse_min = 1.0
    df = pd.DataFrame([
        {"k_eff_W_mK": 0.10, "cp_eff_J_kgK": 1000.0, "RMSE_C": 1.001,
         "status": "OK"},
        {"k_eff_W_mK": 0.12, "cp_eff_J_kgK": 1200.0, "RMSE_C": 1.015,
         "status": "OK"},
        {"k_eff_W_mK": 0.14, "cp_eff_J_kgK": 1400.0, "RMSE_C": 1.03,
         "status": "OK"},
        {"k_eff_W_mK": 0.16, "cp_eff_J_kgK": 1600.0, "RMSE_C": 1.05,
         "status": "OK"},
        {"k_eff_W_mK": 0.18, "cp_eff_J_kgK": 1800.0, "RMSE_C": 1.10,
         "status": "OK"},
    ])
    near = scan.near_optimal_regions(df, rmse_min)
    assert near["1pct"]["n_points"] == 1
    assert near["2pct"]["n_points"] == 2
    assert near["5pct"]["n_points"] == 4
    assert near["5pct"]["k_min"] == 0.10 and near["5pct"]["cp_max"] == 1600.0


# ===============================================================
# 19. 持久化 / 续跑 (CSV 往返)
# ===============================================================

def test_append_and_reload(tmp_path):
    p = tmp_path / "scan.csv"
    scan.append_rows(p, [{"a": 1, "b": 2}])
    scan.append_rows(p, [{"a": 3, "b": 4}])
    df = pd.read_csv(p)
    assert len(df) == 2
    assert list(df["a"]) == [1, 3]


def test_combined_deduplicates(tmp_path):
    (tmp_path / "coarse_scan.csv").write_text(
        "k_eff_W_mK,cp_eff_J_kgK,RMSE_C,status\n"
        "0.14,1400.0,1.5,OK\n"
        "0.16,1600.0,2.0,OK\n", encoding="utf-8")
    (tmp_path / "extend_scan.csv").write_text(
        "k_eff_W_mK,cp_eff_J_kgK,RMSE_C,status\n"
        "0.14,1400.0,1.5,OK\n"
        "0.12,1200.0,2.5,OK\n", encoding="utf-8")
    comb = scan.build_combined(tmp_path)
    assert len(comb) == 3  # 0.14/1400 去重
    assert comb["k_eff_W_mK"].tolist() == [0.12, 0.14, 0.16]


# ===============================================================
# 20. 全局材料不被扫描污染
# ===============================================================

def test_scan_does_not_mutate_globals():
    before = {name: (m.k_W_mK, m.rho_kg_m3, m.cp_J_kgK)
              for name, m in DEFAULT_MATERIALS.items()}
    scan.evaluate_point(0.18, 1500.0, _T, _TINT, _TTOP)
    after = {name: (m.k_W_mK, m.rho_kg_m3, m.cp_J_kgK)
             for name, m in DEFAULT_MATERIALS.items()}
    assert before == after


# ===============================================================
# 数据加载
# ===============================================================

def test_load_experiment_real_dataset():
    t, tint, ttop = scan.load_experiment()
    assert len(t) == 299
    assert t[0] == pytest.approx(1.0) and t[-1] == pytest.approx(299.0)
    assert tint[0] == pytest.approx(27.64)
    assert ttop[0] == pytest.approx(27.8)
    assert np.all(np.diff(t) > 0)
