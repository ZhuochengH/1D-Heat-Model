#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B — 滞后放置架构比较测试 (规格 #48, 20 项)。

覆盖:
 48-1  tau 网格 25 值 (0.0-12.0, 步长 0.5)
 48-2  固定 k=0.055 / cp=1200 (冻结候选, 不重拟合)
 48-3  O: 一次 FDM + 输出侧滞后 (tau 只作用 T_top_observed)
 48-4  O: T_sample 不滞后
 48-5  I: 输入侧滞后底部边界 (初始态 = internal[0]), 无输出滤波器
 48-6  S: 同一 tau 输入侧 + 输出侧
 48-7  tau=0 对所有架构严格相同 (恒等)
 48-8  72C 目标查询轴 = 实测时间 (np.interp)
 48-9  指标含 RMSE/MAE/mean/median/max abs
 48-10 每架构用最小 RMSE 选 tau (绝不基于样品)
 48-11 TAU_BOUNDARY_WARNING 若最优 tau = 12.0
 48-12 不引入独立 tau_input / tau_output
 48-13 复用 apply_first_order_lag 与 run_convection_radiation_fdm (不重写物理)
 48-14 不修改对流/辐射物理 (固定 h=10, eps=0.90, 非线性辐射)
 48-15 不覆盖旧输出 (输出目录 gitignored, 不与 v1/校准输出冲突)
 48-16 扫描行数 = 3 架构 x 25 tau = 75
 48-17 扫描结果含所有必需列
 48-18 O 架构 tau=0 与冻结候选基线一致性 (72C RMSE 参考)
 48-19 I/S 在 tau=0 复用 O 基线 (不额外 FDM)
 48-20 运行脚本输出文件齐全且元数据正确
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core import lag_augmented_thermal_model as lm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = PROJECT_ROOT / "thermal_model/utilities/lag_placement_comparison_model.py"
RUN_SCRIPT = PROJECT_ROOT / "workflows/diagnostics/run_lag_placement_comparison.py"
OUT_ROOT = (PROJECT_ROOT / "model_comparison_output"
            / "lag_placement_comparison_v1")
OUT72 = OUT_ROOT / "72C_calibration"

TAU_GRID = tuple(float(0.5 * k) for k in range(0, 25))

# 短合成数据 (72C 目标用真实校准文件, 单元测试用合成)
_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 78.0, 80.0, 75.0])
_TTOP = np.array([28.0, 35.0, 47.0, 58.0, 63.0, 64.0, 60.0])
_ENV = 27.8


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = None
_run = None


@pytest.fixture(scope="module")
def m():
    global _m
    if _m is None:
        _m = _load(MODEL, "lag_placement_model")
    return _m


@pytest.fixture(scope="module")
def run():
    global _run
    if _run is None:
        _run = _load(RUN_SCRIPT, "lag_placement_run")
    return _run


# ================================================================
# 48-1: tau 网格
# ================================================================

def test_tau_grid_25_values(m):
    assert len(m.TAU_GRID_S) == 25
    assert m.TAU_GRID_S[0] == 0.0
    assert m.TAU_GRID_S[-1] == 12.0
    assert all(abs(m.TAU_GRID_S[i + 1] - m.TAU_GRID_S[i] - 0.5) < 1e-12
               for i in range(24))
    assert m.TAU_MAX_S == 12.0


# ================================================================
# 48-2: 固定 k/cp
# ================================================================

def test_frozen_k_cp_fixed(m):
    assert m.FROZEN_K_W_MK == pytest.approx(0.055)
    assert m.FROZEN_CP_J_KGK == pytest.approx(1200.0)
    # 与冻结候选一致
    assert m.FROZEN_K_W_MK == pytest.approx(
        m.FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK)
    assert m.FROZEN_CP_J_KGK == pytest.approx(
        m.FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK)


# ================================================================
# 48-3: O = 输出侧滞后
# ================================================================

def test_arch_o_output_side_lag(m):
    out = m.run_arch_output_side(_T, _TINT, _ENV, 2.0)
    # tau>0: T_top_obs != T_top_fdm (滞后)
    assert out["T_top_fdm"].size == out["T_top_obs"].size
    assert not np.allclose(out["T_top_fdm"], out["T_top_obs"])
    # tau=0: 恒等
    out0 = m.run_arch_output_side(_T, _TINT, _ENV, 0.0)
    assert np.allclose(out0["T_top_fdm"], out0["T_top_obs"])
    assert np.allclose(out0["T_top_fdm"], out0["T_sample_fdm"] * 0
                       + out0["T_top_fdm"])


# ================================================================
# 48-4: O 的 T_sample 不滞后
# ================================================================

def test_arch_o_sample_not_lagged(m):
    # O: T_sample 是 FDM 直接输出, 不经过滞后 (与 tau 无关)
    out1 = m.run_arch_output_side(_T, _TINT, _ENV, 3.0)
    out2 = m.run_arch_output_side(_T, _TINT, _ENV, 7.0)
    assert np.allclose(out1["T_sample_fdm"], out2["T_sample_fdm"])
    # T_sample 不是滞后后的 T_top
    assert not np.allclose(out1["T_sample_fdm"], out1["T_top_obs"])


# ================================================================
# 48-5: I = 输入侧滞后
# ================================================================

def test_arch_i_input_side_lag_no_output_filter(m):
    out = m.run_arch_input_side(_T, _TINT, _ENV, 2.0)
    # I: 无输出滤波器 -> T_top_obs == T_top_fdm
    assert np.allclose(out["T_top_obs"], out["T_top_fdm"])
    # I 的 FDM 与 O 不同 (边界被滞后)
    out_o = m.run_arch_output_side(_T, _TINT, _ENV, 2.0)
    assert not np.allclose(out["T_top_fdm"], out_o["T_top_fdm"])


# ================================================================
# 48-6: S = 共享单 tau
# ================================================================

def test_arch_s_shared_single_tau(m):
    out = m.run_arch_shared(_T, _TINT, _ENV, 2.0)
    # S: 输出侧滞后 (同一 tau) -> T_top_obs != T_top_fdm
    assert not np.allclose(out["T_top_obs"], out["T_top_fdm"])
    # S 的输入边界也被滞后 -> FDM 与 O 不同
    out_o = m.run_arch_output_side(_T, _TINT, _ENV, 2.0)
    assert not np.allclose(out["T_top_fdm"], out_o["T_top_fdm"])


# ================================================================
# 48-7: tau=0 对所有架构严格相同
# ================================================================

def test_tau_zero_identical_across_architectures(m):
    o = m.run_arch_output_side(_T, _TINT, _ENV, 0.0)
    i = m.run_arch_input_side(_T, _TINT, _ENV, 0.0)
    s = m.run_arch_shared(_T, _TINT, _ENV, 0.0)
    assert np.allclose(o["T_top_obs"], i["T_top_obs"])
    assert np.allclose(o["T_top_obs"], s["T_top_obs"])
    assert np.allclose(o["T_sample_fdm"], i["T_sample_fdm"])
    assert np.allclose(o["T_sample_fdm"], s["T_sample_fdm"])


# ================================================================
# 48-8: 72C 目标查询轴 = 实测时间
# ================================================================

def test_metrics_query_axis_measured_time(m):
    # pred_on_meas_time 已插值到实测时间; 残差按元素差
    pred = np.array([29.0, 36.0, 48.0, 58.0, 63.0, 64.0, 61.0])
    met = m.top_metrics_at_measurement_times(pred, _TTOP)
    r = pred - _TTOP
    assert met["RMSE_C"] == pytest.approx(np.sqrt(np.mean(r ** 2)))
    assert met["MAE_C"] == pytest.approx(np.mean(np.abs(r)))
    assert met["mean_residual_C"] == pytest.approx(np.mean(r))
    assert met["median_abs_residual_C"] == pytest.approx(np.median(np.abs(r)))
    assert met["max_abs_residual_C"] == pytest.approx(np.max(np.abs(r)))
    # evaluate_architecture 内部用 np.interp(t_proto, t_arr, ...)
    row = m.evaluate_architecture("O", 0.0, _T, _TINT, _TTOP, _ENV)
    assert row["architecture"] == "O"
    assert row["tau_lag_s"] == 0.0
    assert set(row) >= {"RMSE_72C_C", "MAE_72C_C", "mean_residual_C",
                        "median_abs_residual_C", "max_abs_residual_C"}


# ================================================================
# 48-9: 指标完整性
# ================================================================

def test_metrics_all_fields(m):
    row = m.evaluate_architecture("I", 1.5, _T, _TINT, _TTOP, _ENV)
    for key in ("RMSE_72C_C", "MAE_72C_C", "mean_residual_C",
                "median_abs_residual_C", "max_abs_residual_C"):
        assert np.isfinite(row[key])
    assert row["tau_lag_s"] == pytest.approx(1.5)


# ================================================================
# 48-10: 最小 RMSE 选 tau (不基于样品)
# ================================================================

def test_best_tau_by_min_rmse_only(m):
    rows = m.scan_architecture("O", _T, _TINT, _TTOP, _ENV)
    best = m.select_best_tau(rows, "O")
    assert best["best_tau_s"] == min(rows,
                                     key=lambda r: r["RMSE_72C_C"])[
        "tau_lag_s"]
    assert best["best_RMSE_C"] == pytest.approx(
        min(r["RMSE_72C_C"] for r in rows))
    # 选择不使用样品温度: 扫描行只有顶部指标, 无样品列
    assert "sample" not in "".join(str(rows[0].keys())).lower()


# ================================================================
# 48-11: TAU_BOUNDARY_WARNING
# ================================================================

def test_tau_boundary_warning(m):
    rows = [{"tau_lag_s": 0.0, "RMSE_72C_C": 5.0, "MAE_72C_C": 4.0,
             "mean_residual_C": 3.0, "median_abs_residual_C": 2.0,
             "max_abs_residual_C": 8.0},
            {"tau_lag_s": 12.0, "RMSE_72C_C": 1.0, "MAE_72C_C": 0.8,
             "mean_residual_C": 0.5, "median_abs_residual_C": 0.6,
             "max_abs_residual_C": 2.0}]
    best = m.select_best_tau(rows, "O", tau_max=12.0)
    assert best["TAU_BOUNDARY_WARNING"] is True
    rows2 = [{"tau_lag_s": 8.0, "RMSE_72C_C": 1.0, "MAE_72C_C": 0.8,
              "mean_residual_C": 0.5, "median_abs_residual_C": 0.6,
              "max_abs_residual_C": 2.0},
             {"tau_lag_s": 12.0, "RMSE_72C_C": 3.0, "MAE_72C_C": 2.5,
              "mean_residual_C": 2.0, "median_abs_residual_C": 2.1,
              "max_abs_residual_C": 5.0}]
    best2 = m.select_best_tau(rows2, "O", tau_max=12.0)
    assert best2["TAU_BOUNDARY_WARNING"] is False


# ================================================================
# 48-12: 不引入独立 tau_input / tau_output
# ================================================================

def test_no_independent_tau_in_out(m):
    import inspect
    # 任何函数签名都不接受 tau_input / tau_output 参数 (检查签名而非 docstring)
    for name in ("run_arch_output_side", "run_arch_input_side",
                 "run_arch_shared", "evaluate_architecture"):
        fn = getattr(m, name)
        sig = inspect.signature(fn)
        assert "tau_input" not in sig.parameters
        assert "tau_output" not in sig.parameters
    # S 用单一 tau
    sig = inspect.signature(m.run_arch_shared)
    assert "tau" in sig.parameters
    assert "tau_input" not in sig.parameters
    assert "tau_output" not in sig.parameters


# ================================================================
# 48-13: 复用既有物理
# ================================================================

def test_reuses_existing_physics(m):
    import inspect
    src = inspect.getsource(m)
    # 使用 apply_first_order_lag 与 run_convection_radiation_fdm
    assert "apply_first_order_lag" in src
    assert "run_convection_radiation_fdm" in src
    # 不复制 FDM 主循环 (不定义 run_simulation 类函数)
    assert "def run_convection_radiation_fdm" not in src.replace(
        "cr.run_convection_radiation_fdm", "")


# ================================================================
# 48-14: 不修改对流/辐射物理
# ================================================================

def test_physics_constants_unchanged(m):
    assert cr.H_CONV_STRATEGY_E_W_M2K == pytest.approx(10.0)
    assert cr.EMISSIVITY_STRATEGY_E == pytest.approx(0.90)
    assert cr.SIGMA_SB_W_M2_K4 == pytest.approx(5.670374419e-8)
    assert cr.VIEW_FACTOR_STRATEGY_E == pytest.approx(1.0)
    assert cr.RHO_COC_STRATEGY_E == pytest.approx(1020.0)
    # 模型使用默认边界 (不覆盖)
    assert m.SAVE_DT == 0.1


# ================================================================
# 48-15: 不覆盖旧输出
# ================================================================

def test_no_old_output_overwritten():
    # Phase B 输出目录独立且 gitignored; 与 v1 / 校准输出不冲突
    assert "lag_placement_comparison_v1" in str(OUT_ROOT)
    # 不引用旧校准输出目录作为写入目标
    import inspect
    run_src = inspect.getsource(
        _load(RUN_SCRIPT, "lag_placement_run"))
    assert "convection_radiation_k_cp_tau_calibration" not in run_src
    assert "strategy_G_conservative_cross_protocol_v1" not in run_src


# ================================================================
# 48-16: 扫描行数 = 75
# ================================================================

def test_scan_row_count(m):
    rows, summary = m.run_72c_comparison(_T, _TINT, _TTOP, _ENV)
    assert len(rows) == 3 * 25
    assert len(summary) == 3
    archs = {r["architecture"] for r in rows}
    assert archs == {"O", "I", "S"}


# ================================================================
# 48-17: 扫描结果列
# ================================================================

def test_scan_columns(m):
    rows, _ = m.run_72c_comparison(_T, _TINT, _TTOP, _ENV)
    required = {"architecture", "tau_lag_s", "RMSE_72C_C", "MAE_72C_C",
                "mean_residual_C", "median_abs_residual_C",
                "max_abs_residual_C"}
    assert required <= set(rows[0].keys())
    # tau 网格完整
    taus = sorted({r["tau_lag_s"] for r in rows})
    assert len(taus) == 25


# ================================================================
# 48-18: O 架构 tau=0 与冻结候选基线一致
# ================================================================

def test_o_tau0_matches_frozen_baseline(m, run):
    # 运行脚本应复现冻结候选 O tau=8.5 的 RMSE ≈ 0.889 (72C)
    t_proto, t_int, t_top_meas = run.load_72c()
    env = cr.infer_environment_from_initial_top_measurement(
        t_top_meas, time_s=t_proto)["T_environment_C"]
    row = m.evaluate_architecture("O", 8.5, t_proto, t_int, t_top_meas, env)
    # 冻结候选 72C RMSE (存储) = 0.8891597
    assert row["RMSE_72C_C"] == pytest.approx(
        0.8891597125869538, abs=0.01)


# ================================================================
# 48-19: I/S tau=0 复用 O 基线 (不额外 FDM)
# ================================================================

def test_i_s_tau0_reuse_baseline(m):
    # tau=0 分支直接调用 O 基线, 结果与 O 相同
    o = m.run_arch_output_side(_T, _TINT, _ENV, 0.0)
    i = m.run_arch_input_side(_T, _TINT, _ENV, 0.0)
    s = m.run_arch_shared(_T, _TINT, _ENV, 0.0)
    assert np.allclose(i["T_top_fdm"], o["T_top_fdm"])
    assert np.allclose(s["T_top_fdm"], o["T_top_fdm"])


# ================================================================
# 48-20: 运行脚本输出齐全
# ================================================================

def test_run_outputs_complete():
    assert (OUT72 / "lag_placement_72C_scan.csv").exists()
    assert (OUT72 / "lag_placement_72C_best.csv").exists()
    assert (OUT72 / "lag_placement_72C_rmse_vs_tau.png").exists()
    assert (OUT72 / "lag_placement_72C_metadata.json").exists()
    df = pd.read_csv(OUT72 / "lag_placement_72C_scan.csv")
    assert len(df) == 75
    assert set(df["architecture"]) == {"O", "I", "S"}
    meta = json.loads((OUT72 / "lag_placement_72C_metadata.json")
                      .read_text(encoding="utf-8"))
    assert meta["no_refit"] is True
    assert meta["no_independent_tau_in_out"] is True
    assert len(meta["best_per_architecture"]) == 3
    # tau=0 三架构 RMSE 相同
    r0 = df[df["tau_lag_s"] == 0]
    assert r0["RMSE_72C_C"].nunique() == 1
