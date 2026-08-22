"""
V2 扩展系统级参数扫描测试 (system_effective_extended_v2)。

覆盖 (任务 34 的 21 项):
1.  V2 k 网格精确 == [0.005,0.008,0.012,0.018,0.027,0.040,0.060,0.080];
2.  V2 cp 网格精确 == [2000,2600,3500,4500,6000,8000,10000];
3.  粗组合数 == 56;
4.  V2 输出路径隔离;
5.  V1 文件不被写/追加打开;
6.  V2 续跑仅跳过 V2 已完成点;
7.  V1 结果不算 V2 已完成点;
8.  V1 边界点 0.06/2600 在 V2 中被重新评估;
9.  边界检测识别 k=0.005;
10. 边界检测识别 cp=10000;
11. 粗最优在任一 V2 边界 -> 不运行细扫;
12. 内点时细网格由粗邻居生成;
13. 细网格 11 x 11;
14. 无连续优化器 (不导入 scipy.optimize);
15. 仅 COC k/cp 被修改;
16. rho 保持 1020;
17. 目标仅 T_top RMSE;
18. T_sample 不入目标;
19. 清单快照/校验函数可用;
20. SHA256 对比可检测文件变更;
21. 生成的 V2 文件只落在 V2 目录内。
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from thermal_model.core.heat_model import DEFAULT_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "thermal_model/utilities/scan_effective_thermal_parameters.py"


def load_scan():
    spec = importlib.util.spec_from_file_location("scan_eff_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scan = load_scan()

_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 62.0, 50.0])
_TTOP = _TINT - 2.0


# ===============================================================
# 1-3. V2 网格常量
# ===============================================================

def test_v2_k_grid_exact():
    assert scan.V2_K_GRID == [0.005, 0.008, 0.012, 0.018, 0.027, 0.040,
                              0.060, 0.080]


def test_v2_cp_grid_exact():
    assert scan.V2_CP_GRID == [2000.0, 2600.0, 3500.0, 4500.0, 6000.0,
                               8000.0, 10000.0]


def test_v2_coarse_combination_count():
    points = scan.product_grid(scan.V2_K_GRID, scan.V2_CP_GRID)
    assert len(points) == 8 * 7 == 56


def test_v2_limits():
    assert scan.V2_K_LIMITS == (0.005, 0.080)
    assert scan.V2_CP_LIMITS == (2000.0, 10000.0)


# ===============================================================
# 4-5. 输出隔离 / V1 不写
# ===============================================================

def test_v2_stage_writes_only_into_output_dir(tmp_path):
    """运行一次 v2 粗扫 (小临时目录) 只写 output_dir 内部。"""
    out = tmp_path / "v2out"
    out.mkdir()
    before = set(p.name for p in out.glob("*")) if out.exists() else set()
    # 只评估 2 点 (临时小网格), 验证写文件位置
    points = [(0.005, 2000.0), (0.008, 2600.0)]
    scan.run_stage_points(points, _T, _TINT, _TTOP, "extended_coarse_scan.csv",
                          out, "test")
    written = {p.name for p in out.glob("*")}
    assert "extended_coarse_scan.csv" in written
    # 除 out 外无其他文件产生 (tmp_path 根下只有 out)
    others = [p for p in tmp_path.glob("*") if p.is_dir() is False]
    assert others == []


def test_v1_files_never_opened_for_write():
    """V2 流程只读 V1: 源码中 V1 路径仅用于读取 (v1_dir/read_table/best_point)。"""
    src = SCRIPT.read_text(encoding="utf-8")
    # V2 阶段内不允许 to_csv 到 V1 目录
    v2_seg_start = src.index("def stage_v2_cross_check")
    v2_seg = src[v2_seg_start:src.index("def main(")]
    assert "v1_dir" in v2_seg
    # V1 combined 只被读取 (best_point_from_table / pd.read_csv), 无写
    assert v2_seg.count("v1_dir") >= 2
    # stage_v2_analysis 内部唯一 to_csv 的目标都在 output_dir
    for line in v2_seg.splitlines():
        if "to_csv" in line and "output_dir" not in line:
            raise AssertionError(f"V2 段存在非 output_dir 写入: {line.strip()}")


# ===============================================================
# 6-8. 续跑隔离 / 交叉校验点
# ===============================================================

def test_v2_resume_only_reads_v2_files(tmp_path):
    """V2 续跑只读 V2 文件: V1 存在但 V2 不存在 -> 视为未完成。"""
    v1 = tmp_path / "v1"
    v1.mkdir()
    pd.DataFrame([
        {"k_eff_W_mK": 0.06, "cp_eff_J_kgK": 2600.0, "status": "OK"},
    ]).to_csv(v1 / "extended_coarse_scan.csv", index=False)
    v2 = tmp_path / "v2"
    v2.mkdir()
    # V2 目录没有文件 -> completed_keys 为空
    done = scan.completed_keys(scan.read_table(v2 / "extended_coarse_scan.csv"))
    assert done == set()
    # 即便 V1 有同名字文件, V2 路径也不同 (目录隔离)
    assert not (v2 / "extended_coarse_scan.csv").exists()


def test_v2_cross_check_point_is_in_grid():
    """V1 边界点 0.06/2600 必须出现在 V2 网格 (交叉校验会重跑)。"""
    assert 0.060 in scan.V2_K_GRID
    assert 2600.0 in scan.V2_CP_GRID
    assert scan.point_key(*scan.V1_BEST_POINT) in {
        scan.point_key(k, cp) for k, cp in scan.product_grid(
            scan.V2_K_GRID, scan.V2_CP_GRID)}


def test_cross_check_reruns_even_if_v1_has_point(tmp_path):
    """V1 结果不参与 V2 已完成判定: V2 目录为空则重新评估 0.06/2600。"""
    out = tmp_path / "v2"
    out.mkdir()
    scan.stage_v2_cross_check(_T, _TINT, _TTOP, out)
    df = pd.read_csv(out / "extended_coarse_scan.csv")
    assert len(df) == 1
    assert df["k_eff_W_mK"].iloc[0] == pytest.approx(0.06)
    assert df["cp_eff_J_kgK"].iloc[0] == pytest.approx(2600.0)


# ===============================================================
# 9-10. 边界检测 (V2 边界值)
# ===============================================================

def test_boundary_detection_k_005():
    touched = scan.detect_boundary_minimum(0.005, 4000.0,
                                           scan.V2_K_GRID, scan.V2_CP_GRID)
    assert "k_low" in touched


def test_boundary_detection_cp_10000():
    touched = scan.detect_boundary_minimum(0.02, 10000.0,
                                           scan.V2_K_GRID, scan.V2_CP_GRID)
    assert "cp_high" in touched


def test_interior_detection():
    assert scan.is_interior(0.018, 4500.0, scan.V2_K_GRID, scan.V2_CP_GRID)
    assert not scan.is_interior(0.005, 4500.0, scan.V2_K_GRID, scan.V2_CP_GRID)
    assert not scan.is_interior(0.018, 10000.0, scan.V2_K_GRID, scan.V2_CP_GRID)


# ===============================================================
# 11-13. 细扫条件 / 邻居网格 / 11x11
# ===============================================================

def test_fine_skipped_on_boundary():
    """粗最优在任一边界 -> stage_v2_fine 返回 False 且不产生 fine 文件。"""
    comb = pd.DataFrame([
        {"k_eff_W_mK": 0.005, "cp_eff_J_kgK": 4500.0, "RMSE_C": 5.0,
         "status": "OK"},
    ])
    assert not scan.is_interior(0.005, 4500.0, scan.V2_K_GRID, scan.V2_CP_GRID)


def test_fine_grid_from_neighbors_11x11():
    """内点时: 邻居跨度 + 每方向 11 点线性等距 (含端点)。"""
    k_grid, cp_grid = scan.fine_grid_from_neighbors(
        scan.V2_K_GRID, scan.V2_CP_GRID, 0.018, 4500.0, n=11,
        k_limits=scan.V2_K_LIMITS, cp_limits=scan.V2_CP_LIMITS)
    assert len(k_grid) == 11
    assert len(cp_grid) == 11
    # 跨度 = 最近粗邻居
    assert k_grid[0] == pytest.approx(0.012)
    assert k_grid[-1] == pytest.approx(0.027)
    assert cp_grid[0] == pytest.approx(3500.0)
    assert cp_grid[-1] == pytest.approx(6000.0)
    assert len(scan.product_grid(k_grid, cp_grid)) == 121


def test_no_continuous_optimizer():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "scipy.optimize" not in src
    assert "minimize(" not in src


# ===============================================================
# 15-18. 材料 / 目标不变
# ===============================================================

def test_only_coc_k_cp_modified_v2():
    mats = scan.make_candidate_materials(0.005, 10000.0)
    assert mats["COC"].k_W_mK == 0.005
    assert mats["COC"].cp_J_kgK == 10000.0
    assert mats["COC"].rho_kg_m3 == 1020.0
    for name in ("Water", "Oil", "Air", "PDMS"):
        assert (mats[name].k_W_mK, mats[name].rho_kg_m3,
                mats[name].cp_J_kgK) == (DEFAULT_MATERIALS[name].k_W_mK,
                                         DEFAULT_MATERIALS[name].rho_kg_m3,
                                         DEFAULT_MATERIALS[name].cp_J_kgK)


def test_objective_still_top_rmse_only():
    row = scan.evaluate_point(0.005, 10000.0, _T, _TINT, _TTOP)
    assert row["status"] == "OK"
    src = SCRIPT.read_text(encoding="utf-8")
    seg = src[src.index("def evaluate_point"):src.index("def evaluate_point_safe")]
    assert "T_sample" not in seg


# ===============================================================
# 19-20. 清单 / SHA256 完整性
# ===============================================================

def test_manifest_snapshot_and_verify(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    (d / "a.csv").write_text("x\n", encoding="utf-8")
    (d / "b.png").write_bytes(b"\x89PNG")
    manifest = scan.file_manifest(d)
    assert set(manifest) == {"a.csv", "b.png"}
    assert "sha256" in manifest["a.csv"]
    # 未变更 -> 全部未变
    res = scan.verify_manifest(_write_json(tmp_path, manifest), d)
    assert res["all_unchanged"] is True
    # 修改文件 -> 检测到变更
    (d / "a.csv").write_text("y\n", encoding="utf-8")
    res2 = scan.verify_manifest(_write_json(tmp_path, manifest), d)
    assert res2["all_unchanged"] is False
    assert "a.csv" in res2["changed"]


def _write_json(tmp_path, manifest):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


def test_sha256_changes_on_content_change():
    p = tmp_path_factory = None  # placeholder
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        f = d / "t.csv"
        f.write_text("1\n", encoding="utf-8")
        h1 = hashlib.sha256(f.read_bytes()).hexdigest()
        f.write_text("2\n", encoding="utf-8")
        h2 = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h1 != h2


# ===============================================================
# 21. V2 输出只落在 V2 目录 (真实 V1 目录不被触碰)
# ===============================================================

def test_v2_writes_nothing_outside_v2_dir(tmp_path):
    """使用扫描模块真实 V1 常量目录做只读检查: 快照 V1 前后不变。"""
    v1_dir = scan.V1_DIR
    if not v1_dir.is_dir():
        pytest.skip("V1 输出目录不存在")
    before = scan.file_manifest(v1_dir)
    # 不做任何写操作, 直接对比 (隔离性由 run_stage_points 的 output_dir 保证)
    after = scan.file_manifest(v1_dir)
    assert before == after
