#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D k_eff–cp_eff RMSE 景观扫描 (系统级有效热参数标定)。

模型目的
--------
标定的是 SYSTEM-LEVEL EFFECTIVE 热参数 (k_eff, cp_eff), 允许吸收:
    实际 COC 材料性质不确定度 / 内部传感器与顶部测温的动态与不确定度 /
    热接触效应 / 1D 简化几何 / 其他未建模热效应。
NOT 被解释为 TOPAS COC 的固有材料常数。

物理 (全部复用 heat_model 唯一权威求解器, 本脚本不包含任何 FDM 方程):
    几何            : BARE_TOP_COC_LAYERS (850 um, 裸顶)
    底部边界        : 实测 T_internal(t) (Dirichlet, 直接边界)
    拟合目标        : T_top_surface_predicted vs T_top_measured (x=850 um)
    目标函数        : RMSE = sqrt(mean((T_top_pred - T_top_meas)^2)),
                      全部实测时间点等权, 无时间平移 / 无加权重采样
    初始条件        : auto = 第一个实测内部温度 (每个候选点动态解析)
    样品平均        : 修正版控制体积空间加权 (仅作模型输出, 不进目标)

固定参数
--------
    rho_COC = 1020 kg/m^3 (不拟合)
    Water / Oil 材料性质不变
    h_conv = 5.0 W/(m^2 K); T_air_ambient = 25.0 C
    save_dt = 0.1 s; 网格 / 时间步长不变

扫描流程 (可续跑)
----------------
    baseline : 0.13 / 1800
    reference: 0.14 / 1400 (+ 基准计时)
    coarse   : k 9 值 x cp 8 值 = 72 点
    extend   : 若粗网格最优落在边界, 仅沿触碰方向以粗步长扩展,
               直到最优成为内点或到达诊断边界 (k 0.06-0.30, cp 600-2600)
    fine     : 围绕粗最优的局部细网格 (dk=0.004, dcp=50)

每个点完成后立即写 CSV (可续跑); 不做 scipy 优化; 不做独立验证;
不做 k(T)/cp(T); 不拟合 T_sample / h_conv / rho / 时间平移。
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker

import heat_model

# =============================================================
# 常量
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ALIGNED_CSV = (
    PROJECT_ROOT / "temperature_alignment_output" / "72C"
    / "aligned_internal_top_temperature.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "parameter_scan_output" / "72C"

H_CONV = 5.0
T_AMB = 25.0
SAVE_DT = 0.1
RHO_COC = 1020.0

K_GRID_COARSE = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
CP_GRID_COARSE = [800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0]

K_LIMITS = (0.06, 0.30)
CP_LIMITS = (600.0, 2600.0)
K_STEP_COARSE = 0.02
CP_STEP_COARSE = 200.0
FINE_K_STEP = 0.004
FINE_CP_STEP = 50.0
FINE_K_HALF = 0.02
FINE_CP_HALF = 200.0

BASELINE = (0.13, 1800.0)
REFERENCE = (0.14, 1400.0)

# -------------------------------------------------------------
# V2 扩展分析 (system_effective_extended_v2) —— 更宽的系统级有效参数区域
# -------------------------------------------------------------
# k 近似对数覆盖 (0.005..0.080); cp 覆盖 (2000..10000)。
# 这些是 SYSTEM-LEVEL EFFECTIVE 参数, 允许远低于聚合物文献值。
V2_K_GRID = [0.005, 0.008, 0.012, 0.018, 0.027, 0.040, 0.060, 0.080]
V2_CP_GRID = [2000.0, 2600.0, 3500.0, 4500.0, 6000.0, 8000.0, 10000.0]
V2_K_LIMITS = (0.005, 0.080)
V2_CP_LIMITS = (2000.0, 10000.0)

# V1 最优边界点 (交叉校验参考)
V1_BEST_POINT = (0.060, 2600.0)
V1_BEST_RMSE = 15.4314

TIME_COL = "time_s"
TINT_COL = "T_internal_interpolated_C"
TTOP_COL = "T_top_measured_C"


# =============================================================
# 数据加载 / 候选材料 / 指标
# =============================================================

def load_experiment(path=None):
    """加载对齐实验: 返回 (t, T_internal, T_top_measured)。"""
    p = Path(path) if path else ALIGNED_CSV
    if not p.is_file():
        raise FileNotFoundError(
            f"对齐数据集不存在: {p}. 请先用 "
            "align_internal_and_top_temperature.py 生成。"
        )
    df = pd.read_csv(p)
    for col in (TIME_COL, TINT_COL, TTOP_COL):
        if col not in df.columns:
            raise KeyError(f"对齐数据集缺少列 {col!r}; 实际列: {list(df.columns)}")
    t = df[TIME_COL].to_numpy(dtype=float)
    t_int = df[TINT_COL].to_numpy(dtype=float)
    t_top = df[TTOP_COL].to_numpy(dtype=float)
    if not (np.all(np.diff(t) > 0) and len(t) >= 2):
        raise ValueError("实验时间轴必须严格递增且至少 2 点。")
    if not (np.all(np.isfinite(t_int)) and np.all(np.isfinite(t_top))):
        raise ValueError("实验温度列必须为有限数值。")
    return t, t_int, t_top


def make_candidate_materials(k_eff, cp_eff):
    """返回候选材料库: 仅替换 COC 的 k/cp (rho 保持 1020), 其余逐位不变。"""
    mats = heat_model.copy_default_materials()
    coc = mats["COC"]
    mats["COC"] = heat_model.Material(
        name=coc.name, k_W_mK=float(k_eff),
        rho_kg_m3=coc.rho_kg_m3, cp_J_kgK=float(cp_eff),
    )
    return mats


def compute_metrics(t, residual):
    """由残差序列计算 RMSE / MAE / mean / max abs / max pos / max neg。"""
    r = np.asarray(residual, dtype=float)
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(r)
    if not finite.any():
        return {
            "rmse": np.nan, "mae": np.nan, "mean_residual": np.nan,
            "max_abs_error": np.nan, "max_positive": np.nan,
            "max_negative": np.nan,
        }
    r = r[finite]
    i = int(np.argmax(np.abs(r)))
    return {
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "mae": float(np.mean(np.abs(r))),
        "mean_residual": float(np.mean(r)),
        "max_abs_error": float(np.max(np.abs(r))),
        "max_positive": float(np.max(r)),
        "max_negative": float(np.min(r)),
        "time_of_max_abs": float(t[finite][i]),
    }


def point_key(k_eff, cp_eff):
    """确定性参数点键 (量化到 1e-6, 避免浮点表示导致的重复)。"""
    return f"{float(k_eff):.6f}|{float(cp_eff):.6f}"


# =============================================================
# 单点评估
# =============================================================

def evaluate_point(k_eff, cp_eff, t_proto, t_int, t_top_meas,
                   h_conv=H_CONV, t_amb=T_AMB, save_dt=SAVE_DT,
                   return_result=False):
    """对单个 (k_eff, cp_eff) 运行完整 FDM 并计算目标指标。

    顶部预测 = T_top_surface_arr 线性插值到实测 TIME 坐标 (t_proto; 无时间平移)。
    注意: 查询轴必须是实测时间 t_proto, 而不是实测温度值 t_top_meas
    (旧版本误用 t_top_meas 作为查询点, 已修正)。
    初始条件 = auto = 第一个内部温度。
    返回 dict (含所有指标 + runtime + n_points + dt)。
    """
    t0 = time.perf_counter()
    mats = make_candidate_materials(k_eff, cp_eff)
    result = heat_model.run_simulation(
        time_s=t_proto,
        bottom_temperature_C=t_int,
        materials=mats,
        layers=heat_model.BARE_TOP_COC_LAYERS,
        h_conv=h_conv, T_air_ambient=t_amb, save_dt=save_dt,
        T_initial_C=float(t_int[0]),
    )
    wall = time.perf_counter() - t0
    t_arr = result["t_array"]
    # 修正: 在实测时间坐标上插值 (np.interp(实测时间, FDM时间, 预测温度))
    T_top_pred = np.interp(t_proto, t_arr, result["T_top_surface_arr"])
    metrics = compute_metrics(t_proto, T_top_pred - t_top_meas)
    out = {
        "k_eff_W_mK": float(k_eff),
        "cp_eff_J_kgK": float(cp_eff),
        "RMSE_C": metrics["rmse"],
        "MAE_C": metrics["mae"],
        "mean_residual_C": metrics["mean_residual"],
        "max_abs_error_C": metrics["max_abs_error"],
        "max_positive_residual_C": metrics["max_positive"],
        "max_negative_residual_C": metrics["max_negative"],
        "runtime_s": wall,
        "n_measurement_points": int(len(t_top_meas)),
        "dt_FDM_s": float(result["dt"]),
        "status": "OK",
        "error": "",
    }
    if return_result:
        return out, result
    return out


def evaluate_point_safe(k_eff, cp_eff, t_proto, t_int, t_top_meas, **kw):
    """evaluate_point 的容错包装: 失败时记录 status=FAILED + error。"""
    try:
        return evaluate_point(k_eff, cp_eff, t_proto, t_int, t_top_meas, **kw)
    except Exception as exc:  # noqa: BLE001
        return {
            "k_eff_W_mK": float(k_eff),
            "cp_eff_J_kgK": float(cp_eff),
            "RMSE_C": np.nan, "MAE_C": np.nan,
            "mean_residual_C": np.nan, "max_abs_error_C": np.nan,
            "max_positive_residual_C": np.nan, "max_negative_residual_C": np.nan,
            "runtime_s": np.nan, "n_measurement_points": int(len(t_top_meas)),
            "dt_FDM_s": np.nan, "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }


# =============================================================
# 网格 / 续跑 / CSV
# =============================================================

def product_grid(k_values, cp_values):
    """(k, cp) 全部组合, 按 k 再 cp 确定性排序。"""
    return [(float(k), float(cp))
            for k in sorted(k_values) for cp in sorted(cp_values)]


def read_table(path):
    """读已有扫描表; 不存在返回空 DataFrame。"""
    if Path(path).is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def completed_keys(df):
    """已成功完成 (status==OK) 的参数点键集合。"""
    if df.empty or "status" not in df.columns:
        return set()
    ok = df[df["status"] == "OK"]
    return {point_key(r.k_eff_W_mK, r.cp_eff_J_kgK)
            for r in ok.itertuples()}


def append_rows(path, rows):
    """把新行追加写入 CSV (文件不存在则建表头)。"""
    new = pd.DataFrame(rows)
    if Path(path).is_file():
        new.to_csv(path, mode="a", header=False, index=False)
    else:
        new.to_csv(path, index=False)


def run_stage_points(points, t_proto, t_int, t_top_meas, stage_file,
                     output_dir, label, skip_existing=True):
    """顺序评估 points (跳过已完成), 每点立即持久化, 打印进度与 ETA。"""
    out_path = output_dir / stage_file
    done = completed_keys(read_table(out_path)) if skip_existing else set()
    total = len(points)
    t_start_all = time.perf_counter()
    completed_now = 0
    eta_remaining = None
    for i, (k, cp) in enumerate(points, start=1):
        key = point_key(k, cp)
        if skip_existing and key in done:
            print(f"[skip {i}/{total}] k={k} cp={cp} (已完成)")
            continue
        t0 = time.perf_counter()
        row = evaluate_point_safe(k, cp, t_proto, t_int, t_top_meas)
        wall = time.perf_counter() - t0
        append_rows(out_path, [row])
        done.add(key)
        completed_now += 1
        rmse_s = f"{row['RMSE_C']:.4f}" if np.isfinite(row["RMSE_C"]) else "nan"
        if completed_now >= 1:
            eta = (time.perf_counter() - t_start_all) / completed_now * (
                total - i
            )
            eta_s = f"{eta/60:.1f} min"
        else:
            eta_s = "n/a"
        print(f"[{label} {i}/{total}] k={k:.4f} cp={cp:.0f} "
              f"RMSE={rmse_s} C  runtime={wall:.1f} s  ETA={eta_s}")
    print(f"[{label}] 完成: {total} 点, 新评估 {completed_now} 点, "
          f"跳过 {total - completed_now} 点 (含已存在)。")
    return out_path


# =============================================================
# 边界检测 / 扩展 / 细网格
# =============================================================

def detect_boundary_minimum(best_k, best_cp, k_grid, cp_grid):
    """返回最优触碰的边界集合 ('k_low'/'k_high'/'cp_low'/'cp_high')。"""
    touched = set()
    if best_k == min(k_grid):
        touched.add("k_low")
    if best_k == max(k_grid):
        touched.add("k_high")
    if best_cp == min(cp_grid):
        touched.add("cp_low")
    if best_cp == max(cp_grid):
        touched.add("cp_high")
    return touched


def best_point_from_table(df):
    """已完成 OK 点中 RMSE 最小的行 (NaN RMSE 忽略)。"""
    ok = df[(df["status"] == "OK") & df["RMSE_C"].notna()]
    if ok.empty:
        return None
    idx = ok["RMSE_C"].idxmin()
    return ok.loc[idx]


def is_interior(best_k, best_cp, k_vals, cp_vals):
    """最优是否在给定 k/cp 网格内部 (不在首/末值)。"""
    return (min(k_vals) < best_k < max(k_vals)
            and min(cp_vals) < best_cp < max(cp_vals))


def fine_grid_from_neighbors(k_vals, cp_vals, best_k, best_cp, n=11,
                             k_limits=None, cp_limits=None):
    """围绕最优的局部细网格: 跨最近邻居 k_left..k_right / cp_low..cp_high,
    每个方向 n 个线性等距点 (含端点), 裁剪到诊断边界。

    用于 V2: 粗网格 k 近似对数/不规则, 因此用邻居跨度而非固定步长。
    """
    ks = sorted(set(float(v) for v in k_vals))
    cps = sorted(set(float(v) for v in cp_vals))
    ki = ks.index(float(best_k))
    ci = cps.index(float(best_cp))
    k_lo = ks[ki - 1] if ki > 0 else float(best_k)
    k_hi = ks[ki + 1] if ki < len(ks) - 1 else float(best_k)
    cp_lo = cps[ci - 1] if ci > 0 else float(best_cp)
    cp_hi = cps[ci + 1] if ci < len(cps) - 1 else float(best_cp)
    k_grid = np.round(np.linspace(k_lo, k_hi, n), 12).tolist()
    cp_grid = np.round(np.linspace(cp_lo, cp_hi, n), 12).tolist()
    if k_limits is not None:
        k_grid = [v for v in k_grid
                  if k_limits[0] - 1e-12 <= v <= k_limits[1] + 1e-12]
    if cp_limits is not None:
        cp_grid = [v for v in cp_grid
                   if cp_limits[0] - 1e-12 <= v <= cp_limits[1] + 1e-12]
    return k_grid, cp_grid


def file_manifest(directory):
    """目录内全部文件的 {name: {size, mtime, sha256}} 快照 (V1 完整性校验)。"""
    d = Path(directory)
    out = {}
    for p in sorted(d.iterdir()):
        if p.is_file():
            out[p.name] = {
                "size": int(p.stat().st_size),
                "mtime": float(p.stat().st_mtime),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
    return out


def verify_manifest(before_path, directory):
    """对比 before 清单与当前目录, 返回变更文件列表。"""
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = file_manifest(directory)
    changed = []
    for name, info in before.items():
        if after.get(name) != info:
            changed.append(name if name in after else name + " (MISSING)")
    return {
        "files_checked": sorted(before),
        "changed": changed,
        "all_unchanged": not changed,
    }


def clipped_fine_grid(best_k, best_cp, k_limits, cp_limits):
    """围绕最优的内点细网格 (dk=0.004, dcp=50), 裁剪到诊断边界。"""
    k_grid = np.round(np.arange(best_k - FINE_K_HALF,
                                best_k + FINE_K_HALF + 1e-12,
                                FINE_K_STEP), 12)
    cp_grid = np.round(np.arange(best_cp - FINE_CP_HALF,
                                 best_cp + FINE_CP_HALF + 1e-12,
                                 FINE_CP_STEP), 12)
    k_grid = [float(x) for x in k_grid
              if k_limits[0] - 1e-12 <= x <= k_limits[1] + 1e-12]
    cp_grid = [float(x) for x in cp_grid
               if cp_limits[0] - 1e-12 <= x <= cp_limits[1] + 1e-12]
    return k_grid, cp_grid


# =============================================================
# 近最优区域 / 剖面
# =============================================================

def near_optimal_regions(df, rmse_min, thresholds=(0.01, 0.02, 0.05)):
    """Delta_RMSE = RMSE - RMSE_min; 返回各阈值内的点集统计。

    仅作 near-optimal RMSE region 描述, 不是统计置信区间。
    """
    ok = df[(df["status"] == "OK") & df["RMSE_C"].notna()].copy()
    if ok.empty or not np.isfinite(rmse_min):
        return {}
    ok["delta_rmse"] = ok["RMSE_C"] - rmse_min
    out = {}
    for th in thresholds:
        # 加 1e-12 容差: 避免 1.05-1.0=0.05000000000000004 之类的浮点边界问题
        sub = ok[ok["delta_rmse"] <= rmse_min * th + 1e-12]
        out[f"{th*100:.0f}pct"] = {
            "n_points": int(len(sub)),
            "k_min": float(sub["k_eff_W_mK"].min()) if len(sub) else None,
            "k_max": float(sub["k_eff_W_mK"].max()) if len(sub) else None,
            "cp_min": float(sub["cp_eff_J_kgK"].min()) if len(sub) else None,
            "cp_max": float(sub["cp_eff_J_kgK"].max()) if len(sub) else None,
        }
    return out


def profile_minima(df):
    """k / cp 方向的最小 RMSE 剖面 (仅 OK 点)。"""
    ok = df[(df["status"] == "OK") & df["RMSE_C"].notna()]
    prof_k = (ok.groupby("k_eff_W_mK")["RMSE_C"]
              .min().sort_index().reset_index())
    prof_cp = (ok.groupby("cp_eff_J_kgK")["RMSE_C"]
               .min().sort_index().reset_index())
    return prof_k, prof_cp


# =============================================================
# 绘图
# =============================================================

def _grid_matrix(df, value_col="RMSE_C"):
    """把 (k, cp, value) 行集转成结构化矩阵 (缺位 NaN)。"""
    k_vals = sorted(df["k_eff_W_mK"].unique())
    cp_vals = sorted(df["cp_eff_J_kgK"].unique())
    Z = np.full((len(cp_vals), len(k_vals)), np.nan)
    for r in df.itertuples():
        ki = k_vals.index(r.k_eff_W_mK)
        ci = cp_vals.index(r.cp_eff_J_kgK)
        Z[ci, ki] = getattr(r, value_col)
    return np.asarray(k_vals), np.asarray(cp_vals), Z


def plot_landscape(df, title, path, best=None, markers=(),
                   value_col="RMSE_C", cmap="viridis_r"):
    """2D RMSE 景观: 填充等高线 + 标注最优与参考点。"""
    if df.empty:
        return False
    k_vals, cp_vals, Z = _grid_matrix(df, value_col)
    fig, ax = plt.subplots(figsize=(10, 7))
    cf = ax.contourf(k_vals, cp_vals, Z, levels=24, cmap=cmap)
    ax.contour(k_vals, cp_vals, Z, levels=12, colors="k", linewidths=0.4,
               alpha=0.5)
    fig.colorbar(cf, ax=ax, label="RMSE [°C]")
    ax.set_xlabel("k_eff [W/(m·K)]")
    ax.set_ylabel("cp_eff [J/(kg·K)]")
    ax.set_title(title)
    if best is not None:
        ax.plot(best[0], best[1], "r*", ms=16, label="best grid point")
    for (mk, mcp, mlab) in markers:
        ax.plot(mk, mcp, "o", ms=8, mfc="none", label=mlab)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_landscape_v2(df, value_col, title, path, best=None, markers=(),
                      log_k=True, cmap="viridis_r"):
    """V2 景观: 保留实际非均匀 k 坐标, x 轴用对数 (k 跨一个数量级以上)。"""
    if df.empty:
        return False
    k_vals, cp_vals, Z = _grid_matrix(df, value_col)
    fig, ax = plt.subplots(figsize=(10, 7))
    cf = ax.contourf(k_vals, cp_vals, Z, levels=24, cmap=cmap)
    ax.contour(k_vals, cp_vals, Z, levels=12, colors="k", linewidths=0.4,
               alpha=0.5)
    label = {"RMSE_C": "RMSE [°C]", "mean_residual_C": "mean residual [°C]"}
    fig.colorbar(cf, ax=ax, label=label.get(value_col, value_col))
    ax.set_xlabel("k_eff [W/(m·K)]")
    ax.set_ylabel("cp_eff [J/(kg·K)]")
    if log_k:
        ax.set_xscale("log")
        ax.set_xticks(k_vals)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title(title)
    if best is not None:
        ax.plot(best[0], best[1], "r*", ms=16, label="V2 best grid point")
    for (mk, mcp, mlab) in markers:
        ax.plot(mk, mcp, "o", ms=8, mfc="none", label=mlab)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_profiles(prof_k, prof_cp, path_k, path_cp):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(prof_k["k_eff_W_mK"], prof_k["RMSE_C"], "o-")
    ax[0].set_xlabel("k_eff [W/(m·K)]"); ax[0].set_ylabel("min RMSE [°C]")
    ax[0].set_title("RMSE profile vs k (min over cp)")
    ax[1].plot(prof_cp["cp_eff_J_kgK"], prof_cp["RMSE_C"], "o-")
    ax[1].set_xlabel("cp_eff [J/(kg·K)]"); ax[1].set_ylabel("min RMSE [°C]")
    ax[1].set_title("RMSE profile vs cp (min over k)")
    fig.tight_layout()
    fig.savefig(path_k)
    plt.close(fig)


def plot_best_trace(t_meas, t_int, t_top_meas, t_arr, t_top_pred, t_sample,
                    path):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t_meas, t_int, color="#7f7f7f", lw=1.2, ls=":",
            label="Measured internal T (boundary)")
    ax.plot(t_meas, t_top_meas, color="#d62728", lw=1.5,
            label="Measured Top COC surface T")
    ax.plot(t_arr, t_top_pred, color="#1f77b4", lw=2,
            label="Best-fit predicted Top COC surface T")
    ax.plot(t_arr, t_sample, color="#2ca02c", lw=1.5, ls="--",
            label="Predicted sample T (estimated, not measured)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("Best-fit system-level effective model vs 72°C experiment")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_residuals(t_meas, residual, path, regimes=None):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t_meas, residual, color="#1f77b4", lw=1.0)
    ax.axhline(0, color="k", lw=0.8)
    if regimes is not None:
        for reg, color in (("TRANSIENT_HEATING", "r"),
                           ("TRANSIENT_COOLING", "b"),
                           ("SETTLING", "g"), ("TRANSITION_OTHER", "k")):
            mask = regimes == reg
            if mask.any():
                ax.scatter(t_meas[mask], residual[mask], s=6, color=color,
                           label=reg, alpha=0.7)
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Residual (pred - meas) (°C)")
    ax.set_title("Best-fit top-surface residual vs time")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =============================================================
# 元数据
# =============================================================

def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_metadata(extra=None):
    meta = {
        "input_dataset_path": str(ALIGNED_CSV),
        "git_head": git_head(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry_preset": "BARE_TOP_COC_LAYERS",
        "fixed_rho_COC_kg_m3": RHO_COC,
        "h_conv_W_m2K": H_CONV,
        "T_air_ambient_C": T_AMB,
        "initial_condition_mode": "auto (first T_internal value per run)",
        "coarse_k_range": [min(K_GRID_COARSE), max(K_GRID_COARSE)],
        "coarse_cp_range": [min(CP_GRID_COARSE), max(CP_GRID_COARSE)],
        "diagnostic_limits": {"k_eff": list(K_LIMITS),
                              "cp_eff": list(CP_LIMITS)},
        "fine_steps": {"k_eff": FINE_K_STEP, "cp_eff": FINE_CP_STEP},
        "fitting_objective": (
            "RMSE of T_top_surface_predicted (interpolated to measured "
            "times) vs T_top_measured; equal weight, no time shift, "
            "no sample-temperature term"
        ),
        "solver_save_dt_s": SAVE_DT,
        "note": (
            "k_eff/cp_eff are SYSTEM-LEVEL EFFECTIVE thermal parameters, "
            "NOT intrinsic COC material constants."
        ),
    }
    if extra:
        meta.update(extra)
    (OUTPUT_DIR / "parameter_scan_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


# =============================================================
# 主流程
# =============================================================

def load_regime_labels():
    path = PROJECT_ROOT / "temperature_regime_output" / "72C" \
        / "temperature_regime_labeled.csv"
    if path.is_file():
        df = pd.read_csv(path)
        return df["regime"].to_numpy(dtype=str)
    return None


def build_combined_v2(output_dir):
    """合并 V2 阶段表 -> extended_combined_scan.csv (只读 V2 目录, 去重排序)。

    只读取 V2 自己的文件: V1 结果绝不参与 V2 已完成点判定。
    """
    frames = []
    for name in ("extended_coarse_scan.csv", "extended_fine_scan.csv"):
        p = output_dir / name
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    comb = pd.concat(frames, ignore_index=True)
    comb = comb.drop_duplicates(
        subset=["k_eff_W_mK", "cp_eff_J_kgK"], keep="first"
    )
    comb = comb.sort_values(["k_eff_W_mK", "cp_eff_J_kgK"],
                            ignore_index=True)
    comb.to_csv(output_dir / "extended_combined_scan.csv", index=False)
    return comb


def build_combined(output_dir):
    """合并所有阶段表 -> combined_parameter_scan.csv (去重, 确定性排序)。"""
    frames = []
    for name in ("baseline_scan.csv", "reference_scan.csv", "coarse_scan.csv",
                 "extend_scan.csv", "fine_scan.csv"):
        p = output_dir / name
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    comb = pd.concat(frames, ignore_index=True)
    comb = comb.drop_duplicates(
        subset=["k_eff_W_mK", "cp_eff_J_kgK"], keep="first"
    )
    comb = comb.sort_values(["k_eff_W_mK", "cp_eff_J_kgK"],
                            ignore_index=True)
    comb.to_csv(output_dir / "combined_parameter_scan.csv", index=False)
    return comb


def stage_baseline_reference(t_proto, t_int, t_top_meas, output_dir):
    """基线 0.13/1800 与参考 0.14/1400 (+ 基准计时)。"""
    for (k, cp), fname in ((BASELINE, "baseline_scan.csv"),
                           (REFERENCE, "reference_scan.csv")):
        out = output_dir / fname
        done = completed_keys(read_table(out))
        if point_key(k, cp) in done:
            print(f"[baseline/reference] 已存在: k={k} cp={cp}")
            continue
        t0 = time.perf_counter()
        row = evaluate_point_safe(k, cp, t_proto, t_int, t_top_meas)
        wall = time.perf_counter() - t0
        append_rows(out, [row])
        print(f"[{fname}] k={k} cp={cp} RMSE={row['RMSE_C']:.4f} C "
              f"MAE={row['MAE_C']:.4f} MaxAbs={row['max_abs_error_C']:.4f} "
              f"runtime={wall:.1f} s")


def stage_coarse(t_proto, t_int, t_top_meas, output_dir):
    points = product_grid(K_GRID_COARSE, CP_GRID_COARSE)
    return run_stage_points(points, t_proto, t_int, t_top_meas,
                            "coarse_scan.csv", output_dir, "coarse")


def stage_extend(t_proto, t_int, t_top_meas, output_dir):
    """边界扩展: 触碰方向逐线外推, 直到最优为内点或到达诊断边界。"""
    out = output_dir / "extend_scan.csv"
    done = completed_keys(read_table(out))
    k_grid = set(K_GRID_COARSE)
    cp_grid = set(CP_GRID_COARSE)
    added = 0
    while True:
        comb = build_combined(output_dir)
        best = best_point_from_table(comb)
        if best is None:
            print("[extend] 无可用最优, 跳过。")
            break
        bk, bcp = float(best["k_eff_W_mK"]), float(best["cp_eff_J_kgK"])
        touched = detect_boundary_minimum(bk, bcp, sorted(k_grid),
                                          sorted(cp_grid))
        if not touched:
            print(f"[extend] 最优 k={bk} cp={bcp} 为内点, 停止扩展。")
            break
        new_lines = []
        for edge in sorted(touched):
            if edge == "k_low" and min(k_grid) - K_STEP_COARSE >= K_LIMITS[0]:
                new_lines.append(("k", min(k_grid) - K_STEP_COARSE))
            elif edge == "k_high" and max(k_grid) + K_STEP_COARSE <= K_LIMITS[1]:
                new_lines.append(("k", max(k_grid) + K_STEP_COARSE))
            elif edge == "cp_low" and min(cp_grid) - CP_STEP_COARSE >= CP_LIMITS[0]:
                new_lines.append(("cp", min(cp_grid) - CP_STEP_COARSE))
            elif edge == "cp_high" and max(cp_grid) + CP_STEP_COARSE <= CP_LIMITS[1]:
                new_lines.append(("cp", max(cp_grid) + CP_STEP_COARSE))
        if not new_lines:
            print("[extend] 触碰边界但已达诊断极限, 停止扩展 "
                  "(最优未被当前物理合理区域包围)。")
            break
        for axis, val in new_lines:
            if axis == "k":
                k_grid.add(val)
                points = [(val, cp) for cp in sorted(cp_grid)]
            else:
                cp_grid.add(val)
                points = [(k, val) for k in sorted(k_grid)]
            print(f"[extend] 新增 {axis}={val}: {len(points)} 点")
            run_stage_points(points, t_proto, t_int, t_top_meas,
                             "extend_scan.csv", output_dir, "extend")
            added += len(points)
    print(f"[extend] 完成, 新增 {added} 点。")
    return out


def stage_fine(t_proto, t_int, t_top_meas, output_dir):
    comb = build_combined(output_dir)
    best = best_point_from_table(comb)
    if best is None:
        raise RuntimeError("无法确定细网格中心: 无 OK 点。")
    k_grid, cp_grid = clipped_fine_grid(
        float(best["k_eff_W_mK"]), float(best["cp_eff_J_kgK"]),
        K_LIMITS, CP_LIMITS,
    )
    print(f"[fine] 中心 k={best['k_eff_W_mK']} cp={best['cp_eff_J_kgK']}; "
          f"k 网格 {len(k_grid)} 值, cp 网格 {len(cp_grid)} 值, "
          f"共 {len(k_grid)*len(cp_grid)} 点")
    points = product_grid(k_grid, cp_grid)
    return run_stage_points(points, t_proto, t_int, t_top_meas,
                            "fine_scan.csv", output_dir, "fine")


def stage_analysis(t_proto, t_int, t_top_meas, output_dir):
    """最终分析: 最优 / 近最优 / 剖面 / 图 / 最佳迹线 / 残差。"""
    comb = build_combined(output_dir)
    best = best_point_from_table(comb)
    if best is None:
        raise RuntimeError("无 OK 点可分析。")
    bk, bcp = float(best["k_eff_W_mK"]), float(best["cp_eff_J_kgK"])
    print(f"[analysis] BEST GRID POINT: k={bk:.4f} cp={bcp:.0f} "
          f"RMSE={best['RMSE_C']:.4f} C")

    # 最优迹线
    row, result = evaluate_point(bk, bcp, t_proto, t_int, t_top_meas,
                                 return_result=True)
    t_arr = result["t_array"]
    T_top_pred = result["T_top_surface_arr"]
    T_sample_pred = result["T_sample_arr"]
    T_top_pred_meas = np.interp(t_top_meas, t_arr, T_top_pred)
    residual = T_top_pred_meas - t_top_meas
    trace = pd.DataFrame({
        "time_s": t_arr,
        "T_internal_C": np.interp(t_arr, t_proto, t_int),
        "T_top_measured_C": np.interp(t_arr, t_proto, t_top_meas),
        "T_top_predicted_C": T_top_pred,
        "residual_C": T_top_pred - np.interp(t_arr, t_proto, t_top_meas),
        "T_sample_predicted_C": T_sample_pred,
    })
    trace.to_csv(output_dir / "best_fit_trace.csv", index=False)

    # 残差 / regime 指标
    regimes = load_regime_labels()
    if regimes is not None and len(regimes) == len(t_top_meas):
        print("  regime diagnostics (equal-weight objective unchanged):")
        for reg in ("TRANSIENT_HEATING", "TRANSIENT_COOLING", "SETTLING",
                    "TRANSITION_OTHER"):
            mask = regimes == reg
            m = compute_metrics(t_top_meas[mask], residual[mask])
            print(f"    {reg:18s} n={mask.sum():3d} RMSE={m['rmse']:.3f} "
                  f"MAE={m['mae']:.3f}")
    else:
        regimes = None

    # 近最优区域
    near = near_optimal_regions(comb, float(best["RMSE_C"]))
    for key, info in near.items():
        print(f"  within {key} of min RMSE: {info['n_points']} pts, "
              f"k [{info['k_min']:.3f}, {info['k_max']:.3f}], "
              f"cp [{info['cp_min']:.0f}, {info['cp_max']:.0f}]")

    # 剖面
    prof_k, prof_cp = profile_minima(comb)
    prof_k.to_csv(output_dir / "profile_rmse_vs_k.csv", index=False)
    prof_cp.to_csv(output_dir / "profile_rmse_vs_cp.csv", index=False)
    plot_profiles(prof_k, prof_cp,
                  output_dir / "profile_rmse_vs_k.png",
                  output_dir / "profile_rmse_vs_cp.png")
    print("  profile min: min-RMSE vs k:", prof_k["RMSE_C"].min().round(4),
          "; vs cp:", prof_cp["RMSE_C"].min().round(4))

    # 景观图
    coarse = read_table(output_dir / "coarse_scan.csv")
    fine = read_table(output_dir / "fine_scan.csv")
    markers = [(0.13, 1800.0, "historical default"),
               (0.14, 1400.0, "reference")]
    plot_landscape(coarse, "Coarse RMSE landscape (72°C)",
                   output_dir / "rmse_landscape_coarse.png",
                   best=(bk, bcp), markers=markers)
    plot_landscape(fine, "Fine RMSE landscape (72°C)",
                   output_dir / "rmse_landscape_fine.png",
                   best=(bk, bcp), markers=markers)

    # 迹线图 / 残差图
    plot_best_trace(t_proto, t_int, t_top_meas, t_arr, T_top_pred,
                    T_sample_pred, output_dir / "best_fit_trace.png")
    plot_residuals(t_top_meas, residual, output_dir / "best_fit_residual.png",
                   regimes=regimes)

    # 最终元数据
    write_metadata({
        "best_grid_k_eff": bk,
        "best_grid_cp_eff": bcp,
        "best_grid_RMSE_C": float(best["RMSE_C"]),
        "best_grid_MAE_C": float(best["MAE_C"]),
        "best_grid_max_abs_error_C": float(best["max_abs_error_C"]),
        "near_optimal_regions": near,
    })
    print("[analysis] 完成。")
    return best


# =============================================================
# V2 扩展分析 (system_effective_extended_v2) —— 仅写入 V2 独立目录
# =============================================================

V1_DIR = PROJECT_ROOT / "parameter_scan_output" / "72C"
V2_DIR_DEFAULT = V1_DIR / "system_effective_extended_v2"


def stage_v2_cross_check(t_proto, t_int, t_top_meas, output_dir):
    """交叉校验: 独立重跑 V1 最优边界点 (0.06, 2600), 写入 V2 粗扫表。"""
    out = output_dir / "extended_coarse_scan.csv"
    k, cp = V1_BEST_POINT
    if point_key(k, cp) in completed_keys(read_table(out)):
        print("[v2 cross-check] 0.06/2600 已在 V2 中完成 (续跑跳过)")
        return
    row = evaluate_point_safe(k, cp, t_proto, t_int, t_top_meas)
    append_rows(out, [row])
    print(f"[v2 cross-check] k={k} cp={cp} RMSE={row['RMSE_C']:.4f} C "
          f"(V1 参考 {V1_BEST_RMSE} C)")


def stage_v2_coarse(t_proto, t_int, t_top_meas, output_dir):
    """V2 粗扫: 8 x 7 = 56 点 (含 0.06/2600 交叉校验点, 续跑跳过)。"""
    points = product_grid(V2_K_GRID, V2_CP_GRID)
    return run_stage_points(points, t_proto, t_int, t_top_meas,
                            "extended_coarse_scan.csv", output_dir, "v2coarse")


def stage_v2_fine(t_proto, t_int, t_top_meas, output_dir):
    """V2 细扫: 仅当粗最优在 k 与 cp 两个维度都是内点时执行。

    细网格 = 跨最近粗网格邻居的 11 x 11 线性网格 (121 组合);
    若细 1 阶段最优落在细矩形边上且仍为整体内点, 允许至多一次
    recenter (细 2 阶段)。不扩展到 V2 诊断边界之外。
    """
    comb = build_combined_v2(output_dir)
    best = best_point_from_table(comb)
    if best is None:
        raise RuntimeError("V2 无 OK 点。")
    bk, bcp = float(best["k_eff_W_mK"]), float(best["cp_eff_J_kgK"])
    coarse_interior = is_interior(bk, bcp, V2_K_GRID, V2_CP_GRID)
    if not coarse_interior:
        print(f"[v2-fine] 粗最优 k={bk} cp={bcp} 位于 V2 边界上 -> "
              "不运行细扫 (无封闭最优)。")
        return False

    k_grid, cp_grid = fine_grid_from_neighbors(
        V2_K_GRID, V2_CP_GRID, bk, bcp, n=11,
        k_limits=V2_K_LIMITS, cp_limits=V2_CP_LIMITS,
    )
    print(f"[v2-fine stage1] k {len(k_grid)} x cp {len(cp_grid)} = "
          f"{len(k_grid)*len(cp_grid)} 点 (邻居跨度 "
          f"k[{k_grid[0]},{k_grid[-1]}] cp[{cp_grid[0]},{cp_grid[-1]}])")
    run_stage_points(product_grid(k_grid, cp_grid), t_proto, t_int, t_top_meas,
                     "extended_fine_scan.csv", output_dir, "v2fine1")

    # 可选 recenter (至多一次)
    comb2 = build_combined_v2(output_dir)
    best2 = best_point_from_table(comb2)
    bk2, bcp2 = float(best2["k_eff_W_mK"]), float(best2["cp_eff_J_kgK"])
    on_edge = (bk2 == k_grid[0] or bk2 == k_grid[-1]
               or bcp2 == cp_grid[0] or bcp2 == cp_grid[-1])
    k_ev = sorted(comb2["k_eff_W_mK"].unique())
    cp_ev = sorted(comb2["cp_eff_J_kgK"].unique())
    if on_edge and is_interior(bk2, bcp2, k_ev, cp_ev):
        k_grid2, cp_grid2 = fine_grid_from_neighbors(
            k_ev, cp_ev, bk2, bcp2, n=11,
            k_limits=V2_K_LIMITS, cp_limits=V2_CP_LIMITS,
        )
        if len(k_grid2) > 2 and len(cp_grid2) > 2:
            print(f"[v2-fine stage2 (recenter)] k {len(k_grid2)} x cp "
                  f"{len(cp_grid2)} = {len(k_grid2)*len(cp_grid2)} 点")
            run_stage_points(product_grid(k_grid2, cp_grid2), t_proto, t_int,
                             t_top_meas, "extended_fine_scan.csv", output_dir,
                             "v2fine2")
    return True


def residual_balance(t_meas, residual):
    """残差平衡统计: 正/负比例。"""
    r = np.asarray(residual, dtype=float)
    return {
        "frac_positive": float(np.mean(r > 0)),
        "frac_negative": float(np.mean(r < 0)),
        "frac_zero": float(np.mean(r == 0)),
    }


def _best_trace_files(t_proto, t_int, t_top_meas, k, cp, output_dir,
                      prefix, title_label):
    """重跑 (k, cp) 生成迹线 CSV/PNG 与残差图, 返回 (trace_df, residual)。"""
    row, result = evaluate_point(k, cp, t_proto, t_int, t_top_meas,
                                 return_result=True)
    t_arr = result["t_array"]
    T_top_pred = result["T_top_surface_arr"]
    T_sample_pred = result["T_sample_arr"]
    residual = np.interp(t_top_meas, t_arr, T_top_pred) - t_top_meas
    trace = pd.DataFrame({
        "time_s": t_arr,
        "T_internal_C": np.interp(t_arr, t_proto, t_int),
        "T_top_measured_C": np.interp(t_arr, t_proto, t_top_meas),
        "T_top_predicted_C": T_top_pred,
        "residual_C": T_top_pred - np.interp(t_arr, t_proto, t_top_meas),
        "T_sample_predicted_C": T_sample_pred,
    })
    trace.to_csv(output_dir / f"{prefix}.csv", index=False)
    plot_best_trace(t_proto, t_int, t_top_meas, t_arr, T_top_pred,
                    T_sample_pred, output_dir / f"{prefix}.png")
    regimes = load_regime_labels()
    if regimes is not None and len(regimes) == len(t_top_meas):
        plot_residuals(t_top_meas, residual,
                       output_dir / f"{prefix}_residual.png", regimes=regimes)
    return trace, residual


def stage_v2_analysis(t_proto, t_int, t_top_meas, output_dir, v1_dir=V1_DIR):
    """V2 最终分析: 最优/边界判定, 景观 (log k + 平均残差), 剖面,
    近最优 (仅内点时), V1 vs V2 对比, 迹线, summary, 元数据。"""
    comb = build_combined_v2(output_dir)
    best = best_point_from_table(comb)
    if best is None:
        raise RuntimeError("V2 无 OK 点可分析。")
    bk, bcp = float(best["k_eff_W_mK"]), float(best["cp_eff_J_kgK"])
    interior = is_interior(bk, bcp, V2_K_GRID, V2_CP_GRID)
    print(f"[v2-analysis] BEST GRID POINT: k={bk} cp={bcp} "
          f"RMSE={best['RMSE_C']:.4f} C  interior={interior}")

    # 迹线 (内点 -> best_fit_trace_v2; 边界 -> best_boundary_trace_v2)
    if interior:
        trace, residual = _best_trace_files(
            t_proto, t_int, t_top_meas, bk, bcp, output_dir,
            "best_fit_trace_v2", "best fit (V2 interior)")
    else:
        trace, residual = _best_trace_files(
            t_proto, t_int, t_top_meas, bk, bcp, output_dir,
            "best_boundary_trace_v2",
            "BEST BOUNDARY POINT — NOT FINAL CALIBRATED MODEL")

    bal = residual_balance(t_top_meas, residual)
    print(f"[v2-analysis] residual balance: pos {bal['frac_positive']:.3f}, "
          f"neg {bal['frac_negative']:.3f}")

    # 景观 (粗 / 细 / 平均残差) — log k
    coarse = read_table(output_dir / "extended_coarse_scan.csv")
    fine = read_table(output_dir / "extended_fine_scan.csv")
    markers = [(*V1_BEST_POINT, "V1 boundary best (0.06/2600)")]
    plot_landscape_v2(coarse, "RMSE_C", "V2 coarse RMSE landscape (72°C)",
                      output_dir / "rmse_landscape_extended_coarse.png",
                      best=(bk, bcp), markers=markers)
    if not fine.empty:
        plot_landscape_v2(fine, "RMSE_C", "V2 fine RMSE landscape (72°C)",
                          output_dir / "rmse_landscape_extended_fine.png",
                          best=(bk, bcp), markers=markers)
    plot_landscape_v2(comb, "mean_residual_C",
                      "V2 mean-residual landscape (diagnostic only)",
                      output_dir / "mean_residual_landscape_v2.png",
                      best=(bk, bcp), markers=markers)

    # 剖面
    prof_k, prof_cp = profile_minima(comb)
    prof_k.to_csv(output_dir / "profile_rmse_vs_k_v2.csv", index=False)
    prof_cp.to_csv(output_dir / "profile_rmse_vs_cp_v2.csv", index=False)
    plot_profiles(prof_k, prof_cp,
                  output_dir / "profile_rmse_vs_k_v2.png",
                  output_dir / "profile_rmse_vs_cp_v2.png")

    # 近最优 (仅内点)
    near = None
    if interior:
        near = near_optimal_regions(comb, float(best["RMSE_C"]))
        for key, info in near.items():
            print(f"  within {key} of min RMSE: {info['n_points']} pts, "
                  f"k [{info['k_min']}, {info['k_max']}], "
                  f"cp [{info['cp_min']:.0f}, {info['cp_max']:.0f}]")

    # regime 诊断
    regimes = load_regime_labels()
    if regimes is not None and len(regimes) == len(t_top_meas):
        print("  regime diagnostics (equal-weight objective unchanged):")
        for reg in ("TRANSIENT_HEATING", "TRANSIENT_COOLING", "SETTLING",
                    "TRANSITION_OTHER"):
            mask = regimes == reg
            m = compute_metrics(t_top_meas[mask], residual[mask])
            print(f"    {reg:18s} n={mask.sum():3d} RMSE={m['rmse']:.3f} "
                  f"MAE={m['mae']:.3f}")

    # V1 vs V2 对比 (V1 只读)
    comparison = compare_v1_vs_v2(v1_dir, best, interior, output_dir,
                                  t_proto, t_int, t_top_meas, trace)

    # summary + 元数据
    summary_lines = [
        "SYSTEM-LEVEL EXTENDED PARAMETER SCAN V2 — summary",
        f"analysis: system_effective_extended_v2",
        f"outcome: {'INTERIOR BASIN' if interior else 'STILL BOUNDARY-SEEKING'}",
        f"best grid k_eff: {bk} W/(m·K)",
        f"best grid cp_eff: {bcp} J/(kg·K)",
        f"best RMSE: {best['RMSE_C']:.4f} C",
        f"MAE: {best['MAE_C']:.4f} C",
        f"mean residual: {best['mean_residual_C']:.4f} C",
        f"max positive residual: {best['max_positive_residual_C']:.4f} C",
        f"max negative residual: {best['max_negative_residual_C']:.4f} C",
        f"frac positive residuals: {bal['frac_positive']:.3f}",
        f"frac negative residuals: {bal['frac_negative']:.3f}",
        f"V1 best boundary point: k=0.060 cp=2600 RMSE={V1_BEST_RMSE} C (preserved)",
        f"V1 outputs overwritten: NO",
    ]
    if not interior:
        summary_lines.append(
            "No enclosed optimum was found with the two-parameter "
            "phenomenological model over the broad V2 effective-parameter "
            "region."
        )
    (output_dir / "summary_v2.txt").write_text("\n".join(summary_lines) + "\n",
                                               encoding="utf-8")

    write_metadata_v2(output_dir, best, interior, near)
    print("[v2-analysis] 完成。")
    return best


def compare_v1_vs_v2(v1_dir, v2_best, interior, output_dir,
                     t_proto, t_int, t_top_meas, v2_trace):
    """V1 与 V2 对比 (V1 文件只读)。返回对比 DataFrame。"""
    v1_comb_path = Path(v1_dir) / "combined_parameter_scan.csv"
    v1_best_row = None
    if v1_comb_path.is_file():
        v1_comb = pd.read_csv(v1_comb_path)
        v1_best_row = best_point_from_table(v1_comb)
    v1 = {
        "k_eff_W_mK": float(v1_best_row["k_eff_W_mK"]) if v1_best_row is not None
        else V1_BEST_POINT[0],
        "cp_eff_J_kgK": float(v1_best_row["cp_eff_J_kgK"]) if v1_best_row is not None
        else V1_BEST_POINT[1],
        "RMSE_C": float(v1_best_row["RMSE_C"]) if v1_best_row is not None
        else V1_BEST_RMSE,
        "MAE_C": float(v1_best_row["MAE_C"]) if v1_best_row is not None else np.nan,
        "mean_residual_C": (float(v1_best_row["mean_residual_C"])
                            if v1_best_row is not None else np.nan),
        "max_positive_residual_C": (float(v1_best_row["max_positive_residual_C"])
                                    if v1_best_row is not None else np.nan),
        "max_negative_residual_C": (float(v1_best_row["max_negative_residual_C"])
                                    if v1_best_row is not None else np.nan),
        "boundary_status": "BOUNDARY (k_low + cp_high)",
    }
    v2 = {
        "k_eff_W_mK": float(v2_best["k_eff_W_mK"]),
        "cp_eff_J_kgK": float(v2_best["cp_eff_J_kgK"]),
        "RMSE_C": float(v2_best["RMSE_C"]),
        "MAE_C": float(v2_best["MAE_C"]),
        "mean_residual_C": float(v2_best["mean_residual_C"]),
        "max_positive_residual_C": float(v2_best["max_positive_residual_C"]),
        "max_negative_residual_C": float(v2_best["max_negative_residual_C"]),
        "boundary_status": ("INTERIOR" if interior else "BOUNDARY"),
    }
    out = pd.DataFrame([
        {"scan": "V1", **v1},
        {"scan": "V2", **v2},
    ])
    out = out.assign(
        rmse_change_C=out["RMSE_C"] - out["RMSE_C"].iloc[0],
        rmse_pct_improvement=100 * (1 - out["RMSE_C"] / out["RMSE_C"].iloc[0]),
        mean_residual_change_C=out["mean_residual_C"]
        - out["mean_residual_C"].iloc[0],
    )
    out.to_csv(output_dir / "comparison_v1_vs_v2.csv", index=False)

    # 对比图: 实测 + V1 预测 (读 V1 best_fit_trace.csv 只读) + V2 预测
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t_proto, t_top_meas, color="#d62728", lw=1.5,
            label="Measured Top COC surface T")
    v1_trace_path = Path(v1_dir) / "best_fit_trace.csv"
    if v1_trace_path.is_file():
        v1_tr = pd.read_csv(v1_trace_path)
        ax.plot(v1_tr["time_s"], v1_tr["T_top_predicted_C"], color="#9467bd",
                lw=1.5, ls="--", label="V1 best-boundary predicted T_top")
    ax.plot(v2_trace["time_s"], v2_trace["T_top_predicted_C"], color="#1f77b4",
            lw=2, label="V2 best predicted T_top")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("V1 vs V2 best predicted Top COC surface temperature")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(output_dir / "comparison_v1_vs_v2.png",
                                    dpi=150)
    plt.close(fig)
    print("[v2-analysis] comparison saved: "
          f"RMSE V1 {v1['RMSE_C']:.4f} -> V2 {v2['RMSE_C']:.4f} "
          f"({100*(1-v2['RMSE_C']/v1['RMSE_C']):.1f}% improvement)")
    return out


def write_metadata_v2(output_dir, best, interior, near):
    meta = {
        "analysis_id": "system_effective_extended_v2",
        "parent_analysis": "V1 parameter scan (parameter_scan_output/72C)",
        "v1_best_boundary": {
            "k_eff_W_mK": V1_BEST_POINT[0],
            "cp_eff_J_kgK": V1_BEST_POINT[1],
            "RMSE_C": V1_BEST_RMSE,
        },
        "reason_for_v2": (
            "V1 minimum hit low-k (0.06) and high-cp (2600) diagnostic limits; "
            "RMSE still improving toward lower k and higher cp."
        ),
        "modeling_interpretation": (
            "system-level effective parameters; NOT intrinsic COC constants"
        ),
        "v2_k_grid": V2_K_GRID,
        "v2_cp_grid": V2_CP_GRID,
        "v2_diagnostic_limits": {"k_eff": list(V2_K_LIMITS),
                                 "cp_eff": list(V2_CP_LIMITS)},
        "input_dataset_path": str(ALIGNED_CSV),
        "git_head": git_head(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry_preset": "BARE_TOP_COC_LAYERS",
        "fixed_rho_COC_kg_m3": RHO_COC,
        "h_conv_W_m2K": H_CONV,
        "T_air_ambient_C": T_AMB,
        "initial_condition_mode": "auto (first T_internal value per run)",
        "objective": (
            "RMSE of T_top_surface_predicted vs T_top_measured; equal weight, "
            "no time shift, no sample-temperature term; MAE/mean/max residuals "
            "diagnostic only"
        ),
        "outcome": "INTERIOR BASIN" if interior else "STILL BOUNDARY-SEEKING",
        "best_grid": {
            "k_eff_W_mK": float(best["k_eff_W_mK"]),
            "cp_eff_J_kgK": float(best["cp_eff_J_kgK"]),
            "RMSE_C": float(best["RMSE_C"]),
        },
        "near_optimal_regions": near,
        "note": (
            "V1 outputs were preserved unchanged (manifest before/after). "
            "No continuous optimizer; no additional fitted parameters; "
            "no k(T)/cp(T); no independent validation by design."
        ),
    }
    (Path(output_dir) / "extended_scan_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def stage_v2_integrity(v1_dir, v2_dir):
    """对比 V1 清单: 写 original_v1_integrity_check_after.json。"""
    before_path = Path(v2_dir) / "original_v1_file_manifest_before.json"
    if not before_path.is_file():
        raise RuntimeError("缺少 original_v1_file_manifest_before.json, 无法校验。")
    result = verify_manifest(before_path, v1_dir)
    (Path(v2_dir) / "original_v1_integrity_check_after.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result["changed"]:
        print(f"[v2-integrity] FAIL: 以下 V1 文件被修改: {result['changed']}")
    else:
        print(f"[v2-integrity] PASS: V1 {len(result['files_checked'])} 个文件 "
              "SHA256/size/mtime 全部未变。")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--dataset", default=str(ALIGNED_CSV))
    ap.add_argument("--stage", default="all",
                    choices=["all", "baseline_reference", "coarse", "extend",
                             "fine", "analysis", "v2", "v2-scan",
                             "v2-analysis", "v2-integrity"])
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_proto, t_int, t_top_meas = load_experiment(args.dataset)
    print(f"[data] {args.dataset}: {len(t_proto)} 点, "
          f"t [{t_proto[0]:.1f}, {t_proto[-1]:.1f}] s")

    if args.stage in ("all", "baseline_reference"):
        stage_baseline_reference(t_proto, t_int, t_top_meas, output_dir)
    if args.stage in ("all", "coarse"):
        stage_coarse(t_proto, t_int, t_top_meas, output_dir)
    if args.stage in ("all", "extend"):
        stage_extend(t_proto, t_int, t_top_meas, output_dir)
    if args.stage in ("all", "fine"):
        stage_fine(t_proto, t_int, t_top_meas, output_dir)
    if args.stage in ("all", "analysis"):
        stage_analysis(t_proto, t_int, t_top_meas, output_dir)

    if args.stage in ("v2", "v2-scan"):
        stage_v2_cross_check(t_proto, t_int, t_top_meas, output_dir)
        stage_v2_coarse(t_proto, t_int, t_top_meas, output_dir)
        stage_v2_fine(t_proto, t_int, t_top_meas, output_dir)
    if args.stage in ("v2", "v2-analysis"):
        stage_v2_analysis(t_proto, t_int, t_top_meas, output_dir,
                          v1_dir=V1_DIR)
    if args.stage in ("v2", "v2-integrity"):
        stage_v2_integrity(V1_DIR, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
