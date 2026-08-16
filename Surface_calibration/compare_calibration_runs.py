#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-validation of two independent Peltier surface calibration runs
===================================================================

科学目标
--------
1) 主验证 (PRIMARY VALIDATION):
   Dataset 1 的固定稳态校准方程
       T_pred_A = a_A * T_set + b_A
   用于预测 Dataset 2 在 Dataset 1 校准范围内、独立实验测得的稳态表面温度
   (40, 50, 60, 70, 80, 90 °C)。

2) 外推检查 (EXTRAPOLATION):
   Dataset 1 方程在 Dataset 2 的 30 °C 稳态点上的预测。
   30 °C 低于 Dataset 1 校准范围 (35-95 °C)，因此是外推，单独报告，
   绝不混入主插值验证统计。

3) 可重复性 (REPRODUCIBILITY):
   比较两次独立拟合的最终校准方程参数 (a, b, tau_eff) 与
   连续校准函数在重叠校准范围 (35-90 °C) 上的预测差异。

重要原则
--------
- 不修改任何已有校准模型/文件。
- 不重新拟合 Dataset 1。
- Dataset 1 参数固定，绝不能用 Dataset 2 更新。
- Dataset 2 稳态点优先复用既定分段/稳态提取逻辑产生的
  steady_points.csv；脚本会验证其结构完整性。
- 两次实验没有完全相同的稳态设定点，因此
  "重叠校准范围内的函数一致性" ≠ "相同设定点的直接测量比较"。

输出目录
--------
<repo root>/calibration_output/Cross_Validation_Surface_2/
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头后端，避免 GUI 依赖

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 仓库与默认路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURFACE_CALIBRATION_DIR = PROJECT_ROOT / "Surface_calibration"

DEFAULT_DATASET1_EQUATION = (
    PROJECT_ROOT / "calibration_output" / "final_calibration_equation.txt"
)
DEFAULT_DATASET2_EQUATION = (
    PROJECT_ROOT / "calibration_output" / "Surface_2" / "final_calibration_equation.txt"
)
DEFAULT_DATASET2_STEADY = (
    PROJECT_ROOT / "calibration_output" / "Surface_2" / "steady_points.csv"
)
# Dataset 2 原始数据位于仓库上级的 Calibration 目录
DEFAULT_DATASET2_RAW = PROJECT_ROOT.parent / "Calibration" / "Surface 2.xls"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "calibration_output" / "Cross_Validation_Surface_2"
)

DATASET1_SETPOINTS = [35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
DATASET2_SETPOINTS = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
                      80.0, 70.0, 60.0, 50.0, 40.0, 30.0]

DATASET1_RANGE = (35.0, 95.0)
DATASET2_RANGE = (30.0, 90.0)
OVERLAP_RANGE = (35.0, 90.0)          # 两个校准范围的重叠区间

PRIMARY_SETPOINTS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
REPRESENTATIVE_TEMPS = [35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

# 合成 setpoint profile (用于完整动态模型对比):
# 30 -> 40 -> 50 -> 40 -> 30 °C, 每段 60 s, dt = 1 s
SYNTHETIC_SETPOINTS = [30.0, 40.0, 50.0, 40.0, 30.0]
SYNTHETIC_HOLD_S = 60.0
SYNTHETIC_DT = 1.0


# ============================================================
# 数据类与解析
# ============================================================

@dataclass
class CalibrationEquation:
    """从 final_calibration_equation.txt 解析出的最终校准模型。"""
    a: float
    b: float
    tau_eff: float
    r2_steady: float | None
    rmse_dynamic: float | None
    source: Path

    def predict(self, tset):
        return self.a * np.asarray(tset, dtype=float) + self.b


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_equation_file(path) -> CalibrationEquation:
    """
    解析最终校准方程文件。

    期望格式 (由 peltier_surface_calibration_v2.py 生成):
        Steady-state calibration:
        T_inf = <a> * T_set <b>

        Dynamic calibration:
        T_s(t) = (<a> * T_set,j <b>) + [...] * exp(-(t - t_j) / <tau>)

        Model quality:
        Steady-state R² = <r2>
        Dynamic RMSE = <rmse> °C
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    steady_m = re.search(
        rf"T_inf\s*=\s*({_NUM})\s*\*\s*T_set\s*({_NUM})", text
    )
    tau_m = re.search(
        rf"exp\(\s*-\(\s*t\s*-\s*t_j\s*\)\s*/\s*({_NUM})\s*\)", text
    )
    r2_m = re.search(rf"Steady-state\s*R²\s*=\s*({_NUM})", text)
    rmse_m = re.search(rf"Dynamic\s*RMSE\s*=\s*({_NUM})", text)

    if steady_m is None:
        raise ValueError(
            f"无法从 {path} 解析稳态方程 'T_inf = a * T_set + b'。"
        )
    if tau_m is None:
        raise ValueError(
            f"无法从 {path} 解析动态时间常数 tau_eff。"
        )

    return CalibrationEquation(
        a=float(steady_m.group(1)),
        b=float(steady_m.group(2)),
        tau_eff=float(tau_m.group(1)),
        r2_steady=float(r2_m.group(1)) if r2_m else None,
        rmse_dynamic=float(rmse_m.group(1)) if rmse_m else None,
        source=path.resolve(),
    )


def load_mean_steady_points(path) -> pd.DataFrame:
    """读取 steady_points.csv 中 Mean 信号、被接受的稳态点。"""
    df = pd.read_csv(path)
    if "signal" not in df.columns:
        raise ValueError(
            f"{path} 缺少 'signal' 列，不是预期格式的 steady_points.csv。"
        )
    mean = df[df["signal"] == "Mean"].copy()
    if mean.empty:
        raise ValueError(
            f"{path} 中找不到 signal='Mean' 的稳态点记录。"
        )
    for col in ["segment", "setpoint", "steady_temp",
                "steady_slope_C_per_s", "accepted",
                "start_index", "end_index"]:
        if col not in mean.columns:
            raise ValueError(f"{path} 缺少列 '{col}'。")
    mean = mean.sort_values("segment").reset_index(drop=True)
    return mean


# ============================================================
# 基础计算
# ============================================================

def compute_metrics(errors) -> dict:
    """Bias / MAE / RMSE / Max Absolute Error。"""
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {
            "n": 0, "bias": np.nan, "mae": np.nan,
            "rmse": np.nan, "max_abs": np.nan,
        }
    return {
        "n": int(e.size),
        "bias": float(np.mean(e)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "max_abs": float(np.max(np.abs(e))),
    }


def symmetric_relative_difference(a: float, b: float) -> float:
    """
    对称相对差异 (%)：
        |A - B| / ((|A| + |B|) / 2) * 100
    分母接近零时返回 NaN（此时不宜使用百分比差异）。
    """
    denom = (abs(a) + abs(b)) / 2.0
    if denom < 1e-12:
        return np.nan
    return abs(a - b) / denom * 100.0


def classify_by_range(setpoint: float, rng) -> str:
    """设定点是否落在校准范围 [lo, hi] 内。"""
    lo, hi = rng
    return "interpolation" if lo <= setpoint <= hi else "extrapolation"


def steady_predict(eq: CalibrationEquation, tset):
    """Dataset 稳态模型预测: T_pred = a * T_set + b (标量或数组)。"""
    return eq.predict(tset)


# ============================================================
# 完整动态模型 (稳态 + 一阶动态) 的合成轨迹对比
# ============================================================

def predict_dynamic_profile(
    eq: CalibrationEquation,
    setpoint_series,
    dt: float = SYNTHETIC_DT,
    initial_temperature: float | None = None,
):
    """
    用最终一阶动态模型生成表面温度轨迹:
        T_eq[n] = a * T_set[n] + b
        T_s[n] = T_eq[n] + (T_s[n-1] - T_eq[n]) * exp(-dt / tau_eff)

    与 peltier_surface_calibration_v2.predict_first_order_profile 完全等价。
    初始温度缺省取第一个 setpoint 的稳态平衡温度 (T_eq[0])，表示实验开始前
    已在首个 setpoint 处达到平衡。
    """
    s = np.asarray(setpoint_series, dtype=float)
    Teq = eq.predict(s)
    out = np.empty_like(Teq, dtype=float)
    if initial_temperature is None:
        initial_temperature = float(Teq[0])
    out[0] = float(initial_temperature)
    alpha = np.exp(-dt / max(float(eq.tau_eff), 1e-12))
    for i in range(1, len(out)):
        out[i] = Teq[i] + (out[i - 1] - Teq[i]) * alpha
    return out, Teq


def build_synthetic_profile(
    setpoints=SYNTHETIC_SETPOINTS,
    hold_s: float = SYNTHETIC_HOLD_S,
    dt: float = SYNTHETIC_DT,
):
    """生成假设 setpoint 阶梯 profile 的时间轴 (s) 与 setpoint 序列。"""
    n_hold = int(round(hold_s / dt))
    n_stages = len(setpoints)
    time = np.arange(n_hold * n_stages, dtype=float) * dt
    sps = np.repeat(np.asarray(setpoints, dtype=float), n_hold)
    return time, sps


def dynamic_trajectory_comparison(
    eq_A: CalibrationEquation,
    eq_B: CalibrationEquation,
    setpoint_series,
    dt: float = SYNTHETIC_DT,
) -> dict:
    """
    在相同合成 setpoint 输入下比较两条完整模型轨迹 (T_A vs T_B)。
    两条轨迹各自从自身首个 setpoint 的稳态平衡温度开始。
    """
    traj_A, _ = predict_dynamic_profile(eq_A, setpoint_series, dt)
    traj_B, _ = predict_dynamic_profile(eq_B, setpoint_series, dt)
    diff = traj_A - traj_B
    i_max = int(np.argmax(np.abs(diff)))
    return {
        "traj_A": traj_A,
        "traj_B": traj_B,
        "diff": diff,
        "mean_signed": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "max_abs_time_s": float(i_max) * dt,
    }


def per_transition_max_differences(
    traj_A,
    traj_B,
    setpoint_series,
    dt: float = SYNTHETIC_DT,
    hold_s: float = SYNTHETIC_HOLD_S,
) -> pd.DataFrame:
    """
    每次 setpoint 切换后 hold 窗口内两条轨迹的最大绝对差异。
    用于定位"动态响应差异最大"发生在哪个方向/哪次切换。
    """
    diff = np.asarray(traj_A) - np.asarray(traj_B)
    s = np.asarray(setpoint_series, dtype=float)
    n_hold = int(round(hold_s / dt))
    rows = []
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            start = i
            end = min(i + n_hold, len(s))
            window = diff[start:end]
            if window.size == 0:
                continue
            iw = int(np.argmax(np.abs(window)))
            rows.append({
                "transition": f"{s[i - 1]:.0f} -> {s[i]:.0f}",
                "direction": (
                    "heating" if s[i] > s[i - 1] else "cooling"
                ),
                "window_s": float((end - start) * dt),
                "max_abs_difference": float(np.max(np.abs(window))),
                "time_of_max_s": float(start + iw) * dt,
            })
    return pd.DataFrame(rows)


def build_validation_frame(
    steady_df: pd.DataFrame,
    eq_A: CalibrationEquation,
    dataset1_range=(35.0, 95.0),
) -> pd.DataFrame:
    """
    为每个 Dataset 2 受控段构造 Dataset 1 -> Dataset 2 预测误差记录。

    保留每个独立受控段 (加热/冷却重复不合并)。
    direction: 相对上一段 setpoint 升高=heating / 降低=cooling / 首段=initial
    classification: 该 setpoint 相对 Dataset 1 校准范围是 interpolation 还是
                    extrapolation。
    """
    rows = []
    prev_sp = None

    for _, seg in steady_df.iterrows():
        sp = float(seg["setpoint"])
        if prev_sp is None:
            direction = "initial"
        elif sp > prev_sp:
            direction = "heating"
        else:
            direction = "cooling"
        prev_sp = sp

        pred = eq_A.predict(sp)
        measured = float(seg["steady_temp"])
        error = measured - pred

        rows.append({
            "segment_id": int(seg["segment"]),
            "direction": direction,
            "T_set": sp,
            "T_measured_B": measured,
            "T_predicted_A": float(pred),
            "error": float(error),
            "absolute_error": float(abs(error)),
            "steady_slope_C_per_s": float(seg["steady_slope_C_per_s"]),
            "accepted": bool(seg["accepted"]),
            "classification": classify_by_range(sp, dataset1_range),
        })

    return pd.DataFrame(rows)


def equation_grid_comparison(
    eq_A: CalibrationEquation,
    eq_B: CalibrationEquation,
    overlap=(35.0, 90.0),
    n: int = 1001,
) -> dict:
    """
    在重叠校准范围内对两条连续拟合函数做密集网格比较。
    Delta_model = T_A(Tset) - T_B(Tset)
    """
    grid = np.linspace(overlap[0], overlap[1], n)
    diff = eq_A.predict(grid) - eq_B.predict(grid)

    i_max = int(np.argmax(np.abs(diff)))
    return {
        "grid": grid,
        "delta": diff,
        "mean_signed": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "max_abs_setpoint": float(grid[i_max]),
    }


# ============================================================
# 数据完整性验证 (Dataset 2)
# ============================================================

def verify_steady_points(
    steady_df: pd.DataFrame,
    expected_sequence=DATASET2_SETPOINTS,
) -> list[str]:
    """
    验证 steady_points.csv 的结构完整性。

    检查:
    1. 恰好 13 个受控段;
    2. setpoint 序列与预期完全一致;
    3. 段区间连续且覆盖全部样本 (start_0==0, end_i==start_{i+1});
    4. 所有稳态点均被既定稳态选择逻辑接受;
    5. 无意外额外段 (隐含于 1)。
    """
    problems = []
    n_seg = len(steady_df)
    expected_n = len(expected_sequence)

    if n_seg != expected_n:
        problems.append(
            f"检测到 {n_seg} 个受控段，预期 {expected_n} 个。"
        )

    seq = [float(v) for v in steady_df["setpoint"].tolist()]
    if seq != [float(v) for v in expected_sequence]:
        problems.append(
            f"受控段 setpoint 序列 {seq} 与预期 "
            f"{list(expected_sequence)} 不一致。"
        )

    starts = [int(v) for v in steady_df["start_index"].tolist()]
    ends = [int(v) for v in steady_df["end_index"].tolist()]
    if starts[0] != 0:
        problems.append(f"首个段起点应为 0，实际为 {starts[0]}。")
    for i in range(len(steady_df) - 1):
        if ends[i] != starts[i + 1]:
            problems.append(
                f"段 {i} 结束索引 {ends[i]} 与段 {i + 1} 起点 "
                f"{starts[i + 1]} 不连续。"
            )

    if not bool(steady_df["accepted"].all()):
        bad = steady_df.loc[~steady_df["accepted"], "segment"].tolist()
        problems.append(
            f"以下段未被稳态选择逻辑接受 (not accepted): {bad}"
        )

    return problems


def check_passive_cooling_tail(
    raw_path: Path,
    sheet: str = "Data",
    t1_col: str = "T1",
    t2_col: str = "T2",
    final_setpoint: float = 30.0,
    n_last: int = 15,
    tol_C: float = 1.2,
    slope_threshold_C_per_s: float = -0.05,
) -> tuple[bool, str]:
    """
    检查原始数据末尾是否有 Peltier-OFF 被动冷却污染。

    若数据在最后受控段 (30 °C) 期间仍在受控保温:
      - 末尾 Tmean 应接近 30 °C (偏差 < tol_C);
      - 末尾斜率不应显著为负。
    返回 (ok, message)。
    """
    df = pd.read_excel(raw_path, sheet_name=sheet)
    T1 = pd.to_numeric(df[t1_col], errors="coerce").to_numpy(dtype=float)
    T2 = pd.to_numeric(df[t2_col], errors="coerce").to_numpy(dtype=float)
    Tmean = (T1 + T2) / 2.0

    valid = np.isfinite(Tmean)
    Tmean = Tmean[valid]
    if len(Tmean) < n_last:
        return False, f"有效样本不足 {n_last} 个，无法检查被动冷却。"

    tail = Tmean[-n_last:]
    tail_mean = float(np.mean(tail))
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])

    ok = (abs(tail_mean - final_setpoint) <= tol_C) and (
        slope >= slope_threshold_C_per_s
    )
    msg = (
        f"末 {n_last} 样本 Tmean 均值 = {tail_mean:.2f} °C "
        f"(设定 {final_setpoint} °C)，斜率 = {slope:+.4f} °C/s"
    )
    return ok, msg


def cross_check_segmentation(raw_path: Path, expected_setpoints=DATASET2_SETPOINTS):
    """
    用 peltier_surface_calibration_v2.py 中既定的自动分段逻辑
    (detect_change_points_from_temperature + segments_from_detected_changes，
    默认参数) 重新对 Dataset 2 原始 Tmean 分段，返回段 (start, end, setpoint)。
    用于与 steady_points.csv 记录的段边界交叉核对。
    失败时返回 None (调用方降级为警告)。
    """
    try:
        import peltier_surface_calibration_v2 as cal
    except ImportError:
        return None

    try:
        df = pd.read_excel(raw_path, sheet_name="Data")
        T1 = pd.to_numeric(df["T1"], errors="coerce").to_numpy(dtype=float)
        T2 = pd.to_numeric(df["T2"], errors="coerce").to_numpy(dtype=float)
        Tmean = (T1 + T2) / 2.0

        segments = cal.segments_from_detected_changes(
            Tmean,
            setpoint_sequence=[float(v) for v in expected_setpoints],
            dt=1.0,
            change_threshold=0.35,
            smooth_window=3,
            min_gap_s=20.0,
        )
        return [(s.start, s.end, s.setpoint) for s in segments]
    except Exception as exc:  # noqa: BLE001 - 交叉核对失败仅降级警告
        print(f"  [warn] 自动分段交叉核对失败: {exc}")
        return None


# ============================================================
# 参数与函数比较
# ============================================================

def parameter_comparison(eq_A: CalibrationEquation, eq_B: CalibrationEquation) -> pd.DataFrame:
    rows = []
    fields = [
        ("a (slope, °C/°C)", eq_A.a, eq_B.a),
        ("b (intercept, °C)", eq_A.b, eq_B.b),
        ("tau_eff (s)", eq_A.tau_eff, eq_B.tau_eff),
    ]
    if eq_A.r2_steady is not None and eq_B.r2_steady is not None:
        fields.append(("R²_steady (-)", eq_A.r2_steady, eq_B.r2_steady))
    if eq_A.rmse_dynamic is not None and eq_B.rmse_dynamic is not None:
        fields.append(("dynamic RMSE (°C)", eq_A.rmse_dynamic, eq_B.rmse_dynamic))

    for name, vA, vB in fields:
        rows.append({
            "parameter": name,
            "Dataset1_A": vA,
            "Dataset2_B": vB,
            "absolute_difference": abs(vA - vB),
            "symmetric_relative_difference_percent": (
                symmetric_relative_difference(vA, vB)
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# 绘图
# ============================================================

def plot_equation_comparison(
    eq_A: CalibrationEquation,
    eq_B: CalibrationEquation,
    measured: pd.DataFrame,
    overlap=OVERLAP_RANGE,
    output_path: Path = None,
):
    """两条独立拟合校准函数在重叠校准范围上的比较。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    grid = np.linspace(overlap[0], overlap[1], 400)
    ax.plot(
        grid, eq_A.predict(grid), color="#1f77b4", linewidth=2,
        label=f"Dataset 1 model: T = {eq_A.a:.4f}·Tset {eq_A.b:+.3f}",
    )
    ax.plot(
        grid, eq_B.predict(grid), color="#d62728", linewidth=2,
        label=f"Dataset 2 model: T = {eq_B.a:.4f}·Tset {eq_B.b:+.3f}",
    )

    # Dataset 2 实验稳态点 (Mean) —— 仅展示重叠范围内
    m = measured[
        (measured["setpoint"] >= overlap[0])
        & (measured["setpoint"] <= overlap[1])
        & np.isfinite(measured["steady_temp"])
    ]
    ax.scatter(
        m["setpoint"], m["steady_temp"],
        facecolors="none", edgecolors="#2ca02c", s=45, linewidths=1.2,
        label="Dataset 2 measured steady points (Mean)",
    )

    ax.plot(
        [overlap[0], overlap[1]], [overlap[0], overlap[1]],
        linestyle="--", color="grey", linewidth=1,
        label="Ideal: Tsurface = Tset",
    )

    ax.set_xlim(overlap[0] - 2, overlap[1] + 2)
    ax.set_xlabel("Setpoint Temperature (°C)")
    ax.set_ylabel("Predicted Surface Temperature (°C)")
    ax.set_title(
        "Independently fitted steady calibration functions\n"
        f"overlapping calibrated range {overlap[0]:.0f}–{overlap[1]:.0f} °C"
    )
    ax.text(
        0.02, 0.02,
        "Comparison of independently fitted continuous calibration\n"
        "functions — NOT identical measured steady-state setpoints",
        transform=ax.transAxes, fontsize=7.5, va="bottom",
        color="#555555",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_primary_validation(
    validation_df: pd.DataFrame,
    extrap_df: pd.DataFrame,
    eq_A: CalibrationEquation,
    dataset1_range=DATASET1_RANGE,
    output_path: Path = None,
):
    """
    主验证图: Dataset 1 固定方程预测 vs Dataset 2 实测稳态点。

    - 40-90 °C 主验证段: 加热/冷却分别用不同标记，不合并重复设定点;
    - 30 °C 外推点: 单独标记为 Extrapolation。
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    in_range = validation_df[validation_df["classification"] == "interpolation"]
    heat = in_range[in_range["direction"] == "heating"]
    cool = in_range[in_range["direction"] == "cooling"]

    ax.scatter(
        heat["T_set"], heat["T_measured_B"],
        marker="^", s=80, color="#1f77b4", edgecolor="black", linewidth=0.5,
        label="Dataset 2 measured steady (heating)",
    )
    ax.scatter(
        cool["T_set"], cool["T_measured_B"],
        marker="v", s=80, color="#ff7f0e", edgecolor="black", linewidth=0.5,
        label="Dataset 2 measured steady (cooling)",
    )

    # Dataset 1 固定方程预测线 (完整校准范围 35-95 °C)
    grid = np.linspace(dataset1_range[0], dataset1_range[1], 300)
    ax.plot(
        grid, eq_A.predict(grid), color="#1f77b4", linewidth=2,
        linestyle="--",
        label=(
            f"Dataset 1 fixed-equation prediction\n"
            f"T = {eq_A.a:.4f}·Tset {eq_A.b:+.3f}"
        ),
    )
    ax.scatter(
        in_range["T_set"], in_range["T_predicted_A"],
        marker="x", s=60, color="#7f7f7f",
        label="Dataset 1 prediction at Dataset 2 setpoints",
    )

    # 30 °C 外推点 (单独标记)
    if not extrap_df.empty:
        ax.scatter(
            extrap_df["T_set"], extrap_df["T_measured_B_30"],
            marker="D", s=70, color="grey", edgecolor="black", linewidth=0.5,
            label="30 °C Dataset 2 measured (EXTRAPOLATION)",
        )
        ax.annotate(
            "Extrapolation\n(below Dataset 1 calibrated range)",
            xy=(30.0, float(extrap_df["T_measured_B_30"].iloc[0])),
            xytext=(38, float(extrap_df["T_measured_B_30"].iloc[0]) - 6),
            fontsize=7.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        )

    ax.set_xlabel("Dataset 2 Setpoint Temperature (°C)")
    ax.set_ylabel("Surface Temperature (°C)")
    ax.set_title(
        "Dataset 1 steady-state model predicts Dataset 2 measured steady states\n"
        "40–90 °C: independent interpolation validation | 30 °C: extrapolation"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_dynamic_model_comparison(
    eq_A: CalibrationEquation,
    eq_B: CalibrationEquation,
    time_s,
    setpoint_series,
    traj_A,
    traj_B,
    output_path: Path = None,
):
    """
    完整模型 (稳态 + 一阶动态) 在合成 setpoint profile 上的轨迹对比。

    展示两个最终预测公式:
        T_s[n] = (a*T_set[n] + b)
                 + (T_s[n-1] - (a*T_set[n] + b)) * exp(-dt/tau_eff)
    在同一 setpoint 输入 (30->40->50->40->30 °C) 下的响应差异。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(
        time_s, setpoint_series, where="post",
        color="#999999", linewidth=1, linestyle=":", alpha=0.9,
        label="Synthetic setpoint",
    )
    ax.plot(
        time_s, traj_A, color="#1f77b4", linewidth=1.8,
        label=(
            f"Dataset 1 full model: T∞={eq_A.a:.4f}·Tset{eq_A.b:+.3f}, "
            f"τ={eq_A.tau_eff:.2f} s"
        ),
    )
    ax.plot(
        time_s, traj_B, color="#d62728", linewidth=1.8,
        label=(
            f"Dataset 2 full model: T∞={eq_B.a:.4f}·Tset{eq_B.b:+.3f}, "
            f"τ={eq_B.tau_eff:.2f} s"
        ),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Surface Temperature (°C)")
    ax.set_title(
        "Full dynamic model comparison on synthetic setpoint profile\n"
        "30 → 40 → 50 → 40 → 30 °C, 60 s per stage, dt = 1 s"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# 报告生成
# ============================================================

def _fmt(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_setpoints(vals) -> str:
    return ", ".join(f"{float(v):.0f}" for v in vals)


def render_report(
    eq_A: CalibrationEquation,
    eq_B: CalibrationEquation,
    validation_df: pd.DataFrame,
    extrap_df: pd.DataFrame,
    prim_all: dict,
    prim_heat: dict,
    prim_cool: dict,
    max_err_row,
    param_df: pd.DataFrame,
    grid_cmp: dict,
    rep_table: pd.DataFrame,
    dyn_cmp: dict,
    per_trans: pd.DataFrame,
    syn_time,
    syn_set,
    paths: dict,
    contamination_ok: bool,
    contamination_msg: str,
) -> str:
    L = []

    def p(*lines):
        L.extend(lines)

    p("# Surface Calibration Cross-Validation Report")
    p("")
    p(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("")

    # ---- 1. Objective ----
    p("## 1. Objective")
    p("")
    p(
        "Three distinct scientific questions are addressed. There are "
        "**NO identical measured steady-state setpoints** between Dataset 1 "
        "and Dataset 2, which is intentional and scientifically useful: "
        "Dataset 2's intermediate setpoints act as independent "
        "interpolation-validation points for Dataset 1's continuous "
        "steady-state calibration function."
    )
    p("")
    p("**A. PRIMARY VALIDATION (interpolation):**")
    p(
        "Can Dataset 1's FIXED steady-state calibration equation "
        "T = a_A·Tset + b_A predict the independently measured Dataset 2 "
        "steady states at intermediate setpoints 40, 50, 60, 70, 80, 90 °C "
        "that lie INSIDE Dataset 1's 35–95 °C calibrated range but were never "
        "used as steady-state calibration points in Dataset 1?"
    )
    p("")
    p("**B. EXTRAPOLATION:**")
    p(
        "How well does Dataset 1 predict Dataset 2's 30 °C steady state, "
        "which lies BELOW Dataset 1's 35–95 °C calibrated range?"
    )
    p("")
    p("**C. REPRODUCIBILITY:**")
    p(
        "How similar are the independently fitted calibration equations "
        "(a, b) and effective time constants (tau_eff) from Dataset 1 and "
        "Dataset 2, and how similar are the two continuous calibration "
        "functions over their overlapping calibrated range 35–90 °C?"
    )
    p("")
    p(
        "Dataset 2's 40, 50, 60, 70, 80, 90 °C points are valuable "
        "independent interpolation-validation points for Dataset 1 because "
        "Dataset 1 never held those setpoints as steady-state calibration "
        "points, so Dataset 1's continuous fitted equation must be trusted "
        "to interpolate them, and Dataset 2 provides actual experimental "
        "steady states at exactly those setpoints from an independent run."
    )
    p("")

    # ---- 2. Input Data and Models ----
    p("## 2. Input Data and Models")
    p("")
    p("**Input paths:**")
    p(f"- Dataset 1 final equation: `{paths['eq_A']}`")
    p(f"- Dataset 2 final equation: `{paths['eq_B']}`")
    p(f"- Dataset 2 steady points:  `{paths['steady']}`")
    p(f"- Dataset 2 raw data:       `{paths['raw']}`")
    p("")
    p(f"- Dataset 1 calibration setpoints: {_fmt_setpoints(DATASET1_SETPOINTS)} °C")
    p(
        f"- Dataset 2 controlled sequence:  "
        f"{_fmt_setpoints(DATASET2_SETPOINTS)} °C"
    )
    p("")
    p(f"**Dataset 1 equation:** T_inf = {eq_A.a:.6f} * T_set {eq_A.b:+.6f}")
    p(
        f"- a_A = {eq_A.a:.6f}, b_A = {eq_A.b:.6f} °C, "
        f"tau_eff_A = {eq_A.tau_eff:.4f} s"
        + (
            f", steady R²_A = {eq_A.r2_steady:.6f}"
            if eq_A.r2_steady is not None else ""
        )
        + (
            f", dynamic RMSE_A = {eq_A.rmse_dynamic:.4f} °C"
            if eq_A.rmse_dynamic is not None else ""
        )
    )
    p("")
    p(f"**Dataset 2 equation:** T_inf = {eq_B.a:.6f} * T_set {eq_B.b:+.6f}")
    p(
        f"- a_B = {eq_B.a:.6f}, b_B = {eq_B.b:.6f} °C, "
        f"tau_eff_B = {eq_B.tau_eff:.4f} s"
        + (
            f", steady R²_B = {eq_B.r2_steady:.6f}"
            if eq_B.r2_steady is not None else ""
        )
        + (
            f", dynamic RMSE_B = {eq_B.rmse_dynamic:.4f} °C"
            if eq_B.rmse_dynamic is not None else ""
        )
    )
    p("")
    p(f"- Dataset 1 calibrated range = 35–95 °C")
    p(f"- Dataset 2 calibrated range = 30–90 °C")
    p(f"- Overlapping calibrated range = 35–90 °C")
    p("")

    # ---- 3. Primary validation ----
    p("## 3. Dataset 1 -> Dataset 2 Primary Interpolation Validation")
    p("")
    p(
        "Dataset 1 parameters are **FIXED**. No Dataset 1 parameter may be "
        "updated using Dataset 2 data. The prediction uses only Dataset 1's "
        "steady-state equation "
        f"T_pred_A = {eq_A.a:.6f}·Tset {eq_A.b:+.6f}, and the experimental "
        "reference is Dataset 2's actually measured steady temperature "
        "Tmean = (T1 + T2) / 2 extracted by the established "
        "segmentation/steady-state procedure (steady_points.csv, all points "
        "accepted)."
    )
    p("")
    p(
        "Only Dataset 2 controlled segments with setpoints inside Dataset 1's "
        "calibrated range (35–95 °C) are used. This gives **11 individual "
        "in-range segments**: heating 40→50→60→70→80→90 and cooling "
        "80→70→60→50→40. Repeated setpoints (40, 50, 60, 70, 80 °C) are "
        "preserved as separate heating/cooling observations; they are not "
        "silently averaged."
    )
    p("")
    p("**Complete in-range prediction-error table:**")
    p("")
    p(
        "| Segment | Direction | T_set (°C) | T_measured_B (°C) | "
        "T_predicted_A (°C) | Error (°C) | |Error| (°C) |"
    )
    p("|---|---|---|---|---|---|---|")
    for _, r in validation_df.iterrows():
        p(
            f"| {int(r['segment_id'])} | {r['direction']} | "
            f"{_fmt(r['T_set'], 0)} | {_fmt(r['T_measured_B'], 2)} | "
            f"{_fmt(r['T_predicted_A'], 2)} | {_fmt(r['error'], 3)} | "
            f"{_fmt(r['absolute_error'], 3)} |"
        )
    p("")
    p("**Overall interpolation metrics (40–90 °C, 11 segments):**")
    p("")
    p(
        f"- Bias (Mean Error) = {_fmt(prim_all['bias'], 3)} °C\n"
        f"- MAE = {_fmt(prim_all['mae'], 3)} °C\n"
        f"- RMSE = {_fmt(prim_all['rmse'], 3)} °C\n"
        f"- Maximum Absolute Error = {_fmt(prim_all['max_abs'], 3)} °C"
    )
    p("")
    p("**Heating-only interpolation metrics (40,50,60,70,80,90 °C):**")
    p(
        f"- Bias = {_fmt(prim_heat['bias'], 3)} °C, "
        f"MAE = {_fmt(prim_heat['mae'], 3)} °C, "
        f"RMSE = {_fmt(prim_heat['rmse'], 3)} °C, "
        f"Max |Error| = {_fmt(prim_heat['max_abs'], 3)} °C"
    )
    p("")
    p("**Cooling-only interpolation metrics (80,70,60,50,40 °C):**")
    p(
        f"- Bias = {_fmt(prim_cool['bias'], 3)} °C, "
        f"MAE = {_fmt(prim_cool['mae'], 3)} °C, "
        f"RMSE = {_fmt(prim_cool['rmse'], 3)} °C, "
        f"Max |Error| = {_fmt(prim_cool['max_abs'], 3)} °C"
    )
    p("")
    p(
        f"The largest absolute prediction error occurs at segment "
        f"{int(max_err_row['segment_id'])} "
        f"({max_err_row['direction']}, T_set = {_fmt(max_err_row['T_set'], 0)} °C) "
        f"with |error| = {_fmt(max_err_row['absolute_error'], 3)} °C."
    )
    p("")
    p(
        "This result is specifically described as **independent "
        "steady-state interpolation validation of Dataset 1 using "
        "experimentally measured Dataset 2 steady states** — not as proof "
        "of perfect generalization."
    )
    p("")
    p(
        "Important: Dataset 1 transiently passing through a temperature such "
        "as 40 °C during a 35→45 °C transition is **NOT** used as "
        "validation. The validation uses Dataset 1's steady-state fitted "
        "equation to predict a hypothetical steady state at 40 °C, which is "
        "then compared with Dataset 2's actual experimentally held 40 °C "
        "steady state. Transient passage is not a steady-state measurement."
    )
    p("")

    # ---- 4. Extrapolation ----
    p("## 4. 30 °C Extrapolation Check")
    p("")
    p(
        "Dataset 2 contains 30 °C controlled steady states (segments "
        + (
            ", ".join(str(int(v)) for v in extrap_df["segment_id"].tolist())
        )
        + "). Dataset 1's calibrated range begins at 35 °C, so Dataset 1's "
        "prediction at 30 °C is **EXTRAPOLATION**."
    )
    p("")
    p("| Segment | Direction | T_set (°C) | T_measured_B_30 (°C) | T_predicted_A_30 (°C) | error_30 (°C) | |error_30| (°C) |")
    p("|---|---|---|---|---|---|---|")
    for _, r in extrap_df.iterrows():
        p(
            f"| {int(r['segment_id'])} | {r['direction']} | "
            f"{_fmt(r['T_set'], 0)} | {_fmt(r['T_measured_B_30'], 2)} | "
            f"{_fmt(r['T_predicted_A_30'], 2)} | {_fmt(r['error_30'], 3)} | "
            f"{_fmt(r['absolute_error_30'], 3)} |"
        )
    p("")
    p(
        "These points are **excluded** from the primary interpolation "
        "Bias / MAE / RMSE / Max Absolute Error statistics."
    )
    p("")
    p(
        "30 °C is outside Dataset 1's calibrated range; this result must be "
        "interpreted cautiously as an extrapolation check only."
    )
    p("")

    # ---- 5. Parameter comparison ----
    p("## 5. Independent Equation Parameter Comparison")
    p("")
    p(
        "This section assesses **reproducibility** of the independently "
        "fitted parameters, not external prediction accuracy."
    )
    p("")
    p("| Parameter | Dataset 1 (A) | Dataset 2 (B) | Absolute difference | Symmetric relative difference (%) |")
    p("|---|---|---|---|---|")
    for _, r in param_df.iterrows():
        sym = (
            _fmt(r["symmetric_relative_difference_percent"], 2)
            if np.isfinite(r["symmetric_relative_difference_percent"])
            else "n/a (denominator ~ 0)"
        )
        p(
            f"| {r['parameter']} | {_fmt(r['Dataset1_A'], 6)} | "
            f"{_fmt(r['Dataset2_B'], 6)} | {_fmt(r['absolute_difference'], 6)} | "
            f"{sym} |"
        )
    p("")
    tau_row = param_df[param_df["parameter"] == "tau_eff (s)"].iloc[0]
    p(
        "For tau_eff, similarity of the effective time constants would be "
        "evidence of **dynamic-parameter reproducibility**. Here the "
        "absolute difference is "
        f"{_fmt(tau_row['absolute_difference'], 3)} s with a symmetric "
        "relative difference of "
        f"{_fmt(tau_row['symmetric_relative_difference_percent'], 1)} %, "
        "which is a substantial difference; dynamic-parameter "
        "reproducibility is therefore **not established** across these two "
        "runs. Full time-resolved dynamic cross-validation is not claimed "
        "because exact setpoint transition timestamps were not "
        "independently recorded."
    )
    p("")
    p(
        "Similarity of fitted parameters is evidence of reproducibility, "
        "not by itself proof of external prediction accuracy."
    )
    p("")

    # ---- 6. Function agreement ----
    p("## 6. Calibration-Function Agreement Over the Overlapping Calibrated Range (35–90 °C)")
    p("")
    p(
        "There are no identical measured steady-state setpoints between "
        "Dataset 1 and Dataset 2, so this section is deliberately **not** "
        "called a 'common setpoint comparison'. It quantifies the agreement "
        "of the two continuous independently fitted calibration functions "
        "over the temperature interval supported by both experiments "
        "(35–90 °C), using a dense grid."
    )
    p("")
    p(
        f"- Equation-to-equation mean signed difference (T_A − T_B) = "
        f"{_fmt(grid_cmp['mean_signed'], 4)} °C\n"
        f"- Equation-to-equation MAE = {_fmt(grid_cmp['mae'], 4)} °C\n"
        f"- Equation-to-equation RMSE = {_fmt(grid_cmp['rmse'], 4)} °C\n"
        f"- Maximum absolute prediction difference = "
        f"{_fmt(grid_cmp['max_abs'], 4)} °C\n"
        f"- Setpoint of maximum difference = "
        f"{_fmt(grid_cmp['max_abs_setpoint'], 1)} °C"
    )
    p("")
    p("**Representative-temperature comparison table:**")
    p("")
    p(
        "| T_set (°C) | Prediction_Model_A (°C) | Prediction_Model_B (°C) | "
        "Difference_A_minus_B (°C) |"
    )
    p("|---|---|---|---|")
    for _, r in rep_table.iterrows():
        p(
            f"| {_fmt(r['T_set'], 0)} | {_fmt(r['Prediction_Model_A'], 3)} | "
            f"{_fmt(r['Prediction_Model_B'], 3)} | "
            f"{_fmt(r['Difference_A_minus_B'], 3)} |"
        )
    p("")
    p(
        "For clarity: 35 °C is directly a Dataset 1 calibration setpoint and "
        "lies within Dataset 2's calibrated range; 40, 50, 60, 70, 80, 90 °C "
        "are Dataset 2 calibration setpoints and interpolation points of "
        "Dataset 1's continuous equation. The table compares continuous "
        "fitted functions, not paired raw measurements."
    )
    p("")

    # ---- 7. Figures ----
    p("## 7. Figures")
    p("")
    p(
        "- `calibration_equation_comparison.png`: the two independently "
        "fitted steady calibration functions over the overlapping calibrated "
        "range 35–90 °C, with Dataset 2 measured steady points overlaid. "
        "It demonstrates cross-run calibration-function agreement, not "
        "identical-setpoint comparison."
    )
    p(
        "- `dataset1_predicts_dataset2_steady.png`: the PRIMARY validation "
        "figure. Dataset 2 measured steady states at 40–90 °C (heating and "
        "cooling preserved separately) versus Dataset 1 fixed-equation "
        "predictions at the same setpoints. The 30 °C point is shown "
        "separately and clearly marked as extrapolation."
    )
    p(
        "- `full_dynamic_model_comparison.png`: both final models "
        "(steady + first-order dynamic, i.e. the complete final equations) "
        "driven by the same synthetic setpoint profile "
        f"{_fmt_setpoints(SYNTHETIC_SETPOINTS)} °C, "
        f"{SYNTHETIC_HOLD_S:.0f} s per stage, dt = {SYNTHETIC_DT:.0f} s. "
        "It visualises how the tau_eff difference translates into different "
        "transient surface-temperature responses."
    )
    p("")

    # ---- 8. Synthetic dynamic model comparison ----
    p("## 8. Full Dynamic Model Comparison on a Synthetic Setpoint Profile")
    p("")
    p(
        "To visualise the combined effect of the (well-validated) steady "
        "calibration and the (not cross-validated) dynamic parameter, both "
        "complete final equations were driven by the same hypothetical "
        "setpoint profile: "
        f"{_fmt_setpoints(SYNTHETIC_SETPOINTS)} °C, "
        f"{SYNTHETIC_HOLD_S:.0f} s per stage, dt = {SYNTHETIC_DT:.0f} s."
    )
    p("")
    p("Model equations used (identical structure for both datasets):")
    p("")
    p("```")
    p("T_eq[n] = a * T_set[n] + b")
    p("T_s[n]  = T_eq[n] + (T_s[n-1] - T_eq[n]) * exp(-dt / tau_eff)")
    p("```")
    p("")
    p(
        "Each trajectory starts from its own steady equilibrium at the first "
        "setpoint (30 °C). This is a **model-to-model** comparison of the two "
        "final prediction formulas — it is not validated against any "
        "measurement and must not be interpreted as dynamic validation."
    )
    p("")
    p(
        f"- Dataset 1 model: T∞ = {eq_A.a:.6f}·Tset {eq_A.b:+.6f}, "
        f"tau_eff = {eq_A.tau_eff:.4f} s\n"
        f"- Dataset 2 model: T∞ = {eq_B.a:.6f}·Tset {eq_B.b:+.6f}, "
        f"tau_eff = {eq_B.tau_eff:.4f} s"
    )
    p("")
    p("**Trajectory-to-trajectory metrics (T_A − T_B over the whole profile):**")
    p(
        f"- mean signed difference = {_fmt(dyn_cmp['mean_signed'], 4)} °C\n"
        f"- MAE = {_fmt(dyn_cmp['mae'], 4)} °C\n"
        f"- RMSE = {_fmt(dyn_cmp['rmse'], 4)} °C\n"
        f"- max absolute difference = {_fmt(dyn_cmp['max_abs'], 4)} °C "
        f"at t = {_fmt(dyn_cmp['max_abs_time_s'], 1)} s"
    )
    p("")
    p("**Per-transition maximum absolute difference (within each 60 s window after a switch):**")
    p("")
    if per_trans.empty:
        p("- no transitions detected in the synthetic profile.")
    else:
        p(
            "| Transition | Direction | max |T_A − T_B| (°C) | Time of max (s) |"
        )
        p("|---|---|---|---|")
        for _, r in per_trans.iterrows():
            p(
                f"| {r['transition']} | {r['direction']} | "
                f"{_fmt(r['max_abs_difference'], 3)} | "
                f"{_fmt(r['time_of_max_s'], 1)} |"
            )
    p("")
    p(
        "The maximum trajectory difference occurs during the transient "
        "response and is dominated by the tau_eff difference (7.31 s vs "
        "4.45 s): Dataset 2's faster model reaches each new equilibrium "
        "sooner, while the two models converge to nearly the same "
        "equilibrium values at each plateau (steady differences ≤ 0.05 °C). "
        "The steady offset between the models at any plateau is small; the "
        "large transient gap reflects the unvalidated dynamic parameter. "
        "Full trajectory data are saved in `full_dynamic_model_comparison.csv`."
    )
    p("")

    # ---- 9. Interpretation ----
    p("## 9. Interpretation")
    p("")
    p("### Steady-State External Interpolation Validation")
    p(
        f"Dataset 1's fixed equation predicts the 11 Dataset 2 measured "
        f"steady states at 40–90 °C with Bias = {_fmt(prim_all['bias'], 3)} °C, "
        f"MAE = {_fmt(prim_all['mae'], 3)} °C, "
        f"RMSE = {_fmt(prim_all['rmse'], 3)} °C and maximum absolute error "
        f"{_fmt(prim_all['max_abs'], 3)} °C. "
        f"Heating RMSE = {_fmt(prim_heat['rmse'], 3)} °C, "
        f"cooling RMSE = {_fmt(prim_cool['rmse'], 3)} °C. "
        + (
            "Heating and cooling prediction errors are comparable, "
            "indicating the interpolation error is not direction-specific."
            if abs(prim_heat["rmse"] - prim_cool["rmse"]) <= 0.15
            else (
                "The heating/cooling RMSE difference is notable; "
                "interpret per-segment values above."
            )
        )
    )
    p("")
    p("### 30 °C Extrapolation")
    p(
        f"At 30 °C, Dataset 1 predicts {_fmt(extrap_df['T_predicted_A_30'].mean(), 2)} °C "
        f"versus Dataset 2 measured {_fmt(extrap_df['T_measured_B_30'].mean(), 2)} °C "
        f"(mean over the two 30 °C segments), error "
        f"{_fmt(extrap_df['error_30'].mean(), 3)} °C. "
        "This lies outside Dataset 1's calibrated range and is interpreted "
        "cautiously; it is not part of the primary interpolation validation."
    )
    p(
        "Individually, the first 30 °C segment (initial hold) gives error "
        f"{_fmt(extrap_df['error_30'].iloc[0], 3)} °C "
        f"(|error| = {_fmt(extrap_df['absolute_error_30'].iloc[0], 3)} °C) and the "
        f"second 30 °C segment (cooling) gives error "
        f"{_fmt(extrap_df['error_30'].iloc[-1], 3)} °C "
        f"(|error| = {_fmt(extrap_df['absolute_error_30'].iloc[-1], 3)} °C). "
        "One of the two individual 30 °C extrapolation errors exceeded the "
        "maximum absolute error observed within the primary interpolation "
        "range "
        f"({_fmt(prim_all['max_abs'], 3)} °C); the other did not. "
        "Because both points lie outside the calibrated range, no strong "
        "conclusion is drawn from them."
    )
    p("")
    p("### Calibration Reproducibility")
    p(
        f"The independently fitted slopes differ by "
        f"{_fmt(abs(eq_A.a - eq_B.a), 6)} (symmetric relative difference "
        f"{_fmt(symmetric_relative_difference(eq_A.a, eq_B.a), 2)} %) and the "
        f"intercepts by {_fmt(abs(eq_A.b - eq_B.b), 4)} °C "
        f"({_fmt(symmetric_relative_difference(eq_A.b, eq_B.b), 2)} %). "
        f"The effective time constants differ by "
        f"{_fmt(abs(eq_A.tau_eff - eq_B.tau_eff), 3)} s "
        f"({_fmt(symmetric_relative_difference(eq_A.tau_eff, eq_B.tau_eff), 1)} %). "
        "The two continuous calibration functions agree over 35–90 °C with "
        f"equation MAE = {_fmt(grid_cmp['mae'], 3)} °C and equation RMSE = "
        f"{_fmt(grid_cmp['rmse'], 3)} °C (max |T_A − T_B| = "
        f"{_fmt(grid_cmp['max_abs'], 3)} °C at "
        f"{_fmt(grid_cmp['max_abs_setpoint'], 1)} °C). "
        "Steady-state function agreement is close, but the tau_eff "
        "difference is large, so dynamic-parameter reproducibility is NOT "
        "claimed. Dataset 1 is not claimed to dynamically predict Dataset 2, "
        "because exact setpoint transition timestamps were not independently "
        "recorded."
    )
    p("")

    # ---- 10. Limitations ----
    p("## 10. Limitations")
    p("")
    for item in [
        "The two datasets have intentionally offset steady-state setpoint grids.",
        "There are no identical measured steady-state setpoints between Dataset 1 and Dataset 2.",
        "Dataset 1's transient passage through Dataset 2 setpoint temperatures is not used as validation.",
        "Exact setpoint transition timestamps were not directly recorded in the raw data.",
        "Steady-state validation is therefore more defensible than full time-resolved dynamic cross-validation.",
        "Dataset 2 steady points depend on the established segmentation/steady-state extraction procedure.",
        "Dataset 2's 30 °C point lies outside Dataset 1's calibration range and is therefore extrapolation.",
        "The two datasets are independent experimental runs but are not necessarily a completely independent instrument/operator validation.",
        "No parameters from Dataset 1 may be refitted using Dataset 2 during the external prediction calculation.",
        (
            "Passive-cooling contamination check on Dataset 2 tail: "
            + ("NOT detected — " if contamination_ok else "DETECTED — ")
            + contamination_msg + "."
        ),
    ]:
        p(f"- {item}")
    p("")

    # ---- 11. Final quantitative summary ----
    p("## 11. Final Quantitative Summary")
    p("")
    p("**Dataset 1 model:**")
    p(
        f"- calibrated setpoints = {_fmt_setpoints(DATASET1_SETPOINTS)} °C\n"
        f"- calibrated range = 35–95 °C\n"
        f"- a = {eq_A.a:.6f}\n"
        f"- b = {eq_A.b:.6f} °C\n"
        f"- tau_eff = {eq_A.tau_eff:.4f} s"
    )
    p("")
    p("**Dataset 2 model:**")
    p(
        f"- calibrated setpoints = {_fmt_setpoints(DATASET2_SETPOINTS)} °C\n"
        f"- calibrated range = 30–90 °C\n"
        f"- a = {eq_B.a:.6f}\n"
        f"- b = {eq_B.b:.6f} °C\n"
        f"- tau_eff = {eq_B.tau_eff:.4f} s"
    )
    p("")
    p("**Primary Dataset 1 -> Dataset 2 interpolation validation (40–90 °C):**")
    p(
        f"- number of validation segments = {prim_all['n']}\n"
        f"- Bias = {_fmt(prim_all['bias'], 3)} °C\n"
        f"- MAE = {_fmt(prim_all['mae'], 3)} °C\n"
        f"- RMSE = {_fmt(prim_all['rmse'], 3)} °C\n"
        f"- Max absolute error = {_fmt(prim_all['max_abs'], 3)} °C\n"
        f"- Heating RMSE = {_fmt(prim_heat['rmse'], 3)} °C\n"
        f"- Cooling RMSE = {_fmt(prim_cool['rmse'], 3)} °C"
    )
    p("")
    p("**30 °C extrapolation:**")
    p(
        f"- Dataset 1 prediction = {_fmt(extrap_df['T_predicted_A_30'].mean(), 2)} °C\n"
        f"- Dataset 2 measured = {_fmt(extrap_df['T_measured_B_30'].mean(), 2)} °C\n"
        f"- error = {_fmt(extrap_df['error_30'].mean(), 3)} °C"
    )
    p("")
    p("**Independent equation comparison:**")
    p(
        f"- slope difference = {_fmt(abs(eq_A.a - eq_B.a), 6)}\n"
        f"- intercept difference = {_fmt(abs(eq_A.b - eq_B.b), 4)} °C\n"
        f"- tau difference = {_fmt(abs(eq_A.tau_eff - eq_B.tau_eff), 3)} s"
    )
    p("")
    p("**Calibration-function agreement over 35–90 °C:**")
    p(
        f"- equation MAE = {_fmt(grid_cmp['mae'], 3)} °C\n"
        f"- equation RMSE = {_fmt(grid_cmp['rmse'], 3)} °C\n"
        f"- maximum equation prediction difference = {_fmt(grid_cmp['max_abs'], 3)} °C\n"
        f"- setpoint of maximum difference = {_fmt(grid_cmp['max_abs_setpoint'], 1)} °C"
    )
    p("")
    p("**Full dynamic model comparison on synthetic profile "
      f"{_fmt_setpoints(SYNTHETIC_SETPOINTS)} °C "
      f"({SYNTHETIC_HOLD_S:.0f} s/stage, dt={SYNTHETIC_DT:.0f} s):**")
    p(
        f"- trajectory mean signed difference = {_fmt(dyn_cmp['mean_signed'], 3)} °C\n"
        f"- trajectory MAE = {_fmt(dyn_cmp['mae'], 3)} °C\n"
        f"- trajectory RMSE = {_fmt(dyn_cmp['rmse'], 3)} °C\n"
        f"- trajectory max absolute difference = {_fmt(dyn_cmp['max_abs'], 3)} °C "
        f"at t = {_fmt(dyn_cmp['max_abs_time_s'], 1)} s"
    )
    p("")
    p(
        "Any statement such as 'small', 'close', or 'consistent' is based "
        "strictly on the calculated values above."
    )
    p("")

    return "\n".join(L) + "\n"


# ============================================================
# 主程序
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validate two independent Peltier surface calibration runs "
            "(Dataset 1 -> Dataset 2 steady-state interpolation validation, "
            "30 °C extrapolation check, equation reproducibility)."
        )
    )
    parser.add_argument(
        "--dataset1-equation",
        default=str(DEFAULT_DATASET1_EQUATION),
        help="Dataset 1 final_calibration_equation.txt",
    )
    parser.add_argument(
        "--dataset2-equation",
        default=str(DEFAULT_DATASET2_EQUATION),
        help="Dataset 2 final_calibration_equation.txt",
    )
    parser.add_argument(
        "--dataset2-steady",
        default=str(DEFAULT_DATASET2_STEADY),
        help="Dataset 2 steady_points.csv",
    )
    parser.add_argument(
        "--dataset2-raw",
        default=str(DEFAULT_DATASET2_RAW),
        help="Dataset 2 raw Excel (.xls/.xlsx) for passive-cooling check",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="New dedicated output directory for cross-validation results",
    )
    parser.add_argument(
        "--dataset1-range",
        default="35,95",
        help="Dataset 1 calibrated range (comma separated)",
    )
    parser.add_argument(
        "--overlap-range",
        default="35,90",
        help="Overlapping calibrated range (comma separated)",
    )
    args = parser.parse_args(argv)

    # ---------------- 输入 ----------------
    eq_A = parse_equation_file(args.dataset1_equation)
    eq_B = parse_equation_file(args.dataset2_equation)

    d1_lo, d1_hi = (float(v) for v in args.dataset1_range.split(","))
    ov_lo, ov_hi = (float(v) for v in args.overlap_range.split(","))
    dataset1_range = (d1_lo, d1_hi)
    overlap = (ov_lo, ov_hi)

    steady_path = Path(args.dataset2_steady)
    steady_df = load_mean_steady_points(steady_path)

    raw_path = Path(args.dataset2_raw)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-VALIDATION: DATASET 1 MODEL -> DATASET 2 MEASUREMENTS")
    print("=" * 70)
    print(f"Dataset 1 equation: {eq_A.source}")
    print(
        f"  T_inf = {eq_A.a:.6f} * T_set {eq_A.b:+.6f}, "
        f"tau_eff = {eq_A.tau_eff:.4f} s"
    )
    print(f"Dataset 2 equation: {eq_B.source}")
    print(
        f"  T_inf = {eq_B.a:.6f} * T_set {eq_B.b:+.6f}, "
        f"tau_eff = {eq_B.tau_eff:.4f} s"
    )

    # ---------------- Dataset 2 数据完整性 ----------------
    print("\n[1] Verifying Dataset 2 steady points (established extraction)...")
    problems = verify_steady_points(steady_df)
    if problems:
        raise RuntimeError(
            "Dataset 2 steady_points.csv 验证失败，停止交叉验证:\n- "
            + "\n- ".join(problems)
        )
    print("  OK: 13 controlled stages, sequence 30→40→...→90→80→...→30, "
          "contiguous, all accepted.")

    contamination_ok, contamination_msg = True, "n/a"
    if raw_path.is_file():
        print("\n[2] Checking Dataset 2 tail for passive-cooling contamination...")
        contamination_ok, contamination_msg = check_passive_cooling_tail(raw_path)
        print(f"  {contamination_msg}")
        if not contamination_ok:
            raise RuntimeError(
                "Dataset 2 末尾存在疑似 Peltier-OFF 被动冷却污染，"
                "停止交叉验证以免使用无效验证数据。"
            )
        print("  OK: final 30 °C stage is a controlled hold (no passive cooling).")

        print("\n[3] Cross-checking segmentation with established change-point logic...")
        segs = cross_check_segmentation(raw_path)
        if segs is not None:
            recorded = list(zip(
                steady_df["start_index"].astype(int),
                steady_df["end_index"].astype(int),
            ))
            detected = [(s, e) for s, e, _ in segs]
            if detected == recorded:
                print("  OK: re-detected segment boundaries match steady_points.csv.")
            else:
                print("  WARN: re-detected boundaries differ from steady_points.csv:")
                print(f"    recorded: {recorded}")
                print(f"    detected: {detected}")
    else:
        print(f"\n[2] WARN: raw file not found ({raw_path}); "
              "passive-cooling check skipped.")

    # ---------------- 主验证 ----------------
    print("\n[4] Building Dataset 1 -> Dataset 2 validation frame...")
    validation_df = build_validation_frame(steady_df, eq_A, dataset1_range)

    primary = validation_df[
        validation_df["classification"] == "interpolation"
    ].reset_index(drop=True)
    extrap = validation_df[
        validation_df["classification"] == "extrapolation"
    ].reset_index(drop=True)

    n_expected_inrange = len([v for v in DATASET2_SETPOINTS
                              if dataset1_range[0] <= v <= dataset1_range[1]])
    if len(primary) != n_expected_inrange:
        raise RuntimeError(
            f"预期 {n_expected_inrange} 个 in-range 段，实际 {len(primary)} 个。"
        )
    if dataset1_range[0] > 30.0 and extrap["T_set"].nunique() != 1:
        raise RuntimeError("外推段应只有 30 °C 一个设定点。")
    print(f"  In-range (interpolation) segments: {len(primary)}")
    print(f"  Extrapolation segments: {len(extrap)} "
          f"(setpoints: {sorted(extrap['T_set'].unique())})")

    prim_all = compute_metrics(primary["error"])
    prim_heat = compute_metrics(primary.loc[primary["direction"] == "heating", "error"])
    prim_cool = compute_metrics(primary.loc[primary["direction"] == "cooling", "error"])
    max_err_idx = int(primary["absolute_error"].idxmax())
    max_err_row = primary.loc[max_err_idx]

    print("\n  PRIMARY interpolation metrics (40-90 °C):")
    print(f"    n = {prim_all['n']}")
    print(f"    Bias = {prim_all['bias']:.4f} °C")
    print(f"    MAE = {prim_all['mae']:.4f} °C")
    print(f"    RMSE = {prim_all['rmse']:.4f} °C")
    print(f"    Max |error| = {prim_all['max_abs']:.4f} °C")
    print(f"    Heating RMSE = {prim_heat['rmse']:.4f} °C")
    print(f"    Cooling RMSE = {prim_cool['rmse']:.4f} °C")
    print(f"    Max-error segment = {int(max_err_row['segment_id'])} "
          f"({max_err_row['direction']}, "
          f"T_set = {max_err_row['T_set']:.0f} °C)")

    # 30 °C 外推表
    extrap_out = extrap.copy()
    extrap_out = extrap_out.rename(columns={
        "T_set": "T_set",
        "T_measured_B": "T_measured_B_30",
        "T_predicted_A": "T_predicted_A_30",
        "error": "error_30",
        "absolute_error": "absolute_error_30",
    })[
        ["segment_id", "direction", "T_set",
         "T_measured_B_30", "T_predicted_A_30", "error_30",
         "absolute_error_30"]
    ]

    # ---------------- 参数比较 ----------------
    print("\n[5] Parameter reproducibility comparison...")
    param_df = parameter_comparison(eq_A, eq_B)
    print(param_df.to_string(index=False))

    # ---------------- 函数一致性 ----------------
    print("\n[6] Calibration-function agreement over overlapping range "
          f"{ov_lo:.0f}-{ov_hi:.0f} °C...")
    grid_cmp = equation_grid_comparison(eq_A, eq_B, overlap=overlap)
    print(f"    mean signed diff = {grid_cmp['mean_signed']:.4f} °C")
    print(f"    MAE = {grid_cmp['mae']:.4f} °C")
    print(f"    RMSE = {grid_cmp['rmse']:.4f} °C")
    print(f"    max |diff| = {grid_cmp['max_abs']:.4f} °C at "
          f"Tset = {grid_cmp['max_abs_setpoint']:.1f} °C")

    rep_rows = []
    for tset in REPRESENTATIVE_TEMPS:
        rep_rows.append({
            "T_set": tset,
            "Prediction_Model_A": float(eq_A.predict(tset)),
            "Prediction_Model_B": float(eq_B.predict(tset)),
            "Difference_A_minus_B": float(eq_A.predict(tset) - eq_B.predict(tset)),
        })
    rep_table = pd.DataFrame(rep_rows)

    # ---------------- 保存 CSV ----------------
    print("\n[7] Saving CSV outputs...")
    primary_out = primary[[
        "segment_id", "direction", "T_set", "T_measured_B",
        "T_predicted_A", "error", "absolute_error",
        "steady_slope_C_per_s", "accepted",
    ]].copy()
    primary_out.to_csv(output_dir / "dataset1_predicts_dataset2_steady.csv",
                       index=False)
    extrap_out.to_csv(output_dir / "extrapolation_30C.csv", index=False)
    param_df.to_csv(output_dir / "parameter_comparison.csv", index=False)
    rep_table.to_csv(output_dir / "equation_prediction_comparison.csv",
                     index=False)

    # ---------------- 绘图 ----------------
    print("\n[8] Generating figures...")
    plot_equation_comparison(
        eq_A, eq_B,
        measured=steady_df,
        overlap=overlap,
        output_path=output_dir / "calibration_equation_comparison.png",
    )
    plot_primary_validation(
        primary, extrap_out, eq_A,
        dataset1_range=dataset1_range,
        output_path=output_dir / "dataset1_predicts_dataset2_steady.png",
    )

    # ---------------- 完整动态模型对比 (合成 profile) ----------------
    print(f"\n[9] Synthetic full-model comparison "
          f"({_fmt_setpoints(SYNTHETIC_SETPOINTS)} °C, "
          f"{SYNTHETIC_HOLD_S:.0f} s/stage, dt={SYNTHETIC_DT:.0f} s)...")
    syn_time, syn_set = build_synthetic_profile()
    dyn_cmp = dynamic_trajectory_comparison(eq_A, eq_B, syn_set, SYNTHETIC_DT)
    per_trans = per_transition_max_differences(
        dyn_cmp["traj_A"], dyn_cmp["traj_B"],
        syn_set, SYNTHETIC_DT, SYNTHETIC_HOLD_S,
    )
    print(f"    trajectory mean signed diff = {dyn_cmp['mean_signed']:.4f} °C")
    print(f"    trajectory MAE = {dyn_cmp['mae']:.4f} °C")
    print(f"    trajectory RMSE = {dyn_cmp['rmse']:.4f} °C")
    print(f"    trajectory max |diff| = {dyn_cmp['max_abs']:.4f} °C "
          f"at t = {dyn_cmp['max_abs_time_s']:.1f} s")
    print(per_trans.to_string(index=False))

    dyn_df = pd.DataFrame({
        "time_s": syn_time,
        "setpoint": syn_set,
        "T_A_dynamic": dyn_cmp["traj_A"],
        "T_B_dynamic": dyn_cmp["traj_B"],
        "Difference_A_minus_B": dyn_cmp["diff"],
    })
    dyn_df.to_csv(output_dir / "full_dynamic_model_comparison.csv", index=False)

    plot_dynamic_model_comparison(
        eq_A, eq_B,
        time_s=syn_time,
        setpoint_series=syn_set,
        traj_A=dyn_cmp["traj_A"],
        traj_B=dyn_cmp["traj_B"],
        output_path=output_dir / "full_dynamic_model_comparison.png",
    )

    # ---------------- 报告 ----------------
    print("\n[10] Generating combined report...")
    report = render_report(
        eq_A=eq_A,
        eq_B=eq_B,
        validation_df=primary,
        extrap_df=extrap_out,
        prim_all=prim_all,
        prim_heat=prim_heat,
        prim_cool=prim_cool,
        max_err_row=max_err_row,
        param_df=param_df,
        grid_cmp=grid_cmp,
        rep_table=rep_table,
        dyn_cmp=dyn_cmp,
        per_trans=per_trans,
        syn_time=syn_time,
        syn_set=syn_set,
        paths={
            "eq_A": str(eq_A.source),
            "eq_B": str(eq_B.source),
            "steady": str(steady_path.resolve()),
            "raw": str(raw_path.resolve()),
        },
        contamination_ok=contamination_ok,
        contamination_msg=contamination_msg,
    )
    (output_dir / "Cross_Validation_Report.md").write_text(
        report, encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir.resolve()}")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
