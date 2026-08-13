#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Peltier surface temperature calibration
=======================================

目的
----
1) 稳态校准:
       T_inf = a * T_set + b

2) 动态校准（一阶有效热响应）:
       T_s(t) = T_inf,j + [T_s(t_j) - T_inf,j] * exp(-(t-t_j)/tau_eff)

   对任意随时间变化的 setpoint profile，使用等价的一阶递推:
       T_eq[n] = a*T_set[n] + b
       T_s[n]  = T_eq[n] + (T_s[n-1] - T_eq[n]) * exp(-dt/tau_eff)

3) T1、T2 分别拟合；同时对 T_mean=(T1+T2)/2 单独拟合。
   推荐最终使用 Mean 模型，因为它直接对两传感器平均后的温度轨迹拟合，
   比简单平均非线性参数 tau 更稳健。

输入 Excel
----------
优先格式:
    Time(optional) | Set | T1 | T2

如果 Excel 没有 Set 列:
    需要通过 --setpoints 手动提供每个稳态段对应的设定温度，
    程序会从 T1/T2 平均轨迹自动寻找阶跃起点。

示例
----
有 Set 列:
python peltier_surface_calibration.py data.xlsx --set-col Set --t1-col T1 --t2-col T2 --dt 1

只有 T1/T2:
python peltier_surface_calibration.py data.xlsx --set-col NONE \
    --setpoints 40,50,60,70,80,90,100,90,80,70,60,50,30 \
    --dt 1

输出
----
calibration_output/
    calibration_summary.csv
    steady_points.csv
    dynamic_steps.csv
    trace_with_fit.csv
    calibration_params.json
    steady_calibration.png
    dynamic_calibration.png
    final_calibration_equation.txt
    final_model_validation.csv
    final_model_validation.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# ============================================================
# 功能块 1：基础工具
# ============================================================

@dataclass
class Segment:
    start: int
    end: int                 # Python slice: [start:end)
    setpoint: float


@dataclass
class ModelResult:
    name: str
    a: float
    b: float
    r2_steady: float
    tau_eff: float
    tau_heating: float
    tau_cooling: float
    rmse_dynamic: float


def linear_fit(x: np.ndarray, y: np.ndarray):
    """y = a*x + b"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a, b = np.polyfit(x, y, 1)
    pred = a * x + b

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return float(a), float(b), float(r2)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def parse_setpoints(text: Optional[str]):
    if text is None or str(text).strip() == "":
        return None
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


# ============================================================
# 功能块 2：寻找设定温度分段
# ============================================================

def segments_from_set_column(set_values: np.ndarray, tol: float = 1e-9):
    """
    如果 Excel 有 Set 列，直接按 Set 值变化位置分段。
    """
    s = np.asarray(set_values, dtype=float)

    valid = np.isfinite(s)
    if not valid.all():
        raise ValueError("Set 列中存在空值/非数字，请先补齐。")

    change_idx = np.where(np.abs(np.diff(s)) > tol)[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(s)]

    segments = []
    for st, en in zip(starts, ends):
        segments.append(Segment(int(st), int(en), float(np.median(s[st:en]))))

    return segments


def detect_change_points_from_temperature(
    mean_temp: np.ndarray,
    dt: float,
    change_threshold: float = 0.35,
    smooth_window: int = 3,
    min_gap_s: float = 20.0,
):
    """
    当 Excel 没有 Set 列时，从 T1/T2 平均轨迹自动寻找明显阶跃。

    方法:
    1. rolling median 平滑
    2. 计算相邻点温度变化率 |dT/dt|
    3. 超过阈值视为动态段
    4. 连续超阈值点合并，只保留每段的起点
    """
    y = pd.Series(np.asarray(mean_temp, dtype=float))
    smooth = y.rolling(
        window=max(1, int(smooth_window)),
        center=True,
        min_periods=1
    ).median().to_numpy()

    rate = np.abs(np.diff(smooth)) / dt
    raw = np.where(rate >= change_threshold)[0] + 1

    if len(raw) == 0:
        return []

    # 合并连续/相邻的高变化率点
    groups = [[int(raw[0])]]
    for idx in raw[1:]:
        idx = int(idx)
        if idx <= groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    candidates = [g[0] for g in groups]

    # 再按最小间隔过滤，防止一次阶跃被分成多个 change point
    min_gap_n = max(1, int(round(min_gap_s / dt)))
    accepted = []
    for idx in candidates:
        if not accepted or idx - accepted[-1] >= min_gap_n:
            accepted.append(idx)

    return accepted


def segments_from_detected_changes(
    mean_temp: np.ndarray,
    setpoint_sequence: list[float],
    dt: float,
    change_threshold: float,
    smooth_window: int,
    min_gap_s: float,
):
    cps = detect_change_points_from_temperature(
        mean_temp,
        dt=dt,
        change_threshold=change_threshold,
        smooth_window=smooth_window,
        min_gap_s=min_gap_s,
    )

    starts = [0] + cps
    ends = cps + [len(mean_temp)]

    if len(starts) != len(setpoint_sequence):
        raise ValueError(
            "\n自动检测到的稳态段数量与 --setpoints 数量不一致。\n"
            f"检测到 {len(starts)} 段，但你提供了 {len(setpoint_sequence)} 个 setpoint。\n"
            f"检测到的 change rows (0-based): {cps}\n\n"
            "可以调整:\n"
            "  --change-threshold   (默认 0.35 °C/s)\n"
            "  --min-gap           (默认 20 s)\n"
            "或者在 Excel 中增加 Set 列，推荐这样做。"
        )

    return [
        Segment(int(st), int(en), float(sp))
        for st, en, sp in zip(starts, ends, setpoint_sequence)
    ]


# ============================================================
# 功能块 3：从每一段提取稳态温度
# ============================================================

def extract_steady_points(
    values: np.ndarray,
    segments: list[Segment],
    dt: float,
    steady_window_s: float = 20.0,
    max_steady_slope: float = 0.03,
):
    """
    每个 setpoint 段取最后 steady_window_s 秒作为稳态候选。
    同时检查该窗口内是否仍在明显漂移。

    返回:
        setpoint, steady_temperature, slope, accepted
    """
    y = np.asarray(values, dtype=float)
    nwin = max(3, int(round(steady_window_s / dt)))

    rows = []

    for j, seg in enumerate(segments):
        length = seg.end - seg.start

        if length < 3:
            rows.append({
                "segment": j,
                "setpoint": seg.setpoint,
                "steady_temp": np.nan,
                "steady_slope_C_per_s": np.nan,
                "accepted": False,
                "start_index": seg.start,
                "end_index": seg.end,
            })
            continue

        st = max(seg.start, seg.end - nwin)
        yy = y[st:seg.end]
        tt = np.arange(len(yy), dtype=float) * dt

        mask = np.isfinite(yy)
        yy = yy[mask]
        tt = tt[mask]

        if len(yy) < 3:
            steady = np.nan
            slope = np.nan
            accepted = False
        else:
            slope = float(np.polyfit(tt, yy, 1)[0])
            # 对 0.1°C 分辨率数据，median 比 mean 更不容易受单个抖动点影响
            steady = float(np.median(yy))
            accepted = abs(slope) <= max_steady_slope

        rows.append({
            "segment": j,
            "setpoint": seg.setpoint,
            "steady_temp": steady,
            "steady_slope_C_per_s": slope,
            "accepted": accepted,
            "start_index": seg.start,
            "end_index": seg.end,
        })

    return pd.DataFrame(rows)


# ============================================================
# 功能块 4：动态一阶模型与 tau 拟合
# ============================================================

def first_order_step(t, T0, Tinf, tau):
    tau = max(float(tau), 1e-12)
    return Tinf + (T0 - Tinf) * np.exp(-np.asarray(t) / tau)


def build_transition_records(
    values: np.ndarray,
    segments: list[Segment],
    a: float,
    b: float,
    dt: float,
    dynamic_window_s: float,
):
    """
    为每次设定温度切换创建一个动态拟合数据块。

    为了匹配 1 Hz 采样:
    - t=0 使用切换前最后一个测量点
    - 然后跟随切换后的前 dynamic_window_s 秒数据

    T_inf 不从动态曲线自由拟合，而由稳态关系:
        T_inf = a*T_set + b
    固定下来。
    """
    y = np.asarray(values, dtype=float)
    nwin = max(3, int(round(dynamic_window_s / dt)))

    records = []

    for j in range(1, len(segments)):
        prev_seg = segments[j - 1]
        seg = segments[j]

        # 使用切换前最后一个点作为 t=0
        start = max(0, seg.start - 1)
        end = min(seg.end, start + nwin + 1)

        yy = y[start:end]
        tt = np.arange(len(yy), dtype=float) * dt

        mask = np.isfinite(yy)
        yy = yy[mask]
        tt = tt[mask]

        if len(yy) < 4:
            continue

        T0 = float(yy[0])
        Tinf = float(a * seg.setpoint + b)
        direction = "heating" if seg.setpoint > prev_seg.setpoint else "cooling"

        records.append({
            "transition": j,
            "old_setpoint": prev_seg.setpoint,
            "new_setpoint": seg.setpoint,
            "direction": direction,
            "start_index": start,
            "end_index": end,
            "t": tt,
            "y": yy,
            "T0": T0,
            "Tinf": Tinf,
        })

    return records


def fit_tau_for_records(
    records,
    tau_min: float = 0.05,
    tau_max: float = 120.0,
):
    """
    用所有 step response 的原始温度点一起拟合一个 tau。
    不先求 dT/dt，避免 1 Hz 数据数值微分放大噪声。
    """
    if len(records) == 0:
        return np.nan

    def objective(tau):
        sse = 0.0
        n = 0
        for r in records:
            pred = first_order_step(r["t"], r["T0"], r["Tinf"], tau)
            residual = r["y"] - pred
            sse += float(np.sum(residual ** 2))
            n += len(residual)
        return sse / max(n, 1)

    res = minimize_scalar(
        objective,
        bounds=(tau_min, tau_max),
        method="bounded",
    )

    return float(res.x)


def fit_per_transition_tau(records, tau_min=0.05, tau_max=120.0):
    rows = []

    for r in records:
        def objective(tau):
            pred = first_order_step(r["t"], r["T0"], r["Tinf"], tau)
            return float(np.mean((r["y"] - pred) ** 2))

        res = minimize_scalar(
            objective,
            bounds=(tau_min, tau_max),
            method="bounded",
        )

        tau = float(res.x)
        pred = first_order_step(r["t"], r["T0"], r["Tinf"], tau)

        rows.append({
            "transition": r["transition"],
            "old_setpoint": r["old_setpoint"],
            "new_setpoint": r["new_setpoint"],
            "direction": r["direction"],
            "T0": r["T0"],
            "Tinf_from_steady_fit": r["Tinf"],
            "tau_s": tau,
            "rmse_C": rmse(r["y"], pred),
            "n_points": len(r["y"]),
        })

    return pd.DataFrame(rows)


def predict_first_order_profile(
    setpoint_series: np.ndarray,
    dt: float,
    a: float,
    b: float,
    tau: float,
    initial_temperature: float,
):
    """
    将任意 setpoint waveform 转成动态校准后的表面温度 waveform。

    这是闭式阶跃公式的离散递推形式。
    因此不仅能处理 step，也能处理你 FDM 中的线性 ramp。
    """
    s = np.asarray(setpoint_series, dtype=float)
    Teq = a * s + b

    out = np.empty_like(Teq, dtype=float)
    out[0] = float(initial_temperature)

    alpha = np.exp(-dt / tau)

    for i in range(1, len(out)):
        out[i] = Teq[i] + (out[i - 1] - Teq[i]) * alpha

    return out, Teq


# ============================================================
# 功能块 5：单独拟合 T1 / T2 / Mean
# ============================================================

def fit_one_signal(
    name: str,
    values: np.ndarray,
    segments: list[Segment],
    set_series: np.ndarray,
    dt: float,
    steady_window_s: float,
    max_steady_slope: float,
    dynamic_window_s: float,
    tau_min: float,
    tau_max: float,
):
    # ---- 稳态 ----
    steady_df = extract_steady_points(
        values,
        segments,
        dt=dt,
        steady_window_s=steady_window_s,
        max_steady_slope=max_steady_slope,
    )

    accepted = steady_df[
        steady_df["accepted"]
        & np.isfinite(steady_df["steady_temp"])
    ].copy()

    if len(accepted) < 3:
        raise RuntimeError(
            f"{name}: 可用于稳态线性拟合的稳态点不足 3 个。"
            "请增加稳态停留时间，或放宽 --max-steady-slope。"
        )

    a, b, r2 = linear_fit(
        accepted["setpoint"].to_numpy(),
        accepted["steady_temp"].to_numpy(),
    )

    # ---- 动态 ----
    records = build_transition_records(
        values,
        segments,
        a=a,
        b=b,
        dt=dt,
        dynamic_window_s=dynamic_window_s,
    )

    tau_all = fit_tau_for_records(records, tau_min=tau_min, tau_max=tau_max)

    heating_records = [r for r in records if r["direction"] == "heating"]
    cooling_records = [r for r in records if r["direction"] == "cooling"]

    tau_heat = (
        fit_tau_for_records(heating_records, tau_min, tau_max)
        if heating_records else np.nan
    )
    tau_cool = (
        fit_tau_for_records(cooling_records, tau_min, tau_max)
        if cooling_records else np.nan
    )

    dynamic_steps = fit_per_transition_tau(
        records,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    dynamic_steps.insert(0, "signal", name)

    # ---- 用单一 tau 对整个 setpoint profile 做预测 ----
    predicted, Teq = predict_first_order_profile(
        setpoint_series=set_series,
        dt=dt,
        a=a,
        b=b,
        tau=tau_all,
        initial_temperature=float(values[0]),
    )

    dyn_rmse = rmse(values, predicted)

    result = ModelResult(
        name=name,
        a=a,
        b=b,
        r2_steady=r2,
        tau_eff=tau_all,
        tau_heating=tau_heat,
        tau_cooling=tau_cool,
        rmse_dynamic=dyn_rmse,
    )

    return result, steady_df, dynamic_steps, predicted, Teq


# ============================================================
# 功能块 6：绘图与输出
# ============================================================

def plot_steady(all_steady, results, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    marker_map = {"T1": "o", "T2": "s", "Mean": "^"}

    for name in ["T1", "T2", "Mean"]:
        df = all_steady[name]
        used = df[df["accepted"] & np.isfinite(df["steady_temp"])]

        ax.scatter(
            used["setpoint"],
            used["steady_temp"],
            marker=marker_map[name],
            label=f"{name} steady points",
        )

        r = results[name]
        xfit = np.linspace(
            used["setpoint"].min(),
            used["setpoint"].max(),
            200,
        )
        yfit = r.a * xfit + r.b

        ax.plot(
            xfit,
            yfit,
            label=(
                f"{name}: T∞={r.a:.4f}·Tset{r.b:+.3f}, "
                f"R²={r.r2_steady:.5f}"
            ),
        )

    ax.plot(
        ax.get_xlim(),
        ax.get_xlim(),
        linestyle="--",
        linewidth=1,
        label="Ideal: Tsurface = Tset",
    )

    ax.set_xlabel("Set temperature (°C)")
    ax.set_ylabel("Steady surface temperature (°C)")
    ax.set_title("Steady-state calibration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_dynamic(time_s, measured, predicted, results, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_s, measured["T1"], linewidth=1, alpha=0.7, label="T1 measured")
    ax.plot(time_s, measured["T2"], linewidth=1, alpha=0.7, label="T2 measured")
    ax.plot(time_s, measured["Mean"], linewidth=1.5, label="Mean measured")

    ax.plot(
        time_s,
        predicted["Mean"],
        linewidth=2,
        linestyle="--",
        label=(
            f"Mean first-order fit: τeff={results['Mean'].tau_eff:.2f} s, "
            f"RMSE={results['Mean'].rmse_dynamic:.2f} °C"
        ),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Dynamic calibration: measured vs first-order model")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def plot_final_model_reconstruction(
    time_s,
    set_series,
    measured_mean,
    model_surface,
    mean_result,
    output_path: Path,
):
    """
    用最终 Mean 模型，根据 setpoint profile 反推出表面温度，
    并与原始 T1/T2 平均温度画在同一张图中。

    这里的 model_surface 由最终可用公式生成：
        T_inf = a*T_set + b

        T_s[n] = T_inf[n]
                 + (T_s[n-1] - T_inf[n]) * exp(-dt/tau_eff)

    对分段恒定 setpoint，这与：
        T_s(t) = T_inf,j
                 + [T_s(t_j)-T_inf,j] exp(-(t-t_j)/tau_eff)
    完全等价。
    """
    time_s = np.asarray(time_s, dtype=float)
    set_series = np.asarray(set_series, dtype=float)
    measured_mean = np.asarray(measured_mean, dtype=float)
    model_surface = np.asarray(model_surface, dtype=float)

    residual = measured_mean - model_surface
    model_rmse = rmse(measured_mean, model_surface)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # 原始两传感器平均值
    ax.plot(
        time_s,
        measured_mean,
        linewidth=1.8,
        label="Measured mean: (T1 + T2) / 2",
    )

    # 最终公式根据 setpoint 反推出来的表面温度
    ax.plot(
        time_s,
        model_surface,
        linewidth=2.2,
        linestyle="--",
        label=(
            "Final model from setpoint "
            f"(τeff={mean_result.tau_eff:.2f} s)"
        ),
    )

    # 同时把原始 setpoint 画出来，方便直接看到每次温度切换
    ax.plot(
        time_s,
        set_series,
        linewidth=1.1,
        linestyle=":",
        alpha=0.75,
        label="Setpoint",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        "Final calibration model reconstruction vs measured mean\n"
        f"RMSE = {model_rmse:.3f} °C"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



# ============================================================
# 功能块 7：主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit steady + first-order dynamic Peltier surface calibration."
    )

    parser.add_argument("input_file", help="Input .xlsx file")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index")

    parser.add_argument("--time-col", default="NONE")
    parser.add_argument("--set-col", default="Set")
    parser.add_argument("--t1-col", default="T1")
    parser.add_argument("--t2-col", default="T2")

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Sampling interval in seconds when no Time column is used (default 1 s).",
    )

    parser.add_argument(
        "--setpoints",
        default=None,
        help=(
            "Comma-separated setpoint sequence, only used when --set-col NONE. "
            "Example: 40,50,60,70,80,90,100,90,80,70,60,50,30"
        ),
    )

    parser.add_argument("--steady-window", type=float, default=20.0)
    parser.add_argument("--dynamic-window", type=float, default=30.0)
    parser.add_argument("--max-steady-slope", type=float, default=0.03)

    parser.add_argument("--change-threshold", type=float, default=0.35)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--min-gap", type=float, default=20.0)

    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--tau-max", type=float, default=120.0)

    parser.add_argument("--output-dir", default="calibration_output")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # sheet 参数: "0" -> 0，否则作为 sheet name
    sheet_arg = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(input_path, sheet_name=sheet_arg)

    # ---- 检查列 ----
    for col in [args.t1_col, args.t2_col]:
        if col not in df.columns:
            raise KeyError(
                f"Excel 中找不到列 '{col}'。当前列名为: {list(df.columns)}"
            )

    T1 = pd.to_numeric(df[args.t1_col], errors="coerce").to_numpy(dtype=float)
    T2 = pd.to_numeric(df[args.t2_col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(T1) & np.isfinite(T2)
    if not valid.all():
        print(f"Warning: 删除 {np.sum(~valid)} 行 T1/T2 空值或非数字数据。")
        df = df.loc[valid].reset_index(drop=True)
        T1 = T1[valid]
        T2 = T2[valid]

    Tmean = (T1 + T2) / 2.0

    # ---- 时间轴 ----
    if str(args.time_col).upper() != "NONE" and args.time_col in df.columns:
        time_raw = df[args.time_col]

        if np.issubdtype(time_raw.dtype, np.datetime64):
            time_s = (
                (time_raw - time_raw.iloc[0])
                .dt.total_seconds()
                .to_numpy(dtype=float)
            )
        else:
            time_s = pd.to_numeric(time_raw, errors="coerce").to_numpy(dtype=float)

        dt = float(np.nanmedian(np.diff(time_s)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("无法从 Time 列得到有效采样间隔。")
    else:
        dt = float(args.dt)
        time_s = np.arange(len(Tmean), dtype=float) * dt

    # ---- 分段 ----
    use_set_col = (
        str(args.set_col).upper() != "NONE"
        and args.set_col in df.columns
    )

    if use_set_col:
        set_series = pd.to_numeric(
            df[args.set_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        segments = segments_from_set_column(set_series)

    else:
        manual_setpoints = parse_setpoints(args.setpoints)
        if manual_setpoints is None:
            raise ValueError(
                "\n当前数据没有可用的 Set 列，而且没有提供 --setpoints。\n"
                "稳态校准 T_inf=a*T_set+b 必须知道每个稳态段对应的设定温度。\n\n"
                "两种解决方式:\n"
                "1) 推荐: Excel 增加 Set 列；\n"
                "2) 使用 --setpoints 依次写出每个稳态段的 setpoint。"
            )

        segments = segments_from_detected_changes(
            Tmean,
            setpoint_sequence=manual_setpoints,
            dt=dt,
            change_threshold=args.change_threshold,
            smooth_window=args.smooth_window,
            min_gap_s=args.min_gap,
        )

        set_series = np.empty(len(Tmean), dtype=float)
        for seg in segments:
            set_series[seg.start:seg.end] = seg.setpoint

    # ---- 拟合三个 signal ----
    signals = {
        "T1": T1,
        "T2": T2,
        "Mean": Tmean,
    }

    results = {}
    all_steady = {}
    all_dynamic_steps = []
    predictions = {}
    equilibrium = {}

    for name, values in signals.items():
        (
            result,
            steady_df,
            dyn_steps,
            pred,
            Teq,
        ) = fit_one_signal(
            name=name,
            values=values,
            segments=segments,
            set_series=set_series,
            dt=dt,
            steady_window_s=args.steady_window,
            max_steady_slope=args.max_steady_slope,
            dynamic_window_s=args.dynamic_window,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
        )

        results[name] = result
        all_steady[name] = steady_df
        all_dynamic_steps.append(dyn_steps)
        predictions[name] = pred
        equilibrium[name] = Teq

    # ---- 汇总结果 ----
    summary_rows = []
    for name in ["T1", "T2", "Mean"]:
        r = results[name]
        summary_rows.append({
            "signal": name,
            "a": r.a,
            "b_C": r.b,
            "R2_steady": r.r2_steady,
            "tau_eff_s": r.tau_eff,
            "tau_heating_s_diagnostic": r.tau_heating,
            "tau_cooling_s_diagnostic": r.tau_cooling,
            "dynamic_full_trace_RMSE_C": r.rmse_dynamic,
            "t95_approx_s": 3.0 * r.tau_eff,
        })

    summary_df = pd.DataFrame(summary_rows)

    # 参数直接平均，仅作为诊断；推荐正式使用 Mean 直接拟合结果
    avg_parameter_row = {
        "signal": "Average_of_T1_T2_parameters (diagnostic only)",
        "a": (results["T1"].a + results["T2"].a) / 2,
        "b_C": (results["T1"].b + results["T2"].b) / 2,
        "R2_steady": np.nan,
        "tau_eff_s": (results["T1"].tau_eff + results["T2"].tau_eff) / 2,
        "tau_heating_s_diagnostic": np.nanmean([
            results["T1"].tau_heating,
            results["T2"].tau_heating,
        ]),
        "tau_cooling_s_diagnostic": np.nanmean([
            results["T1"].tau_cooling,
            results["T2"].tau_cooling,
        ]),
        "dynamic_full_trace_RMSE_C": np.nan,
        "t95_approx_s": 3.0 * (
            (results["T1"].tau_eff + results["T2"].tau_eff) / 2
        ),
    }
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([avg_parameter_row])],
        ignore_index=True,
    )

    # ---- steady points 合并 ----
    steady_out = []
    for name, sdf in all_steady.items():
        tmp = sdf.copy()
        tmp.insert(0, "signal", name)
        steady_out.append(tmp)
    steady_out = pd.concat(steady_out, ignore_index=True)

    dynamic_steps_out = pd.concat(all_dynamic_steps, ignore_index=True)

    trace_df = pd.DataFrame({
        "time_s": time_s,
        "setpoint": set_series,
        "T1_measured": T1,
        "T2_measured": T2,
        "Tmean_measured": Tmean,
        "T1_equilibrium_from_steady": equilibrium["T1"],
        "T2_equilibrium_from_steady": equilibrium["T2"],
        "Tmean_equilibrium_from_steady": equilibrium["Mean"],
        "T1_dynamic_fit": predictions["T1"],
        "T2_dynamic_fit": predictions["T2"],
        "Tmean_dynamic_fit": predictions["Mean"],

        # Final usable model reconstruction from setpoint:
        # uses the final Mean a, b and tau_eff.
        "Tsurface_final_model_from_setpoint": predictions["Mean"],
        "Tmean_minus_final_model_C": Tmean - predictions["Mean"],
    })

    # ---- 保存 ----
    summary_df.to_csv(output_dir / "calibration_summary.csv", index=False)
    steady_out.to_csv(output_dir / "steady_points.csv", index=False)
    dynamic_steps_out.to_csv(output_dir / "dynamic_steps.csv", index=False)
    trace_df.to_csv(output_dir / "trace_with_fit.csv", index=False)

    mean_result = results["Mean"]

    # ========================================================
    # Final usable calibration equation
    # ========================================================
    # IMPORTANT:
    # predictions["Mean"] was generated using exactly these final
    # Mean-model parameters (a, b, tau_eff), so it is the surface
    # temperature reconstructed directly from the final equation
    # and the provided setpoint profile.
    final_surface_from_setpoint = predictions["Mean"]
    final_model_rmse = rmse(Tmean, final_surface_from_setpoint)

    final_equation = (
        "FINAL CALIBRATION MODEL\n"
        "=======================\n\n"
        "Steady-state calibration:\n"
        f"T_inf = {mean_result.a:.6f} * T_set "
        f"{mean_result.b:+.6f}\n\n"
        "Dynamic calibration:\n"
        "T_s(t) = "
        f"({mean_result.a:.6f} * T_set,j "
        f"{mean_result.b:+.6f}) "
        "+ [T_s(t_j) - "
        f"({mean_result.a:.6f} * T_set,j "
        f"{mean_result.b:+.6f})] "
        f"* exp(-(t - t_j) / {mean_result.tau_eff:.4f})\n\n"
        "Model quality:\n"
        f"Steady-state R² = {mean_result.r2_steady:.6f}\n"
        f"Dynamic RMSE = {final_model_rmse:.4f} °C\n"
    )

    with open(
        output_dir / "final_calibration_equation.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(final_equation)

    final_validation_df = pd.DataFrame({
        "time_s": time_s,
        "setpoint": set_series,
        "T1_measured": T1,
        "T2_measured": T2,
        "Tmean_measured": Tmean,
        "Tsurface_final_model_from_setpoint": final_surface_from_setpoint,
        "residual_measured_minus_model_C": (
            Tmean - final_surface_from_setpoint
        ),
    })

    final_validation_df.to_csv(
        output_dir / "final_model_validation.csv",
        index=False,
    )

    plot_final_model_reconstruction(
        time_s=time_s,
        set_series=set_series,
        measured_mean=Tmean,
        model_surface=final_surface_from_setpoint,
        mean_result=mean_result,
        output_path=output_dir / "final_model_validation.png",
    )

    params = {
        "recommended_model": "Mean",
        "sampling_interval_s": dt,
        "steady_model": {
            "equation": "T_inf = a*T_set + b",
            "a": mean_result.a,
            "b_C": mean_result.b,
            "R2": mean_result.r2_steady,
        },
        "dynamic_model": {
            "equation": (
                "T_s(t)=T_inf+[T_s(t_j)-T_inf]*exp(-(t-t_j)/tau_eff)"
            ),
            "tau_eff_s": mean_result.tau_eff,
            "tau_heating_s_diagnostic": mean_result.tau_heating,
            "tau_cooling_s_diagnostic": mean_result.tau_cooling,
            "full_trace_RMSE_C": mean_result.rmse_dynamic,
        },
        "note": (
            "Use the Mean fit as the final model. T1/T2 fits are retained "
            "to assess spatial/sensor consistency. tau_eff is an effective "
            "response constant and may include thermometer lag and 1 Hz "
            "sampling effects."
        ),
    }

    with open(output_dir / "calibration_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    plot_steady(
        all_steady,
        results,
        output_dir / "steady_calibration.png",
    )

    plot_dynamic(
        time_s,
        measured={"T1": T1, "T2": T2, "Mean": Tmean},
        predicted=predictions,
        results=results,
        output_path=output_dir / "dynamic_calibration.png",
    )

    # ---- 控制台输出 ----
    print("\n" + "=" * 70)
    print("Calibration completed")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\nRecommended final model: Mean(T1,T2)")
    print(
        f"T_inf = {mean_result.a:.6f} * T_set "
        f"{mean_result.b:+.6f}"
    )
    print(f"tau_eff = {mean_result.tau_eff:.4f} s")
    print(f"steady R² = {mean_result.r2_steady:.6f}")
    print(f"dynamic full-trace RMSE = {mean_result.rmse_dynamic:.4f} °C")

    print("\n" + "=" * 70)
    print("FINAL CALIBRATION EQUATION")
    print("=" * 70)
    print(final_equation)

    print("Final model reconstruction:")
    print(
        f"RMSE between measured mean and final-model reconstruction = "
        f"{final_model_rmse:.4f} °C"
    )
    print(
        "Saved comparison plot: "
        "calibration_output/final_model_validation.png"
    )

    if np.isfinite(mean_result.tau_heating) and np.isfinite(mean_result.tau_cooling):
        ratio = max(
            mean_result.tau_heating,
            mean_result.tau_cooling,
        ) / min(
            mean_result.tau_heating,
            mean_result.tau_cooling,
        )

        print(
            f"Diagnostic: tau_heat={mean_result.tau_heating:.3f} s, "
            f"tau_cool={mean_result.tau_cooling:.3f} s"
        )

        if ratio > 1.5:
            print(
                "NOTE: heating/cooling tau differ by >50%. "
                "A two-tau model may be worth considering, "
                "although the default final model still uses one tau_eff."
            )

    print(f"\nOutputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
