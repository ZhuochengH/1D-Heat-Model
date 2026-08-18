#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct internal-sensor FDM comparison model
=============================================

Sensitivity/comparison model — 直接内置传感器边界假设:

    Zone 1 内置传感器实测温度 (T_internal)
        ↓  DIRECTLY (无稳态校准, 无 tau 滤波)
    FDM 底部边界  T[0] = T_internal_fdm[n]
        ↓
    T_sample_direct

科学目的
--------
当前校准版稳态+动态表面模型 (sample and heater T in one plot.py) 预测的
样品层温度似乎偏低。本脚本提供一个对比/敏感性模型, 回答:

    样品温度预测的降低有多少来自表面校准/动态边界模型,
    而不是 FDM 通过芯片自身的热传导?

注意
----
- 本脚本不是实验验证, 也不声称内置传感器温度物理上等于真实 Peltier 表面
  温度。这是模型假设敏感性实验。
- FDM 物理/几何/网格/材料/数值方案与参考脚本完全一致。
- 唯一物理差异: 底部边界从 T_surface_dynamic 改为 T_internal。

数据流 (与校准版对比):
    校准版:  T_internal -> T_surface_eq -> T_surface_dynamic -> T[0] -> T_sample
    本脚本:  T_internal -> T[0] -> T_sample_direct
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头后端

import matplotlib.pyplot as plt

import pandas as pd

import heat_model

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fdm_protocol_output" / "direct_internal_first300"
# 校准版模型输出 (用于对比; 若存在)
DEFAULT_CALIBRATED_CSV = (
    PROJECT_ROOT / "fdm_protocol_output" / "dynamic_first300"
    / "protocol_fdm_output.csv"
)

# ==========================================
# 材料 / 层叠 / 网格 / FDM 求解器全部复用 heat_model 模块 (唯一权威实现)。
# 本脚本不再定义任何材料属性、几何厚度、网格构造或 FDM 时间循环。
# ==========================================

# 顶部自然对流边界参数 (与校准版脚本一致)
H_CONV        = 5.0     # W/(m²·K)
T_AIR_AMBIENT = 25.0    # °C

# ==========================================
# 1. 协议加载 (T_internal) — 与参考脚本相同的验证逻辑
# ==========================================
# 无静默回退: 本脚本只支持 excel 模式 (直接边界假设必须基于实测数据)。

def load_protocol_from_excel(xlsx_path, column="Zone 1   Avg (°C)",
                             sheet=0, time_col=None, dt=1.0):
    """
    从 Excel (.xlsx/.xls) 读取实测 Zone 1 内置传感器温度协议 (T_internal)。

    返回 (t_protocol, T_internal, resolved_column):
      - t_protocol: 时间轴 (秒), 严格递增, 从 0 开始;
      - T_internal: 内置传感器实测温度 (缺失值前向填充, 删除前导 NaN);
      - resolved_column: 实际匹配到的列名 (精确或空格折叠容错)。

    列名匹配: 精确 -> 空格折叠容错 (处理 'Zone 1 Avg (°C)' 与
    'Zone 1   Avg (°C)' 等变体)。

    时间轴: 若 time_col 存在 -> 解析为自首点起算的秒; 否则按 dt 生成。
    数值时间列优先; datetime 列用相对秒。
    """
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"协议文件不存在: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    col = _find_column(df, column)
    if col is None:
        raise KeyError(
            f"找不到协议列 {column!r} (含空格折叠容错); 可用列: {list(df.columns)}"
        )
    s = pd.to_numeric(df[col], errors="coerce").ffill()
    mask = s.notna().to_numpy()
    if mask.sum() == 0:
        raise ValueError(f"协议列 {column!r} 没有有效数值。")
    T_internal = s.to_numpy(dtype=float)[mask]

    if time_col is not None and time_col in df.columns:
        raw = df[time_col]
        is_datetime_col = pd.api.types.is_datetime64_any_dtype(raw)
        t_num = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
        if (not is_datetime_col) and np.isfinite(t_num).sum() >= 0.5 * len(t_num):
            # 数值时间列 (如 'Relative time(s)' 0.0, 1.009, ...) -> 秒
            t_vals = t_num.copy()
        else:
            # 日期时间字符串列 (如 'RECTime' 06/25/2026 16:22:06) -> 相对首点秒
            t_dt = pd.to_datetime(raw, errors="coerce")
            t_vals = np.full(len(raw), np.nan, dtype=float)
            ok = t_dt.notna().to_numpy()
            if ok.any():
                base = t_dt[ok].iloc[0]
                t_vals[ok] = (t_dt[ok] - base).dt.total_seconds().to_numpy(
                    dtype=float
                )
        t_vals = t_vals[mask]
        finite = np.isfinite(t_vals)
        if not finite.any():
            raise ValueError(f"时间列 {time_col!r} 没有有效数值。")
        t_protocol = t_vals - t_vals[finite][0]
    else:
        t_protocol = np.arange(len(T_internal), dtype=float) * float(dt)

    if len(t_protocol) < 2 or np.any(np.diff(t_protocol) <= 0):
        raise ValueError("协议时间轴必须严格递增 (数据未按时间排序或采样间隔异常)。")
    return t_protocol, T_internal, col


def _find_column(df, column):
    """列名查找: 先精确, 再空格折叠容错。"""
    if column in df.columns:
        return column
    col_norm = re.sub(r"\s+", " ", str(column).strip())
    for c in df.columns:
        if re.sub(r"\s+", " ", str(c).strip()) == col_norm:
            return c
    return None


def truncate_protocol(t_protocol, T_internal, max_rows=None):
    """
    保留协议前 max_rows 个有效数据行 (时间戳与温度一一对应)。

    返回 (t_cut, T_cut, n_used)。
    """
    t = np.asarray(t_protocol, dtype=float)
    T = np.asarray(T_internal, dtype=float)
    n = len(t)
    if max_rows is not None:
        n_use = min(n, int(max_rows))
        return t[:n_use], T[:n_use], n_use
    return t, T, n


# ==========================================
# 2. 直接边界模型 — 关键差异点
# ==========================================
# 无稳态校准 (a*T+b), 无 tau 滤波, 无 T_surface_eq, 无动态表面模型。
# T_internal 直接作为 FDM 底部边界 (由 heat_model.run_simulation 插值到
# FDM 时间网格并施加 Dirichlet BC)。


# ==========================================
# 3. 对比指标 (直接 vs 校准动态模型)
# ==========================================

def compute_sample_comparison_metrics(
    t_common, T_sample_direct, T_sample_calibrated
):
    """
    计算样品温度预测差异指标:
        Delta = T_sample_direct - T_sample_calibrated

    返回 dict:
        mean_signed / mae / rmse / max_abs / max_abs_time /
        calibrated_max / direct_max / max_difference
    """
    d = np.asarray(T_sample_direct, dtype=float) - np.asarray(
        T_sample_calibrated, dtype=float
    )
    t = np.asarray(t_common, dtype=float)
    finite = np.isfinite(d)
    d = d[finite]
    t = t[finite]
    if d.size == 0:
        return {
            "n": 0, "mean_signed": np.nan, "mae": np.nan, "rmse": np.nan,
            "max_abs": np.nan, "max_abs_time": np.nan,
            "calibrated_max": np.nan, "direct_max": np.nan,
            "max_difference": np.nan,
        }
    i_max = int(np.argmax(np.abs(d)))
    return {
        "n": int(d.size),
        "mean_signed": float(np.mean(d)),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max_abs": float(np.max(np.abs(d))),
        "max_abs_time": float(t[i_max]),
        "calibrated_max": float(np.max(T_sample_calibrated)),
        "direct_max": float(np.max(T_sample_direct)),
        "max_difference": float(np.max(T_sample_direct)
                                - np.max(T_sample_calibrated)),
    }


def align_and_compare(direct_csv_path, calibrated_csv_path, output_dir):
    """
    读取直接模型与校准模型 CSV, 对齐到共同时间戳, 计算差异指标,
    生成对比 CSV 与对比图。

    返回 (metrics_dict, comparison_df)。
    任一输入缺失时返回 (None, None)。
    """
    if not Path(direct_csv_path).is_file() or not Path(calibrated_csv_path).is_file():
        return None, None

    d = pd.read_csv(direct_csv_path)
    c = pd.read_csv(calibrated_csv_path)

    t_cal = c["time_s"].to_numpy(dtype=float)
    # 以校准模型时间戳为基准, 把直接模型插值对齐 (两者都 0.1s 采样, 基本一致)
    T_direct_aligned = np.interp(
        t_cal,
        d["time_s"].to_numpy(dtype=float),
        d["T_sample_direct_C"].to_numpy(dtype=float),
    )
    T_cal_aligned = np.interp(
        t_cal,
        t_cal,
        c["T_sample_C"].to_numpy(dtype=float),
    )
    T_internal_aligned = np.interp(
        t_cal,
        c["time_s"].to_numpy(dtype=float),
        c["T_internal_C"].to_numpy(dtype=float),
    )

    comparison_df = pd.DataFrame({
        "time_s": t_cal,
        "T_internal_C": T_internal_aligned,
        "T_sample_calibrated_dynamic_C": T_cal_aligned,
        "T_sample_direct_internal_C": T_direct_aligned,
        "sample_difference_direct_minus_calibrated_C": (
            T_direct_aligned - T_cal_aligned
        ),
    })
    comparison_csv = Path(output_dir) / "sample_temperature_model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    metrics = compute_sample_comparison_metrics(
        t_cal, T_direct_aligned, T_cal_aligned
    )

    _plot_sample_comparison(
        t_cal, T_internal_aligned, T_cal_aligned, T_direct_aligned,
        output_path=Path(output_dir) / "sample_temperature_model_comparison.png",
    )
    return metrics, comparison_df


def _plot_sample_comparison(t, T_internal, T_cal, T_direct, output_path):
    """三条曲线同一坐标: T_internal / 校准样品 / 直接样品。"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, T_internal, color="#7f7f7f", linewidth=1.2, linestyle=":",
            label="Zone 1 Internal Sensor Temperature")
    ax.plot(t, T_cal, color="#1f77b4", linewidth=2,
            label="Sample Temperature — Calibrated Dynamic Surface BC")
    ax.plot(t, T_direct, color="#d62728", linewidth=2, linestyle="--",
            label="Sample Temperature — Direct Internal Sensor BC")
    ax.set_title(
        "Sample Prediction Sensitivity: Calibrated Dynamic Surface BC "
        "vs Direct Internal-Sensor BC\n"
        "(model-assumption comparison, NOT experimental validation)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Comparison plot: {output_path.resolve()}")


# ==========================================
# 6. CLI
# ==========================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Direct internal-sensor boundary FDM (sensitivity comparison)"
    )
    p.add_argument("--protocol-xlsx", required=True,
                   help="实测 T_internal 协议 .xlsx 路径 (必需)")
    p.add_argument("--protocol-col", default="Zone 1   Avg (°C)",
                   help="协议列名 (含空格/度数符号, 支持空格折叠容错)")
    p.add_argument("--protocol-sheet", default=0,
                   help="协议 sheet 名或索引 (真实文件常用 'Extracted_Data')")
    p.add_argument("--protocol-time-col", default="Relative time(s)",
                   help="时间列 (缺省 'Relative time(s)'); 缺列时按 --protocol-dt 生成")
    p.add_argument("--protocol-dt", type=float, default=1.0,
                   help="无时间列时的采样间隔 (秒)")
    p.add_argument("--max-protocol-rows", type=int, default=None,
                   help="最多使用前 N 个有效协议行 (开发/测试限流)")
    p.add_argument("--calibrated-csv", default=str(DEFAULT_CALIBRATED_CSV),
                   help="校准动态模型输出 CSV (用于对比; 可选)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help="输出目录 (plot/csv/metadata)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- 加载协议 (T_internal) ----
    t_protocol, T_internal, resolved_col = load_protocol_from_excel(
        args.protocol_xlsx, column=args.protocol_col,
        sheet=args.protocol_sheet, time_col=args.protocol_time_col,
        dt=args.protocol_dt,
    )
    protocol_file = str(Path(args.protocol_xlsx).resolve())

    n_original = len(t_protocol)
    t_protocol, T_internal, n_used = truncate_protocol(
        t_protocol, T_internal, args.max_protocol_rows
    )
    if args.max_protocol_rows is not None and n_original < args.max_protocol_rows:
        print(f"WARN: 协议仅 {n_original} 行, 少于 --max-protocol-rows "
              f"{args.max_protocol_rows}, 使用全部有效行。")

    t_start = float(t_protocol[0])
    t_end = float(t_protocol[-1])
    t_total = t_end - t_start

    print(f"Protocol file: {protocol_file}")
    print(f"Requested column: {args.protocol_col!r}")
    print(f"Resolved column: {resolved_col!r}")
    print(f"Time column: {args.protocol_time_col!r}")
    print(f"Original protocol rows: {n_original}")
    print(f"Rows used: {n_used}")
    print(f"Protocol start time: {t_start:.3f} s")
    print(f"Protocol end time: {t_end:.3f} s")
    print(f"Effective simulated duration: {t_total:.3f} s")

    # ==========================================
    # 4. 运行共享 FDM 求解器 (底部边界 = T_internal, 无校准/无滤波)
    # ==========================================
    # 直接模式: T_internal 直接作为底部 Dirichlet 边界; 求解器内部插值到
    # FDM 时间网格, 不施加任何 a/b 校准或 tau 滤波。
    result = heat_model.run_simulation(
        time_s=t_protocol,
        bottom_temperature_C=T_internal,
        materials=heat_model.DEFAULT_MATERIALS,
        layers=heat_model.DEFAULT_LAYERS,
        h_conv=H_CONV, T_air_ambient=T_AIR_AMBIENT,
    )
    t_array         = result["t_array"]
    T_internal_arr  = result["T_bottom_arr"]   # 直接边界 = T_internal (插值后)
    T_sample_arr    = result["T_sample_arr"]
    dt              = result["dt"]
    Nt              = result["Nt"]
    Nx              = result["Nx"]
    save_interval   = result["save_interval"]
    L_total         = float(result["mesh"].boundaries[-1])

    print(f"网格节点数: {Nx}（原均匀5μm网格: {int(round(L_total/5e-6))+1}）")
    print(f"FDM 时间步长 dt = {dt*1e6:.1f} μs，时间点数 Nt = {Nt:,}")
    print("[align] len(time_fdm) == len(bottom_temperature_fdm) == "
          f"{len(result['time_fdm'])}")

    # ==========================================
    # 5. 输出
    # ==========================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 9a. 直接模型图
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t_array, T_internal_arr, color="#7f7f7f", linewidth=1.5,
            linestyle=":", label="Zone 1 Internal Sensor / Direct FDM Boundary")
    ax.plot(t_array, T_sample_arr, color="#1f77b4", linewidth=2,
            label="Predicted Sample Temperature — Direct Boundary Model")
    ax.set_title(
        "Direct Internal-Sensor Boundary Model\n"
        "(sensitivity comparison: no surface calibration, no tau filter)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, max(t_array))
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    plot_path = output_dir / "direct_internal_fdm.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Output plot: {plot_path.resolve()}")

    # 9b. 数值输出 (0.1 s 下采样; 内部全分辨率)
    out = pd.DataFrame({
        "time_s": t_array,
        "T_internal_boundary_C": T_internal_arr,
        "T_sample_direct_C": T_sample_arr,
    })
    csv_path = output_dir / "direct_internal_fdm_output.csv"
    out.to_csv(csv_path, index=False)
    print(f"Output CSV: {csv_path.resolve()}")
    print(f"(CSV 输出间隔: {save_interval * dt:.3f} s ≈ 0.1 s)")

    # 9c. 元数据 (明确区分于校准模型)
    metadata = {
        "model_type": "direct_internal_sensor_boundary",
        "protocol_excel_path": protocol_file,
        "requested_protocol_column": args.protocol_col,
        "resolved_protocol_column": resolved_col,
        "time_column": args.protocol_time_col,
        "original_protocol_rows": n_original,
        "protocol_rows_used": n_used,
        "protocol_start_time": t_start,
        "protocol_end_time": t_end,
        "simulated_duration": t_total,
        "FDM_dt": dt,
        "FDM_time_points": Nt,
        "boundary_source": "T_internal",
        "calibration_applied": False,
        "dynamic_tau_applied": False,
        "initial_condition": (
            "T field initialized to 25.0 °C (identical to reference "
            "calibrated model); boundary driven by T_internal from t=0"
        ),
        "CSV_downsampling_interval": float(save_interval * dt),
        "note": (
            "Sensitivity/model-assumption comparison only. This model does "
            "NOT claim the internal sensor temperature equals the physical "
            "Peltier surface temperature."
        ),
    }
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata file: {meta_path.resolve()}")

    # ==========================================
    # 10. 与校准动态模型输出对比 (若存在)
    # ==========================================
    metrics, comparison_df = align_and_compare(
        csv_path, args.calibrated_csv, output_dir
    )

    # ---- 诊断汇总 ----
    print("\n[diagnostics]")
    print(f"Model type: direct_internal_sensor_boundary")
    print(f"Protocol file: {protocol_file}")
    print(f"Resolved protocol column: {resolved_col!r}")
    print(f"Time column: {args.protocol_time_col!r}")
    print(f"Original Excel rows: {n_original}")
    print(f"Rows used: {n_used}")
    print(f"Protocol start time: {t_start:.3f} s")
    print(f"Protocol end time: {t_end:.3f} s")
    print(f"Simulated duration: {t_total:.3f} s")
    print(f"FDM dt: {dt * 1e6:.1f} μs")
    print(f"FDM time points: {Nt:,}")
    print("FDM boundary variable: T_internal_direct (T[0] = T_internal_fdm[n])")
    print("Calibration applied: NO")
    print("Dynamic tau applied: NO")
    print(f"T_internal boundary min/max: [{np.nanmin(T_internal_arr):.3f}, "
          f"{np.nanmax(T_internal_arr):.3f}] °C")
    print(f"T_sample_direct min/max: [{np.nanmin(T_sample_arr):.3f}, "
          f"{np.nanmax(T_sample_arr):.3f}] °C")

    if metrics is not None:
        print("\n[sample comparison vs calibrated dynamic model]")
        print(f"  mean signed difference: {metrics['mean_signed']:.4f} °C")
        print(f"  MAE: {metrics['mae']:.4f} °C")
        print(f"  RMSE: {metrics['rmse']:.4f} °C")
        print(f"  max absolute difference: {metrics['max_abs']:.4f} °C "
              f"at t = {metrics['max_abs_time']:.1f} s")
        print(f"  calibrated-model max T_sample: {metrics['calibrated_max']:.3f} °C")
        print(f"  direct-model max T_sample: {metrics['direct_max']:.3f} °C")
        print(f"  max-temperature difference: {metrics['max_difference']:+.3f} °C")
        print("  (MODEL SENSITIVITY comparison, NOT experimental validation)")
    else:
        print("\n[sample comparison] 校准模型输出不可用, 跳过对比。")

    print(f"Output plot: {plot_path.resolve()}")
    print(f"Output CSV: {csv_path.resolve()}")
    print(f"Metadata file: {meta_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
