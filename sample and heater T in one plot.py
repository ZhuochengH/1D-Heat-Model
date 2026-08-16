#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peltier 表面温度边界条件下的 FDM 瞬态热模型
=============================================

v3 修改 (完整 稳态+动态 表面模型管道):

数据流:
    Zone 1 内置传感器实测温度 (Excel 列 "Zone 1   Avg (°C)")
        ↓  T_internal
    稳态表面校准  T_surface_eq = a*T_internal + b
        ↓
    一阶动态表面模型 (tau_eff)
        ↓  T_surface(t)
    FDM 底部边界  T[0] = T_surface(t)
        ↓
    样品层温度 T_sample(t)

关键设计:
- 显式协议模式 (--protocol-mode excel|builtin), 无静默回退;
- Excel 列 "Zone 1   Avg (°C)" = T_internal (Zone 1 内置传感器实测温度),
  不是理论设定温度 (支持空格折叠容错匹配);
- 稳态校准参数 a/b 默认从 calibration_output/final_calibration_equation.txt
  解析 (--calibration-file), 可用 --calibration-a/-b 成对覆盖;
- tau_eff 必须显式提供 (--tau-eff);
- 动态模型在 FDM 时间轴上以实际局部时间差积分 (Option A);
- 初始条件: T_surface[0] = T_surface_eq[0]
  (假设表面初始与第一个记录的内置传感器温度平衡);
- 注意: 当前 a/b/tau_eff 为 PROVISIONAL (暂定), 未来实验将替换,
  替换只需改校准文件或 CLI 参数, 无需修改 FDM 求解器。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头后端: 始终保存 PNG, 不依赖 GUI 弹窗

import matplotlib.pyplot as plt

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_FILE = (
    PROJECT_ROOT / "calibration_output" / "final_calibration_equation.txt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fdm_protocol_output" / "dynamic_first300"

# ==========================================
# 1. 定义几何厚度与空间网格（非均匀网格）
# ==========================================
L_coc_bot = 180e-6      # 底层COC
L_sample   = 20e-6      # 水性样品层
L_oil      = 50e-6      # 矿物油层
L_coc_top  = 600e-6     # 顶层COC
L_air      = 3000e-6    # Air gap层（隔热层）
L_pdms     = 200e-6     # Cap PDMS层

L_total = L_coc_bot + L_sample + L_oil + L_coc_top + L_air + L_pdms

# 各层界面位置
x_coc_bot_end = L_coc_bot
x_sample_end  = L_coc_bot + L_sample
x_oil_end     = L_coc_bot + L_sample + L_oil
x_coc_top_end = L_coc_bot + L_sample + L_oil + L_coc_top
x_air_end     = x_coc_top_end + L_air

# 非均匀网格步长：
#   关注区（COC_bot/Sample/Oil/COC_top）: dx_fine=5 μm，保留原精度
#   隔热层（Air gap）: dx_air=200 μm，粗化约40倍
#   顶盖（PDMS）:      dx_pdms=50 μm，适度粗化
# 核心加速原理：Air的热扩散率α≈2.1e-5 m²/s，远大于其他层（Water: 1.4e-7）。
# 均匀5μm网格时 dt≈5.9e-7 s（被Air约束）；Air粗化后 dt≈8.7e-5 s（被Water约束），加速约147×。
# 节点数同时从811减至~190，每步计算量再降约4×，综合加速≈700×。
dx_fine = 5e-6    # 精细区域（关注区）
dx_air  = 200e-6  # Air gap（隔热层，不关注温度分布）
dx_pdms = 50e-6   # PDMS顶盖

def make_layer(x0, x1, dx):
    n = max(1, int(round((x1 - x0) / dx)))
    return np.linspace(x0, x1, n + 1)

x = np.unique(np.concatenate([
    make_layer(0,             x_coc_bot_end, dx_fine),
    make_layer(x_coc_bot_end, x_sample_end,  dx_fine),
    make_layer(x_sample_end,  x_oil_end,     dx_fine),
    make_layer(x_oil_end,     x_coc_top_end, dx_fine),
    make_layer(x_coc_top_end, x_air_end,     dx_air ),
    make_layer(x_air_end,     L_total,       dx_pdms),
]))
Nx = len(x)
h  = np.diff(x)  # 节点间距数组，长度 Nx-1

# ==========================================
# 2. 定义材料属性并分配到对应网格
# ==========================================
rho_coc,  k_coc,  cp_coc  = 1020.0, 0.13,   1800.0
rho_w,    k_w,    cp_w    = 1000.0, 0.60,   4180.0
rho_oil,  k_oil,  cp_oil  = 876.0,  0.142,  1962.0
rho_air,  k_air,  cp_air  = 1.204,  0.0257, 1005.0
rho_pdms, k_pdms, cp_pdms = 970.0,  0.15,   1460.0

h_conv        = 5.0    # 顶部自然对流换热系数 W/(m²·K)
T_air_ambient = 25.0   # 环境温度 °C

rho = np.zeros(Nx)
k   = np.zeros(Nx)
cp  = np.zeros(Nx)

for i, xi in enumerate(x):
    if xi <= x_coc_bot_end + 1e-9:
        rho[i], k[i], cp[i] = rho_coc,  k_coc,  cp_coc
    elif xi <= x_sample_end + 1e-9:
        rho[i], k[i], cp[i] = rho_w,    k_w,    cp_w
    elif xi <= x_oil_end + 1e-9:
        rho[i], k[i], cp[i] = rho_oil,  k_oil,  cp_oil
    elif xi <= x_coc_top_end + 1e-9:
        rho[i], k[i], cp[i] = rho_coc,  k_coc,  cp_coc
    elif xi <= x_air_end + 1e-9:
        rho[i], k[i], cp[i] = rho_air,  k_air,  cp_air
    else:
        rho[i], k[i], cp[i] = rho_pdms, k_pdms, cp_pdms

idx_sample = np.where((x > L_coc_bot) & (x <= L_coc_bot + L_sample + 1e-9))[0]

# ==========================================
# 3. 热循环协议 (Thermal Protocol) —— 显式模式, 无静默回退
# ==========================================
# 模式:
#   A) excel   : 从 .xlsx 读取实测 Zone 1 内置传感器温度 (T_internal)
#                默认列名 "Zone 1   Avg (°C)", 支持空格折叠容错;
#   B) builtin : 理想热循环 profile (仅显式选择时使用, 用于测试/兼容)。
# 两种模式都走完整管道: T_internal -> T_surface_eq -> T_surface_dynamic -> FDM。

def load_protocol_from_excel(xlsx_path, column="Zone 1   Avg (°C)",
                             sheet=0, time_col=None, dt=1.0):
    """
    从 Excel (.xlsx/.xls) 读取实测 Zone 1 内置传感器温度协议 (T_internal)。

    返回 (t_protocol, T_internal):
      - t_protocol: 时间轴 (秒), 严格递增, 从 0 开始;
      - T_internal: 内置传感器实测温度 (缺失值前向填充, 删除前导 NaN)。

    列名匹配:
      1) 精确匹配;
      2) 空格折叠容错 (连续空格折叠为单空格), 处理
         'Zone 1 Avg (°C)' 与 'Zone 1   Avg (°C)' 等变体。

    时间轴:
      - 若 time_col 存在 -> 解析为自首点起算的秒;
      - 否则按采样间隔 dt 生成 t = arange(n) * dt。
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
    return t_protocol, T_internal


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

    返回 (t_cut, T_cut, n_used)。max_rows=None 或超过行数时保留全部,
    是否发生截断由调用方据 n_used 与原长度判断。
    """
    t = np.asarray(t_protocol, dtype=float)
    T = np.asarray(T_internal, dtype=float)
    n = len(t)
    if max_rows is not None:
        n_use = min(n, int(max_rows))
        return t[:n_use], T[:n_use], n_use
    return t, T, n


def load_surface_calibration(calibration_file):
    """
    从最终校准方程文件解析稳态参数 a, b:
        T_inf = a * T_set + b
    (文件内旧术语 T_set 即本脚本 T_internal 输入侧, 数值关系不变)
    """
    path = Path(calibration_file)
    if not path.is_file():
        raise FileNotFoundError(f"校准文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    _NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    m = re.search(rf"T_inf\s*=\s*({_NUM})\s*\*\s*T_set\s*({_NUM})", text)
    if m is None:
        raise ValueError(f"无法从 {path} 解析稳态方程 'T_inf = a * T_set + b'。")
    return float(m.group(1)), float(m.group(2))


def resolve_calibration_parameters(calibration_file, a_override=None, b_override=None):
    """
    校准参数来源规则:
      - a, b 都提供   -> CLI override
      - 都不提供     -> 校准文件 (默认来源)
      - 只提供其一   -> 报错 (不允许与文件混用)
    返回 (a, b, source_str)
    """
    if (a_override is not None) != (b_override is not None):
        raise ValueError(
            "必须同时提供 --calibration-a 与 --calibration-b, "
            "或都不提供 (从 --calibration-file 读取); 不允许只提供一个。"
        )
    if a_override is not None:
        assert b_override is not None  # 与 a_override 同时提供 (上方已校验)
        a_val = float(a_override)
        b_val = float(b_override)
        return a_val, b_val, "CLI override"
    a, b = load_surface_calibration(calibration_file)
    return a, b, f"file: {Path(calibration_file).resolve()}"


def apply_surface_calibration(T_internal, a, b):
    """稳态表面校准: T_surface_eq = a * T_internal + b。"""
    return float(a) * np.asarray(T_internal, dtype=float) + float(b)


def validate_tau_eff(tau_eff):
    """校验 tau_eff: 必须是有穷正数 (秒)。"""
    if tau_eff is None:
        raise ValueError("动态表面模型需要 tau_eff, 请用 --tau-eff 显式提供。")
    try:
        tau = float(tau_eff)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tau_eff 无效: {tau_eff!r}") from exc
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"tau_eff 必须是有穷正数 (秒), 收到: {tau_eff!r}")
    return tau


def apply_dynamic_surface_model(T_surface_eq, t_axis, tau_eff):
    """
    一阶动态表面模型, 在给定时间轴上以实际局部时间差积分:
        T_surface[n] = T_eq[n]
                       + (T_surface[n-1] - T_eq[n]) * exp(-dt_n / tau_eff)
    其中 dt_n = t_axis[n] - t_axis[n-1] (支持非均匀时间轴)。

    初始条件: T_surface[0] = T_eq[0]
    (假设表面初始与第一个记录的内置传感器温度平衡)
    """
    Teq = np.asarray(T_surface_eq, dtype=float)
    t = np.asarray(t_axis, dtype=float)
    tau = validate_tau_eff(tau_eff)
    out = np.empty_like(Teq, dtype=float)
    out[0] = float(Teq[0])
    for i in range(1, len(out)):
        dt_n = t[i] - t[i - 1]
        alpha = np.exp(-dt_n / tau)
        out[i] = Teq[i] + (out[i - 1] - Teq[i]) * alpha
    return out


def prepare_fdm_boundary(T_internal, t_protocol, a, b, tau_eff, time_fdm):
    """
    完整表面模型管道 (Option A):
        T_internal 插值到 FDM 时间轴
        -> T_surface_eq_fdm = a*T_internal_fdm + b
        -> 在 FDM 时间轴上积分一阶动态模型 -> T_surface_fdm

    选择理由: 动态模型在 FDM 均匀时间轴上以真实 dt 积分, 避免协议
    1 Hz 采样与 78 μs FDM 步长之间的步长不一致; 所有数组自然对齐。

    返回 (T_internal_fdm, T_surface_eq_fdm, T_surface_fdm)。
    """
    T_internal_fdm = np.interp(time_fdm, t_protocol, T_internal)
    T_surface_eq_fdm = apply_surface_calibration(T_internal_fdm, a, b)
    T_surface_fdm = apply_dynamic_surface_model(T_surface_eq_fdm, time_fdm, tau_eff)
    return T_internal_fdm, T_surface_eq_fdm, T_surface_fdm


def resolve_protocol_mode(mode, protocol_xlsx):
    """协议模式显式解析: 无静默回退。"""
    mode = str(mode).lower()
    if mode not in ("excel", "builtin"):
        raise ValueError(f"未知协议模式: {mode!r} (可选: excel / builtin)")
    if mode == "excel" and not protocol_xlsx:
        raise ValueError(
            "--protocol-mode excel 必须提供 --protocol-xlsx; "
            "不会自动回退到内置 profile。"
        )
    return mode


def builtin_ideal_protocol():
    """回退 (仅显式 --protocol-mode builtin): 理想热循环 (10 °C/s ramp + 阶梯保持)。"""
    ramp_rate = 10.0
    times = [0.0]
    temps = [25.0]

    def add_step(target_T, hold_time):
        ramp_time = abs(target_T - temps[-1]) / ramp_rate
        times.append(times[-1] + ramp_time)
        temps.append(target_T)
        times.append(times[-1] + hold_time)
        temps.append(target_T)

    t_set_top, t_set_bottom, t_set_extension, extention_time = 100.0, 50.0, 80.0, 1
    add_step(90.0, 120.0)
    for _ in range(3):
        add_step(t_set_bottom, 1)
        add_step(t_set_extension, extention_time)
        add_step(t_set_top, 0.5)
    return np.asarray(times, dtype=float), np.asarray(temps, dtype=float)


def find_intersections(x_data, y_data, y_level):
    intersections = []
    for i in range(len(y_data) - 1):
        y0, y1 = y_data[i], y_data[i + 1]
        if (y0 - y_level) * (y1 - y_level) < 0:
            t_cross = x_data[i] + (y_level - y0) / (y1 - y0) * (x_data[i + 1] - x_data[i])
            intersections.append(t_cross)
        elif abs(y0 - y_level) < 1e-10:
            intersections.append(x_data[i])
    return intersections


class DraggableHLine:
    def __init__(self, ax, y_init, x_data, y_data, color, label_prefix):
        self.ax = ax
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.label_prefix = label_prefix

        self.line = ax.axhline(y_init, color=color, linestyle='-', linewidth=2, picker=5, alpha=0.8)

        self.temp_label = ax.text(
            max(x_data) * 1.01, y_init,
            f'{y_init:.1f} °C',
            color=color, fontweight='bold', va='center', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1)
        )

        self.markers = []
        self.texts = []
        self._update_intersections(y_init)
        self.press = None
        self.connect()

    def connect(self):
        self.cidpress   = self.line.figure.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion  = self.line.figure.canvas.mpl_connect('motion_notify_event',  self.on_motion)

    def _update_intersections(self, y_level):
        for m in self.markers: m.remove()
        for t in self.texts:   t.remove()
        self.markers.clear()
        self.texts.clear()

        cross_times = find_intersections(self.x_data, self.y_data, y_level)
        for i, t_cross in enumerate(cross_times):
            marker, = self.ax.plot([t_cross], [y_level], marker='o', color=self.color, markersize=7, zorder=5)
            self.markers.append(marker)

            offset = 4 if i % 2 == 0 else -4
            va     = 'bottom' if i % 2 == 0 else 'top'
            text   = self.ax.text(
                t_cross, y_level + offset, f'{t_cross:.1f}s',
                color=self.color, ha='center', va=va, fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )
            self.texts.append(text)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        contains, _ = self.line.contains(event)
        if not contains: return
        self.press = self.line.get_ydata()[0], event.ydata

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.ax: return

        y0, ypress = self.press
        new_y = y0 + (event.ydata - ypress)
        new_y = max(min(self.y_data), min(new_y, max(self.y_data)))

        self.line.set_ydata([new_y, new_y])
        self.temp_label.set_position((max(self.x_data) * 1.01, new_y))
        self.temp_label.set_text(f'{new_y:.1f} °C')
        self._update_intersections(new_y)
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw_idle()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Peltier 表面温度 FDM: T_internal -> 稳态+动态表面模型 -> FDM"
    )
    p.add_argument("--protocol-mode", required=True,
                   choices=["excel", "builtin"],
                   help="协议模式: excel (实测) 或 builtin (理想热循环), 必须显式选择")
    p.add_argument("--protocol-xlsx", default=None,
                   help="实测 T_internal 协议 .xlsx 路径 (excel 模式必需)")
    p.add_argument("--protocol-col", default="Zone 1   Avg (°C)",
                   help="协议列名 (含空格/度数符号, 支持空格折叠容错)")
    p.add_argument("--protocol-sheet", default=0,
                   help="协议 sheet 名或索引 (真实文件常用 'Extracted_Data')")
    p.add_argument("--protocol-time-col", default=None,
                   help="时间列 (如 'Relative time(s)' / 'RECTime'); 缺省按 --protocol-dt 生成")
    p.add_argument("--protocol-dt", type=float, default=1.0,
                   help="无时间列时的采样间隔 (秒)")
    p.add_argument("--max-protocol-rows", type=int, default=None,
                   help="最多使用前 N 个有效协议行 (开发/测试限流)")
    p.add_argument("--calibration-file", default=str(DEFAULT_CALIBRATION_FILE),
                   help="稳态校准文件 (解析 a, b), 默认仓库校准结果")
    p.add_argument("--calibration-a", type=float, default=None,
                   help="显式覆盖斜率 a (必须与 --calibration-b 同时提供)")
    p.add_argument("--calibration-b", type=float, default=None,
                   help="显式覆盖截距 b (°C) (必须与 --calibration-a 同时提供)")
    p.add_argument("--tau-eff", type=float, default=None,
                   help="有效动态时间常数 (秒), 动态表面模型必需")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help="输出目录 (plot/csv/metadata)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- 协议模式 (显式, 无静默回退) ----
    mode = resolve_protocol_mode(args.protocol_mode, args.protocol_xlsx)
    print(f"Protocol mode: {'EXCEL' if mode == 'excel' else 'BUILTIN'}")

    # ---- 加载协议 (T_internal) ----
    if mode == "excel":
        t_protocol, T_internal = load_protocol_from_excel(
            args.protocol_xlsx, column=args.protocol_col,
            sheet=args.protocol_sheet, time_col=args.protocol_time_col,
            dt=args.protocol_dt,
        )
        protocol_file = str(Path(args.protocol_xlsx).resolve())
        protocol_column = args.protocol_col
    else:
        t_protocol, T_internal = builtin_ideal_protocol()
        protocol_file = "(builtin ideal thermocycling profile)"
        protocol_column = "(ideal T_internal)"

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
    print(f"Protocol column: {protocol_column}")
    print(f"Original protocol rows: {n_original}")
    print(f"Rows used: {n_used}")
    print(f"Protocol start time: {t_start:.3f} s")
    print(f"Protocol end time: {t_end:.3f} s")
    print(f"Effective simulated duration: {t_total:.3f} s")

    # ---- 校准参数 (默认来源: 校准文件; 或 CLI 成对覆盖) ----
    a, b, calib_source = resolve_calibration_parameters(
        args.calibration_file, args.calibration_a, args.calibration_b
    )
    print(f"Calibration source: {calib_source}")
    print(f"Steady surface calibration: T_surface_eq = {a:.6f}*T_internal {b:+.6f}")

    # ---- 动态模型参数 (必须显式提供) ----
    tau = validate_tau_eff(args.tau_eff)
    print(f"tau_eff: {tau:.4f} s (PROVISIONAL)")

    # ==========================================
    # 4. 预计算非均匀网格 FDM 系数
    # ==========================================
    # (模块级已构建 x, h, k, rho, cp 等网格/材料数组)

    # 各界面处调和平均导热系数 k_{i+1/2}（长度 Nx-1）
    k_half = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])

    # 内部节点（i=1..Nx-2）的前后间距与界面导热系数
    h_m = h[:-1]       # x[i] - x[i-1]，长度 Nx-2
    h_p = h[1:]        # x[i+1] - x[i]，长度 Nx-2
    k_m = k_half[:-1]  # k_{i-1/2}，长度 Nx-2
    k_p = k_half[1:]   # k_{i+1/2}，长度 Nx-2

    rho_int = rho[1:-1]
    cp_int  = cp[1:-1]

    # 非均匀显式 FDM 稳定性条件（逐节点计算 dt 上限）：
    # dt_i ≤ ρ_i·c_i·(h_m+h_p) / [2·(k_p/h_p + k_m/h_m)]
    dt_stable = rho_int * cp_int * (h_m + h_p) / (2 * (k_p / h_p + k_m / h_m))
    dt = np.min(dt_stable) * 0.9

    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)   # 协议起点映射为 t=0

    print(f"网格节点数: {Nx}（原均匀5μm网格: {int(round(L_total/5e-6))+1}）")
    print(f"FDM 时间步长 dt = {dt*1e6:.1f} μs，时间点数 Nt = {Nt:,}")

    # ---- 完整表面模型 (Option A: 先插值到 FDM 时间轴, 再积分动态模型) ----
    # 数据流: T_internal_fdm -> T_surface_eq_fdm -> T_surface_fdm
    T_internal_fdm, T_surface_eq_fdm, T_surface_fdm = prepare_fdm_boundary(
        T_internal, t_protocol, a, b, tau, time_fdm
    )
    if not (len(time_fdm) == len(T_internal_fdm) == len(T_surface_eq_fdm)
            == len(T_surface_fdm)):
        raise RuntimeError("FDM 时间轴与表面模型数组长度不一致。")
    print("[align] len(time_fdm) == len(T_internal_fdm) == "
          "len(T_surface_eq_fdm) == len(T_surface_fdm) == "
          f"{len(time_fdm)}")

    # 预计算三对角更新系数（完全向量化，时间循环内仅三次向量乘加）
    # T_new[i] = c_c[i]*T[i] + c_m[i]*T[i-1] + c_p[i]*T[i+1]
    fac = 2 * dt / ((h_m + h_p) * rho_int * cp_int)
    c_p = fac * k_p / h_p   # 上邻居权重
    c_m = fac * k_m / h_m   # 下邻居权重
    c_c = 1.0 - c_p - c_m   # 对角权重（稳定时 ≥ 0）

    # 顶部 Robin BC 预计算（PDMS顶面 → 自然对流）
    # -k*(T[-1]-T[-2])/h[-1] = h_conv*(T[-1]-T_amb)
    # → T[-1] = bc_A * T[-2] + bc_B
    bc_A = (k[-1] / h[-1]) / (k[-1] / h[-1] + h_conv)
    bc_B = h_conv * T_air_ambient / (k[-1] / h[-1] + h_conv)

    # ==========================================
    # 5. FDM 热传导主循环
    # ==========================================
    # 底部边界: 完整动态表面温度 T_surface_fdm
    # (物理路径: T_internal -> T_surface_eq -> T_surface_dynamic -> T[0])
    T = np.ones(Nx) * 25.0

    save_interval = max(1, int(0.1 / dt))
    plot_times        = []
    plot_T_internal   = []
    plot_T_surface_eq = []
    plot_T_surface    = []
    plot_T_sample     = []

    for n in range(Nt):
        if n % save_interval == 0:
            plot_times.append(time_fdm[n])
            plot_T_internal.append(T_internal_fdm[n])
            plot_T_surface_eq.append(T_surface_eq_fdm[n])
            plot_T_surface.append(T_surface_fdm[n])
            plot_T_sample.append(np.mean(T[idx_sample]))

        T[0]    = T_surface_fdm[n]                       # 动态表面温度 BC
        T[1:-1] = c_c * T[1:-1] + c_m * T[:-2] + c_p * T[2:]  # 内部节点更新
        T[-1]   = bc_A * T[-2] + bc_B                    # 顶部 Robin BC

    t_array = np.array(plot_times)
    T_internal_arr = np.array(plot_T_internal)
    T_surface_eq_arr = np.array(plot_T_surface_eq)
    T_surface_arr = np.array(plot_T_surface)
    T_sample_arr = np.array(plot_T_sample)

    # ==========================================
    # 6. 绘图（物理分层, 科学标签）
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 7))

    if mode == "excel":
        ax.plot(t_array, T_internal_arr, color="#7f7f7f", linewidth=1.2,
                linestyle=":", label="Internal Zone 1 Sensor Temperature (T_internal)")
        ax.plot(t_array, T_surface_eq_arr, color="#2ca02c", linewidth=1.2,
                linestyle="-.", alpha=0.8,
                label="Surface Equilibrium Target (T_surface_eq)")
    else:
        ax.plot(t_array, T_internal_arr, color="#7f7f7f", linewidth=1.2,
                linestyle=":", label="Ideal Protocol T_internal")
    ax.plot(t_array, T_surface_arr, color="#d62728", linewidth=1.8,
            linestyle="--", label="Dynamic Peltier Surface Temperature (FDM BC, T_surface)")
    ax.plot(t_array, T_sample_arr, color="#1f77b4", linewidth=2,
            label="Predicted Sample Temperature (T_sample)")

    ax.set_title(
        "Thermal Cycling: Internal Sensor -> Dynamic Surface BC -> Sample\n"
        f"Protocol mode: {mode.upper()} | tau_eff = {tau:.2f} s (provisional)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, max(t_array))
    ax.set_ylim(20, 110)
    ax.legend(fontsize=9, loc="lower right")

    hline1 = DraggableHLine(ax, y_init=90.0, x_data=t_array, y_data=T_sample_arr,
                             color="darkorange", label_prefix="Denaturation")
    hline2 = DraggableHLine(ax, y_init=60.0, x_data=t_array, y_data=T_sample_arr,
                             color="purple", label_prefix="Annealing")

    # ---- 输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "protocol_fdm_dynamic.png"
    csv_path = output_dir / "protocol_fdm_output.csv"
    meta_path = output_dir / "run_metadata.json"

    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Output plot: {plot_path.resolve()}")

    # ---- 数值输出 (0.1 s 下采样; 内部 FDM 数组保持全分辨率) ----
    out = pd.DataFrame({
        "time_s": t_array,
        "T_internal_C": T_internal_arr,
        "T_surface_equilibrium_C": T_surface_eq_arr,
        "T_surface_dynamic_C": T_surface_arr,
        "T_sample_C": T_sample_arr,
    })
    out.to_csv(csv_path, index=False)
    print(f"Output CSV: {csv_path.resolve()}")
    print(f"(CSV 输出间隔: {save_interval * dt:.3f} s ≈ 0.1 s; "
          "FDM 内部数组保持全分辨率)")

    # ---- 运行元数据 (当前参数为 PROVISIONAL) ----
    metadata = {
        "protocol_mode": mode,
        "protocol_excel_path": protocol_file,
        "protocol_column": protocol_column,
        "original_protocol_rows": n_original,
        "protocol_rows_used": n_used,
        "protocol_start_time": t_start,
        "protocol_end_time": t_end,
        "simulated_duration": t_total,
        "calibration_source": calib_source,
        "calibration_a": a,
        "calibration_b": b,
        "tau_eff": tau,
        "FDM_dt": dt,
        "FDM_time_points": Nt,
        "initial_condition": (
            "T_surface[0] = T_surface_eq[0] "
            "(假设表面初始与第一个记录的内部传感器温度平衡)"
        ),
        "output_downsampling_interval": float(save_interval * dt),
        "provisional_parameters": True,
        "note": (
            "a/b/tau_eff 为 PROVISIONAL (暂定), 未来校准实验将替换; "
            "替换无需修改 FDM 求解器。"
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata file: {meta_path.resolve()}")

    # ---- 诊断汇总 ----
    print("\n[diagnostics]")
    print(f"Protocol mode: {mode}")
    print(f"Protocol file: {protocol_file}")
    print(f"Protocol column: {protocol_column}")
    print(f"Original Excel rows: {n_original}")
    print(f"Rows used: {n_used}")
    print(f"Protocol start time: {t_start:.3f} s")
    print(f"Protocol end time: {t_end:.3f} s")
    print(f"Simulated duration: {t_total:.3f} s")
    print(f"Calibration source: {calib_source}")
    print(f"a: {a:.6f}")
    print(f"b: {b:.6f} °C")
    print(f"tau_eff: {tau:.4f} s (PROVISIONAL)")
    print(f"T_internal min/max: [{np.nanmin(T_internal_arr):.3f}, "
          f"{np.nanmax(T_internal_arr):.3f}] °C")
    print(f"T_surface_eq min/max: [{np.nanmin(T_surface_eq_arr):.3f}, "
          f"{np.nanmax(T_surface_eq_arr):.3f}] °C")
    print(f"T_surface_dynamic min/max: [{np.nanmin(T_surface_arr):.3f}, "
          f"{np.nanmax(T_surface_arr):.3f}] °C")
    print(f"T_sample min/max: [{np.nanmin(T_sample_arr):.3f}, "
          f"{np.nanmax(T_sample_arr):.3f}] °C")
    print(f"FDM dt: {dt * 1e6:.1f} μs")
    print(f"FDM time points: {Nt:,}")
    print("FDM boundary variable: T_surface_dynamic (T[0] = T_surface_fdm[n])")
    print(f"Output plot: {plot_path.resolve()}")
    print(f"Output CSV: {csv_path.resolve()}")
    print(f"Metadata file: {meta_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
