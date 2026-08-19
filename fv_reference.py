#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立 FV 参考求解器 —— 仅供对比 / 验证用 (非产品代码)。

网格构造复用 heat_model.build_layer_stack (几何 / 节点归属与产品代码一致),
但时间推进的数值公式在此独立实现, 用于交叉验证 run_simulation:

- old_run          : 旧界面处理 (区间导热 = 端点材料标签的调和平均;
                     界面节点热容 = 左侧材料整段 rho*cp) 的独立复刻;
- corrected_run    : 修正版有限体积 (区间导热 = 区间物理材料 k;
                     界面节点热容 = 左右半宽体积加权) 的独立复刻;
- analytical_steady: 多层串联热阻的解析稳态 (q 与各界面温度)。

二者与 heat_model.run_simulation 使用完全相同的:
    底部 Dirichlet 插值 / 顶部 Robin 公式 / 显式 Euler 时间积分 /
    dt = 0.9 * min(stability) / 时间网格构造 / 输出下采样。
"""

import numpy as np

import heat_model


def _layer_props(materials, layers):
    """每层的 k 与 rho*cp 查找表 (独立于 heat_model 内部实现)。"""
    layer_k = np.empty(len(layers), dtype=float)
    layer_rc = np.empty(len(layers), dtype=float)
    for i, layer in enumerate(layers):
        m = materials[layer.material]
        layer_k[i] = m.k_W_mK
        layer_rc[i] = m.rho_kg_m3 * m.cp_J_kgK
    return layer_k, layer_rc


def _run_loop(mesh, k_face, rc, time_s, bottom_temperature_C,
              h_conv, T_air_ambient, save_dt, T_initial_C,
              sample_weights=None, return_fields=False):
    """通用显式 FV 时间循环 (参考实现核心, 与 run_simulation 同构)。

    sample_weights: 若提供 (长度 Nx), T_sample 用空间加权平均;
                    否则退化为旧式算术平均 np.mean(T[idx_sample])。
    return_fields : 为 True 时在返回中附加 T_fields (形状 [n_saved, Nx])。
    """
    x = mesh.x
    h = mesh.h
    Nx = mesh.Nx
    time_s = np.asarray(time_s, dtype=float)
    bottom = np.asarray(bottom_temperature_C, dtype=float)
    if sample_weights is None:
        sample_weights = np.zeros(Nx)
        sample_weights[mesh.idx_sample] = 1.0 / max(1, len(mesh.idx_sample))
    sample_weights = np.asarray(sample_weights, dtype=float)

    h_m = h[:-1]
    h_p = h[1:]
    k_m = k_face[:-1]
    k_p = k_face[1:]
    rc_int = rc[1:-1]
    dt_stable = rc_int * (h_m + h_p) / (2 * (k_p / h_p + k_m / h_m))
    dt = float(np.min(dt_stable) * 0.9)

    t_total = float(time_s[-1] - time_s[0])
    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)
    bottom_fdm = np.interp(time_fdm, time_s, bottom)

    fac = 2 * dt / ((h_m + h_p) * rc_int)
    c_p = fac * k_p / h_p
    c_m = fac * k_m / h_m
    c_c = 1.0 - c_p - c_m

    bc_A = (k_face[-1] / h[-1]) / (k_face[-1] / h[-1] + h_conv)
    bc_B = h_conv * T_air_ambient / (k_face[-1] / h[-1] + h_conv)

    T = np.ones(Nx) * float(T_initial_C)
    save_interval = max(1, int(save_dt / dt))
    ts, Tb, Ts, To, Tts = [], [], [], [], []
    T_fields = []
    has_ts = mesh.idx_top_surface.size > 0
    for n in range(Nt):
        if n % save_interval == 0:
            ts.append(time_fdm[n])
            Tb.append(bottom_fdm[n])
            Ts.append(np.dot(sample_weights, T))
            To.append(T[-1])
            if has_ts:
                Tts.append(float(T[mesh.idx_top_surface[0]]))
            if return_fields:
                T_fields.append(T.copy())
        T[0] = bottom_fdm[n]
        T[1:-1] = c_c * T[1:-1] + c_m * T[:-2] + c_p * T[2:]
        T[-1] = bc_A * T[-2] + bc_B

    out = {
        "time_fdm": time_fdm,
        "bottom_temperature_fdm": bottom_fdm,
        "t_array": np.array(ts),
        "T_bottom_arr": np.array(Tb),
        "T_sample_arr": np.array(Ts),
        "T_outer_surface_arr": np.array(To),
        "T_top_arr": np.array(To),
        "T_top_surface_arr": np.array(Tts),
        "T_final": T,
        "dt": dt,
        "Nt": Nt,
        "mesh": mesh,
    }
    if return_fields:
        out["T_fields"] = np.array(T_fields)
    return out


def corrected_run(materials, layers, time_s, bottom_temperature_C,
                  h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
                  T_initial_C=25.0, return_fields=False):
    """修正版 FV 的独立复刻: 区间导热 = 区间物理材料 k; 界面节点热容体积加权;
    样品温度 = 节点控制体积空间加权平均 (与 heat_model 一致)。"""
    mesh = heat_model.build_layer_stack(materials, layers)
    layer_k, layer_rc = _layer_props(materials, layers)
    nl = mesh.node_layer_index
    k_face = layer_k[nl[1:]]
    rc = np.empty(mesh.Nx)
    rc[0] = layer_rc[nl[0]]
    rc[-1] = layer_rc[nl[-1]]
    rc[1:-1] = (layer_rc[nl[1:-1]] * mesh.h[:-1]
                + layer_rc[nl[2:]] * mesh.h[1:]) / (mesh.h[:-1] + mesh.h[1:])
    return _run_loop(mesh, k_face, rc, time_s, bottom_temperature_C,
                     h_conv, T_air_ambient, save_dt, T_initial_C,
                     sample_weights=mesh.sample_weights,
                     return_fields=return_fields)


def old_run(materials, layers, time_s, bottom_temperature_C,
            h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
            T_initial_C=25.0, return_fields=False):
    """旧界面处理的独立复刻: 调和平均区间导热 + 界面节点整段左侧材料热容 +
    样品温度算术平均 (旧观测定义, 仅供历史对比)。"""
    mesh = heat_model.build_layer_stack(materials, layers)
    k = mesh.k
    rho = mesh.rho
    cp = mesh.cp
    k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    rc = rho * cp
    return _run_loop(mesh, k_face, rc, time_s, bottom_temperature_C,
                     h_conv, T_air_ambient, save_dt, T_initial_C,
                     return_fields=return_fields)


def analytical_steady(materials, layers, Tb, h_conv, T_amb):
    """多层串联热阻解析稳态。

    返回 (q, pos, T): q = 面热流 (W/m^2); pos = 各界面位置 (m), 首项 0,
    T = 对应温度, 末项为顶表面对流面温度。
    """
    R_total = (
        sum(l.thickness_m / materials[l.material].k_W_mK for l in layers)
        + 1.0 / h_conv
    )
    q = (Tb - T_amb) / R_total
    pos = [0.0]
    T = [Tb]
    x = 0.0
    for l in layers:
        x += l.thickness_m
        T.append(T[-1] - q * l.thickness_m / materials[l.material].k_W_mK)
        pos.append(x)
    return q, np.array(pos), np.array(T)


def interface_indices(mesh, positions_m):
    """返回 mesh.x 中与给定位置重合的节点索引 (界面必须在节点上)。"""
    idx = []
    for x_target in positions_m:
        j = int(np.argmin(np.abs(mesh.x - x_target)))
        if abs(mesh.x[j] - x_target) > 1e-9:
            raise ValueError(f"位置 {x_target} m 不在节点上 (最近节点 {mesh.x[j]})")
        idx.append(j)
    return idx
