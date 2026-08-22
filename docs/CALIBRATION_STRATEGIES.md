# CALIBRATION STRATEGIES — 标定策略

> 最后更新: 2026-08-20
>
> 本文档定义四条**并行**标定策略。策略 A 是当前唯一被接受的 72°C 标定;
> 策略 B 是面向 fast-PCR 瞬态重建的实验性/候选策略; 策略 C 是
> 滞后分离模型的可行性实验; 策略 D 是三参数 (k,cp,tau) 表征。
> B / C / D **均未验证、未接受**。

---

## 概览

| | Strategy A | Strategy B | Strategy C | Strategy D |
|---|---|---|---|---|
| ID | `top_rmse_optimal_k_cp_v1` | `fast_pcr_oriented_alpha_cp_v1` | `lag_separated_alpha_k_tau_v1` | `lag_separated_k_cp_tau_v1` |
| 状态 | **CURRENT ACCEPTED** | **EXPERIMENTAL / PROVISIONAL** | **EXPERIMENTAL / FEASIBILITY ONLY** | **EXPERIMENTAL / 3-PARAM CHARACTERIZATION** |
| 参数化 | (k_eff, cp_eff) | (alpha_eff, cp_eff) | (alpha_eff, k_eff, tau_ext) | (k_eff, cp_eff, tau_lag) |
| 目标 | 最小化 72°C 修正时间 RMSE | 近最优 72°C 顶部拟合 + 更高 alpha_eff | 分离芯片内瞬态传播与外部系统热滞后 | 高 k、非极端 cp 解是否可复现 72C 顶部动力学 |
| 选参规则 | 绝对最小 RMSE | 近最优带 + Pareto 前沿 | 固定 k 扫 alpha/tau | 分层: 拟合质量 → 物理合理性 → 高 k 优先 |
| 接受值 | k=0.0165, cp=900 | 未选定 (候选) | 未选定 (可行性) | 未选定 (表征) |
| 用途 | 当前 72°C 顶部响应模型 | 未来 fast-PCR 瞬态样品重建候选 | 可行性测试 | 三参数可行性/可辨识性 |
| 局限性 | 低 alpha_eff 可能压制快速瞬态 | 尚无快速瞬态验证数据 | 单阶滞后是降阶近似 | 单阶滞后是降阶近似; 参数退化风险待查 |

**四条策略必须保持分离**: B / C / D 不替换、不修改、不贬低策略 A。

---

## Strategy A — top_rmse_optimal_k_cp_v1 (当前接受)

- 目的: 尽可能精确复现 72°C 实测 Top COC 响应。
- 参数化: `k_eff`, `cp_eff` (rho_COC = 1020 kg/m³ 固定)。
- 选参标准: 修正测量时间目标下的最小 RMSE (`corrected_measurement_time_rmse`)。
- 已接受结果:
  - `k_eff = 0.0165 W/(m·K)`
  - `cp_eff = 900 J/(kg·K)`
  - `alpha_eff = k/(rho·cp) ≈ 1.797e-8 m²/s`
  - RMSE_top ≈ 0.7337 °C, MAE_top ≈ 0.5628 °C
- 来源: `corrected_time_objective_v3` (V3)。
- 代码位置: `calibrated_model_config.py::NOMINAL_BARE_TOP_CALIBRATION_V1`。
- Git: tag `bare-top-calibrated-model-v1`。
- 局限性:
  - 标定协议含较长平台期; 对 1–3 s 快速 PCR 阶段, 低 alpha_eff
    (≈1.8e-8 m²/s) 可能显著压制样品层温度的预测上升/下降幅度。

## Strategy B — fast_pcr_oriented_alpha_cp_v1 (实验性/候选)

- 目的: 在保持可接受的 72°C 顶部拟合的同时, 显式考察更高的有效热扩散率
  alpha_eff, 以更好地支持 fast-PCR 瞬态样品温度预测。
- 参数化: `alpha_eff` [m²/s], `cp_eff` [J/(kg·K)]; 求解器输入由
  `k_eff = alpha_eff · rho_COC · cp_eff` 派生 (rho = 1020 固定)。
- 选参哲学: **近最优拟合区 + Pareto 前沿** (不做任意加权复合分数)。
  - 拟合带 (绝对 RMSE 增量): STRICT ≤0.05 °C, MODERATE ≤0.10 °C,
    APPLICATION ≤0.20 °C; 另加内部约定 RMSE ≤ 1 °C 候选。
  - 每个带内取 **最高 alpha_eff** 候选。
- 状态: **未接受**。禁止在得到可靠的快速瞬态验证数据前将其提升为最终标定。
- 来源: 本目录 `parameter_scan_output/72C/fast_pcr_oriented_alpha_cp_v1/`
  (`alpha_cp_calibration_strategy.py`)。
- 局限性:
  - 尚无同步快速瞬态实测 (样品/顶部) 可用于拟合或验证;
  - DOE11 样品预测**仅用于敏感性图示**, 不能作为拟合目标或真值。

### 重参数化的数学事实 (必须明确)

alpha_eff = k_eff / (rho_COC · cp_eff), 因此 k_eff = alpha_eff · rho_COC · cp_eff。
在 rho 固定时, (k, cp) ↔ (alpha, cp) 是**一一对应**的。

因此: **仅改变扫描坐标 (k,cp) → (alpha,cp) 本身不会产生不同的 RMSE 最优点。**

策略 B 与策略 A 的差异**不在坐标变换**, 而在于选参规则:
- 策略 A: 只看绝对最小 RMSE;
- 策略 B: 在近最优区/Pareto 前沿内考察**更高 alpha_eff** 的候选。

---

## 拟合质量 vs 瞬态传播 vs 应用后果 (三分立)

最终分析必须区分三件独立的事情, 不得合并成一个"最佳模型"分数:

1. **FIT QUALITY**: 该 (alpha, cp) 对 72°C Top COC 的拟合质量 (RMSE 等)。
2. **TRANSIENT PROPAGATION**: 该对隐含的 alpha_eff (瞬态传播速率) 及
   L²/alpha 特征时间尺度。
3. **APPLICATION CONSEQUENCE**: 该候选如何改变预测的 DOE11 样品温度
   (仅敏感性说明, 不构成验证)。

## 语言约定

- 策略 A = "72C top-RMSE-optimal calibration" (当前接受)。
- 策略 B = "fast-PCR-oriented high-diffusivity candidate strategy" 或
  "application-oriented effective-diffusivity strategy"。
- 对策略 B 用词: candidate / provisional / experimental / near-optimal。
- **禁止**把策略 B 描述为: validated / physically correct / true sample
  model / improved final calibration (除非未来快速瞬态测量支持)。

## Strategy C — lag_separated_alpha_k_tau_v1 (实验性/可行性)

- 状态: **EXPERIMENTAL / FEASIBILITY ONLY** (不作为名义模型)。
- 参数: `alpha_eff` [m²/s], `k_eff` [W/(m·K)], `tau_ext` [s];
  `rho_COC = 1020` 固定。
- 派生: `cp_eff = k_eff / (rho * alpha_eff)`。
- 观察模型 (仅作用于顶部观测, 不作用于样品):
  `tau_ext · dT_top_obs/dt + T_top_obs = T_top_FDM`。
- 样品温度: 直接来自 FDM (`T_sample_predicted = T_sample_FDM`),
  **不经过 tau_ext**。
- `tau_ext` 解释: **有效外部/未解析系统热滞后**
  (可吸收内部传感器响应、Peltier/芯片接触动力学、顶部温度计响应等
  芯片外动力学)。是降阶唯象参数, 不是"温度计时间常数"。
- 重要局限: 若真实系统含两个独立一阶滞后
  (1/(1+tau_in·s) × 1/(1+tau_out·s)), 其乘积是二阶, 不能被单个一阶
  tau_ext 精确表示 — 单滞后是有意的降阶近似。
- 实现: `lag_augmented_thermal_model.py` (新并行模块, 复用
  `heat_model.run_simulation`, 不重复 FDM 物理)。
- 本任务: 固定 `k_eff = 0.0165`, 只扫 `alpha_eff × tau_ext` 可行性;
  完整 alpha/k/tau 标定是未来任务 (仅在可行性成立时)。

## Strategy D — lag_separated_k_cp_tau_v1 (实验性/三参数表征)

- 状态: **EXPERIMENTAL / THREE-PARAMETER CHARACTERIZATION** (不作为名义模型)。
- 参数 (直接拟合/扫描): `k_eff` [W/(m·K)], `cp_eff` [J/(kg·K)], `tau_lag` [s];
  `rho_COC = 1020` 固定。
- 派生: `alpha_eff = k/(rho·cp)`; `effusivity e = sqrt(k·rho·cp)`;
  `Rth_area_bottom = L_bottom/k` (L_bottom = 180 µm)。
  **不直接拟合 alpha**。
- 滞后架构: 沿用策略 C 的输出侧一阶观察滞后
  `tau_lag·dT_top_obs/dt + T_top_obs = T_top_FDM` (复用
  `lag_augmented_thermal_model.py` 的精确分段线性递推; tau=0 恒等;
  只作用于顶部观测, 不作用于样品)。
- 目的: 判断当未解析的外部/系统滞后被显式表示时, 是否存在
  **更高 k、cp 不过度极端** (cp ≥ 800 子集) 的拟合良好解。
- 选参哲学 (分层, 无复合分数):
  1. 拟合质量 (RMSE);
  2. 物理合理性 (cp ≥ 800 J/(kg·K) 为 NON_EXTREME_CP_SUBSET);
  3. 在近等价且物理合理的模型中, 倾向更高 k。
- **目标函数不含任何 k 奖励/惩罚**: 不把样品温度 "调得好看"。
- 探索性 cp 扫描下界 = 600 J/(kg·K) (模型化探索边界, 非文献置信区间);
  cp ≥ 800 为项目级报告子集。文档已明确这是建模选择。
- 滞后警告: 最优 tau ≥ 16 s 标 `LARGE_LAG_WARNING`; tau = 20 s 额外标
  `TAU_SCAN_BOUNDARY_WARNING` (20 s 仅为诊断, 不假定物理可行)。
- 计算: 每个 (k,cp) 只跑一次 FDM (63 次), 11 个 tau 复用同一 T_top_FDM
  迹线做滞后剖面 (693 个指标组合)。
- 本任务不调用任何连续优化器; 不做 DOE11 / PCR 样品拟合。

## 未来计划 (PLANNED FUTURE STRATEGY)

未来获得可靠的同步快速瞬态数据后, 可在策略 B 的 Pareto 候选之间选择。
概念性多时间尺度目标 (仅记录, 不在本任务实现/运行):

    J = 0.5 · RMSE_slow² + 0.5 · RMSE_fast²

其中 RMSE_slow = 72°C 长平台拟合, RMSE_fast = 快速瞬态拟合。
实现前必须已有可靠的 fast 验证数据。
