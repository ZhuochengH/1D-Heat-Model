#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""续跑 V2 工作流: 复用已完成的重标定结果, 重跑验证/预测/分解/对比/图。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import workflows.v2_fc70_no_pdms.v2_candidate_workflow as w


def main():
    # 读取已保存的标定 best (k=0.07/cp=700/tau=8.0)
    params = json.loads((w.OUTPUT_ROOT / "v2_calibration_params.json")
                        .read_text(encoding="utf-8"))
    best = {
        "k_eff_W_mK": params["k_eff_W_mK"],
        "cp_eff_J_kgK": params["cp_eff_J_kgK"],
        "tau_top_s": params["tau_top_s"],
        "RMSE_C": params["calibration_RMSE_C"],
        "final_warnings": params.get("final_warnings", []),
        "refined_warnings": params.get("refined_warnings", []),
    }

    # 重建标定 eval (66C 数据, V2 vs V1)
    top = w.load_top_series(w.DS_CALIB_66C_TOP)
    internal = w.load_internal_series(w.DS_CALIB_66C_INT)
    new_ev = w._evaluate_v2_bare(top, internal, best["k_eff_W_mK"],
                                 best["cp_eff_J_kgK"], best["tau_top_s"])
    old_ev = w._evaluate_v1_66c(top, internal)
    cal = {"best": best, "new_ev": new_ev, "old_ev": old_ev}

    # 验证
    val = w.phase_c_validation(best, n_workers=12)

    # 预测
    pred = w.run_sample_prediction(best)
    w.save_json({"cyc_v2": {k: v for k, v in pred["cyc_v2"].items()
                            if k not in ("highs", "lows", "summary")},
                 "cyc_v1": {k: v for k, v in pred["cyc_v1"].items()
                            if k not in ("highs", "lows", "summary")},
                 "sample_max_v2_C": pred["sample_max_v2_C"],
                 "sample_max_v1_C": pred["sample_max_v1_C"],
                 "newton_max_iter": pred["newton_max_iter"],
                 "newton_residual_W_m2": pred["newton_residual"]},
                w.OUTPUT_ROOT / "sample_prediction_summary.json")

    # 分解
    decomp = w.run_decomposition(best)
    w.save_json(decomp, w.OUTPUT_ROOT / "decomposition.json")

    # 对比表 + 图
    num = w.numerical_check()
    v2_params = params
    w.write_comparison(cal, val, pred, decomp, num, v2_params)
    w.write_figures(cal, val, pred)

    print("DONE. Output ->", w.OUTPUT_ROOT)


if __name__ == "__main__":
    main()
