# Repository Manifest

Concise map of the reorganized repository (2026-08-22 finalization).

## Core model (`thermal_model/`)

| module | role |
|---|---|
| `core/heat_model.py` | authoritative 1D multilayer FDM solver + layer/material presets |
| `core/convection_radiation_thermal_model.py` | convection + nonlinear radiation top boundary (Strategy E) |
| `core/lag_augmented_thermal_model.py` | first-order output lag + lag-augmented model |
| `core/fv_reference.py` | independent FV reference implementation (tests) |
| `config/calibrated_model_config.py` | legacy nominal calibration config (V3 0.0165/900) |
| `config/final_frozen_model.py` | **FINAL_FROZEN_THERMAL_MODEL_V1** (source of truth) |
| `config/final_frozen_model_metadata.json` | machine-readable final model metadata |
| `utilities/draggable_hlines.py` | interactive draggable threshold lines (import-safe) |
| `utilities/predict_sample_from_internal_temperature.py` | internal-workbook loader (authoritative) |
| `utilities/classify_temperature_regimes.py` | regime classifier (heating/cooling/settling) |
| `utilities/phase_thermal_dwell.py` | phase-specific dwell statistics |
| `utilities/align_internal_and_top_temperature.py` | timestamp alignment utilities |
| `utilities/scan_effective_thermal_parameters.py` | effective-parameter scan framework |
| `utilities/lag_placement_comparison_model.py` | lag-placement architecture comparison |
| `utilities/analyze_frozen_sample_peak.py` | repeated-cycle peak detector (reused downstream) |
| `utilities/validate_frozen_model_two_new_bare_top_datasets.py` | loaders + regime labels (reused) |
| `historical/frozen_strategy_G_candidate.py` | superseded frozen candidate (0.055/1200/8.5) — preserved |

## Calibration workflows (`workflows/calibration/`)

- `recalibrate_thermal_model_66C_redo.py` — corrected 66 C recalibration
  (PHASE A/C/D/E anti-circularity); produced the promoted candidate.
- `k_cp_tau_calibration_strategy.py`, `alpha_cp_calibration_strategy.py`
- `convection_radiation_k_cp_tau_calibration.py`
- `convection_radiation_k_cp_tau_local_refinement.py`
- `lag_separated_feasibility_scan.py`
- `run_calibrated_thermal_model.py`, `run_convection_radiation_check.py`

## Validation workflows (`workflows/validation/`)

- `validate_66C_candidate_known_offset.py` — **authoritative** known-offset
  validation (SETPOINT_90C_EVENT_PLUS_1S): 60 C / 72 C.
- `validate_66C_candidate_multi_dataset.py` — earlier multi-dataset
  validation (60/72 C diagnostic under assumed t0; 3 s authoritative).
- `validate_calibrated_thermal_model.py` — generic validation runner.

## Prediction / application (`workflows/prediction/`)

- `predict_sample_temperature_frozen_model.py` — **primary end-user tool**
  (bare/insulated/both, windowing, interactive thresholds).
- `run_frozen_strategy_G_cross_protocol.py` — cross-protocol sample peak runs.
- `run_corrected_dwell_v2.py`, `run_lag_placement_pcr.py` — PCR interpretation.

## Diagnostics (`workflows/diagnostics/`)

- `evaluate_bare_vs_insulated_top.py`
- `run_lag_placement_comparison.py`
- `compare_interface_fv_72C.py`, `compare_sample_initial_72C.py`
- `sample and heater T in one plot.py`, `sample and internal sensor T in one plot.py`

## Legacy (`workflows/legacy/`)

- `main.py` — boilerplate stub (initial uv scaffold).

## Documentation (`docs/`)

- `CODE_ARCHITECTURE.md` — technical architecture guide (physics, FDM,
  data flow, module responsibilities, modification guide).
- `CALIBRATION_HISTORY.md` — full calibration history incl. final promotion.
- `CALIBRATION_STRATEGIES.md` — calibration strategy write-ups.
- `FINAL_MODEL_RELEASE.md` — release notes.
- `REPOSITORY_MANIFEST.md` — this file.
- `HANDOFF.md` — surface-calibration handoff (legacy).

## Tests (`tests/`)

Single pytest suite (`uv run pytest -q`), covering numerical regression,
calibration/validation structure, anti-circularity, and CLI behavior.

## Output directories (generated, gitignored)

`calibration_output/`, `fdm_protocol_output/`,
`temperature_alignment_output/`, `temperature_regime_output/`,
`parameter_scan_output/`, `calibrated_model_output/`,
`model_comparison_output/`, `sample_temperature_output/`.

## Data

Experimental workbooks live outside the repository
(`../Calibration/...`); they are not committed.
