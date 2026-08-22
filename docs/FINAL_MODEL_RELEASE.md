# FINAL MODEL RELEASE — FINAL_FROZEN_THERMAL_MODEL_V1

Release date: 2026-08-22
Git tag: `thermal-model-final-v1`

## Final model parameters

| parameter | value |
|---|---|
| k_eff | 0.0675 W/(m K) |
| cp_eff | 700 J/(kg K) |
| rho_COC | 1020 kg/m3 |
| tau_top | 8.0 s (output-side lag, Top observation only; never sample) |
| h_conv | 10.0 W/(m2 K) |
| epsilon | 0.90 |
| sigma_SB | 5.670374419e-8 W/(m2 K4) |
| F_view | 1.0 |
| radiation | nonlinear Stefan-Boltzmann |

These are system-level reduced-order effective parameters, not intrinsic
material constants. Model source of truth:
`thermal_model/config/final_frozen_model.py` (+ machine-readable
`thermal_model/config/final_frozen_model_metadata.json`).

## Calibration

Corrected 66 C redo dataset (Top COC + internal, synchronized relative t0):
**RMSE = 0.6368 C**.

## External validation (zero-refit)

| dataset | sync rule | RMSE |
|---|---|---|
| 60 C | SETPOINT_90C_EVENT_PLUS_1S | 1.3749 C |
| 72 C | SETPOINT_90C_EVENT_PLUS_1S | 3.0817 C |
| 3 s extension | SIMULTANEOUS_START_RELATIVE_T0 | 1.0643 C |
| mean external | — | 1.8403 C |

All validation used the same locked parameters with no refitting, no
time-shift optimization, and no qPCR/sample information.

## Known limitation

72 C protocol: larger cooling-phase mismatch (cooling RMSE ~4.09 C),
interpreted as a documented reduced-order transient limitation. Parameters
were NOT altered to improve this dataset.

## Prediction tool

```bash
uv run python workflows/prediction/predict_sample_temperature_frozen_model.py \
    --input "<xlsx>" --model bare --no-gui
```

## Bare / insulated interpretation

- Bare: calibrated + externally validated.
- Insulated (3 mm sealed air + 200 um PDMS): forward extension only, not
  independently validated.
- Sample temperature: model-predicted hidden state, not directly measured.
  External Top RMSE is not a direct sample-temperature uncertainty.

## Scientific limitations

1. k_eff / cp_eff / tau_top are effective, not intrinsic properties.
2. tau_top is an effective observation-chain lag, not a physical sensor
   time constant.
3. Insulated geometry is a conduction-only forward extension.
4. 72 C cooling-phase transient error.

## Freeze rule

Any future change to k, cp, tau, h, epsilon, geometry, or lag placement
constitutes a **new model version**, not a silent modification of V1.

## History (Git)

- previous frozen candidate (0.055 / 1200 / 8.5) preserved as
  `thermal_model/historical/frozen_strategy_G_candidate.py`
  (superseded by the corrected 66 C recalibration + multi-dataset validation).
