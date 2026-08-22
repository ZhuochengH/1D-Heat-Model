# Thermal Model for Fast dPCR Chip

## Purpose

Reduced-order 1D multilayer thermal model translating **measured internal
device temperature** into:

- **predicted sample-layer temperature**, and
- **predicted Top COC observation**.

## Final model

`FINAL_FROZEN_THERMAL_MODEL_V1` (promoted 2026-08-22):

| parameter | value |
|---|---|
| k_eff | 0.0675 W/(m K) |
| cp_eff | 700 J/(kg K) |
| rho_COC | 1020 kg/m3 |
| tau_top | 8.0 s (output-side lag, Top observation only) |
| h_conv | 10 W/(m2 K) |
| epsilon | 0.90 |
| sigma_SB | 5.670374419e-8 W/(m2 K4) |
| F_view | 1.0 |

Radiation is nonlinear Stefan-Boltzmann. k_eff / cp_eff / tau_top are
**system-level reduced-order effective parameters**, not intrinsic TOPAS
material constants.

## Architecture

```
measured internal temperature
    -> 1D multilayer FDM + convection + nonlinear radiation
    -> raw Top COC
    -> output-side first-order effective lag (tau_top = 8.0 s)
    -> predicted measured Top COC
```

Sample temperature is the raw FDM sample-layer temperature (control-volume
weighted) and is **never** lagged.

## Calibration

Corrected 66 C redo dataset: **RMSE = 0.6368 C**.

## External validation (zero-refit, authoritative)

| dataset | RMSE |
|---|---|
| 60 C | 1.3749 C |
| 72 C | 3.0817 C |
| 3 s extension | 1.0643 C |
| **mean external RMSE** | **1.8403 C** |

Synchronization: 60 C / 72 C use the experimentally known
`SETPOINT_90C_EVENT_PLUS_1S` rule; 3 s extension uses
`SIMULTANEOUS_START_RELATIVE_T0`.

## Known limitation

The 72 C protocol shows a larger **cooling-phase mismatch**
(cooling RMSE ~4.09 C) — a documented reduced-order transient limitation.
Parameters were not altered to improve it.

## Sample prediction

Primary end-user entry point:

```bash
uv run python workflows/prediction/predict_sample_temperature_frozen_model.py \
    --input "<internal_temperature_workbook.xlsx>" --model bare --no-gui
```

Modes: `--model bare | insulated | both`, `--start-s` / `--end-s` windowing
(preserving full thermal history), `--output-dir`, `--no-gui`.

## Bare vs insulated

- **Bare** (`BARE_TOP_COC_LAYERS`, 850 um): calibrated + externally validated.
- **Insulated** (`LEGACY_INSULATED_LAYERS`, 4050 um: + 3 mm sealed air +
  200 um PDMS cap): forward extension only, **not independently validated**
  as an insulated geometry. Sealed air is treated as pure conduction.

## Repository structure

```
thermal_model/     core physics, config, utilities (importable package)
workflows/         calibration / validation / prediction / diagnostics / legacy
tests/             pytest suite
docs/              history, strategies, release notes, manifest
Surface_calibration/  legacy Peltier-surface calibration package
*/output*          generated scientific outputs (gitignored)
```

## Reproducibility

```bash
uv sync
uv run pytest -q
```

## Scientific status

- Bare model: calibrated + externally validated.
- Insulated model: forward extension, not independently validated.
- Sample temperature: model-predicted hidden thermal state, **not directly
  measured**; external Top COC RMSE is NOT equal to direct sample-temperature
  uncertainty.
