# 1D Thermal Model for Fast Digital PCR

A reduced-order, one-dimensional multilayer thermal model for a fast
digital-PCR chip. The model translates **measured internal device
temperature** into **predicted sample-layer temperature** and a **predicted
Top COC (cyclic olefin copolymer) surface temperature** that can be compared
against independent thermometer measurements.

The model is calibrated and validated exclusively at the **Top COC** level.
The **sample-layer temperature is a model-predicted hidden state** — it was
not directly measured in this experiment.

## What This Repository Does

The device logs the temperature of an internal (Peltier-associated) sensor.
The chip itself — a thin COC cartridge holding a water-based PCR sample
under a mineral-oil layer — is not instrumented. This repository provides a
fast 1D finite-difference thermal model that:

1. simulates the chip's thermal response from the measured internal trace,
2. predicts the **Top COC surface temperature** (validated against
   independent Top thermocouple data), and
3. predicts the **sample-layer temperature** (model-derived, not measured).

All analysis is driven by frozen, versioned parameters; the final model is
`FINAL_FROZEN_THERMAL_MODEL_V1`.

## Model Overview

```
measured internal temperature
        |
        v
  1D multilayer FDM
  (conduction + top convection + nonlinear Stefan-Boltzmann radiation)
        |
        +----------------+----------------+
        |                |                |
        v                v                v
   Sample layer    raw Top COC       bottom boundary
   (layer-averaged,   |                (Dirichlet)
    never lagged)     v
                 output-side first-order observation lag
                 (tau_top = 8.0 s)
                       |
                       v
              predicted measured Top COC
```

- **Bottom boundary**: measured internal temperature (time-dependent
  Dirichlet condition).
- **Top boundary**: natural convection plus nonlinear radiation to a
  constant ambient (Robin condition).
- **Observation lag**: the raw FDM Top temperature is passed through a
  first-order output-side lag before being compared with thermometer data.
  This lag is applied to the **Top observation only** — never to the sample
  layer or to the chip physics.

## Final Frozen Model

`FINAL_FROZEN_THERMAL_MODEL_V1` (promoted 2026-08-22):

| parameter | value | meaning |
|---|---|---|
| k_eff | 0.0675 W/(m K) | effective COC thermal conductivity |
| cp_eff | 700 J/(kg K) | effective COC specific heat capacity |
| rho_COC | 1020 kg/m3 | COC density (fixed) |
| tau_top | 8.0 s | output-side Top observation lag |
| h_conv | 10 W/(m2 K) | top natural-convection coefficient (fixed) |
| epsilon | 0.90 | top surface emissivity (fixed) |
| sigma_SB | 5.670374419e-8 W/(m2 K4) | Stefan-Boltzmann constant |
| F_view | 1.0 | radiation view factor (fixed) |

> **Important**: `k_eff` and `cp_eff` are **system-level reduced-order
> effective parameters** obtained by calibration. They are **not** newly
> measured intrinsic TOPAS/COC material constants.

The authoritative configuration lives in
[`thermal_model/config/final_frozen_model.py`](thermal_model/config/final_frozen_model.py)
with a machine-readable copy in
[`thermal_model/config/final_frozen_model_metadata.json`](thermal_model/config/final_frozen_model_metadata.json).

## Model Validation

All RMSE values below are **predicted Top COC vs measured Top COC**
(bare-top validation architecture). There is no directly measured
sample-layer temperature, hence no sample-layer RMSE.

| stage | dataset | RMSE |
|---|---|---|
| calibration | 66 C Top COC | 0.6368 C |
| external validation | 60 C Top COC | 1.3749 C |
| external validation | 72 C Top COC | 3.0817 C |
| external validation | 3 s-extension Top COC | 1.0643 C |
| mean external-validation RMSE | 60 C + 72 C + 3 s extension | 1.8403 C |
| median external-validation RMSE | — | 1.3749 C |
| worst external-validation RMSE | — | 3.0817 C |

Synchronization (no fitted time shift for any dataset):

- **60 C / 72 C**: experimentally known `SETPOINT_90C_EVENT_PLUS_1S` rule
  (Top COC recording starts 1.0 s after the internal protocol enters
  Setpoint = 90.000 C).
- **3 s extension**: documented simultaneous-start relative-t0 alignment
  (`SIMULTANEOUS_START_RELATIVE_T0`).

All external validations are **zero-refit**: parameters were frozen after
the 66 C calibration and never retuned.

**Known limitation**: the 72 C protocol shows a larger cooling-phase
mismatch (cooling RMSE ~4.09 C). This is a documented reduced-order
transient limitation; parameters were **not** altered to improve it.

## Repository Structure

```
thermal_model/               importable Python package
  core/                      authoritative FDM solver, convection/radiation
                             boundary, output lag, FV reference
  config/                    final frozen model + legacy calibrated config
  utilities/                 reusable loaders, peak detection, plotting
  historical/                superseded frozen candidate (preserved)
workflows/                   runnable scripts by function
  calibration/               parameter-calibration scripts
  validation/                zero-refit external-validation scripts
  prediction/                sample-temperature prediction (primary)
  diagnostics/               numerical / experimental cross-checks
  legacy/                    provenance-only scripts
tests/                       pytest suite (single suite)
docs/                        release notes, history, architecture, manifest
Surface_calibration/         legacy Peltier-surface calibration package
*/output*                    generated outputs (gitignored)
```

## Installation

Requirements: **Python >= 3.12** and the [`uv`](https://docs.astral.sh/uv/)
package manager. Dependencies are declared in
[`pyproject.toml`](pyproject.toml) (numpy, pandas, scipy, scikit-learn,
matplotlib, openpyxl, xlrd; pytest for development) and resolved via
`uv.lock`.

```bash
git clone https://github.com/ZhuochengH/1D-Heat-Model.git
cd 1D-Heat-Model
uv sync
```

This creates a `.venv/` with the exact locked dependency set.

## Quick Start

The **primary user-facing workflow** is the frozen-model sample-temperature
prediction tool:

```bash
uv run python workflows/prediction/predict_sample_temperature_frozen_model.py \
    --input /path/to/internal_temperature.xlsx \
    --model both \
    --no-gui
```

This predicts the sample-layer temperature for both the bare and the
insulated chip configurations, writes plots, a CSV time series, and a text
summary under `sample_temperature_output/<workbook>/<model>/`.

A minimal bare-mode run:

```bash
uv run python workflows/prediction/predict_sample_temperature_frozen_model.py \
    --input "/path/to/internal_temperature.xlsx" --model bare --no-gui
```

On Windows PowerShell the same command works with a Windows path:

```powershell
uv run python workflows/prediction/predict_sample_temperature_frozen_model.py --input "D:\path\to\internal_temperature.xlsx" --model bare --no-gui
```

## Input Data Requirements

The prediction tool reads the **internal-temperature workbook** exported by
the device software:

- Excel `.xlsx` workbook
- sheet: `Extracted_Data`
- a **time column** named `Time(s)` — strictly increasing, seconds
  (relative time is fine; absolute timestamps are not required)
- a **temperature column** named `Zone 1 Avg (°C)`

| column | required | example |
|---|---|---|
| `Time(s)` | yes | 0.110, 1.196, ... |
| `Zone 1 Avg (°C)` | yes | 25.335, 26.402, ... |

Column names are matched flexibly (whitespace-normalized). Non-finite or
invalid rows are dropped; the remaining time axis must be strictly
increasing.

## Prediction Modes

`--model` selects the chip geometry:

| mode | meaning |
|---|---|
| `bare` (default) | calibrated + externally validated bare-top chip (850 um: 180 um COC / 20 um sample / 50 um oil / 600 um Top COC; top surface exposed to ambient) |
| `insulated` | forward-model extension of the same chip with a 3 mm sealed-air gap and a 200 um PDMS cap (4050 um total); external convection/radiation moves to the outer PDMS surface |
| `both` | runs both geometries and produces a direct comparison |

> The bare geometry is experimentally validated at the Top COC level. The
> insulated geometry is a **forward-model extension only** and was **not**
> independently validated against insulated Top COC measurements.

## Command-Line Options

Primary tool:
`workflows/prediction/predict_sample_temperature_frozen_model.py`

| option | required | default | meaning |
|---|---|---|---|
| `--input` | yes | — | path to the internal-temperature workbook (`.xlsx`) |
| `--model` | no | `bare` | `bare`, `insulated`, or `both` |
| `--start-s` | no | first recorded time | analysis-window start on the workbook `Time(s)` axis (crops output only; the simulation always runs from the full trace start) |
| `--end-s` | no | last valid time | simulation/analysis end on the workbook `Time(s)` axis |
| `--output-dir` | no | `sample_temperature_output/<stem>/<model>/` | output directory override |
| `--no-gui` | no | GUI shown if available | disable the interactive threshold window (use in batch/headless mode) |

Behavior when options are omitted:

- the whole recorded trace is analyzed (no windowing);
- the simulation **always** starts at the first recorded point of the full
  trace and preserves the complete thermal history — `--start-s` / `--end-s`
  only crop the analysis/output, they never reinitialize the physics;
- the environment temperature is taken as the first recorded internal
  temperature (`INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT`);
- descriptive thresholds are reported at 85 / 87 / 90 / 92 / 95 C
  (informational only — not PCR success/failure criteria).

## Output Files

For each run the tool writes into
`sample_temperature_output/<workbook-stem>/<model>/`:

| file | content |
|---|---|
| `*.png` / `*.pdf` | static prediction plot (bare vs insulated for `both`) |
| `*.csv` | time series: original/simulation/analysis time, measured internal T, predicted sample T, raw Top T (and insulated outer-PDMS T where applicable) |
| `sample_temperature_summary.txt` | model identity, geometry, key statistics (max / mean / median / min, threshold crossings) and scientific-status notes |
| interactive threshold window | if a GUI backend is available and `--no-gui` is not given |

## Available Workflows

| category | script | purpose | typical user |
|---|---|---|---|
| prediction | `workflows/prediction/predict_sample_temperature_frozen_model.py` | frozen-model sample-temperature prediction (primary) | primary |
| calibration | `workflows/calibration/recalibrate_thermal_model_66C_redo.py` | 66 C recalibration that produced the final model | reproducibility |
| calibration | `workflows/calibration/k_cp_tau_calibration_strategy.py` | lag-separated 3-parameter strategy (D) | advanced |
| calibration | `workflows/calibration/alpha_cp_calibration_strategy.py` | alpha/cp strategy (B) | advanced |
| calibration | `workflows/calibration/convection_radiation_k_cp_tau_calibration.py` | convection + radiation recalibration (F) | advanced |
| calibration | `workflows/calibration/convection_radiation_k_cp_tau_local_refinement.py` | targeted local refinement (G) | advanced |
| calibration | `workflows/calibration/lag_separated_feasibility_scan.py` | feasibility scan (C) | advanced |
| calibration | `workflows/calibration/run_calibrated_thermal_model.py` | legacy 72 C nominal runner | legacy |
| calibration | `workflows/calibration/run_convection_radiation_check.py` | convection/radiation model check (E) | advanced |
| validation | `workflows/validation/validate_66C_candidate_known_offset.py` | authoritative known-offset validation (60/72 C) | reproducibility |
| validation | `workflows/validation/validate_66C_candidate_multi_dataset.py` | earlier multi-dataset transfer validation | reproducibility |
| validation | `workflows/validation/validate_calibrated_thermal_model.py` | generic validation runner | advanced |
| prediction | `workflows/prediction/run_corrected_dwell_v2.py` | corrected repeated-cycle dwell analysis | advanced |
| prediction | `workflows/prediction/run_frozen_strategy_G_cross_protocol.py` | cross-protocol sample peak prediction | advanced |
| prediction | `workflows/prediction/run_lag_placement_pcr.py` | lag-placement PCR cross-protocol | advanced |
| diagnostics | `workflows/diagnostics/compare_interface_fv_72C.py` | FV vs FDM interface check | advanced |
| diagnostics | `workflows/diagnostics/compare_sample_initial_72C.py` | sample initial-condition check | advanced |
| diagnostics | `workflows/diagnostics/evaluate_bare_vs_insulated_top.py` | bare vs insulated comparison | advanced |
| diagnostics | `workflows/diagnostics/run_lag_placement_comparison.py` | lag-placement comparison | advanced |
| diagnostics | `workflows/diagnostics/sample and heater T in one plot.py` | plotting helper | advanced |
| diagnostics | `workflows/diagnostics/sample and internal sensor T in one plot.py` | plotting helper | advanced |

### Calibration Workflows

The final calibration used the corrected **66 C** dataset only, fitting
exactly three effective parameters (`k_eff`, `cp_eff`, `tau_top`) with all
other physics fixed. qPCR/86 C data were never part of the fitting
objective. The other calibration scripts document earlier strategies
(B/C/D/F/G) and are kept for provenance and advanced study.

### Validation Workflows

Validation is strictly **zero-refit**: frozen parameters are applied to
independent measured internal traces, the Top COC prediction is compared
against independent Top measurements, and no parameter is retuned.
Synchronization uses experimentally known timing (see
[Model Validation](#model-validation)); fitted time shifts are never used.

### Prediction Workflows

The prediction workflows apply the frozen model forward:

- `predict_sample_temperature_frozen_model.py` — the primary tool (see
  [Quick Start](#quick-start));
- the other prediction scripts perform cross-protocol sample-peak and
  dwell-time analyses used in the PCR interpretation study.

### Diagnostic Workflows

Diagnostic scripts cross-check numerical implementation choices (FV vs FDM,
interface treatment, initial conditions, lag placement) or produce
comparison plots. They are optional and intended for developers and
reviewers.

### Legacy Workflows

`workflows/legacy/` contains the original scaffold stub (`main.py`) kept
for provenance. It is not part of the scientific workflow and is not
recommended for normal use.

## Output Files and Directories

Generated outputs are written under gitignored directories at the
repository root:

| directory | content |
|---|---|
| `sample_temperature_output/` | primary prediction outputs |
| `calibrated_model_output/` | calibration / validation / prediction runs |
| `calibration_output/` | earlier calibration scans |
| `parameter_scan_output/` | parameter-scan results |
| `temperature_alignment_output/` | alignment analyses |
| `temperature_regime_output/` | regime classifications |
| `fdm_protocol_output/` | protocol FDM runs |
| `model_comparison_output/` | model-comparison studies |

## Running Tests

```bash
uv run pytest -q
```

At the final V1 release (tag `thermal-model-final-v1`), the full suite
contained **823 passing tests** covering numerical regression,
calibration/validation structure, anti-circularity safeguards, and CLI
behavior.

## Scientific Scope and Limitations

- The model is a **reduced-order 1D** representation; lateral effects are
  neglected.
- `k_eff` / `cp_eff` / `tau_top` are **effective** parameters, not intrinsic
  material constants.
- The **Top COC** temperature is experimentally measured and used for
  calibration/validation.
- The **sample-layer temperature is model-predicted** and was **not
  directly measured**; no sample-temperature RMSE exists. Top-surface
  validation supports the reduced-order model's thermal transfer behavior,
  but the sample layer remains a model-derived hidden state.
- The **insulated geometry** (3 mm sealed air + 200 um PDMS) is a
  conduction-only forward extension and was **not independently validated**
  as an insulated experimental geometry.
- The **72 C** protocol shows a larger cooling-phase mismatch (cooling RMSE
  ~4.09 C).

## Insulated Sample Prediction (3 s-extension protocol)

Model-predicted sample-layer temperature (frozen parameters, insulated
forward geometry, no refitting):

- predicted overall sample maximum: **~89.07 C**
- repeated-cycle sample peak mean: **~85.67 C**
- repeated-cycle sample peak median: **~85.52 C**

These are **model predictions**, not validation measurements. They must not
be associated with any Top COC validation RMSE.

## Documentation

- [Code architecture](docs/CODE_ARCHITECTURE.md) — how the software works
  internally (physics, FDM, data flow, module responsibilities)
- [Final model release](docs/FINAL_MODEL_RELEASE.md) — release notes for
  `FINAL_FROZEN_THERMAL_MODEL_V1`
- [Calibration history](docs/CALIBRATION_HISTORY.md) — full calibration /
  validation record
- [Calibration strategies](docs/CALIBRATION_STRATEGIES.md) — strategy
  write-ups (A-G)
- [Repository manifest](docs/REPOSITORY_MANIFEST.md) — file-by-file map

## Citation / Project Status

This is a research repository; no formal citation exists yet. Please refer
to the repository and the release identifier
**`FINAL_FROZEN_THERMAL_MODEL_V1`** (Git tag `thermal-model-final-v1`,
corrected documentation tag `thermal-model-final-v1.0.1`).
