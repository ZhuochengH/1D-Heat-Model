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
under a fluorinated FC-70 oil layer — is not instrumented. This repository provides a
fast 1D finite-difference thermal model that:

1. simulates the chip's thermal response from the measured internal trace,
2. predicts the **Top COC surface temperature** (validated against
   independent Top thermocouple data), and
3. predicts the **sample-layer temperature** (model-derived, not measured).

The scientific reference for this README is the master's thesis
*Fast digital PCR using picoliter cavities on a Lab-on-a-Disc*.
The thesis-reported parameters, validation errors, and sample-temperature
predictions are summarized below.

> **Version and reproducibility note:** The existing public V1 release
> (`FINAL_FROZEN_THERMAL_MODEL_V1`) is a historical implementation. Its
> configuration and validation results differ from those reported in the
> thesis. The thesis results below are reported values, not a claim that
> the public V1 code reproduces them. The exact thesis analysis revision,
> input records, synchronization rules, and evaluation windows still
> need to be linked to a reproducible release. The legacy commands and
> file paths retained in this README describe the existing V1 workflow.

## Model Overview

Bare-top calibration and validation architecture:

```text
Measured internal temperature
        |
        v
Time-dependent bottom boundary
        |
        v
1D multilayer FDM
(conduction + external convection + radiation)
        |
        +--------------------------+
        |                          |
        v                          v
Layer-averaged sample T       Raw Top COC T
(no observation lag)                |
                                     v
                             First-order observation lag
                             (tau_top = 8 s)
                                     |
                                     v
                             Predicted Top observation
```

- **Bottom boundary**: measured internal temperature (time-dependent
  Dirichlet condition).
- **Top boundary**: assumed external convection plus nonlinear radiation to a
  constant ambient (Robin condition).
- **Observation lag**: the raw FDM Top temperature is passed through a
  first-order output-side lag before being compared with thermometer data.
  This lag is applied to the **Top observation only** — never to the sample
  layer or to the chip physics.

## Thesis-Reported Frozen Model

The thesis uses the following effective parameters and fixed model inputs:

| Parameter | Value | Meaning |
|---|---:|---|
| `k_eff` | 0.0700 W/(m·K) | Effective COC thermal conductivity |
| `cp_eff` | 700 J/(kg·K) | Effective COC specific heat capacity |
| `rho_COC` | 1020 kg/m³ | Fixed COC density |
| `tau_top` | 8.0 s | Output-side Top COC observation lag |
| `h_conv` | 10 W/(m²·K) | Assumed external convection coefficient |
| `epsilon` | 0.90 | Assumed surface emissivity |
| `sigma_SB` | 5.670374419 × 10⁻⁸ W/(m²·K⁴) | Stefan–Boltzmann constant |
| `F_view` | 1.0 | Assumed radiation view factor |

> `k_eff` and `cp_eff` are calibrated, system-level effective parameters,
> not independently measured intrinsic COC material properties. The
> observation lag describes the Top COC measurement response; it is not
> applied to the predicted sample temperature.

The thesis reports an explicit finite-difference timestep of approximately
78 µs. Numerical stability alone does not establish spatial or temporal
convergence.

The public V1 configuration and its machine-readable metadata remain
available for historical reproducibility:

- [`thermal_model/config/final_frozen_model.py`](thermal_model/config/final_frozen_model.py)
- [`thermal_model/config/final_frozen_model_metadata.json`](thermal_model/config/final_frozen_model_metadata.json)

These files should not be treated as the authoritative configuration for
the thesis results until the corresponding analysis version is verified.

## Model Calibration and Validation (Thesis Results)

The model was calibrated using the 66 °C Top COC dataset. Additional
60 °C, 72 °C and 3 s-extension records were used to evaluate the frozen
model against measured Top COC temperatures.

| Stage | Dataset | Thesis-reported RMSE |
|---|---|---:|
| Calibration | 66 °C Top COC | 0.64 °C |
| Validation | 60 °C Top COC | 1.74 °C |
| Validation | 72 °C Top COC | 1.24 °C |
| Validation | 3 s-extension Top COC | 1.06 °C |

All four **thesis-reported Top COC RMSE values** are below 2 °C.
These values describe agreement between predicted Top COC observations
and measured Top COC temperatures in the bare-top configuration only.
They do **not** quantify sample-temperature prediction accuracy.

### Historical V1 results (not thesis results)

For transparency, the archived V1 release reports the following values:

| Dataset | V1 RMSE |
|---|---:|
| 66 °C calibration | 0.6368 °C |
| 60 °C validation | 1.3749 °C |
| 72 °C validation | 3.0817 °C |
| 3 s-extension validation | 1.0643 °C |

V1 uses `k_eff = 0.0675 W/(m·K)`. Its recorded synchronization rules are
`SETPOINT_90C_EVENT_PLUS_1S` for the 60/72 °C datasets and
`SIMULTANEOUS_START_RELATIVE_T0` for the 3 s-extension dataset. V1 reports
zero-refit validation and a larger cooling-phase mismatch for its 72 °C
record (cooling RMSE approximately 4.09 °C). These details are retained
for V1 provenance only; they are not established as the exact analysis
settings or residuals underlying the thesis table.

The difference between the thesis and archived results remains a
reproducibility issue until the specific thesis scripts, input files,
alignment rules, and evaluation windows are identified. No model has
been recalibrated and no archived numerical results have been altered
by this documentation update.

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

## Quick Start (Existing V1 Workflow)

The commands below describe the existing public implementation, not a verified reproduction of the thesis results.

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

## Model Geometries

The thesis represents the chip as a one-dimensional multilayer stack:

| Layer (bottom to top) | Thickness |
|---|---:|
| Bottom COC | 180 µm |
| PCR sample | 20 µm |
| FC-70 oil | 50 µm |
| Top COC | 600 µm |
| **Bare chip total** | **850 µm** |
| Air gap (insulated configuration only) | 3000 µm |
| **Insulated model total** | **3850 µm** |

The bare geometry is used for calibration and validation at the Top COC
surface. The insulated configuration adds a 3 mm air layer above the
Top COC as a forward-model extension. Air-gap heat transfer is modelled
by conduction only; radiation across the gap and the sealing foil are
omitted in the thesis model. The insulated configuration was not
independently validated against insulated temperature measurements.

### Existing V1 prediction modes

The legacy command-line option `--model` selects the following V1
geometries:

| Mode | Meaning in V1 |
|---|---|
| `bare` (default) | 850 µm bare-top chip |
| `insulated` | Historical forward model with 3 mm air gap and 200 µm PDMS cap (4050 µm total) |
| `both` | Runs both V1 geometries for comparison |

> The 200 µm PDMS cap belongs to the historical V1 insulated model and
> is not part of the thesis geometry described above. Running the legacy
> `insulated` mode is not equivalent to running the thesis model.

## Command-Line Options (Existing V1 Workflow)

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

## Output Files (Existing V1 Workflow)

For each run the tool writes into
`sample_temperature_output/<workbook-stem>/<model>/`:

| file | content |
|---|---|
| `*.png` / `*.pdf` | static prediction plot (bare vs insulated for `both`) |
| `*.csv` | time series: original/simulation/analysis time, measured internal T, predicted sample T, raw Top T (and insulated outer-PDMS T where applicable) |
| `sample_temperature_summary.txt` | model identity, geometry, key statistics (max / mean / median / min, threshold crossings) and scientific-status notes |
| interactive threshold window | if a GUI backend is available and `--no-gui` is not given |

## Available Workflows (Existing V1 Workflow)

The following script inventory is retained from the existing repository documentation. It is not a verified manifest of the thesis analysis pipeline.

| category | script | purpose | typical user |
|---|---|---|---|
| prediction | `workflows/prediction/predict_sample_temperature_frozen_model.py` | frozen-model sample-temperature prediction (primary) | primary |
| calibration | `workflows/calibration/recalibrate_thermal_model_66C_redo.py` | Historical 66 °C recalibration | reproducibility |
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

The documented V1 calibration used the corrected **66 C** dataset only, fitting
exactly three effective parameters (`k_eff`, `cp_eff`, `tau_top`) with all
other physics fixed. qPCR/86 C data were never part of the fitting
objective. The other calibration scripts document earlier strategies
(B/C/D/F/G) and are kept for provenance and advanced study.

### Validation Workflows

The documented V1 validation is **zero-refit**: frozen parameters are applied to
independent measured internal traces, the Top COC prediction is compared
against independent Top measurements, and no parameter is retuned.
Synchronization uses experimentally known timing (see
[Historical V1 results](#historical-v1-results-not-thesis-results)); fitted time shifts are never used.

### Prediction Workflows

The prediction workflows apply the frozen model forward:

- `predict_sample_temperature_frozen_model.py` — the primary tool (see
  [Quick Start](#quick-start-existing-v1-workflow));
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

## Thesis Sample-Temperature Predictions

The thesis applies the frozen model to the insulated configuration to
compare the original and fast thermocycling records. The following
values are model predictions from the analyzed complete cycles, not
independent sample-temperature measurements.

| Metric | Original protocol (2026-09-10) | Fast protocol (2026-09-15) |
|---|---:|---:|
| Complete cycles analyzed | 39 | 29 |
| Mean predicted sample HIGH | 89.23 °C | 85.78 °C |
| Mean predicted sample LOW | 59.58 °C | 60.66 °C |
| Mean exposure near sample HIGH | 14.55 s | 2.64 s |
| Mean exposure near sample LOW | 59.01 s | 6.10 s |

HIGH and LOW denote the local predicted sample-temperature extrema for
each complete cycle. Exposure durations are evaluated within 5 °C of
the respective cycle-specific extrema. They are descriptive thermal
metrics, not programmed holding times or experimentally established
PCR denaturation or annealing thresholds. The recorded/output time
resolution is approximately 1 s, which limits the precision of these
exposure-duration estimates.

The records analyzed here contain 39 and 29 complete cycles from the
original 40-cycle and fast 30-cycle programs, respectively. The incomplete
cycles at the record boundaries are excluded from cycle-level analysis.
The analyzed thermal records should not be assumed to represent every
PCR experiment performed with the corresponding protocol name.

## Sensitivity Analysis (Thesis Results)

The thesis evaluates two complementary forms of local sensitivity:

- **Frozen-model sensitivity:** perturb a model parameter while holding
  the remaining parameters fixed, and evaluate the change in predicted
  sample HIGH and LOW temperatures.
- **Calibration-propagated sensitivity:** perturb a fixed model input,
  recalibrate the effective parameters, and use the resulting model for
  sample-temperature prediction.

Selected thesis-reported combined sensitivity values are:

| Analysis | Parameter | Combined sensitivity |
|---|---|---:|
| Frozen model | Effective COC thermal conductivity | 0.88 °C |
| Frozen model | Effective COC specific heat capacity | 0.53 °C |
| Calibration-propagated | COC density | 3.22 °C |
| Calibration-propagated | Top COC thickness | 0.85 °C |
| Calibration-propagated | Bottom COC thickness | 0.66 °C |

These results describe the specified perturbations under the analyzed
input record. They are not confidence intervals or independently measured
prediction errors. In particular, the density perturbation reached a
calibration search boundary; its large response may reflect parameter
non-identifiability or search behavior rather than an independently
established physical sensitivity. The exact input and settings for each
thesis sensitivity figure should be documented with the reproducible
analysis release.

## Scientific Scope and Limitations

- The model is a reduced-order 1D representation; lateral temperature
  gradients are neglected.
- The calibrated effective thermal properties are not intrinsic material
  constants; the observation lag applies only to Top COC observations.
- Calibration and validation are based on bare Top COC measurements.
- Sample-layer temperature is a model-predicted hidden state. It was not
  independently measured or validated; no sample-temperature RMSE exists.
- The insulated geometry is a forward extension of the bare-top model.
  Its assumptions about the air gap, sealing foil and external heat
  transfer introduce additional model-form uncertainty.
- The assumed boundary conditions and constant material properties are
  simplifications. Stability of the explicit time-stepping scheme does
  not by itself demonstrate numerical convergence.
- Thesis-reported results and archived V1 outputs must not be mixed or
  treated as interchangeable. The exact thesis analysis version needs
  to be established before claiming numerical reproducibility.

## Documentation (Historical V1)

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

The scientific reference for the results summarized in this README is the
master's thesis *Fast digital PCR using picoliter cavities on a
Lab-on-a-Disc*. A formal publication citation is not yet provided here.

For the historical software implementation, refer to the GitHub repository
and `FINAL_FROZEN_THERMAL_MODEL_V1` (Git tag `thermal-model-final-v1`,
corrected documentation tag `thermal-model-final-v1.0.1`). These tags identify
V1, not a verified thesis-result release.

A dedicated thesis-reproduction release should identify the matching
configuration, geometry, source-code revision, input records, synchronization
rules, evaluation windows, and scripts that generate the reported tables and
figures. Until that provenance is established, the numerical results above
are presented as thesis-reported values rather than independently reproduced
outputs of this repository.
