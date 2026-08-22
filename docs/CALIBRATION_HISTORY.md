# Thermal Model Calibration History

This document records the evolution of the thermal-model calibration for the
72 °C bare-top COC experiment. It distinguishes **valid** results from
**historical** results whose objective function contained an implementation
error later identified.

## Numerical model baseline

- Tag `pre-fdm-numerical-review-v1`: architecture state before the numerical
  corrections. Historical rollback baseline only.

## Geometry / FDM corrections

- Bare-top 850 µm experimental geometry (`BARE_TOP_COC_LAYERS`):
  Bottom COC 180 µm / Sample 20 µm / Oil 50 µm / Top COC 600 µm; no Air or
  PDMS layer; Top COC outer surface (x = 850 µm) exposed to ambient via Robin
  convection.
- Interface finite-volume correction: interval face conductivity = physical
  interval material k; interface-node volumetric heat capacity volume-weighted.
- Sample spatial averaging: control-volume weighted mean over the full sample
  layer (180–200 µm).
- Experimental auto initial condition: `T_initial = first T_internal`.

## V1 parameter scan

Status:
    HISTORICAL — INVALID OBJECTIVE

Reason:
    Model predictions were interpolated using measured **temperature values**
    as the `np.interp` query coordinates instead of **measurement time**
    coordinates (temperature-as-time query).

Do not use:
    V1 parameter ranking, reported RMSE, or selected parameters for thesis
    calibration.

## V2 extended system-effective scan

Status:
    HISTORICAL — INVALID OBJECTIVE

Legacy selected point (from the invalid objective):
    k_eff = 0.068 W/(m·K)
    cp_eff = 9200 J/(kg·K)
    RMSE_legacy = 4.7449 °C  (INVALID objective semantics)

Corrected evaluation at the same pair (measurement-time objective):
    RMSE_corrected ≈ 7.4345 °C
    MAE_corrected ≈ 6.0083 °C
    mean residual ≈ +2.2075 °C

Interpretation:
    The pair 0.068 / 9200 is a **provisional / historical** parameter pair
    only. It must NOT be treated as the final calibrated model.

## Objective bug correction

Correct formula:

    T_predicted_i
      =
    interpolate(
        measurement_time_i,
        FDM_time,
        FDM_top_temperature
    )

Objective:

    RMSE =
    sqrt(mean((T_predicted_i - T_measured_i)^2))

All future objectives (V3+) must query the FDM output at the **measurement
time** coordinates. The authoritative helper is
`scan_effective_thermal_parameters.sample_prediction_at_measurement_times`.

## V3 corrected-time-objective calibration

Status:
    COMPLETE — ACCEPTED (interior optimum)

Corrected objective:
    - interpolation query axis = measurement time coordinates;
    - equal-weight RMSE on T_top_surface vs T_top_measured (299 points);
    - no time shift, no offset correction, no sample-temperature term.

Coarse search:
    12 k values × 8 cp values = 96 points (k 0.005–0.240; cp 800–10000),
    all with the corrected measurement-time objective.

Fine search:
    Local 11 × 11 grid (121 points) spanning the coarse neighbors of the
    coarse best; no recenter required (fine best interior).

Accepted V3 parameters (system-level effective):
    k_eff = 0.0165 W/(m·K)
    cp_eff = 900 J/(kg·K)

Corrected-objective metrics at the accepted point:
    RMSE = 0.7337 °C
    MAE = 0.5628 °C
    mean residual = -0.2727 °C
    max positive residual = +1.02 °C
    max negative residual = -2.08 °C

Near-optimal region (Delta-RMSE, not confidence interval):
    within 5% of min RMSE: k [0.0165, 0.018], cp [900, 1000] (2 points).

Comparison (same corrected objective):
    legacy-selected 0.068/9200 corrected RMSE ≈ 7.4345 °C
    V3 selected 0.0165/900 corrected RMSE ≈ 0.7337 °C

Note:
    V1/V2 remain HISTORICAL invalid-objective scans. Their legacy RMSE
    (e.g., 4.7449 °C) must NOT be compared directly with corrected-objective
    RMSE values.

Final configuration:
    `NOMINAL_BARE_TOP_CALIBRATION_V1` now holds the V3 parameters
    (k_eff=0.0165, cp_eff=900). The old 0.068/9200 pair is preserved under
    `LEGACY_OBJECTIVE_PROVISIONAL_CALIBRATION` (historical, not valid for
    final calibration).

Final output:
    calibrated_model_output/72C_corrected_objective_v1/

---

## Key distinction

- V1/V2 were analysis iterations whose objective contained an implementation
  error (temperature-as-time query). They remain useful for traceability and
  comparison, but their parameter selection is not valid for the intended
  time-domain fitting problem.
- V3 uses the corrected measurement-time objective and, if an enclosed optimum
  is found, defines the current calibrated configuration.

---

## 66C REDO recalibration candidate (2026-08-22)

New task: recalibrate the bare-top model on the higher-quality corrected 66C
redo dataset, then forward-predict the insulated 3s-extension sample
temperature WITHOUT further tuning.

Candidate ID: `66C_RECALIBRATED_CANDIDATE_V1` (NOT promoted; pending review).
`frozen_strategy_G_candidate.py` and all frozen outputs remain untouched.

Calibration dataset (synchronized, SIMULTANEOUS_START_RELATIVE_T0, shift=0):
- corrected Top COC: `Calibration/extension 66°C_redo.xls` (T Avg, no artifact
  filtering; only structural NaN/empty/T<=0 excluded)
- internal: `08.17 COC top_66°C_zone1_temperature_analysis.xlsx`

Fixed physics: h=10, eps=0.90, sigma=5.670374419e-8, F=1.0, nonlinear
Stefan-Boltzmann, rho_COC=1020, BARE_TOP_COC_LAYERS.
Fitted ONLY: k_eff, cp_eff, tau_top (output-side lag, Top observation only;
never affects sample). qPCR/86C never in objective.

Result:
- old frozen (0.055/1200/8.5) on 66C: RMSE = 3.0134 C (== historical ext-val)
- NEW optimum: k=0.0675, cp=700, tau=8.0 s
  - RMSE 0.6368 C, MAE 0.4860, mean residual +0.0316, R2 0.9903
  - heating RMSE 0.511 (mean -0.119), cooling 0.819 (mean +0.249),
    settling 0.795 (mean +0.243)
  - improvement vs old: 2.377 C (78.9%)
  - boundary: coarse CP_MIN hit -> expanded to cp=500/600; final optimum
    interior (no warnings). k optimum broad [0.0650, 0.0700]; cp sharp
    [600, 800]; tau interior 8.0 s.
- Independent 3s-extension validation (no refit): RMSE 1.0643 C
  (old frozen 2.3941 C == historical) -> transfer IMPROVED by 1.330 C.
- Classification (thermal only): PROMISING_NEW_CANDIDATE.

Insulated 3s sample forward prediction (locked params; LEGACY_INSULATED_LAYERS;
h=10 + nonlinear radiation on outer PDMS; air = pure conduction only):
- internal max 97.32 C; overall sample max 89.07 C @ t=34.9 s
- 16 repeated cycles (activation phase: none detected by existing detector;
  first 90C hold peaks at t=7.6 s < 30 s so counted as repeated cycle 1)
- repeated sample peaks: min 84.71 / mean 85.67 / median 85.52 /
  max 89.04 / std 0.92 C
- >=85 C: 15/16 (93.8%); >=86 C: 1/16 (6.2%); >=87 C: 1/16 (6.2%);
  >=90/92/95 C: 0/16
- majority (>50%) >=86 C: NO -> model does NOT predict that most insulated
  repeated-cycle sample peaks reach the ~86 C qPCR functional reference.
  This is a remaining thermal-vs-biochemical discrepancy to interpret
  scientifically; NO parameter was tuned toward 86 C.
- old vs new insulated sample: old mean 83.51 / new mean 85.67 C;
  old frac>=86 0.0625 / new frac>=86 0.0625 (both 1/16).

Output (new directory, no historical overwrite):
    calibrated_model_output/66C_recalibrated_candidate_v1/
        calibration_66C/  validation_3s_extension/
        insulated_3s_sample_prediction/  comparison/

Tests: tests/test_recalibrate_thermal_model_66C_redo.py (39 items).
Script: recalibrate_thermal_model_66C_redo.py (PHASE A/C/D/E anti-circularity).

---

## 66C candidate — multi-dataset zero-refit transfer validation (2026-08-22)

New task: test whether 66C_RECALIBRATED_CANDIDATE_V1 (k=0.0675, cp=700,
rho=1020, tau=8.0) transfers zero-refit to three additional dataset pairs.
VALIDATION ONLY; no fitting, no time-shift optimization, no cross-correlation.

Script: validate_66C_candidate_multi_dataset.py
Output: calibrated_model_output/66C_candidate_multi_dataset_v1/

Synchronization determination (experimental timing evidence only):
- ALL internal files lack absolute timestamps (only Time(s) / Relative time(s));
  Top files have absolute RECTime -> ABSOLUTE_TIMESTAMP alignment impossible.
- 60C/72C internal under "Recording when reach setting": no documentation of
  simultaneous start -> synchronization_status = UNCERTAIN,
  validation_role = DIAGNOSTIC_ONLY, excluded from authoritative score.
  Evaluated under stated SIMULTANEOUS_START_RELATIVE_T0 assumption.
- 3s pair: internal under "Recording at the start"; same file pair as the
  previously documented synchronized 3s validation (RMSE 1.0643);
  task premise that Top is under "Recording when reach setting" is wrong
  (file is at Calibration root) -> AUTHORITATIVE, SIMULTANEOUS_START_RELATIVE_T0.

Results (zero-refit, same locked parameters):
- 60C (diagnostic): RMSE 1.7405, MAE 1.2996, mean -0.6348, R2 0.9350 (GOOD)
- 72C (diagnostic): RMSE 1.2447, MAE 0.9011, mean -0.1217, R2 0.9643 (EXCELLENT)
- 3s (authoritative): RMSE 1.0643 (bit-exact regression vs previous task),
  MAE 0.9099, mean +0.0359, R2 0.9739 (EXCELLENT)
- Cross-temperature bias: NO monotonic systematic bias
  (60C -0.63, 72C -0.12, 66C +0.03, 3s +0.04 mean residual);
  slight negative bias at lowest-temperature protocol (60C), no monotonic trend.
- Classification: INSUFFICIENT_SYNCHRONIZED_DATA (2 of 3 additional datasets
  cannot be objectively synchronized; only 3s authoritative). Diagnostic RMSEs
  (1.24-1.74 C) are GOOD/EXCELLENT supporting evidence but cannot certify
  transfer without objective sync.

Tests: tests/test_validate_66C_candidate_multi_dataset.py (37 items).
Frozen model and all historical outputs unchanged; no commits.

---

## 66C candidate — known-offset multi-dataset validation V2 (2026-08-22)

New experimental timing information: Top COC thermometer recording starts
1.0 s after the internal protocol Setpoint reaches 90.000 C, for BOTH 60C and
72C. Synchronization is therefore KNOWN_PHYSICAL_OFFSET, rule
SETPOINT_90C_EVENT_PLUS_1S (hard +1.0 s input, NOT fitted; optimized shift
0.0 s).

Anchor: Setpoint column only (never Zone-1 measured temperature); both
workbooks have exactly ONE transition into Setpoint=90 (protocol start):
- 60C: t90 Time(s)=1.138 -> t90_rel=1.047 s; Top start on model axis 2.047 s
- 72C: t90 Time(s)=1.140 -> t90_rel=1.036 s; Top start on model axis 2.036 s

Model fully locked (k=0.0675, cp=700, rho=1020, tau=8.0, h=10, eps=0.90);
complete internal history preserved (FDM from t=0, no reinit at Top start);
environment = first Top value (still ambient 25.35/26.40 C).

Results (zero-refit, known offset):
- 60C: RMSE 1.3749 (was 1.7405 assumed-t0, -0.366), MAE 1.1702,
  mean -0.5124 (was -0.6348), R2 0.9595 -> EXCELLENT, AUTHORITATIVE
- 72C: RMSE 3.0817 (was 1.2447 assumed-t0, +1.837), MAE 2.4851,
  mean -0.0012 (was -0.1217), R2 0.7818 -> MODERATE, AUTHORITATIVE
  (degradation = late-protocol cooling-phase mismatch: model ~72 C vs
  measured ~65 C during 20C-setpoint dips; near-zero mean residual =
  symmetric phase errors, not bias)
- External validation mean/median/worst RMSE: 1.8403 / 1.3749 / 3.0817 C
  (computed from the three external datasets 60C, 72C, and 3s ONLY;
  66C calibration RMSE 0.6368 is NOT included; 3s authoritative under
  SIMULTANEOUS_START_RELATIVE_T0)
- Cross-temperature bias: WEAK (60C -0.512, 66C +0.032, 72C -0.001,
  3s +0.036; no monotonic trend, max consecutive |diff| 0.544 C)
- Classification: ACCEPTABLE_MULTI_DATASET_TRANSFER
- Promotion recommendation: YES_WITH_LIMITATION (72C MODERATE cooling-phase
  error must be reviewed; frozen model NOT edited)

The experimentally documented timing rule is authoritative regardless of
RMSE direction; the 72C degradation is a genuine finding, not a tuning
artifact.

Script: validate_66C_candidate_known_offset.py
Output: calibrated_model_output/66C_candidate_known_offset_validation_v2/
  (60C/, 72C/, comparison/; v1 uncertain-sync output preserved)
Tests: tests/test_validate_66C_candidate_known_offset.py (32 items).

---

## FINAL MODEL PROMOTION — 2026-08-22

The corrected 66 C recalibration candidate, after experimentally-documented
timing validation (SETPOINT_90C_EVENT_PLUS_1S), is promoted to:

    FINAL_FROZEN_THERMAL_MODEL_V1

Parameters:
    k_eff   = 0.0675 W/(m K)
    cp_eff  = 700 J/(kg K)
    rho_COC = 1020 kg/m3
    tau_top = 8.0 s   (output-side lag, Top observation only)

Validation evidence:
    66 C calibration        : RMSE 0.6368 C
    60 C external validation: RMSE 1.3749 C  (known offset, authoritative)
    72 C external validation: RMSE 3.0817 C  (known offset, authoritative)
    3 s external validation : RMSE 1.0643 C  (authoritative)
    mean external RMSE      : 1.8403 C

Documented limitation: 72 C cooling-phase mismatch (cooling RMSE ~4.09 C),
a reduced-order transient limitation; parameters were not altered.

Previous frozen candidate (k=0.055, cp=1200, tau=8.5) is preserved as
superseded history at thermal_model/historical/frozen_strategy_G_candidate.py.

Repository reorganized by function (thermal_model/ core + config/utilities,
workflows/ calibration|validation|prediction|diagnostics|legacy, docs/).
Git tag: thermal-model-final-v1.
