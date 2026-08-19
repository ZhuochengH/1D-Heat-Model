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
