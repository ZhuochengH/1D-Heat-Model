# Surface Calibration Handoff

## 1. Objective

The scientific objective is to convert the Peltier setpoint temperature into a
calibrated Peltier-surface temperature boundary condition for the downstream
heat-transfer/FDM model.

The data-selection fix:

The final controlled 35 °C stage was followed by Peltier OFF and passive
cooling toward ambient temperature.

That passive cooling must not participate in calibration.

This work does **NOT** change the underlying mathematical model. It only
changes which samples are fed into the existing calibration pipeline.

---

## 2. Original Failure Mode

Previous approximate (invalid) result:

- tau_eff ≈ 18.44 s
- RMSE ≈ 3.576 °C

Two known input/data-processing errors existed:

1. **The descending 85 °C stage was missing.**

   Incorrect descending sequence:

   ```
   95 -> 75 -> 65 -> 55 -> 45 -> 35
   ```

   Correct descending sequence:

   ```
   95 -> 85 -> 75 -> 65 -> 55 -> 45 -> 35
   ```

2. **Data after Peltier OFF were included** as though they belonged to the
   final controlled 35 °C calibration stage (the passive 35 → ~32 °C cooling
   tail was treated as calibration data).

Why these errors could distort tau and RMSE:

- A missing 85 °C descending stage skips a real thermal transition; the fitted
  single effective tau then has to absorb an artificial jump (85→75 directly),
  inflating tau_eff (≈18.44 s).
- Including the passive cooling ramp as if it were a controlled calibration
  stage feeds a slow monotonic decline into the steady-state and dynamic fit,
  inflating the full-trace RMSE (≈3.576 °C).

**The 18.44 s / 3.576 °C result is NOT considered a valid calibration result.**

---

## 3. Correct Experimental Setpoint Sequence

```
35 -> 45 -> 55 -> 65 -> 75 -> 85 -> 95 -> 85 -> 75 -> 65 -> 55 -> 45 -> 35
```

Total controlled stages: **13**

Do **NOT** include 32 °C as a setpoint. The later decline toward
approximately 32 °C was passive cooling after Peltier control was cancelled.

---

## 4. Analysis Cutoff

Real data:

- File: `D:\桌面\微流控毕设\Calibration\zone 1_model check.xls`
- Sheet: `Data`
- Used temperature columns: `T1`, `T2`
- Sampling interval: 1 s

Rows:

- Original usable rows: **1142**
- Confirmed analysis cutoff: **analysis_end_s = 1065 s**
- Cutoff convention: retain samples where `time_s <= 1065`
- Retained rows: **1066**
- Excluded rows: **76**

The final retained sample corresponds to:

- `time_s = 1065`
- 1-based sample = 1066

Passive cooling begins immediately after this controlled region.

---

## 5. Implementation

New helper:

```python
apply_analysis_cutoff(time_s, *arrays, analysis_end_s=None)
```

New CLI option:

- `--analysis-end-s`
- type: `float`
- default: `None`

Backward compatibility:

- If `--analysis-end-s` is omitted, the original full-data behavior is
  preserved (no truncation; `analysis_end_s=None` keeps every sample).

Processing order:

```
Excel input
-> T1/T2 cleaning
-> Tmean construction
-> time_s construction
-> apply_analysis_cutoff          <-- cutoff applied here
-> automatic change-point detection
-> segment construction
-> steady-state extraction
-> steady-state regression
-> dynamic tau fitting
-> RMSE
-> plots / exports
```

The cutoff is applied **BEFORE** segmentation and all fitting.

The real experiment uses `--set-col NONE`. This automatic-segmentation path is
explicitly covered by an integration regression test
(`test_cutoff_precedes_auto_segmentation_when_set_col_none`).

Output directory resolution (added in repository cleanup):

- Canonical default generated-output directory:
  `<repository root>/calibration_output/`
- Resolved from `__file__` (`PROJECT_ROOT = Path(__file__).resolve().parents[1]`),
  **independent of the shell current working directory**.
- Running from either `<repository root>` or
  `<repository root>/Surface_calibration` uses the **same** default output
  directory.
- Explicit `--output-dir` behavior preserved:
  - absolute explicit path → use the supplied absolute path
  - relative explicit path → resolve relative to cwd
- Helper: `resolve_output_dir(output_dir_arg, project_root=None)`.

---

## 6. Mathematical Model

The mathematical structure was **NOT** changed.

Steady-state model:

```
T_inf = a*T_set + b
```

First-order dynamic model:

```
T_s(t) = T_inf,j + [T_s(t_j) - T_inf,j] * exp(-(t - t_j)/tau_eff)
```

where:

```
T_inf,j = a*T_set,j + b
```

Do **not** introduce additional thermal RC terms.

Do **not** introduce a second-order response model.

Do **not** replace the single effective tau with separate heating/cooling tau
in the final model.

---

## 7. Final Verified Calibration Parameters

Final Mean(T1,T2) model:

| Parameter | Value |
|-----------|-------|
| a | 0.950490 |
| b | 1.811586 °C |
| steady-state R² | 0.999987 |
| tau_eff | 7.3072 s |
| diagnostic tau_heating | 7.2064 s |
| diagnostic tau_cooling | 7.4031 s |
| full-trace RMSE | 0.2151 °C |

Final usable steady-state equation:

```
T_inf = 0.950490*T_set + 1.811586
```

Final usable dynamic equation:

```
T_s(t) = T_inf,j + [T_s(t_j)-T_inf,j] * exp(-(t-t_j)/7.3072)
```

with:

```
T_inf,j = 0.950490*T_set,j + 1.811586
```

---

## 8. Steady-State Results

Extracted Mean steady points (setpoint -> steady_temp °C), all 13 accepted:

```
35  -> 35.2
45  -> 44.6
55  -> 54.05
65  -> 63.5
75  -> 73.05
85  -> 82.55
95  -> 92.2
85  -> 82.7
75  -> 73.1
65  -> 63.55
55  -> 54.05
45  -> 44.55
35  -> 35.1
```

All 13 steady points were accepted (`accepted=True`).

Description of the steady-state behavior:

Surface temperature is **close to the setpoint near 35 °C** and becomes
**progressively lower at higher setpoints**, reaching **approximately 2.8 °C
below the setpoint at 95 °C** (92.2 °C vs 95 °C).

---

## 9. Dynamic Result Interpretation

- tau_heating = 7.2064 s
- tau_cooling = 7.4031 s

Their difference is only approximately **2.7%**.

Therefore the current data do not provide a strong reason to replace the
single tau_eff model with separate heating and cooling constants.

The retained engineering model is:

```
tau_eff = 7.3072 s
```

Important scientific wording:

tau_eff is an **effective response constant**. It may include contributions
from:

- actual Peltier response
- thermal transfer between Peltier and measured surface
- thermometer response lag
- 1 Hz acquisition

It must **NOT** be described as the intrinsic physical time constant of the
Peltier alone.

---

## 10. Reconstruction Result

- RMSE = 0.2151 °C

Verified `final_model_validation.png`:

- measured Mean(T1,T2) and final-model reconstruction nearly overlap
- both heating and cooling transitions are reproduced closely
- no large systematic heating/cooling asymmetry is visible
- remaining discrepancies are small and concentrated around transitions

Caveats:

- Do not claim that RMSE reaches the fundamental measurement limit.
- Do not claim independent validation — this is reconstruction on the
  calibration dataset.

---

## 11. Tests and Verification Layer

pytest result (final):

```
18 passed, 0 failed
```

The verification layer was first built as 13 cutoff/calibration tests, then
**expanded to 18 tests** after output-path standardization.

Original cutoff/calibration tests (13):

Unit tests (7):

- no cutoff preserves every sample
- valid cutoff keeps only samples with `time_s <= analysis_end_s`
- time/T1/T2/Tmean remain aligned after truncation
- passive cooling tail excluded from the final 35 °C plateau
- negative / NaN / ±inf / non-numeric cutoff raise a clear exception
- cutoff beyond the final sample preserves all samples
- cutoff before the first sample (empty result) raises a clear exception

Integration tests (2):

1. `test_cutoff_applies_before_segmentation_in_pipeline` — Set-column branch
   (`segments_from_set_column`).
2. `test_cutoff_precedes_auto_segmentation_when_set_col_none` — **the actual
   `--set-col NONE` automatic-segmentation path**
   (`detect_change_points_from_temperature` →
   `segments_from_detected_changes`).

The `--set-col NONE` test was experimentally checked by temporarily moving the
cutoff **after** segmentation: the test **failed** because automatic
change-point detection received the passive-cooling samples (200 samples
instead of 180). After restoring the correct order, the test passed again.
Therefore the test provides regression protection for the scientifically
important pipeline ordering.

The remaining 4 of the 13 tests are additional cutoff/calibration unit tests
beyond the 7 listed above (see `tests/test_analysis_cutoff.py`).

Counting note: `test_invalid_cutoff_raises_clear_exception` is parametrized
(5 variants: negative, NaN, +inf, -inf, non-numeric), so the 7 listed unit-test
categories expand to 11 counted unit tests; 11 + 2 integration = 13
cutoff/calibration tests, plus 5 output-directory tests = 18 total.

Output-directory regression tests (5, added with output-path standardization):

- `test_project_root_is_repo_root`
- `test_default_output_dir_is_repo_root_calibration_output`
- `test_default_independent_of_cwd`
- `test_explicit_absolute_output_dir_overrides_default`
- `test_explicit_relative_output_dir_resolves_against_cwd`

---

## 12. Real-Data Verification

Automatic segmentation detected exactly **13** controlled stages:

```
seg0   35°C: [0:78)
seg1   45°C: [78:146)
seg2   55°C: [146:210)
seg3   65°C: [210:267)
seg4   75°C: [267:330)
seg5   85°C: [330:396)
seg6   95°C: [396:499)
seg7   85°C: [499:566)
seg8   75°C: [566:654)
seg9   65°C: [654:765)
seg10  55°C: [765:884)
seg11  45°C: [884:976)
seg12  35°C: [976:1066)
```

The final controlled 35 °C stage **ends exactly at the analysis cutoff**
(segment end index 1066 == cutoff 1065 + 1, half-open slice `[976:1066)`).

---

## 13. Generated Outputs

The calibration run generated 10 files in the canonical output directory
`<repository root>/calibration_output/`:

1. `calibration_params.json`
2. `calibration_summary.csv`
3. `dynamic_calibration.png`
4. `dynamic_steps.csv`
5. `final_calibration_equation.txt`
6. `final_model_validation.csv`
7. `final_model_validation.png`
8. `steady_calibration.png`
9. `steady_points.csv`
10. `trace_with_fit.csv`

These are **generated runtime artifacts** and are **not** part of the
scientific implementation commit. The canonical root `calibration_output/`
directory is ignored by Git (`/calibration_output/` in `.gitignore`).

The obsolete `Surface_calibration/calibration_output/` directory was
intentionally deleted (see Section 14) and must not be restored.

---

## 14. Git Verification

Three commits are relevant. **Scientific implementation** is explicitly
distinguished from **repository-maintenance** changes:

### 14.1 Scientific implementation

```
fd75990aa5c4fcd55ff86d689b54ce167e475942
fix: exclude passive cooling from surface calibration
```

Purpose:

- add explicit analysis cutoff (`--analysis-end-s` / `apply_analysis_cutoff`)
- prevent Peltier-OFF passive cooling from contaminating calibration
- add cutoff unit/integration regression tests

Commit contents:

- `Surface_calibration/peltier_surface_calibration_v2.py`
- `pyproject.toml`
- `uv.lock`
- `tests/conftest.py`
- `tests/test_analysis_cutoff.py`

Commit statistics:

```
5 files changed, 563 insertions(+), 4 deletions(-)
```

### 14.2 Repository cleanup — stale tracked outputs removed

```
d950cc5
chore: remove stale tracked calibration outputs
```

Purpose:

- intentionally remove obsolete/test-generated files from
  `Surface_calibration/calibration_output/`
- the deletion was explicitly approved by the user and is **not** an
  accidental deletion
- this directory must **not** be restored

Commit statistics:

```
10 files changed, 2398 deletions(-)
```

### 14.3 Repository cleanup — output-directory standardization

```
b62c64a384b74d2cd9b1ce303deaff941e9565fd
chore: standardize calibration output structure
```

Purpose:

- canonical default output directory is now
  `<repository root>/calibration_output/`
- default location is independent of shell cwd (resolved from `__file__`)
- explicit `--output-dir` still overrides the default
- add 5 output-path regression tests
- ignore root `/calibration_output/` in `.gitignore`
- keep `HANDOFF.md` ignored as a local reviewer document

Commit contents:

- `.gitignore`
- `Surface_calibration/peltier_surface_calibration_v2.py`
- `tests/test_output_dir.py`

Commit statistics:

```
3 files changed, 104 insertions(+), 16 deletions(-)
```

### 14.4 Overall state

Commits `d950cc5` and `b62c64a` did **NOT** change calibration equations,
fitted parameters, segmentation, cutoff behavior, tau fitting, or RMSE.

Push performed: **NO**.

Implementation commits exist locally; **local `main` is ahead of `origin/main`**
(no push has been performed). Repository-maintenance commits after `fd75990`
do not alter the verified calibration result.

Final tracked working tree after `b62c64a` is **clean**. `HANDOFF.md` and
`<repository root>/calibration_output/` are intentionally ignored by Git and
are **not** unresolved untracked changes.

---

## 15. Remaining Assumptions and Limitations

- Sampling interval is 1 s (fixed, not estimated from timestamps in the run).
- T1 and T2 are averaged for the final model.
- Thermometer response lag is not independently identified.
- Analysis cutoff is manually specified rather than automatically inferred.
- One effective tau is used.
- Current comparison uses calibration data rather than an independent
  validation experiment.
- Model is intended as an engineering calibration boundary condition, not a
  detailed physical Peltier internal model.

---

## 16. Current Scientific Conclusion

After correcting the descending setpoint mapping and excluding Peltier-OFF
passive cooling, the simple first-order calibration model closely reconstructs
the measured Mean(T1,T2) surface-temperature trajectory (RMSE 0.215 °C).

The previous approximate result (tau_eff ≈ 18.44 s, RMSE ≈ 3.576 °C) is
**invalid as a calibration result** because it used incorrect setpoint mapping
and contaminated data selection.

The corrected result (tau_eff = 7.3072 s, RMSE = 0.2151 °C) does **not**
provide strong evidence that a higher-order dynamic model or separate
heating/cooling tau values are required.

No independent validation is claimed — the current reconstruction uses the
calibration dataset itself.

The intended next use is as a calibrated Peltier-surface bottom boundary
condition for the downstream heat-transfer/FDM model.

---

## 17. Items for ChatGPT Review

1. Scientific correctness of the analysis cutoff.
2. Correctness of the 13-stage setpoint mapping (descending 85 °C included).
3. Adequacy of the regression tests (unit + integration, incl. the
   `--set-col NONE` path).
4. Correctness of the final calibration equations.
5. Appropriateness of retaining one effective tau_eff (heating/cooling tau
   differ by only ~2.7%).
6. Whether residual structure in `final_model_validation.png` suggests any
   justified model extension.
7. Suitability of the calibration as the FDM bottom boundary input.
8. Recommended independent validation strategy.
9. How best to integrate the calibrated boundary into the existing FDM code.

---

## Final Repository State

- Scientific implementation commit:
  `fd75990aa5c4fcd55ff86d689b54ce167e475942`
  (fix: exclude passive cooling from surface calibration)
- Stale-output cleanup commit:
  `d950cc5` (chore: remove stale tracked calibration outputs)
- Output-structure commit:
  `b62c64a384b74d2cd9b1ce303deaff941e9565fd`
  (chore: standardize calibration output structure)
- Push performed: **NO** — local `main` is ahead of `origin/main`.
