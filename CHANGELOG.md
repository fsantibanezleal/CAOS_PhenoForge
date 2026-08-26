# Changelog

All notable changes to phenoforge. Format follows Keep a Changelog; versions use the
CAOS `X.XX.XXX` display form (manifest carries the unpadded semver twin).

## [0.02.000] - 2026-08-26

### Added

- `phenoforge.bayes`: affine-invariant ensemble MCMC (Goodman-Weare 2010,
  DOI 10.2140/camcos.2010.5.65; emcee-style, reimplemented pure-numpy and
  Pyodide-safe) sampling (theta, sigma) posteriors per family with bounded
  logit parameterization; BIC-approximated Bayesian model averaging across the
  bank (Raftery 1995, DOI 10.2307/271063) with proper mixture predictive draws.
- `phenoforge.hybrid`: exact GP model-discrepancy on a phenomenological backbone
  (Kennedy-O'Hagan 2001, DOI 10.1111/1467-9868.00294; RBF + noise, marginal
  likelihood hyperparameters), with posterior function draws as ensemble members.
- 4 new tests (41 total).

## [0.01.002] - 2026-08-26

### Changed

- release workflow gains workflow_dispatch (push-event triggers were not firing on
  this repository; publish is dispatched explicitly). No library code changes.

## [0.01.001] - 2026-08-26

### Fixed

- Small-sample criterion handling: on very short series (n <= p + 1 for every
  family) AICc is +inf bank-wide and selection died on an all-NaN slice. Added
  `ensemble.ic.choose_criterion` (coherent bank-level AICc-to-BIC fallback) and a
  per-member BIC fallback inside `bape_fit` so tiny bootstrap resamples keep their
  members. Found by the Fragua sparse-n6 case variant; covered by
  `tests/test_small_n_fallback.py` (3 tests).

## [0.01.000] - 2026-08-25

### Added

- Family bank core: `ModelFamily`, `Param`, `Reference`, `DataKind`, `FitResult`
  (with AIC/AICc/BIC), registry with process filtering.
- Flotation kinetics families (8): Garcia-Zuniga first-order, Klimpel rectangular,
  Kelsall, modified Kelsall, gamma rate distribution, second-order, fully mixed,
  bank of N perfect mixers. Comminution energy-size families (4): Rittinger, Kick,
  Bond, Morrell Mi. Every family cited to its primary source.
- Fitting: bounded multistart trust-region NLS (`fit_family`, `fit_bank`), optional
  weighted residuals.
- Ensemble methods: IC selection + Akaike-weight model averaging; GLUE behavioural
  ensembles (NSE informal likelihood, weighted quantiles); bootstrap
  bagging/bragging (paired + moving-block); cross-validated convex stacking (NNLS
  simplex weights); BAPE (bootstrap x family-library subsampling with AICc member
  selection, inclusion probabilities, selection shares).
- Metrics: RMSE/MAE/R2; ensemble CRPS, PIT values/histogram, central-interval
  coverage; weight entropy; structural recovery scoring that raises on a missing
  truth key (can never silently run null).
- Dataset-to-family router over declared DataKind contracts.
- 34 tests; ruff clean; MIT license.
