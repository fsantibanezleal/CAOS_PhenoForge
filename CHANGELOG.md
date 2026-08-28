# Changelog

All notable changes to phenoforge. Format follows Keep a Changelog; versions use the
CAOS `X.XX.XXX` display form (manifest carries the unpadded semver twin).

## [0.04.001] - 2026-08-27

### Fixed

- `hybrid.gp`: escalating-jitter Cholesky. An RBF kernel on a large-valued,
  closely spaced driver (an eleven-year record indexed 1 to 11 with values near
  1.3e4) is numerically singular though positive definite in exact arithmetic;
  plain Cholesky raised LinAlgError and killed a canonical bake mid-run. Jitter
  escalates from 1e-12 to 1e-2 of the mean diagonal and still raises beyond
  that, so a genuinely broken kernel is not silently regularized. Regression
  test on the exact failing series (84 tests).

## [0.04.000] - 2026-08-27

### Added

- Continuous (plant) flotation bank (`families.flotation_continuous`, 5
  families): single perfectly mixed cell, plug flow, N mixers in series,
  gamma-distributed floatability, and fast/slow classes, all as recovery versus
  MEAN RESIDENCE TIME. N mixers nests both the single-mixer and plug-flow
  bounds exactly, so the fitted N measures where a circuit sits between them.
- Interval and calibration metrics (`metrics.intervals`): the Winkler interval
  score (Gneiting and Raftery 2007), sharpness, PIT calibration error as a
  Kolmogorov-Smirnov distance, coverage deviation across nominal levels, the
  reliability curve, the effective family count (Hill number of the weights),
  and per-parameter dispersion for the equifinality readout.
- 20 new tests (83 total), including that widening an already-covering band
  cannot improve the interval score, and that the continuous families reduce to
  each other at their documented limits.

### Removed

- `flot_bank_mixers` from the batch flotation bank: it was the N-mixers model,
  which belongs to (and now lives in) the continuous bank as `flotc_n_mixers`
  with the identical equation. Batch and continuous banks no longer overlap and
  the router keeps them apart.

## [0.03.000] - 2026-08-27

### Added

- Thickening bank (7 families): the batch settling-curve signatures of Kynch
  ideal kinematics, Richardson-Zaki hindered settling, Coe-Clevenger two-zone,
  the Talmage-Fitch tangent construction, Buscall-White compressional
  relaxation, the Usher-Scales series combination, and the Burger-Concha
  hyperbolic-to-parabolic transition. Each docstring states exactly which
  observable consequence of its theory the fittable form represents.
- Leaching bank (6 families): shrinking-core film, surface-reaction and
  product-layer control (the last inverted from its implicit form by monotone
  bisection to double precision), the Dixon-Hendrix two-scale column response,
  the Mellado-Cisternas analytical heap model, and lumped first-order recovery.
- Batch grinding population-balance bank (4 families): Austin first-order
  breakage, the Austin abnormal-breakage rollover, Herbst-Fuerstenau
  energy-specific selection, and the Whiten perfect-mixing residual. The
  rollover and residual families both NEST the first-order form exactly.
- Plant utility bank (5 families): per-tonne coefficient, base load plus
  marginal coefficient, power-law scaling, exponential trend and saturating
  logistic trend, for water and energy period balances.
- 34 families total across 5 processes, every one carrying its primary
  reference; 22 new tests (63 total) covering shape, physical limits, nesting
  identities, the implicit-form inversion, router separation, and parameter and
  structural recovery on the new banks.

## [0.02.001] - 2026-08-27

### Fixed

- `bayes.gw._log_jacobian`: numerically stable logaddexp form. The naive
  sigmoid evaluation emitted divide-by-zero RuntimeWarnings and collapsed a
  saturated walker's log-Jacobian to -inf where the exact value is finite
  (log(hi-lo) - |z| to leading order); regression test added (42 total).

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
