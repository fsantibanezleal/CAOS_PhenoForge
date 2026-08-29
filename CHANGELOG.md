# Changelog

## 0.05.002 - 2026-08-29

### Fixed
- `__version__` was a hardcoded literal in `__init__.py` and had drifted: the
  0.05.000 and 0.05.001 releases both reported `0.04.002`. The consuming product
  stamps this string into EVERY baked manifest as the engine that produced the
  result, so a whole matrix of artifacts would have recorded an engine version
  that never ran, and the reproducibility claim those artifacts exist to support
  would have been false.

  The version is now DERIVED from the installed distribution metadata, whose
  single source is `pyproject.toml`, so it cannot drift again. Three tests pin
  the mechanism rather than today's value: one asserts equality with
  `importlib.metadata.version`, one fails if a string literal is ever assigned to
  `__version__` in the source again, and one checks the shape.

## 0.05.001 - 2026-08-28

### Fixed
- `bayes.gw.sample_posterior` defaulted to `sigma_bounds=(1e-4, 0.5)` in RAW
  target units, and the log-posterior returns minus infinity outside them. Right
  for a recovery fraction, meaningless for anything else, and worse than tight:
  the walker start is an order of magnitude below the data spread, so for a
  series spanning about 17 units the start sat ABOVE the hard cap and every
  walker was born at minus infinity. The sampler was degenerate, not merely
  overconfident, on every large-scale observable.

  The prior support is now derived from the data when `sigma_bounds` is None
  (the new default): a noise standard deviation cannot sensibly exceed a few
  times the spread of the observable, nor be a millionth of it. Callers that
  know the noise scale independently may still pin it. The walker start is
  clipped inside the support.

  This is the third instance of one pattern found on 2026-08-28, after the
  deep-ensemble sigmoid and the E-SINDy blow-up bound, both in the consuming
  product: a constant written for the first observable a product ever had, hard
  coded into a method, never revisited as the observables multiplied. Eight
  tests pin the behaviour across three orders of magnitude of target scale.

## 0.05.000 - 2026-08-28

### Added
- `families/dynamics.py`: process step-response bank, six competing lumped
  structures over the same reaction curve (first-order lag, first order plus
  dead time, two unequal lags in series, repeated lag, underdamped second
  order, integrating plus lag). New `DataKind.STEP_RESPONSE` and new process
  `dynamics`. Cited to Ziegler and Nichols 1942, Sundaresan and Krishnaswamy
  1978, Ogunnaike and Ray 1994, Marlin 2000, Astrom and Hagglund 2006 and
  Seborg et al. 2016.
- `comminution.fixed_feed_bank(f80_um)`: the four energy-size laws projected
  onto the product size at a constant series feed, which is what a grindability
  or crushing test actually sweeps. Keys, parameters, equations and references
  are unchanged, so the projection is exact rather than an approximation.

### Notes
- The overdamped pair is now continuous at its repeated pole: the textbook
  closed form divides by `tau_1 - tau_2`, and the critically damped expression
  is substituted inside a relative tolerance of 1e-6.
- Bank totals: 48 families across 7 processes.

All notable changes to phenoforge. Format follows Keep a Changelog; versions use the
CAOS `X.XX.XXX` display form (manifest carries the unpadded semver twin).

## [0.04.002] - 2026-08-27

### Fixed

- `hybrid.gp.member_draws`: the POSTERIOR covariance is a difference of kernels,
  so it can be numerically indefinite even where the prior kernel factors
  cleanly. The fixed 1e-10 ridge was insufficient on a large-valued annual
  record and raised inside the draw, killing a canonical bake AFTER 0.04.001
  had fixed the kernel path. The covariance is now symmetrized, factored with
  the same escalating trace-relative jitter, and sampled from that factor
  directly rather than through `multivariate_normal`, so the jitter that
  succeeded is the one actually used. Regression test covers both the data grid
  and a dense extrapolation grid (85 tests).

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
