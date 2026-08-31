# phenoforge

Phenomenological model-family bank and ensemble calibration engine for industrial
processes. The reference implementation of **BAPE (Bootstrap-Aggregated
Phenomenological Ensembles)**: instead of selecting one phenomenological model for a
process dataset, fit many realizations (model families x parameter multistarts x
bootstrap resamples) from a curated family bank and aggregate them into a calibrated
ensemble with structural inclusion probabilities.

Consumed by the Fragua research product (`CAOS_RES_Fragua`); designed as a
standalone, domain-agnostic library. The core is pure numpy/scipy (Pyodide-safe).

## What it provides

- **A family bank** (`phenoforge.families`): 48 named, cited, bounded
  phenomenological models with physical parameter ranges and declared
  calibration-data contracts, across seven unit processes.

  | Process | Families | Observable |
  |---|---|---|
  | `flotation` | 7 batch (Garcia-Zuniga first order, Klimpel, Kelsall, modified Kelsall, gamma rate distribution, second order, fully mixed) + 5 continuous (perfect mixer, plug flow, N mixers in series, gamma residence-time distribution, two-class) | recovery versus flotation time, or versus mean residence time |
  | `comminution` | 4 energy-size laws (Rittinger, Kick, Bond, Morrell Mi) + 4 batch grinding population-balance forms (Austin first order, rollover, Herbst-Fuerstenau, Whiten plateau) | specific energy versus size reduction; top-size mass fraction versus grind time |
  | `thickening` | 7 settling-curve signatures (Kynch, Richardson-Zaki, Coe-Clevenger, Talmage-Fitch, Buscall-White, Usher-Scales, Burger-Concha) | mud-line height versus settling time |
  | `leaching` | 6 (shrinking-core film, reaction and product-layer control, Dixon-Hendrix, Mellado-Cisternas, first order) | conversion versus leach time |
  | `dynamics` | 6 step responses (first-order lag, first order plus dead time, two lags in series, repeated lag, underdamped second order, integrating plus lag) | a measured variable versus time since a step |
  | `thermal` | 4 ambient-derating forms | net power versus ambient conditions |
  | `utility` | 5 (per tonne, affine, power law, exponential, logistic) | period consumption versus tonnage or time |

  The comminution energy laws are functions of the (feed, product) size PAIR, but a
  grindability or crushing test holds the feed fixed and sweeps only the closing
  screen. `comminution.fixed_feed_bank(f80_um)` returns the same four laws as
  functions of the product size alone at that feed: keys, parameters, equations and
  references are unchanged, so the projection is exact rather than an approximation.
- **Fitting** (`phenoforge.fit`): bounded multistart trust-region nonlinear least
  squares per family; information criteria (AIC/AICc/BIC) on every fit.
- **Ensembles** (`phenoforge.ensemble`):
  - `select` / `akaike_weights` / `averaged_prediction`: IC selection and multimodel
    averaging (Burnham-Anderson).
  - `glue_fit`: GLUE behavioural parameter-set ensembles (Beven-Binley).
  - `bootstrap_fit`: bagging/bragging of one family (paired or moving-block).
  - `stack_fit`: cross-validated convex stacking over the bank (super-learner
    recipe; M-open rationale per Yao-Vehtari-Simpson-Gelman).
  - `bape_fit`: the BAPE ensemble (bootstrap x family-library subsampling, one
    fitted model per member, inclusion probabilities over families).
- **Metrics** (`phenoforge.metrics`): point (RMSE/MAE/R2), probabilistic calibration
  (ensemble CRPS, PIT, interval coverage), and structural readouts (weight entropy,
  known-truth structural recovery that refuses to run silently null).
- **Routing** (`phenoforge.route`): match a dataset's declared data kinds to the
  families they can calibrate.

## What this package deliberately does not contain

phenoforge is pure numpy and scipy, and stays that way. Two capabilities that
appear in the Fragua results matrix live in the PRODUCT repo rather than here,
and the split is deliberate:

- **The learned tier** (deep ensembles, and the mixture of phenomenological
  experts whose gate weights frozen closed-form curves) needs torch and a GPU
  lane. Putting it here would make a deep-learning stack a hard dependency of a
  package whose entire value is that it is small, deterministic and installable
  anywhere, including inside a browser through Pyodide, which is how the Fragua
  live lane runs it.
- **Ensemble-SINDy** needs pysindy. It is in the comparison as the generic
  term-library CONTRAST to a curated family bank, so it is by definition not a
  phenoforge method: the whole point of the comparison is that one side does not
  use this package's premise.

Everything that is phenomenological in the sense this package means (a bounded,
cited, closed-form family, and any aggregation over such families) belongs here
and is here. If a method needs to know what the equations MEAN, it is a
phenoforge method; if it only needs numbers, it is a baseline and lives with the
experiment.

## Quick start

```python
import numpy as np
from phenoforge import list_families
from phenoforge.families import flotation
from phenoforge.ensemble import bape_fit

t = np.array([0.5, 1, 2, 4, 8, 12, 16, 20], dtype=float)
r = 0.85 * (1 - np.exp(-1.2 * t))  # a batch flotation test

ens = bape_fit(flotation.BATCH_FAMILIES, t, r, n_members=200, seed=0)
print(ens.selection_shares())        # which family explains the data, with what share
print(ens.quantiles(t))              # calibrated predictive bands
```

## Method background (primary sources)

- Fasel, Kutz, Brunton, Brunton 2022. Ensemble-SINDy. Proc. R. Soc. A 478:20210904.
  DOI 10.1098/rspa.2021.0904.
- Pinto, de Azevedo, Oliveira, von Stosch 2019. Bioprocess Biosyst. Eng.
  42:1853-1865. DOI 10.1007/s00449-019-02181-y.
- Duan, Ajami, Gao, Sorooshian 2007. Adv. Water Resour. 30:1371-1386.
  DOI 10.1016/j.advwatres.2006.11.014.
- Beven, Binley 1992. Hydrol. Process. 6:279-298; Beven 2006. J. Hydrol. 320:18-36.
  DOI 10.1016/j.jhydrol.2005.07.007.
- Yao, Vehtari, Simpson, Gelman 2018. Bayesian Anal. 13:917-1007.
  DOI 10.1214/17-BA1091.
- Burnham, Anderson 2004. Sociol. Methods Res. 33:261-304.
  DOI 10.1177/0049124104268644.
- Polat, Chander 2000. Int. J. Miner. Process. 58:145-166.
  DOI 10.1016/S0301-7516(99)00069-1 (flotation family library).

## Development

```
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest tests -v
.venv/Scripts/python -m ruff check src tests
```

## License

MIT. See [LICENSE](LICENSE).
