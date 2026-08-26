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

- **A family bank** (`phenoforge.families`): named, cited, bounded phenomenological
  models with physical parameter ranges and declared calibration-data contracts.
  Shipping now: the flotation kinetics zoo (Garcia-Zuniga first-order, Klimpel,
  Kelsall, modified Kelsall, gamma rate distribution, second-order, fully mixed, bank
  of N mixers) and the comminution energy-size laws (Rittinger, Kick, Bond, Morrell
  Mi). The bank grows with the Fragua build (thickening, leaching, utilities, PBM).
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
