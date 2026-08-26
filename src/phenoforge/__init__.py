"""phenoforge: phenomenological model-family bank + ensemble calibration engine.

The package implements the BAPE methodology (Bootstrap-Aggregated Phenomenological
Ensembles): instead of selecting ONE phenomenological model for a process dataset,
fit MANY realizations (model families x parameter multistarts x bootstrap resamples)
from a curated family bank and aggregate them into a calibrated ensemble with
structural inclusion probabilities.

Nearest prior art, cited and differentiated (see the Fragua research dossiers):
- Fasel, Kutz, Brunton, Brunton 2022, Ensemble-SINDy, Proc. R. Soc. A 478:20210904,
  DOI 10.1098/rspa.2021.0904 (bagging over generic term libraries).
- Pinto, de Azevedo, Oliveira, von Stosch 2019, Bioprocess Biosyst. Eng. 42:1853-1865,
  DOI 10.1007/s00449-019-02181-y (bootstrap-aggregated hybrid models, one family).
- Duan, Ajami, Gao, Sorooshian 2007, Adv. Water Resour. 30:1371-1386,
  DOI 10.1016/j.advwatres.2006.11.014 (BMA across model structures, hydrology).
- Beven, Binley 1992, Hydrol. Process. 6:279-298 (GLUE); Beven 2006,
  J. Hydrol. 320:18-36, DOI 10.1016/j.jhydrol.2005.07.007 (equifinality).
- Yao, Vehtari, Simpson, Gelman 2018, Bayesian Anal. 13:917-1007,
  DOI 10.1214/17-BA1091 (stacking in the M-open setting).

The core is pure numpy/scipy (Pyodide-safe by design).
"""

__version__ = "0.01.000"

from phenoforge.families.base import DataKind, FitResult, ModelFamily, Param
from phenoforge.families.registry import get_family, list_families
from phenoforge.fit.nls import fit_family
from phenoforge.router import route

__all__ = [
    "DataKind",
    "FitResult",
    "ModelFamily",
    "Param",
    "__version__",
    "fit_family",
    "get_family",
    "list_families",
    "route",
]
