"""Plant water and energy consumption families (annual/period balances).

All families map x -> consumption. For the intensity families x is processed
tonnage (Mt per period); for the trend families x is time (years since the
series start). Which one a dataset can calibrate is declared through DataKind,
so the router never offers a tonnage model to a pure time series.

Primary sources (transcribed from the Fragua families dossier):
- Gunson, A.J., Klein, B., Veiga, M., Dunbar, S., 2012. Reducing mine water
  requirements. J. Clean. Prod. 21(1), 71-81. DOI 10.1016/j.jclepro.2011.08.020
  (site water balance: makeup = evaporation + tailings retention + product
  moisture + seepage - recycle).
- COCHILCO annual reports, "Consumo de agua en la mineria del cobre" and
  "Proyeccion del consumo de energia electrica de la mineria del cobre"
  (per-tonne unit coefficients by route; portfolio-times-coefficient
  projections). Public, attribute COCHILCO.
- Bond, F.C., 1952. The third theory of comminution. Trans. AIME 193, 484-494,
  and Morrell, S., 2004. Int. J. Miner. Process. 74(1-4), 133-141,
  DOI 10.1016/j.minpro.2003.10.002 (specific energy consumption; the operating
  work index as the efficiency KPI behind the SEC families).
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_GUNSON = Reference(
    "Gunson, Klein, Veiga, Dunbar 2012, J. Clean. Prod. 21(1), 71-81",
    "10.1016/j.jclepro.2011.08.020",
)
_COCHILCO = Reference(
    "COCHILCO, Consumo de agua / Proyeccion de consumo de energia electrica en "
    "la mineria del cobre (annual editions; public, attribute COCHILCO)",
    None,
)
_MORRELL = Reference(
    "Morrell 2004, Int. J. Miner. Process. 74(1-4), 133-141",
    "10.1016/j.minpro.2003.10.002",
)
_BOND = Reference("Bond 1952, Trans. AIME 193, 484-494", None)


def _proportional(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    q = th[..., 0]
    return q * x


def _affine(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    base, q = th[..., 0], th[..., 1]
    return base + q * x


def _power_law(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    a, b = th[..., 0], th[..., 1]
    return a * np.power(np.maximum(x, 1e-9), b)


def _exp_trend(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    c0, g = th[..., 0], th[..., 1]
    return c0 * np.exp(g * x)


def _logistic_trend(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    c_inf, c0, k = th[..., 0], th[..., 1], th[..., 2]
    return c_inf / (1.0 + (c_inf / np.maximum(c0, 1e-9) - 1.0) * np.exp(-k * x))


PER_TONNE = ModelFamily(
    key="util_per_tonne",
    name="Per-tonne unit coefficient (proportional)",
    process="utility",
    params=(Param("q", "unit/Mt", 1e-4, 1e4, 1.0, "unit consumption per unit throughput"),),
    fn=_proportional,
    needs=(DataKind.ANNUAL_BALANCE,),
    equation=r"C = q\,T",
    assumptions=(
        "Consumption strictly proportional to throughput with a stationary unit "
        "coefficient per process route (the COCHILCO reporting model). No fixed "
        "base load, so any nonzero intercept in the data falsifies it against "
        "the affine family."
    ),
    references=(_COCHILCO, _GUNSON),
    x_doc="x: processed tonnage per period",
)

AFFINE_BASELOAD = ModelFamily(
    key="util_affine_baseload",
    name="Base load plus per-tonne coefficient",
    process="utility",
    params=(
        Param("C_base", "unit", 0.0, 1e5, 1.0, "throughput-independent base load"),
        Param("q", "unit/Mt", 1e-4, 1e4, 1.0, "marginal unit consumption"),
    ),
    fn=_affine,
    needs=(DataKind.ANNUAL_BALANCE,),
    equation=r"C = C_{base} + q\,T",
    assumptions=(
        "A fixed site demand (camp, pumping, evaporation from ponds sized by "
        "area not throughput) plus a marginal per-tonne term. The Gunson water "
        "balance makes the base term explicit; the split is what distinguishes "
        "an efficiency gain from a throughput change."
    ),
    references=(_GUNSON, _COCHILCO),
    x_doc="x: processed tonnage per period",
)

POWER_LAW_SCALING = ModelFamily(
    key="util_power_law",
    name="Power-law scaling with throughput",
    process="utility",
    params=(
        Param("a", "unit", 1e-4, 1e5, 1.0, "scale coefficient"),
        Param("b", "-", 0.3, 1.6, 1.0, "scaling exponent"),
    ),
    fn=_power_law,
    needs=(DataKind.ANNUAL_BALANCE,),
    equation=r"C = a\,T^{\,b}",
    assumptions=(
        "Economies (b < 1) or penalties (b > 1) of scale in utility demand; b = 1 "
        "recovers the proportional family exactly, so the pair is a nested test "
        "of whether unit intensity is really constant across operations."
    ),
    references=(_COCHILCO, _BOND),
    x_doc="x: processed tonnage per period",
)

EXP_TREND = ModelFamily(
    key="util_exp_trend",
    name="Exponential consumption trend",
    process="utility",
    params=(
        Param("C0", "unit", 1e-4, 1e5, 1.0, "consumption at series start"),
        Param("g", "1/period", -0.3, 0.3, 0.02, "growth rate"),
    ),
    fn=_exp_trend,
    needs=(DataKind.ANNUAL_BALANCE,),
    equation=r"C(t) = C_0 e^{g t}",
    assumptions=(
        "Constant fractional growth (the headline form of the COCHILCO "
        "projections). Unbounded by construction, so it is the family whose "
        "extrapolation the bounded logistic family is meant to challenge."
    ),
    references=(_COCHILCO,),
    x_doc="x: periods since series start (years)",
)

LOGISTIC_TREND = ModelFamily(
    key="util_logistic_trend",
    name="Saturating (logistic) consumption trend",
    process="utility",
    params=(
        Param("C_inf", "unit", 1e-3, 1e6, 10.0, "saturation level"),
        Param("C0", "unit", 1e-4, 1e5, 1.0, "consumption at series start"),
        Param("k", "1/period", 1e-3, 1.5, 0.15, "approach rate"),
    ),
    fn=_logistic_trend,
    needs=(DataKind.ANNUAL_BALANCE,),
    equation=r"C(t) = \frac{C_\infty}{1 + \left(\frac{C_\infty}{C_0}-1\right)e^{-kt}}",
    assumptions=(
        "Growth bounded by a physical or licensing ceiling (desalination "
        "capacity, grid allocation, water rights). Indistinguishable from the "
        "exponential family early in a series: the honest readout on short "
        "utility records is the equivalence class, not a winner."
    ),
    references=(_COCHILCO, _MORRELL),
    x_doc="x: periods since series start (years)",
)

ALL: tuple[ModelFamily, ...] = (
    PER_TONNE,
    AFFINE_BASELOAD,
    POWER_LAW_SCALING,
    EXP_TREND,
    LOGISTIC_TREND,
)
