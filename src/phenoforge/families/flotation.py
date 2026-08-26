"""Flotation kinetics family bank (batch and continuous).

Equations, parameter ranges, and assumptions transcribed from the Fragua families
dossier (wip/fragua/research/mining-model-families-2026-08-25.md), whose primary
sources are:

- Garcia-Zuniga, H., 1935. Boletin Minero 47, 83-86 (first-order flotation law).
- Klimpel, R.R., 1980. In: Mineral Processing Plant Design, 2nd ed., AIME, 907-934.
- Kelsall, D.F., 1961. Trans. Inst. Min. Metall. 70, 191-204.
- Imaizumi, T., Inoue, T., 1963. 6th IMPC; Loveday, B.K., 1966. Trans. IMM 75, C219-C225.
- Polat, M., Chander, S., 2000. Int. J. Miner. Process. 58, 145-166,
  DOI 10.1016/S0301-7516(99)00069-1 (the canonical first-order-families review; shows
  fitted rate DISTRIBUTIONS are often artifacts of the two-parameter constraint).
- Bu, X., Xie, G., Peng, Y., Ge, L., Ni, C., 2017. Physicochem. Probl. Miner.
  Process. 53(1), 342-365 (the model-zoo review incl. second-order and fully mixed).

All batch families map t (min) -> cumulative recovery fraction R in [0, 1].
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_POLAT_CHANDER = Reference(
    "Polat and Chander 2000, Int. J. Miner. Process. 58, 145-166",
    "10.1016/S0301-7516(99)00069-1",
)
_BU_2017 = Reference(
    "Bu, Xie, Peng, Ge, Ni 2017, Physicochem. Probl. Miner. Process. 53(1), 342-365",
    None,
)


def _fo(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    rinf, k = th[..., 0], th[..., 1]
    return rinf * (1.0 - np.exp(-k * x))


def _klimpel(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    rinf, k = th[..., 0], th[..., 1]
    kt = k * x
    # (1 - exp(-kt)) / kt -> 1 as kt -> 0; series-safe evaluation.
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(kt > 1e-8, -np.expm1(-kt) / np.where(kt == 0.0, 1.0, kt), 1.0 - kt / 2.0)
    return rinf * (1.0 - frac)


def _kelsall(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    phi, kf, ks = th[..., 0], th[..., 1], th[..., 2]
    return (1.0 - phi) * (1.0 - np.exp(-kf * x)) + phi * (1.0 - np.exp(-ks * x))


def _kelsall_mod(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    rinf = th[..., 0]
    return rinf * _kelsall(x, th[..., 1:])


def _gamma(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    rinf, a, b = th[..., 0], th[..., 1], th[..., 2]
    return rinf * (1.0 - np.power(1.0 + b * x, -a))


def _second_order(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    rinf, k = th[..., 0], th[..., 1]
    return (rinf * rinf * k * x) / (1.0 + rinf * k * x)


def _fully_mixed(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    rinf, kappa = th[..., 0], th[..., 1]
    return rinf * (x / (x + kappa))


def _bank(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    rinf, k, n = th[..., 0], th[..., 1], th[..., 2]
    return rinf * (1.0 - np.power(1.0 + k * x / n, -n))


FIRST_ORDER = ModelFamily(
    key="flot_first_order",
    name="Classical first-order (Garcia-Zuniga)",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("k", "1/min", 1e-3, 10.0, 0.8, "first-order rate constant"),
    ),
    fn=_fo,
    needs=(DataKind.TIMED_RECOVERY,),
    equation=r"R(t) = R_\infty\left(1 - e^{-kt}\right)",
    assumptions=(
        "Single rate constant for the whole floatable population; perfectly mixed batch "
        "cell; constant bubble flux and froth behavior over the test. Misfits the "
        "long-time tail when floatability is distributed."
    ),
    references=(
        Reference("Garcia-Zuniga 1935, Boletin Minero 47, 83-86", None),
        _POLAT_CHANDER,
    ),
    x_doc="x: flotation time t (min)",
)

KLIMPEL = ModelFamily(
    key="flot_klimpel",
    name="Klimpel rectangular rate distribution",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("k", "1/min", 1e-3, 20.0, 1.5, "upper bound of the rectangular rate distribution"),
    ),
    fn=_klimpel,
    needs=(DataKind.TIMED_RECOVERY,),
    equation=r"R(t) = R_\infty\left[1 - \frac{1}{kt}\left(1 - e^{-kt}\right)\right]",
    assumptions=(
        "Floatability uniformly distributed on [0, k]; first-order removal per class. "
        "The distribution shape is not identifiable from R(t) alone (Polat-Chander 2000)."
    ),
    references=(
        Reference("Klimpel 1980, Mineral Processing Plant Design 2nd ed., AIME, 907-934", None),
        _POLAT_CHANDER,
    ),
    x_doc="x: flotation time t (min)",
)

KELSALL = ModelFamily(
    key="flot_kelsall",
    name="Kelsall two-rate (fast/slow), R_inf = 1",
    process="flotation",
    params=(
        Param("phi", "fraction", 0.0, 1.0, 0.3, "slow-floating mass fraction"),
        Param("k_f", "1/min", 1e-2, 20.0, 2.0, "fast rate constant"),
        Param("k_s", "1/min", 1e-4, 0.5, 0.05, "slow rate constant"),
    ),
    fn=_kelsall,
    needs=(DataKind.TIMED_RECOVERY,),
    equation=r"R(t) = (1-\varphi)\left(1-e^{-k_f t}\right) + \varphi\left(1-e^{-k_s t}\right)",
    assumptions=(
        "Floatable population is a discrete mixture of exactly two kinetic classes that do "
        "not interconvert; ultimate recovery fixed at 1. Fragile below ~7 time points."
    ),
    references=(Reference("Kelsall 1961, Trans. Inst. Min. Metall. 70, 191-204", None),),
    x_doc="x: flotation time t (min)",
)

KELSALL_MOD = ModelFamily(
    key="flot_kelsall_mod",
    name="Modified Kelsall (free R_inf)",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.9, "ultimate recovery"),
        Param("phi", "fraction", 0.0, 1.0, 0.3, "slow-floating mass fraction"),
        Param("k_f", "1/min", 1e-2, 20.0, 2.0, "fast rate constant"),
        Param("k_s", "1/min", 1e-4, 0.5, 0.05, "slow rate constant"),
    ),
    fn=_kelsall_mod,
    needs=(DataKind.TIMED_RECOVERY,),
    equation=(
        r"R(t) = R_\infty\left[(1-\varphi)\left(1-e^{-k_f t}\right)"
        r" + \varphi\left(1-e^{-k_s t}\right)\right]"
    ),
    assumptions=(
        "Kelsall with a free ultimate recovery; 4 parameters against typically 5-8 batch "
        "points, over-parameterization is expected and is exactly what AICc must expose."
    ),
    references=(
        Reference("Kelsall 1961, Trans. Inst. Min. Metall. 70, 191-204", None),
        _POLAT_CHANDER,
    ),
    x_doc="x: flotation time t (min)",
)

GAMMA_DIST = ModelFamily(
    key="flot_gamma",
    name="Gamma rate-constant distribution",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("a", "-", 0.05, 10.0, 1.0, "gamma shape"),
        Param("b", "min", 1e-3, 60.0, 1.0, "gamma scale (mean rate = a*b)"),
    ),
    fn=_gamma,
    needs=(DataKind.TIMED_RECOVERY,),
    equation=r"R(t) = R_\infty\left[1 - (1 + bt)^{-a}\right]",
    assumptions=(
        "Floatability continuously gamma-distributed. Treat distributed-k families as "
        "interchangeable smoothers unless size-by-size data constrain the shape "
        "(Polat-Chander 2000): an honesty rule this bank inherits."
    ),
    references=(
        Reference("Imaizumi and Inoue 1963, 6th IMPC, Cannes", None),
        Reference("Loveday 1966, Trans. IMM 75, C219-C225", None),
        _POLAT_CHANDER,
    ),
    x_doc="x: flotation time t (min)",
)

SECOND_ORDER = ModelFamily(
    key="flot_second_order",
    name="Second-order kinetics",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("k", "1/(min*fraction)", 1e-3, 20.0, 1.0, "second-order rate constant"),
    ),
    fn=_second_order,
    needs=(DataKind.TIMED_RECOVERY,),
    equation=r"R(t) = \frac{R_\infty^2 k t}{1 + R_\infty k t}",
    assumptions=(
        "Removal rate proportional to the square of the remaining floatable fraction "
        "(particle-bubble encounter framing); one of the standard zoo members compared "
        "in the discrimination literature."
    ),
    references=(_BU_2017, _POLAT_CHANDER),
    x_doc="x: flotation time t (min)",
)

FULLY_MIXED = ModelFamily(
    key="flot_fully_mixed",
    name="Fully mixed reactor analogy",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("kappa", "min", 1e-3, 120.0, 2.0, "characteristic time"),
    ),
    fn=_fully_mixed,
    needs=(DataKind.TIMED_RECOVERY,),
    equation=r"R(t) = R_\infty\,\frac{t}{t + \kappa}",
    assumptions=(
        "Batch response shaped like a perfectly mixed continuous vessel (exponential "
        "rate distribution limit); heavy long-time tail."
    ),
    references=(_BU_2017, _POLAT_CHANDER),
    x_doc="x: flotation time t (min)",
)

BANK_MIXERS = ModelFamily(
    key="flot_bank_mixers",
    name="Bank of N perfect mixers (continuous)",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("k", "1/min", 1e-3, 10.0, 0.5, "collection rate constant"),
        Param("N", "-", 1.0, 12.0, 4.0, "effective number of mixers in series"),
    ),
    fn=_bank,
    needs=(DataKind.CONTINUOUS_RECOVERY,),
    equation=r"R = R_\infty\left[1 - \left(1 + \frac{k\,\tau}{N}\right)^{-N}\right]",
    assumptions=(
        "Each cell a perfect mixer; kinetics transferable from batch via a scale-up "
        "factor (industrial k typically 0.4-1.0 x batch k); RTD validated by tracer work "
        "(Yianatos and coworkers)."
    ),
    references=(
        _POLAT_CHANDER,
        Reference(
            "Yianatos et al. 2008, Minerals Engineering 21(12-14), 817-825",
            "10.1016/j.mineng.2007.12.012",
        ),
    ),
    x_doc="x: mean residence time tau (min)",
)

BATCH_FAMILIES: tuple[ModelFamily, ...] = (
    FIRST_ORDER,
    KLIMPEL,
    KELSALL,
    KELSALL_MOD,
    GAMMA_DIST,
    SECOND_ORDER,
    FULLY_MIXED,
)

ALL: tuple[ModelFamily, ...] = BATCH_FAMILIES + (BANK_MIXERS,)
