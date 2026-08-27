"""Continuous (plant) flotation family bank: recovery versus residence time.

A batch test gives R(t) along one charge; a CONTINUOUS circuit gives R(tau),
where tau is the mean residence time set by cell volume over volumetric feed
rate. The distinction is not cosmetic: the same first-order collection rate
produces a different recovery curve depending on the residence time
distribution of the vessel, which is why a plant bank needs its own families.

General form combining a rate distribution f(k) with a residence time
distribution E(t):

    R = R_inf * int int (1 - exp(-k t)) E(t) f(k) dt dk

The families below are the standard closed forms of that integral for the RTDs
that mineral processing actually uses.

Primary sources (transcribed from the Fragua families dossier):
- Levenspiel, O., 1999. Chemical Reaction Engineering, 3rd ed., Wiley,
  chapters 11 to 14 (perfect mixer, tanks in series, plug flow, axial
  dispersion).
- Polat, M., Chander, S., 2000. Int. J. Miner. Process. 58, 145-166,
  DOI 10.1016/S0301-7516(99)00069-1 (the distributed-rate integral form).
- Yianatos, J., Bergh, L., et al., 2008. Minerals Engineering 21(12-14),
  817-825, DOI 10.1016/j.mineng.2007.12.012 (residence time distribution of
  industrial flotation cells measured by radioactive tracer; the basis for
  treating a bank as N perfect mixers and for the large/small tank in series
  refinement of a single large cell).
- Finch, J.A., Dobby, G.S., 1990. Column Flotation. Pergamon (two-zone
  collection and froth recovery, the origin of the apparent-rate framing).

Scale-up caveat carried in every docstring: an industrial rate constant is
commonly 0.4 to 1.0 times its batch counterpart, so k here is a PLANT rate and
is not directly comparable with a laboratory k. Where tau is a proxy (cell
volume unknown), k is identifiable only up to that constant; the product states
this per case rather than reporting a spurious absolute rate.
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_LEVENSPIEL = Reference(
    "Levenspiel 1999, Chemical Reaction Engineering 3rd ed., ch. 11-14", None
)
_POLAT = Reference(
    "Polat and Chander 2000, Int. J. Miner. Process. 58, 145-166",
    "10.1016/S0301-7516(99)00069-1",
)
_YIANATOS = Reference(
    "Yianatos et al. 2008, Minerals Engineering 21(12-14), 817-825",
    "10.1016/j.mineng.2007.12.012",
)
_FINCH = Reference("Finch and Dobby 1990, Column Flotation, Pergamon", None)

_XDOC = "x: mean residence time tau (min, or a proxy proportional to it)"


def _perfect_mixer(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    r_inf, k = th[..., 0], th[..., 1]
    kt = k * x
    return r_inf * kt / (1.0 + kt)


def _plug_flow(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    r_inf, k = th[..., 0], th[..., 1]
    return r_inf * (1.0 - np.exp(-k * x))


def _n_mixers(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    r_inf, k, n = th[..., 0], th[..., 1], th[..., 2]
    return r_inf * (1.0 - np.power(1.0 + k * x / n, -n))


def _gamma_rtd(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    r_inf, a, b = th[..., 0], th[..., 1], th[..., 2]
    return r_inf * (1.0 - np.power(1.0 + b * x, -a))


def _two_class(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    r_inf, phi, kf, ks = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    fast = kf * x / (1.0 + kf * x)
    slow = ks * x / (1.0 + ks * x)
    return r_inf * ((1.0 - phi) * fast + phi * slow)


PERFECT_MIXER = ModelFamily(
    key="flotc_perfect_mixer",
    name="Single perfectly mixed cell (continuous)",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("k", "1/min", 1e-4, 10.0, 0.3, "plant collection rate constant"),
    ),
    fn=_perfect_mixer,
    needs=(DataKind.CONTINUOUS_RECOVERY,),
    equation=r"R = R_\infty\,\frac{k\tau}{1 + k\tau}",
    assumptions=(
        "One perfectly mixed vessel at steady state with first-order collection; "
        "the exponential residence time distribution puts substantial mass at "
        "very short times, which is why a single large cell recovers less than a "
        "plug-flow vessel of the same mean residence time."
    ),
    references=(_LEVENSPIEL, _FINCH),
    x_doc=_XDOC,
)

PLUG_FLOW = ModelFamily(
    key="flotc_plug_flow",
    name="Plug flow (continuous)",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("k", "1/min", 1e-4, 10.0, 0.3, "plant collection rate constant"),
    ),
    fn=_plug_flow,
    needs=(DataKind.CONTINUOUS_RECOVERY,),
    equation=r"R = R_\infty\left(1 - e^{-k\tau}\right)",
    assumptions=(
        "No axial mixing: every element spends exactly tau in the vessel, so the "
        "continuous response has the same form as a batch test. The upper bound "
        "of what a first-order circuit can achieve at a given mean residence "
        "time, and therefore the optimistic control in the bank."
    ),
    references=(_LEVENSPIEL, _POLAT),
    x_doc=_XDOC,
)

N_MIXERS = ModelFamily(
    key="flotc_n_mixers",
    name="Bank of N perfect mixers in series",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("k", "1/min", 1e-4, 10.0, 0.3, "plant collection rate constant"),
        Param("N", "-", 1.0, 12.0, 4.0, "effective number of mixers"),
    ),
    fn=_n_mixers,
    needs=(DataKind.CONTINUOUS_RECOVERY,),
    equation=r"R = R_\infty\left[1 - \left(1 + \frac{k\tau}{N}\right)^{-N}\right]",
    assumptions=(
        "A bank behaves as N equal perfect mixers in series, validated by "
        "radioactive tracer work on industrial cells (Yianatos and coworkers). "
        "N = 1 recovers the single-mixer family and N to infinity recovers plug "
        "flow, so this family NESTS both bounds and the fitted N measures how "
        "far the circuit sits between them."
    ),
    references=(_YIANATOS, _LEVENSPIEL),
    x_doc=_XDOC,
)

GAMMA_RTD = ModelFamily(
    key="flotc_gamma_rtd",
    name="Gamma-distributed floatability (continuous)",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.85, "ultimate recovery"),
        Param("a", "-", 0.05, 10.0, 1.0, "gamma shape of the rate distribution"),
        Param("b", "min", 1e-3, 60.0, 0.5, "gamma scale"),
    ),
    fn=_gamma_rtd,
    needs=(DataKind.CONTINUOUS_RECOVERY,),
    equation=r"R = R_\infty\left[1 - (1 + b\tau)^{-a}\right]",
    assumptions=(
        "Heterogeneous floatability integrated over the residence time "
        "distribution; algebraically identical to the batch gamma form, which is "
        "itself the warning: recovery-versus-time data alone cannot separate a "
        "distribution of rates from a distribution of residence times "
        "(Polat and Chander 2000). The ensemble should show that as shared mass."
    ),
    references=(_POLAT, _LEVENSPIEL),
    x_doc=_XDOC,
)

TWO_CLASS = ModelFamily(
    key="flotc_two_class",
    name="Fast and slow floating classes (continuous)",
    process="flotation",
    params=(
        Param("R_inf", "fraction", 0.01, 1.0, 0.9, "ultimate recovery"),
        Param("phi", "fraction", 0.0, 1.0, 0.35, "slow-floating mass fraction"),
        Param("k_f", "1/min", 1e-3, 20.0, 1.2, "fast rate constant"),
        Param("k_s", "1/min", 1e-5, 1.0, 0.03, "slow rate constant"),
    ),
    fn=_two_class,
    needs=(DataKind.CONTINUOUS_RECOVERY,),
    equation=(
        r"R = R_\infty\left[(1-\varphi)\frac{k_f\tau}{1+k_f\tau} + "
        r"\varphi\frac{k_s\tau}{1+k_s\tau}\right]"
    ),
    assumptions=(
        "The continuous counterpart of the Kelsall split: two kinetic classes, "
        "each through a perfectly mixed vessel. Four parameters against a plant "
        "curve binned into a dozen operating points is close to the "
        "identifiability limit, which the information criteria must expose "
        "rather than the analyst assuming."
    ),
    references=(_FINCH, _POLAT),
    x_doc=_XDOC,
)

ALL: tuple[ModelFamily, ...] = (PERFECT_MIXER, PLUG_FLOW, N_MIXERS, GAMMA_RTD, TWO_CLASS)
