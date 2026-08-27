"""Leaching and hydrometallurgy family bank (conversion vs time).

All families map t (h) -> fractional conversion / recovery X in [0, 1].

Primary sources (transcribed from the Fragua families dossier):
- Yagi, S., Kunii, D., 1955. 5th Symposium (International) on Combustion,
  231-244 (shrinking-core regimes).
- Levenspiel, O., 1999. Chemical Reaction Engineering, 3rd ed., Wiley, ch. 25
  (the canonical exposition of the three control regimes).
- Dixon, D.G., Hendrix, J.L., 1993. A general model for leaching of one or more
  solid reactants from porous ore particles. Metall. Trans. B 24(1), 157-169.
  DOI 10.1007/BF02657882; and the heap companion, Metall. Trans. B 24(6),
  1087-1102, DOI 10.1007/BF02661000.
- Mellado, M.E., Cisternas, L.A., Galvez, E.D., 2009. An analytical model
  approach to heap leaching. Hydrometallurgy 95(1-2), 33-38.
  DOI 10.1016/j.hydromet.2008.04.009; Mellado et al. 2011, Comput. Chem. Eng.
  35(2), 220-225 (scalable analytical models).

The product-layer-diffusion regime has no closed-form X(t); it is inverted from
its implicit form by a monotone bisection, which is exact to machine tolerance
and keeps the family a pure function of (t, theta).
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_YK = Reference("Yagi and Kunii 1955, 5th Symp. (Int.) on Combustion, 231-244", None)
_LEV = Reference("Levenspiel 1999, Chemical Reaction Engineering 3rd ed., ch. 25", None)
_DH = Reference(
    "Dixon and Hendrix 1993, Metall. Trans. B 24(1), 157-169", "10.1007/BF02657882"
)
_DH_HEAP = Reference(
    "Dixon and Hendrix 1993, Metall. Trans. B 24(6), 1087-1102", "10.1007/BF02661000"
)
_MC = Reference(
    "Mellado, Cisternas, Galvez 2009, Hydrometallurgy 95(1-2), 33-38",
    "10.1016/j.hydromet.2008.04.009",
)
_MC2 = Reference("Mellado et al. 2011, Comput. Chem. Eng. 35(2), 220-225", None)

_XDOC = "x: leaching time t (h)"


def _film(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    x_inf, tau = th[..., 0], th[..., 1]
    return x_inf * np.clip(x / np.maximum(tau, 1e-12), 0.0, 1.0)


def _reaction(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    x_inf, tau = th[..., 0], th[..., 1]
    u = np.clip(x / np.maximum(tau, 1e-12), 0.0, 1.0)
    return x_inf * (1.0 - np.power(1.0 - u, 3.0))


def _product_layer(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    """t/tau = 1 - 3(1-X)^{2/3} + 2(1-X), inverted by bisection on X in [0,1]."""
    x_inf, tau = th[..., 0], th[..., 1]
    u = np.clip(x / np.maximum(tau, 1e-12), 0.0, 1.0)

    def g(conv: np.ndarray) -> np.ndarray:
        r = 1.0 - conv
        return 1.0 - 3.0 * np.power(np.maximum(r, 0.0), 2.0 / 3.0) + 2.0 * r

    lo = np.zeros_like(u)
    hi = np.ones_like(u)
    for _ in range(60):  # 60 bisections: interval 2^-60, exact to double precision
        mid = 0.5 * (lo + hi)
        too_small = g(mid) < u
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    return x_inf * 0.5 * (lo + hi)


def _dixon_hendrix(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Two-scale (intraparticle diffusion inside advective irrigation) response.

    The Dixon-Hendrix column model in its quasi-exponential regime: a series
    combination of a grain-scale first-order step and a particle-scale diffusive
    step, which produces the characteristic sigmoidal early lag that the pure
    shrinking-core forms cannot express.
    """
    x_inf, k_g, k_p = th[..., 0], th[..., 1], th[..., 2]
    kg = np.maximum(k_g, 1e-12)
    kp = np.maximum(k_p, 1e-12)
    same = np.abs(kg - kp) < 1e-9
    safe = np.where(same, kg * 1.000001, kp)
    two_step = 1.0 - (safe * np.exp(-kg * x) - kg * np.exp(-safe * x)) / (safe - kg)
    degenerate = 1.0 - (1.0 + kg * x) * np.exp(-kg * x)
    return x_inf * np.where(same, degenerate, two_step)


def _mellado(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    """R(t) = R_inf [1 - Z exp(-k theta(t))], theta = t - omega (transport delay)."""
    r_inf, z, k, omega = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    theta = np.maximum(x - omega, 0.0)
    return r_inf * np.clip(1.0 - z * np.exp(-k * theta), 0.0, 1.0)


def _heap_first_order(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    r_inf, k = th[..., 0], th[..., 1]
    return r_inf * (1.0 - np.exp(-k * x))


SC_FILM = ModelFamily(
    key="leach_sc_film",
    name="Shrinking core, film-diffusion control",
    process="leaching",
    params=(
        Param("X_inf", "fraction", 0.05, 1.0, 0.9, "ultimate conversion"),
        Param("tau", "h", 0.05, 5000.0, 24.0, "time for complete conversion"),
    ),
    fn=_film,
    needs=(DataKind.CONVERSION_TIME,),
    equation=r"\frac{t}{\tau} = X",
    assumptions=(
        "External mass transfer through the fluid film is rate limiting; sharp "
        "reaction front; isothermal; single reactant. Linear until exhaustion, "
        "so it is the control that any curvature in the data must beat."
    ),
    references=(_YK, _LEV),
    x_doc=_XDOC,
)

SC_REACTION = ModelFamily(
    key="leach_sc_reaction",
    name="Shrinking core, surface-reaction control",
    process="leaching",
    params=(
        Param("X_inf", "fraction", 0.05, 1.0, 0.9, "ultimate conversion"),
        Param("tau", "h", 0.05, 5000.0, 48.0, "time for complete conversion"),
    ),
    fn=_reaction,
    needs=(DataKind.CONVERSION_TIME,),
    equation=r"\frac{t}{\tau} = 1 - (1-X)^{1/3}",
    assumptions=(
        "Chemical reaction at the receding core surface is rate limiting; tau "
        "scales linearly with particle radius (the size-dependence test that "
        "distinguishes this regime from product-layer control)."
    ),
    references=(_YK, _LEV),
    x_doc=_XDOC,
)

SC_PRODUCT_LAYER = ModelFamily(
    key="leach_sc_product",
    name="Shrinking core, product-layer diffusion control",
    process="leaching",
    params=(
        Param("X_inf", "fraction", 0.05, 1.0, 0.9, "ultimate conversion"),
        Param("tau", "h", 0.05, 5000.0, 96.0, "time for complete conversion"),
    ),
    fn=_product_layer,
    needs=(DataKind.CONVERSION_TIME,),
    equation=r"\frac{t}{\tau} = 1 - 3(1-X)^{2/3} + 2(1-X)",
    assumptions=(
        "Diffusion through the porous product (ash) layer is rate limiting; tau "
        "scales with the SQUARE of particle radius. No closed form for X(t): "
        "inverted numerically by monotone bisection to double precision."
    ),
    references=(_YK, _LEV),
    x_doc=_XDOC,
)

DIXON_HENDRIX = ModelFamily(
    key="leach_dixon_hendrix",
    name="Dixon-Hendrix two-scale column response",
    process="leaching",
    params=(
        Param("X_inf", "fraction", 0.05, 1.0, 0.85, "ultimate recovery"),
        Param("k_grain", "1/h", 1e-4, 5.0, 0.05, "grain-scale rate constant"),
        Param("k_part", "1/h", 1e-4, 5.0, 0.15, "particle diffusion rate constant"),
    ),
    fn=_dixon_hendrix,
    needs=(DataKind.CONVERSION_TIME,),
    equation=(
        r"X(t) = X_\infty\left[1 - \frac{k_p e^{-k_g t} - k_g e^{-k_p t}}{k_p - k_g}\right]"
    ),
    assumptions=(
        "Advective irrigation coupled to intraparticle diffusion and grain-scale "
        "reaction, in the regime where the full PDE system is quasi-exponential; "
        "1D plug flow, uniform irrigation, isothermal, no channelling. The full "
        "coupled PDE is a case generator, not this fittable form."
    ),
    references=(_DH, _DH_HEAP),
    x_doc=_XDOC,
)

MELLADO_CISTERNAS = ModelFamily(
    key="leach_mellado",
    name="Mellado-Cisternas analytical heap model",
    process="leaching",
    params=(
        Param("R_inf", "fraction", 0.05, 1.0, 0.8, "ultimate recovery"),
        Param("Z", "-", 0.2, 1.0, 1.0, "initial-deficit coefficient"),
        Param("k_theta", "1/h", 1e-4, 5.0, 0.03, "generalized-time rate constant"),
        Param("omega", "h", 0.0, 500.0, 12.0, "solution transport delay"),
    ),
    fn=_mellado,
    needs=(DataKind.CONVERSION_TIME,),
    equation=r"R(t) = R_\infty\left[1 - Z\,e^{-k_\theta \theta(t)}\right],\ \ \theta = t - \omega",
    assumptions=(
        "First-order lumping of the Dixon-Hendrix mechanisms at two scales, with "
        "a generalized time that absorbs irrigation rate and heap height through "
        "the delay omega. Being analytical it supports optimization and scaling "
        "directly; validity inherited from the quasi-exponential regime."
    ),
    references=(_MC, _MC2),
    x_doc=_XDOC,
)

HEAP_FIRST_ORDER = ModelFamily(
    key="leach_first_order",
    name="Lumped first-order heap recovery",
    process="leaching",
    params=(
        Param("R_inf", "fraction", 0.05, 1.0, 0.8, "ultimate recovery"),
        Param("k", "1/h", 1e-4, 5.0, 0.04, "lumped rate constant"),
    ),
    fn=_heap_first_order,
    needs=(DataKind.CONVERSION_TIME,),
    equation=r"R(t) = R_\infty\left(1 - e^{-kt}\right)",
    assumptions=(
        "Single lumped rate for the whole heap; the industrial reporting default "
        "and the parsimony control every mechanistic leaching family must beat "
        "by a measured margin."
    ),
    references=(_MC, _LEV),
    x_doc=_XDOC,
)

ALL: tuple[ModelFamily, ...] = (
    SC_FILM,
    SC_REACTION,
    SC_PRODUCT_LAYER,
    DIXON_HENDRIX,
    MELLADO_CISTERNAS,
    HEAP_FIRST_ORDER,
)
