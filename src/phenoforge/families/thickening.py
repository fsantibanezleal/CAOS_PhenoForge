"""Thickening and sedimentation family bank (batch settling curves).

SCOPE, stated honestly. The thickening literature's models are constitutive
theories (Kynch kinematic waves, Buscall-White compressional rheology,
Usher-Scales steady-state integration, Burger-Concha degenerate parabolic PDE).
What a BATCH SETTLING TEST observes is one scalar trajectory: the mud-line
(interface) height h(t). The families below are the interface-height SIGNATURES
those theories imply, parameterized so they are identifiable from a settling
curve alone. They are NOT the full PDE solvers: a family here is the observable
consequence of its theory in a batch cylinder, and the docstring of each names
exactly which consequence. The full Burger-Concha simulator is a case generator
in the product repo, not a fittable family.

Primary sources (transcribed from the Fragua families dossier):
- Kynch, G.J., 1952. A theory of sedimentation. Trans. Faraday Soc. 48, 166-176.
  DOI 10.1039/TF9524800166.
- Richardson, J.F., Zaki, W.N., 1954. Sedimentation and fluidisation Part I.
  Trans. Inst. Chem. Eng. 32, 35-53.
- Coe, H.S., Clevenger, G.H., 1916. Methods for determining the capacities of
  slime-settling tanks. Trans. AIME 55, 356-384.
- Talmage, W.P., Fitch, E.B., 1955. Determining thickener unit areas. Ind. Eng.
  Chem. 47(1), 38-41. DOI 10.1021/ie50541a022.
- Buscall, R., White, L.R., 1987. The consolidation of concentrated suspensions.
  J. Chem. Soc. Faraday Trans. 1 83(3), 873-891. DOI 10.1039/f19878300873.
- Usher, S.P., Scales, P.J., 2005. Steady state thickener modelling from the
  compressive yield stress and hindered settling function. Chem. Eng. J.
  111(2-3), 253-261. DOI 10.1016/j.cej.2005.02.015.
- Burger, R., Concha, F., 1998. Mathematical model and numerical simulation of
  the settling of flocculated suspensions. Int. J. Multiphase Flow 24(6),
  1005-1023. DOI 10.1016/S0301-9322(98)00026-3.

All families map t (min) -> interface height h (m). Heights are absolute, so a
series must ship its own initial height; h0 is a fitted parameter bounded around
laboratory cylinder scales (0.05 to 1.5 m).
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_KYNCH = Reference("Kynch 1952, Trans. Faraday Soc. 48, 166-176", "10.1039/TF9524800166")
_RZ = Reference("Richardson and Zaki 1954, Trans. Inst. Chem. Eng. 32, 35-53", None)
_CC = Reference("Coe and Clevenger 1916, Trans. AIME 55, 356-384", None)
_TF = Reference(
    "Talmage and Fitch 1955, Ind. Eng. Chem. 47(1), 38-41", "10.1021/ie50541a022"
)
_BW = Reference(
    "Buscall and White 1987, J. Chem. Soc. Faraday Trans. 1 83(3), 873-891",
    "10.1039/f19878300873",
)
_US = Reference(
    "Usher and Scales 2005, Chem. Eng. J. 111(2-3), 253-261", "10.1016/j.cej.2005.02.015"
)
_BC = Reference(
    "Burger and Concha 1998, Int. J. Multiphase Flow 24(6), 1005-1023",
    "10.1016/S0301-9322(98)00026-3",
)

_XDOC = "x: settling time t (min)"


def _kynch_ideal(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    h0, v0, h_inf = th[..., 0], th[..., 1], th[..., 2]
    return np.maximum(h0 - v0 * x, h_inf)


def _rz_decel(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    h0, v0, h_inf, n = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    # Mass conservation in a batch cylinder gives phi(t) h(t) = phi0 h0, so the
    # local concentration below the interface rises as the mud line falls and
    # the Richardson-Zaki velocity decays with it. Writing u = (h - h_inf) /
    # (h0 - h_inf) as the normalized free-settling gap, the resulting decay
    # integrates to a power-law approach with exponent set by n (the RZ index).
    gap = np.maximum(h0 - h_inf, 1e-9)
    tau = gap / np.maximum(v0, 1e-12)
    return h_inf + gap * np.power(1.0 + (n - 1.0) * x / tau, -1.0 / np.maximum(n - 1.0, 1e-6))


def _cc_two_zone(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    h0, v0, t_c, h_inf, tau = th[..., 0], th[..., 1], th[..., 2], th[..., 3], th[..., 4]
    h_c = np.maximum(h0 - v0 * t_c, h_inf)
    free = h0 - v0 * x
    comp = h_inf + (h_c - h_inf) * np.exp(-(x - t_c) / np.maximum(tau, 1e-9))
    return np.where(x <= t_c, np.maximum(free, h_inf), comp)


def _talmage_fitch(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    # Same two-zone construction as Coe-Clevenger but with the compression
    # point tied to the free-settling line by the tangent condition used in the
    # Talmage-Fitch unit-area method: the compression zone is entered where the
    # curve's tangent meets the underflow height, giving t_c = (h0 - h_u) / v0.
    h0, v0, h_u, tau = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    t_c = np.maximum((h0 - h_u) / np.maximum(v0, 1e-12), 1e-9)
    free = h0 - v0 * x
    comp = h_u + (h_u * 0.0 + (h0 - v0 * t_c) - h_u) * np.exp(-(x - t_c) / np.maximum(tau, 1e-9))
    return np.where(x <= t_c, np.maximum(free, h_u), np.maximum(comp, h_u))


def _buscall_white(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    h0, h_inf, tau, m = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    # Network consolidation above the gel point: the excess height relaxes as a
    # power law whose exponent m carries the compressive yield stress index
    # P_y ~ phi^m (Buscall-White); m -> large recovers a near-exponential.
    return h_inf + (h0 - h_inf) * np.power(1.0 + x / np.maximum(tau, 1e-9), -m)


def _usher_scales(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    h0, v0, h_inf, tau = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    # Hindered settling and network compression acting in series: the mud line
    # follows the smaller of the two rates at every instant, which is what the
    # Usher-Scales integration of R(phi) and P_y(phi) predicts for a batch test.
    free = h0 - v0 * x
    comp = h_inf + (h0 - h_inf) * np.exp(-x / np.maximum(tau, 1e-9))
    return np.maximum(np.minimum(free, comp), h_inf)


def _burger_concha(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    h0, v0, h_inf, t_c, p = th[..., 0], th[..., 1], th[..., 2], th[..., 3], th[..., 4]
    # The hyperbolic-to-parabolic transition of the degenerate PDE: linear
    # kinematic descent while sigma_e = 0, then a diffusive (parabolic) tail in
    # which the excess height decays as a stretched exponential of exponent p.
    h_c = np.maximum(h0 - v0 * t_c, h_inf)
    free = h0 - v0 * x
    dt = np.maximum(x - t_c, 0.0)
    scale = np.maximum((h_c - h_inf) / np.maximum(v0, 1e-12), 1e-9)
    comp = h_inf + (h_c - h_inf) * np.exp(-np.power(dt / scale, p))
    return np.where(x <= t_c, np.maximum(free, h_inf), comp)


KYNCH_IDEAL = ModelFamily(
    key="sett_kynch_ideal",
    name="Kynch ideal (kinematic, incompressible)",
    process="thickening",
    params=(
        Param("h0", "m", 0.05, 1.5, 0.4, "initial interface height"),
        Param("v0", "m/min", 1e-5, 0.5, 0.01, "hindered settling velocity"),
        Param("h_inf", "m", 0.005, 1.0, 0.08, "final sediment height"),
    ),
    fn=_kynch_ideal,
    needs=(DataKind.SETTLING_CURVE,),
    equation=r"h(t) = \max\left(h_0 - v_0 t,\; h_\infty\right)",
    assumptions=(
        "No compressive stress transmission (incompressible sediment); settling "
        "velocity depends on local concentration only; 1D monodisperse "
        "equivalent suspension. Fails in the compression zone of flocculated "
        "tailings, which is exactly what the compressional families below add."
    ),
    references=(_KYNCH, _RZ),
    x_doc=_XDOC,
)

RICHARDSON_ZAKI = ModelFamily(
    key="sett_richardson_zaki",
    name="Richardson-Zaki hindered settling (decelerating)",
    process="thickening",
    params=(
        Param("h0", "m", 0.05, 1.5, 0.4, "initial interface height"),
        Param("v0", "m/min", 1e-5, 0.5, 0.01, "initial hindered settling velocity"),
        Param("h_inf", "m", 0.005, 1.0, 0.08, "final sediment height"),
        Param("n", "-", 1.5, 8.0, 4.65, "Richardson-Zaki index (4.65 at low Re)"),
    ),
    fn=_rz_decel,
    needs=(DataKind.SETTLING_CURVE,),
    equation=(
        r"v_s(\phi) = v_0 (1-\phi)^n,\quad "
        r"h(t) = h_\infty + (h_0-h_\infty)\left[1 + (n-1)\frac{t}{\tau}\right]^{-\frac{1}{n-1}}"
    ),
    assumptions=(
        "Hindered settling with the concentration below the mud line rising by "
        "batch mass conservation; no network yield stress. The RZ index n is "
        "fitted rather than fixed at 4.65 so the family can express non-Stokes "
        "regimes, and the fitted value is reported."
    ),
    references=(_RZ, _KYNCH),
    x_doc=_XDOC,
)

COE_CLEVENGER = ModelFamily(
    key="sett_coe_clevenger",
    name="Coe-Clevenger two-zone (free settling then compression)",
    process="thickening",
    params=(
        Param("h0", "m", 0.05, 1.5, 0.4, "initial interface height"),
        Param("v0", "m/min", 1e-5, 0.5, 0.01, "free-settling velocity"),
        Param("t_c", "min", 0.1, 500.0, 20.0, "compression point time"),
        Param("h_inf", "m", 0.005, 1.0, 0.08, "final sediment height"),
        Param("tau", "min", 0.1, 2000.0, 60.0, "compression relaxation time"),
    ),
    fn=_cc_two_zone,
    needs=(DataKind.SETTLING_CURVE,),
    equation=(
        r"h(t) = \begin{cases} h_0 - v_0 t & t \le t_c \\ "
        r"h_\infty + (h_c - h_\infty)e^{-(t-t_c)/\tau} & t > t_c \end{cases}"
    ),
    assumptions=(
        "Each concentration settles at the rate measured in its own batch test; "
        "free settling until an explicit compression point, then relaxation. "
        "Systematically undersizes thickeners for compressible flocculated feeds "
        "(the documented weakness of the method)."
    ),
    references=(_CC, _KYNCH),
    x_doc=_XDOC,
)

TALMAGE_FITCH = ModelFamily(
    key="sett_talmage_fitch",
    name="Talmage-Fitch tangent construction",
    process="thickening",
    params=(
        Param("h0", "m", 0.05, 1.5, 0.4, "initial interface height"),
        Param("v0", "m/min", 1e-5, 0.5, 0.01, "free-settling velocity"),
        Param("h_u", "m", 0.005, 1.0, 0.08, "underflow-equivalent height"),
        Param("tau", "min", 0.1, 2000.0, 60.0, "compression relaxation time"),
    ),
    fn=_talmage_fitch,
    needs=(DataKind.SETTLING_CURVE,),
    equation=(
        r"t_c = \frac{h_0 - h_u}{v_0},\quad "
        r"h(t) = h_u + (h_0 - v_0 t_c - h_u)\,e^{-(t-t_c)/\tau}\ \ (t>t_c)"
    ),
    assumptions=(
        "Kynch behaviour holds through the test and one curve is representative; "
        "the compression point is tied to the underflow height by the tangent "
        "construction rather than fitted freely. Known to be sensitive to that "
        "construction, which is why it is kept as a distinct family instead of "
        "being merged with Coe-Clevenger."
    ),
    references=(_TF, _CC),
    x_doc=_XDOC,
)

BUSCALL_WHITE = ModelFamily(
    key="sett_buscall_white",
    name="Buscall-White compressional relaxation",
    process="thickening",
    params=(
        Param("h0", "m", 0.05, 1.5, 0.4, "initial interface height"),
        Param("h_inf", "m", 0.005, 1.0, 0.08, "equilibrium sediment height"),
        Param("tau", "min", 0.1, 2000.0, 30.0, "consolidation time scale"),
        Param("m", "-", 0.2, 8.0, 1.5, "compressive yield stress index"),
    ),
    fn=_buscall_white,
    needs=(DataKind.SETTLING_CURVE,),
    equation=r"h(t) = h_\infty + (h_0 - h_\infty)\left(1 + t/\tau\right)^{-m}",
    assumptions=(
        "Irreversible consolidation of a flocculated network above its gel "
        "point; the power-law exponent m carries the compressive yield stress "
        "index P_y(phi) ~ ((phi/phi_g)^m - 1). Applies to polymer-flocculated "
        "mineral tailings; not to free-settling rigid suspensions."
    ),
    references=(_BW,),
    x_doc=_XDOC,
)

USHER_SCALES = ModelFamily(
    key="sett_usher_scales",
    name="Usher-Scales series (hindered settling and compression)",
    process="thickening",
    params=(
        Param("h0", "m", 0.05, 1.5, 0.4, "initial interface height"),
        Param("v0", "m/min", 1e-5, 0.5, 0.01, "hindered settling velocity"),
        Param("h_inf", "m", 0.005, 1.0, 0.08, "equilibrium sediment height"),
        Param("tau", "min", 0.1, 2000.0, 60.0, "compression time scale"),
    ),
    fn=_usher_scales,
    needs=(DataKind.SETTLING_CURVE,),
    equation=(
        r"h(t) = \max\left[\min\left(h_0 - v_0 t,\; "
        r"h_\infty + (h_0-h_\infty)e^{-t/\tau}\right),\; h_\infty\right]"
    ),
    assumptions=(
        "Hindered settling function R(phi) and compressive yield stress P_y(phi) "
        "acting in series, the rate-limiting one governing at each instant; 1D, "
        "no rake shear enhancement. The batch-test consequence of the "
        "Usher-Scales steady-state integration."
    ),
    references=(_US, _BW),
    x_doc=_XDOC,
)

BURGER_CONCHA = ModelFamily(
    key="sett_burger_concha",
    name="Burger-Concha hyperbolic-parabolic transition",
    process="thickening",
    params=(
        Param("h0", "m", 0.05, 1.5, 0.4, "initial interface height"),
        Param("v0", "m/min", 1e-5, 0.5, 0.01, "kinematic (hyperbolic) velocity"),
        Param("h_inf", "m", 0.005, 1.0, 0.08, "equilibrium sediment height"),
        Param("t_c", "min", 0.1, 500.0, 20.0, "transition time (phi reaches phi_c)"),
        Param("p", "-", 0.3, 2.0, 0.7, "parabolic-tail stretching exponent"),
    ),
    fn=_burger_concha,
    needs=(DataKind.SETTLING_CURVE,),
    equation=(
        r"\partial_t\phi + \partial_z\!\left(q\phi + f_{bk}(\phi)\right) = "
        r"\partial_z\!\left(\frac{f_{bk}(\phi)\sigma_e'(\phi)}{\Delta\rho\, g\, \phi}"
        r"\partial_z\phi\right);\ \ h(t>t_c) = h_\infty + (h_c-h_\infty)"
        r"e^{-\left((t-t_c)/\tau\right)^{p}}"
    ),
    assumptions=(
        "The interface signature of the degenerate parabolic-hyperbolic model: "
        "hyperbolic (kinematic) descent while phi < phi_c with sigma_e = 0, then "
        "a diffusive tail once compression switches on. The full entropy-solution "
        "PDE is a case GENERATOR in the product, not this fittable curve."
    ),
    references=(_BC, _KYNCH, _BW),
    x_doc=_XDOC,
)

ALL: tuple[ModelFamily, ...] = (
    KYNCH_IDEAL,
    RICHARDSON_ZAKI,
    COE_CLEVENGER,
    TALMAGE_FITCH,
    BUSCALL_WHITE,
    USHER_SCALES,
    BURGER_CONCHA,
)
