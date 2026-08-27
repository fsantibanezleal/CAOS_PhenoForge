"""Batch grinding population-balance families (top-size disappearance kinetics).

All families map t (min) -> mass fraction REMAINING in the top size interval,
w1(t), the directly observed quantity of a batch grind test at each sieve time.
The full multi-size population balance is a case generator in the product; these
are the top-size signatures that discriminate the mechanisms from one series.

Primary sources (transcribed from the Fragua families dossier):
- Austin, L.G., Klimpel, R.R., Luckie, P.T., 1984. Process Engineering of Size
  Reduction: Ball Milling. SME-AIME (first-order breakage, selection and
  breakage functions, and the abnormal-breakage rollover).
- Herbst, J.A., Fuerstenau, D.W., 1980. Scale-up procedure for continuous
  grinding mill design using population balance models. Int. J. Miner. Process.
  7(1), 1-31. DOI 10.1016/0301-7516(80)90034-4 (energy-specific selection
  function S^E, mill-size invariant).
- Whiten, W.J., 1974. A matrix theory of comminution machines. Chem. Eng. Sci.
  29(2), 589-599 (perfect mixing; the r/d ratio identifiable from feed/product).
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_AUSTIN = Reference(
    "Austin, Klimpel, Luckie 1984, Process Engineering of Size Reduction: Ball Milling",
    None,
)
_HF = Reference(
    "Herbst and Fuerstenau 1980, Int. J. Miner. Process. 7(1), 1-31",
    "10.1016/0301-7516(80)90034-4",
)
_WHITEN = Reference("Whiten 1974, Chem. Eng. Sci. 29(2), 589-599", None)

_XDOC = "x: grinding time t (min)"


def _first_order(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    w0, s1 = th[..., 0], th[..., 1]
    return w0 * np.exp(-s1 * x)


def _non_first_order(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    w0, s1, alpha = th[..., 0], th[..., 1], th[..., 2]
    return w0 * np.exp(-np.power(s1 * x, alpha))


def _energy_specific(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    # S_i = S_i^E * (P/H): with constant specific power the top size decays
    # first-order in SPECIFIC ENERGY, so time enters through P/H.
    w0, se, p_over_h = th[..., 0], th[..., 1], th[..., 2]
    return w0 * np.exp(-se * p_over_h * x)


def _slowing_plateau(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    # Perfect-mixing/discharge-limited behaviour: a fraction w_r resists
    # breakage over the test window (the r/d ratio saturating), leaving a
    # plateau the pure first-order form cannot express.
    w0, s1, w_r = th[..., 0], th[..., 1], th[..., 2]
    return w_r + (w0 - w_r) * np.exp(-s1 * x)


AUSTIN_FIRST_ORDER = ModelFamily(
    key="grind_austin_first_order",
    name="Austin first-order breakage (top size)",
    process="comminution",
    params=(
        Param("w0", "fraction", 0.2, 1.0, 1.0, "initial top-size mass fraction"),
        Param("S1", "1/min", 1e-3, 5.0, 0.3, "selection (specific breakage rate)"),
    ),
    fn=_first_order,
    needs=(DataKind.PSD_TIME,),
    equation=r"\frac{dm_1}{dt} = -S_1 m_1 \;\Rightarrow\; w_1(t) = w_1(0)e^{-S_1 t}",
    assumptions=(
        "First-order breakage per size class; environment-independent parameters "
        "within the fitted regime; batch mill. The top-size class has no "
        "appearance term, so its kinetics isolate the selection function."
    ),
    references=(_AUSTIN,),
    x_doc=_XDOC,
)

AUSTIN_NON_FIRST_ORDER = ModelFamily(
    key="grind_austin_rollover",
    name="Austin non-first-order (abnormal breakage rollover)",
    process="comminution",
    params=(
        Param("w0", "fraction", 0.2, 1.0, 1.0, "initial top-size mass fraction"),
        Param("S1", "1/min", 1e-3, 5.0, 0.3, "selection rate scale"),
        Param("alpha", "-", 0.3, 2.5, 1.0, "kinetic order exponent"),
    ),
    fn=_non_first_order,
    needs=(DataKind.PSD_TIME,),
    equation=r"w_1(t) = w_1(0)\,e^{-\left(S_1 t\right)^{\alpha}}",
    assumptions=(
        "Departure from first-order kinetics observed for very coarse particles "
        "and overfilled mills (Austin's abnormal-breakage region): alpha < 1 "
        "slows with time, alpha > 1 accelerates. alpha = 1 recovers the "
        "first-order family exactly, so the pair is a nested test."
    ),
    references=(_AUSTIN,),
    x_doc=_XDOC,
)

HERBST_FUERSTENAU = ModelFamily(
    key="grind_herbst_fuerstenau",
    name="Herbst-Fuerstenau energy-specific selection",
    process="comminution",
    params=(
        Param("w0", "fraction", 0.2, 1.0, 1.0, "initial top-size mass fraction"),
        Param("SE", "t/kWh", 1e-3, 20.0, 0.5, "energy-specific selection function"),
        Param("P_over_H", "kW/t", 0.5, 60.0, 8.0, "mill power per unit holdup"),
    ),
    fn=_energy_specific,
    needs=(DataKind.PSD_TIME,),
    equation=r"S_1 = S_1^{E}\,\frac{P}{H} \;\Rightarrow\; w_1(t) = w_1(0)e^{-S_1^{E}(P/H)t}",
    assumptions=(
        "Breakage kinetics governed by specific energy absorption rate rather "
        "than mill dimension, so a lab-fitted S^E scales to plant through the "
        "plant power/holdup ratio. The dominant PBM scale-up route in American "
        "practice. On a single batch series S^E and P/H are only identifiable "
        "jointly; the product reports the identifiable combination."
    ),
    references=(_HF, _AUSTIN),
    x_doc=_XDOC,
)

WHITEN_PLATEAU = ModelFamily(
    key="grind_whiten_plateau",
    name="Whiten perfect-mixing residual (breakage-resistant fraction)",
    process="comminution",
    params=(
        Param("w0", "fraction", 0.2, 1.0, 1.0, "initial top-size mass fraction"),
        Param("S1", "1/min", 1e-3, 5.0, 0.4, "breakage rate of the breakable part"),
        Param("w_res", "fraction", 0.0, 0.6, 0.05, "breakage-resistant residual"),
    ),
    fn=_slowing_plateau,
    needs=(DataKind.PSD_TIME,),
    equation=r"w_1(t) = w_{res} + \left(w_1(0) - w_{res}\right)e^{-S_1 t}",
    assumptions=(
        "Perfect-mixing mill contents with a discharge-limited or intrinsically "
        "competent fraction that does not break over the test window (the r/d "
        "ratio saturating in Whiten's formulation). w_res = 0 recovers the "
        "first-order family, so this is the second nested test in the bank."
    ),
    references=(_WHITEN, _AUSTIN),
    x_doc=_XDOC,
)

ALL: tuple[ModelFamily, ...] = (
    AUSTIN_FIRST_ORDER,
    AUSTIN_NON_FIRST_ORDER,
    HERBST_FUERSTENAU,
    WHITEN_PLATEAU,
)
