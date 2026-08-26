"""Comminution energy-size law family bank.

The four classical energy-size relationships as competing families over the same
(F80, P80) -> specific energy data. Transcribed from the Fragua families dossier;
primary sources:

- von Rittinger, P.R., 1867. Lehrbuch der Aufbereitungskunde. Ernst and Korn, Berlin.
- Kick, F., 1885. Das Gesetz der proportionalen Widerstande. Arthur Felix, Leipzig.
- Bond, F.C., 1952. The third theory of comminution. Trans. AIME 193, 484-494.
- Hukki, R.T., 1962. Trans. AIME 223, 403-408 (the size-dependent-exponent
  reconciliation the Morrell form operationalizes).
- Morrell, S., 2004. Int. J. Miner. Process. 74(1-4), 133-141,
  DOI 10.1016/j.minpro.2003.10.002.

x is (n, 2) with columns [F80_um, P80_um]; y is specific energy W (kWh/t).
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_XDOC = "x: (n, 2) columns [F80 (um), P80 (um)]"


def _bond(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    wi = th[..., 0]
    f80, p80 = x[..., 0], x[..., 1]
    return 10.0 * wi * (1.0 / np.sqrt(p80) - 1.0 / np.sqrt(f80))


def _rittinger(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    c = th[..., 0]
    f80, p80 = x[..., 0], x[..., 1]
    return c * (1.0 / p80 - 1.0 / f80)


def _kick(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    c = th[..., 0]
    f80, p80 = x[..., 0], x[..., 1]
    return c * np.log(f80 / p80)


def _morrell(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    mi = th[..., 0]
    f80, p80 = x[..., 0], x[..., 1]

    def f(s: np.ndarray) -> np.ndarray:
        return -(0.295 + s / 1.0e6)

    return mi * 4.0 * (np.power(p80, f(p80)) - np.power(f80, f(f80)))


BOND = ModelFamily(
    key="comm_bond",
    name="Bond third theory",
    process="comminution",
    params=(Param("W_i", "kWh/t", 4.0, 35.0, 14.0, "Bond work index"),),
    fn=_bond,
    needs=(DataKind.SIZE_ENERGY,),
    equation=r"W = 10\,W_i\left(\frac{1}{\sqrt{P_{80}}} - \frac{1}{\sqrt{F_{80}}}\right)",
    assumptions=(
        "Energy proportional to new crack tip length (exponent -1/2); valid roughly "
        "25 mm to 100 um (rod/ball milling); degrades outside that range."
    ),
    references=(Reference("Bond 1952, Trans. AIME 193, 484-494", None),),
    x_doc=_XDOC,
)

RITTINGER = ModelFamily(
    key="comm_rittinger",
    name="Rittinger (new surface area)",
    process="comminution",
    params=(Param("C_R", "kWh*um/t", 1.0, 1.0e6, 1.0e4, "Rittinger coefficient"),),
    fn=_rittinger,
    needs=(DataKind.SIZE_ENERGY,),
    equation=r"W = C_R\left(\frac{1}{P_{80}} - \frac{1}{F_{80}}\right)",
    assumptions="Energy proportional to new surface area; the fine-grinding regime (< ~100 um).",
    references=(Reference("von Rittinger 1867, Lehrbuch der Aufbereitungskunde", None),),
    x_doc=_XDOC,
)

KICK = ModelFamily(
    key="comm_kick",
    name="Kick (volume/strain energy)",
    process="comminution",
    params=(Param("C_K", "kWh/t", 0.01, 100.0, 2.0, "Kick coefficient"),),
    fn=_kick,
    needs=(DataKind.SIZE_ENERGY,),
    equation=r"W = C_K \ln\frac{F_{80}}{P_{80}}",
    assumptions="Energy proportional to reduction ratio; the coarse-crushing regime (> ~10 mm).",
    references=(Reference("Kick 1885, Das Gesetz der proportionalen Widerstande", None),),
    x_doc=_XDOC,
)

MORRELL_MI = ModelFamily(
    key="comm_morrell_mi",
    name="Morrell Mi (size-dependent exponent)",
    process="comminution",
    params=(Param("M_i", "kWh/t", 4.0, 40.0, 18.0, "Morrell work index"),),
    fn=_morrell,
    needs=(DataKind.SIZE_ENERGY,),
    equation=(
        r"W = M_i\,4\left(x_2^{f(x_2)} - x_1^{f(x_1)}\right),\quad "
        r"f(x) = -\left(0.295 + \frac{x}{10^6}\right)"
    ),
    assumptions=(
        "Continuous interpolation between Kick-like and Rittinger-like regimes (the Hukki "
        "reconciliation); sizes in micrometers; the modern power-based alternative to Bond."
    ),
    references=(
        Reference(
            "Morrell 2004, Int. J. Miner. Process. 74(1-4), 133-141",
            "10.1016/j.minpro.2003.10.002",
        ),
        Reference("Hukki 1962, Trans. AIME 223, 403-408", None),
    ),
    x_doc=_XDOC,
)

ALL: tuple[ModelFamily, ...] = (BOND, RITTINGER, KICK, MORRELL_MI)
