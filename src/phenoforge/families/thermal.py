"""Thermal plant output and electrical load family bank.

Two industrial observables that are phenomenological in exactly the same sense
as a flotation rate law: the ambient derating of a thermal power block, and the
load response of an electricity-consuming plant.

AMBIENT DERATING. A gas turbine breathes air, so its output follows the mass
flow it can ingest. Air density falls as ambient temperature rises, and the
competing descriptions of that fall are the families here: a linear derating
coefficient (the vendor curve convention), a quadratic correction, an
ideal-gas density law in which output tracks P_amb / T_abs, and a Carnot-shaped
ceiling in which output falls with the sink temperature ratio. They are
distinguishable only over a wide ambient range, which is why the case matters.

LOAD RESPONSE. A plant's electrical demand against its production rate splits
into a base load that runs whether or not the line is producing and a marginal
term that scales with output; the alternatives are a pure proportional law and
a power-law with economies of scale. These share their algebra with the utility
bank but are fitted per period rather than per year, so they are declared for
the same observable kind and can compete directly.

Primary sources:
- Kehlhofer, R., Hannemann, F., Stirnimann, F., Rukes, B., 2009. Combined-Cycle
  Gas and Steam Turbine Power Plants, 3rd ed., PennWell (ambient correction
  curves; density-driven output).
- Tufekci, P., 2014. Prediction of full load electrical power output of a base
  load operated combined cycle power plant using machine learning methods.
  International Journal of Electrical Power and Energy Systems 60, 126-140,
  DOI 10.1016/j.ijepes.2014.02.027 (the UCI combined cycle dataset and its
  ambient variables).
- Moran, M.J., Shapiro, H.N., Boettner, D.D., Bailey, M.B., 2018. Fundamentals
  of Engineering Thermodynamics, 9th ed., Wiley (Brayton and Rankine cycle
  temperature dependence; the Carnot ceiling).
- VDI 4661 and standard industrial practice for specific energy consumption
  with a base-load term.

x is the driving variable named per family (ambient temperature in Celsius, or
production rate in the plant's own units); y is electrical power or energy.
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_KEHLHOFER = Reference(
    "Kehlhofer, Hannemann, Stirnimann, Rukes 2009, Combined-Cycle Gas and Steam "
    "Turbine Power Plants, 3rd ed., PennWell",
    None,
)
_TUFEKCI = Reference(
    "Tufekci 2014, Int. J. Electr. Power Energy Syst. 60, 126-140",
    "10.1016/j.ijepes.2014.02.027",
)
_MORAN = Reference(
    "Moran, Shapiro, Boettner, Bailey 2018, Fundamentals of Engineering "
    "Thermodynamics, 9th ed., Wiley",
    None,
)

_XDOC_T = "x: ambient temperature (degrees Celsius)"


def _linear_derate(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    p0, a = th[..., 0], th[..., 1]
    return p0 - a * x


def _quadratic_derate(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    p0, a, b = th[..., 0], th[..., 1], th[..., 2]
    return p0 - a * x - b * x * x


def _density_law(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    # output proportional to inlet air density: rho = P / (R T_abs)
    c, p_amb = th[..., 0], th[..., 1]
    return c * p_amb / (x + 273.15)


def _carnot_ceiling(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    # P = P_ref * (1 - T_cold / T_hot) with T_cold the ambient sink
    p_ref, t_hot = th[..., 0], th[..., 1]
    return p_ref * (1.0 - (x + 273.15) / t_hot)


LINEAR_DERATE = ModelFamily(
    key="therm_linear_derate",
    name="Linear ambient derating (vendor curve)",
    process="thermal",
    params=(
        Param("P0", "MW", 100.0, 900.0, 500.0, "output at 0 degrees Celsius"),
        Param("a", "MW/K", 0.0, 20.0, 2.0, "derating coefficient"),
    ),
    fn=_linear_derate,
    needs=(DataKind.XY_RESPONSE,),
    equation=r"P = P_0 - a\,T_{amb}",
    assumptions=(
        "Constant derating slope over the operating range, the convention of "
        "vendor correction curves. Adequate over a narrow ambient band and the "
        "parsimony control that the physically motivated families must beat."
    ),
    references=(_KEHLHOFER, _TUFEKCI),
    x_doc=_XDOC_T,
)

QUADRATIC_DERATE = ModelFamily(
    key="therm_quadratic_derate",
    name="Quadratic ambient derating",
    process="thermal",
    params=(
        Param("P0", "MW", 100.0, 900.0, 500.0, "output at 0 degrees Celsius"),
        Param("a", "MW/K", -5.0, 20.0, 2.0, "linear derating coefficient"),
        Param("b", "MW/K2", -0.5, 0.5, 0.01, "curvature coefficient"),
    ),
    fn=_quadratic_derate,
    needs=(DataKind.XY_RESPONSE,),
    equation=r"P = P_0 - a\,T_{amb} - b\,T_{amb}^{2}",
    assumptions=(
        "Second-order correction capturing the steepening of derating at high "
        "ambient temperature (compressor and condenser limits). b = 0 recovers "
        "the linear family exactly, so the pair is a nested hypothesis test."
    ),
    references=(_KEHLHOFER,),
    x_doc=_XDOC_T,
)

DENSITY_LAW = ModelFamily(
    key="therm_density_law",
    name="Inlet air density law (ideal gas)",
    process="thermal",
    params=(
        Param("c", "MW*K/mbar", 1.0, 1e4, 150.0, "machine constant"),
        Param("P_amb", "mbar", 800.0, 1100.0, 1013.0, "effective ambient pressure"),
    ),
    fn=_density_law,
    needs=(DataKind.XY_RESPONSE,),
    equation=r"P = c\,\frac{P_{amb}}{T_{amb} + 273.15}",
    assumptions=(
        "Output limited by ingested air mass flow, which for an ideal gas scales "
        "as P / T_abs. The mechanistic alternative to a fitted slope: it predicts "
        "the curvature rather than fitting it, and its failure is informative."
    ),
    references=(_MORAN, _KEHLHOFER),
    x_doc=_XDOC_T,
)

CARNOT_CEILING = ModelFamily(
    key="therm_carnot_ceiling",
    name="Carnot-shaped sink-temperature ceiling",
    process="thermal",
    params=(
        Param("P_ref", "MW", 200.0, 3000.0, 900.0, "reference output scale"),
        Param("T_hot", "K", 700.0, 2000.0, 1200.0, "effective hot-side temperature"),
    ),
    fn=_carnot_ceiling,
    needs=(DataKind.XY_RESPONSE,),
    equation=r"P = P_{ref}\left(1 - \frac{T_{amb} + 273.15}{T_{hot}}\right)",
    assumptions=(
        "Output proportional to the Carnot efficiency with the ambient as the "
        "cold sink; a thermodynamic upper-bound shape rather than a machine "
        "curve. Over a narrow ambient band it is nearly linear, which is exactly "
        "the equifinality the ensemble should report instead of a false winner."
    ),
    references=(_MORAN,),
    x_doc=_XDOC_T,
)

ALL: tuple[ModelFamily, ...] = (
    LINEAR_DERATE,
    QUADRATIC_DERATE,
    DENSITY_LAW,
    CARNOT_CEILING,
)
