"""Process-dynamics family bank: step-response models of a unit operation.

When a disturbance or a setpoint moves, an industrial unit answers with a
transient, and the engineer's job is to name the lumped model behind it. That
choice is made in practice by looking at a reaction curve and picking one
structure: a single lag, a lag with transport delay, two lags in series, an
oscillatory pair, or a non-self-regulating integrator. The structures are not
nested and the data rarely separates them, which is exactly the situation the
rest of this package exists to handle.

x is time since the step (any consistent unit, hours here); y is the measured
variable in its own engineering unit. Every family is written so that y(0) = y0
and the gain K carries the sign of the response, so the same bank fits a rise
and a fall without reparameterization.

Primary sources, transcribed from the Fragua dynamics dossier:

- Ziegler, J.G., Nichols, N.B., 1942. Optimum settings for automatic
  controllers. Trans. ASME 64, 759-768. (the reaction-curve reading of a lag
  plus a dead time)
- Sundaresan, K.R., Krishnaswamy, P.R., 1978. Estimation of time delay time
  constant parameters in time, frequency, and Laplace domains. Can. J. Chem.
  Eng. 56(2), 257-262, DOI 10.1002/cjce.5450560215.
- Ogunnaike, B.A., Ray, W.H., 1994. Process Dynamics, Modeling, and Control.
  Oxford University Press.
- Marlin, T.E., 2000. Process Control: Designing Processes and Control Systems
  for Dynamic Performance, 2nd ed. McGraw-Hill.
- Astrom, K.J., Hagglund, T., 2006. Advanced PID Control. ISA.
- Seborg, D.E., Edgar, T.F., Mellichamp, D.A., Doyle, F.J., 2016. Process
  Dynamics and Control, 4th ed. Wiley.
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import DataKind, ModelFamily, Param, Reference

_XDOC = "x: time since the step (h)"
_EPS = 1.0e-9


def _y0(th: np.ndarray) -> np.ndarray:
    return th[..., 0]


def _first_order(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    y0, k, tau = th[..., 0], th[..., 1], th[..., 2]
    return y0 + k * (1.0 - np.exp(-x / np.maximum(tau, _EPS)))


def _fopdt(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    y0, k, tau, theta = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    shifted = np.maximum(x - theta, 0.0)
    resp = k * (1.0 - np.exp(-shifted / np.maximum(tau, _EPS)))
    return y0 + np.where(x >= theta, resp, 0.0)


def _second_order_overdamped(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Two unequal first-order lags in series (Seborg et al. 2016, s5.4).

    The repeated-pole limit tau1 -> tau2 is singular in the textbook closed
    form, so it is evaluated with the critically damped expression whenever the
    two lags are within a relative 1e-6 of each other.
    """
    y0, k, t1, t2 = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    t1 = np.maximum(t1, _EPS)
    t2 = np.maximum(t2, _EPS)
    gap = t1 - t2
    near = np.abs(gap) < 1.0e-6 * np.maximum(t1, t2)
    safe_gap = np.where(near, 1.0, gap)
    distinct = 1.0 - (t1 * np.exp(-x / t1) - t2 * np.exp(-x / t2)) / safe_gap
    tm = 0.5 * (t1 + t2)
    repeated = 1.0 - (1.0 + x / tm) * np.exp(-x / tm)
    return y0 + k * np.where(near, repeated, distinct)


def _critically_damped(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    y0, k, tau = th[..., 0], th[..., 1], th[..., 2]
    tau = np.maximum(tau, _EPS)
    return y0 + k * (1.0 - (1.0 + x / tau) * np.exp(-x / tau))


def _underdamped(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Oscillatory second order, the signature of a unit under feedback control.

    zeta is bounded strictly inside (0, 1) so the damped frequency is real.
    """
    y0, k, tau, zeta = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    tau = np.maximum(tau, _EPS)
    zeta = np.clip(zeta, 1.0e-4, 0.999)
    wd = np.sqrt(1.0 - zeta**2) / tau
    phi = np.arctan2(np.sqrt(1.0 - zeta**2), zeta)
    env = np.exp(-zeta * x / tau) / np.sqrt(1.0 - zeta**2)
    return y0 + k * (1.0 - env * np.sin(wd * x + phi))


def _integrating(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Non-self-regulating: an integrator behind one lag (a level, a hold-up).

    Written as the ramp minus its startup transient so that y(0) = y0 and the
    slope tends to K far from the step.
    """
    y0, k, tau = th[..., 0], th[..., 1], th[..., 2]
    tau = np.maximum(tau, _EPS)
    return y0 + k * (x - tau * (1.0 - np.exp(-x / tau)))


_SEBORG = Reference(
    "Seborg, Edgar, Mellichamp and Doyle 2016, Process Dynamics and Control, 4th ed.", None
)
_MARLIN = Reference("Marlin 2000, Process Control, 2nd ed., McGraw-Hill", None)
_OGUNNAIKE = Reference(
    "Ogunnaike and Ray 1994, Process Dynamics, Modeling, and Control, Oxford", None
)

# Baselines and gains are left wide because the bank is applied to variables in
# their own engineering units (degrees C, kPa, kscmh, percent), and the fitter
# rescales internally; the DYNAMIC parameters are what carry physical bounds.
_Y0 = Param("y_0", "unit", -1.0e4, 1.0e4, 0.0, "pre-step baseline of the measured variable")
_K = Param("K", "unit", -1.0e4, 1.0e4, 1.0, "steady-state gain (signed magnitude of the response)")

FIRST_ORDER = ModelFamily(
    key="dyn_first_order",
    name="First-order lag",
    process="dynamics",
    params=(_Y0, _K, Param("tau", "h", 1.0e-3, 50.0, 1.0, "time constant")),
    fn=_first_order,
    needs=(DataKind.STEP_RESPONSE,),
    equation=r"y(t) = y_0 + K\left(1 - e^{-t/\tau}\right)",
    assumptions=(
        "One dominant energy or material hold-up, perfectly mixed, no transport "
        "delay; the response starts moving at its maximum rate immediately after "
        "the step."
    ),
    references=(_SEBORG, _OGUNNAIKE),
    x_doc=_XDOC,
)

FOPDT = ModelFamily(
    key="dyn_fopdt",
    name="First order plus dead time",
    process="dynamics",
    params=(
        _Y0,
        _K,
        Param("tau", "h", 1.0e-3, 50.0, 1.0, "time constant"),
        Param("theta", "h", 0.0, 10.0, 0.1, "transport delay before any response"),
    ),
    fn=_fopdt,
    needs=(DataKind.STEP_RESPONSE,),
    equation=(
        r"y(t) = y_0 + K\left(1 - e^{-(t-\theta)/\tau}\right)\,"
        r"\mathbf{1}\!\left[t \ge \theta\right]"
    ),
    assumptions=(
        "The industrial default: one lag plus pure transport delay, the model "
        "behind reaction-curve tuning rules. The delay absorbs both true "
        "transport and the higher-order lags the fit does not resolve, which is "
        "why it competes so closely with the second-order families."
    ),
    references=(
        Reference("Ziegler and Nichols 1942, Trans. ASME 64, 759-768", None),
        Reference(
            "Sundaresan and Krishnaswamy 1978, Can. J. Chem. Eng. 56(2), 257-262",
            "10.1002/cjce.5450560215",
        ),
    ),
    x_doc=_XDOC,
)

SECOND_ORDER = ModelFamily(
    key="dyn_second_order",
    name="Two lags in series (overdamped)",
    process="dynamics",
    params=(
        _Y0,
        _K,
        Param("tau_1", "h", 1.0e-3, 50.0, 2.0, "slow lag"),
        Param("tau_2", "h", 1.0e-3, 50.0, 0.5, "fast lag"),
    ),
    fn=_second_order_overdamped,
    needs=(DataKind.STEP_RESPONSE,),
    equation=(
        r"y(t) = y_0 + K\left(1 - \frac{\tau_1 e^{-t/\tau_1} - "
        r"\tau_2 e^{-t/\tau_2}}{\tau_1 - \tau_2}\right)"
    ),
    assumptions=(
        "Two hold-ups in series (a jacket and a vessel, a sensor behind a "
        "reactor); the response leaves the baseline with zero slope, which is "
        "the feature that separates it from a single lag when the data is dense "
        "enough near the step."
    ),
    references=(_SEBORG, _MARLIN),
    x_doc=_XDOC,
)

CRITICALLY_DAMPED = ModelFamily(
    key="dyn_critically_damped",
    name="Repeated lag (critically damped)",
    process="dynamics",
    params=(_Y0, _K, Param("tau", "h", 1.0e-3, 50.0, 1.0, "repeated time constant")),
    fn=_critically_damped,
    needs=(DataKind.STEP_RESPONSE,),
    equation=r"y(t) = y_0 + K\left(1 - \left(1 + \frac{t}{\tau}\right)e^{-t/\tau}\right)",
    assumptions=(
        "Two identical lags: the one-parameter-cheaper sibling of the "
        "overdamped pair, and the reason an information criterion often prefers "
        "it on short records."
    ),
    references=(_SEBORG,),
    x_doc=_XDOC,
)

UNDERDAMPED = ModelFamily(
    key="dyn_underdamped",
    name="Underdamped second order",
    process="dynamics",
    params=(
        _Y0,
        _K,
        Param("tau", "h", 1.0e-3, 50.0, 1.0, "natural period scale"),
        Param("zeta", "-", 1.0e-3, 0.999, 0.5, "damping ratio"),
    ),
    fn=_underdamped,
    needs=(DataKind.STEP_RESPONSE,),
    equation=(
        r"y(t) = y_0 + K\left(1 - \frac{e^{-\zeta t/\tau}}{\sqrt{1-\zeta^2}}"
        r"\sin\!\left(\frac{\sqrt{1-\zeta^2}}{\tau}t + \phi\right)\right),\quad "
        r"\phi = \arctan\frac{\sqrt{1-\zeta^2}}{\zeta}"
    ),
    assumptions=(
        "Overshoot and ringing. On a plant record this is usually the signature "
        "of the CONTROLLER rather than the process, so a win for this family is "
        "read as closed-loop identification, not as an open-loop mechanism."
    ),
    references=(_SEBORG, Reference("Astrom and Hagglund 2006, Advanced PID Control, ISA", None)),
    x_doc=_XDOC,
)

INTEGRATING = ModelFamily(
    key="dyn_integrating",
    name="Integrating plus lag (non-self-regulating)",
    process="dynamics",
    params=(
        _Y0,
        Param("K", "unit/h", -1.0e4, 1.0e4, 1.0, "asymptotic ramp rate"),
        Param("tau", "h", 1.0e-3, 50.0, 1.0, "lag before the ramp establishes"),
    ),
    fn=_integrating,
    needs=(DataKind.STEP_RESPONSE,),
    equation=r"y(t) = y_0 + K\left(t - \tau\left(1 - e^{-t/\tau}\right)\right)",
    assumptions=(
        "No steady state: a level, an inventory or an unbalanced hold-up that "
        "ramps until something else acts. Distinguishing it from a very slow "
        "self-regulating lag is impossible on a record shorter than a few time "
        "constants, and reporting that ambiguity is more useful than resolving "
        "it by assertion."
    ),
    references=(_MARLIN, _OGUNNAIKE),
    x_doc=_XDOC,
)

ALL: tuple[ModelFamily, ...] = (
    FIRST_ORDER,
    FOPDT,
    SECOND_ORDER,
    CRITICALLY_DAMPED,
    UNDERDAMPED,
    INTEGRATING,
)
