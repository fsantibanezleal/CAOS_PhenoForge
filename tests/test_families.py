import numpy as np
import pytest

from phenoforge import get_family, list_families
from phenoforge.families import comminution, flotation

T = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])


def test_registry_counts_and_lookup():
    from phenoforge.families import grinding, leaching, thickening, utility

    expected = (
        len(flotation.ALL) + len(comminution.ALL) + len(grinding.ALL)
        + len(thickening.ALL) + len(leaching.ALL) + len(utility.ALL)
    )
    assert len(list_families()) == expected
    assert len(list_families("flotation")) == 8
    # comminution holds the energy-size laws AND the batch grinding PBM families
    assert len(list_families("comminution")) == len(comminution.ALL) + len(grinding.ALL)
    fam = get_family("flot_first_order")
    assert fam.name.startswith("Classical first-order")
    with pytest.raises(KeyError):
        get_family("nope")


@pytest.mark.parametrize("fam", flotation.BATCH_FAMILIES)
def test_batch_families_start_at_zero_and_are_monotone(fam):
    theta = fam.inits
    r = fam.predict(T, theta)
    assert r.shape == T.shape
    assert abs(r[0]) < 1e-12
    assert np.all(np.diff(r) >= -1e-12)
    assert np.all(r <= 1.0 + 1e-9)


def test_klimpel_small_t_continuity():
    fam = flotation.KLIMPEL
    theta = np.array([0.9, 2.0])
    t = np.array([0.0, 1e-10, 1e-6, 1e-3])
    r = fam.predict(t, theta)
    assert np.all(np.isfinite(r))
    assert np.all(np.diff(r) >= 0.0)


def test_first_order_known_value():
    fam = flotation.FIRST_ORDER
    r = fam.predict(np.array([1.0]), np.array([0.8, 1.0]))
    assert r[0] == pytest.approx(0.8 * (1.0 - np.exp(-1.0)))


def test_kelsall_reduces_to_first_order_when_phi_zero():
    fam = flotation.KELSALL
    r = fam.predict(T, np.array([0.0, 1.2, 0.05]))
    ref = 1.0 - np.exp(-1.2 * T)
    assert np.allclose(r, ref)


def test_bond_hand_value():
    # W = 10 * 14 * (1/sqrt(100) - 1/sqrt(10000)) = 140 * (0.1 - 0.01) = 12.6 kWh/t
    x = np.array([[10000.0, 100.0]])
    w = comminution.BOND.predict(x, np.array([14.0]))
    assert w[0] == pytest.approx(12.6)


def test_theta_length_checked():
    with pytest.raises(ValueError):
        flotation.FIRST_ORDER.predict(T, np.array([0.9]))
