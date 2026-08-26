from phenoforge import route
from phenoforge.families.base import DataKind


def test_timed_recovery_routes_to_batch_flotation_only():
    fams = route((DataKind.TIMED_RECOVERY,))
    keys = {f.key for f in fams}
    assert "flot_first_order" in keys
    assert "flot_gamma" in keys
    assert "flot_bank_mixers" not in keys  # needs CONTINUOUS_RECOVERY
    assert not any(k.startswith("comm_") for k in keys)


def test_size_energy_routes_to_comminution():
    fams = route((DataKind.SIZE_ENERGY,))
    keys = {f.key for f in fams}
    assert keys == {"comm_bond", "comm_rittinger", "comm_kick", "comm_morrell_mi"}


def test_process_filter():
    fams = route((DataKind.TIMED_RECOVERY, DataKind.SIZE_ENERGY), process="comminution")
    assert all(f.process == "comminution" for f in fams)
