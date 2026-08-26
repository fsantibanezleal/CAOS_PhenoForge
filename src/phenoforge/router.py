"""Dataset-to-family routing.

A dataset advertises the DataKind tags it provides; route() returns the families the
bank can calibrate with them. This encodes the calibration-data contracts cataloged
in the Fragua families dossier (each family declares what data it needs) and is
itself a product feature (the web app's router view).
"""

from __future__ import annotations

from phenoforge.families.base import DataKind, ModelFamily
from phenoforge.families.registry import list_families


def route(
    provides: tuple[DataKind, ...] | list[DataKind],
    process: str | None = None,
) -> tuple[ModelFamily, ...]:
    """Families whose EVERY declared need is satisfied by `provides`."""
    have = set(provides)
    fams = list_families(process)
    return tuple(f for f in fams if set(f.needs) <= have)
