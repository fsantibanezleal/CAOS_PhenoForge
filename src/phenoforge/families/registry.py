"""Family registry: every family the bank ships, addressable by key.

The bank spans the mineral-processing unit operations cataloged in the Fragua
families dossier: flotation kinetics, comminution energy-size laws and batch
grinding population balances, thickening settling-curve signatures, leaching
conversion kinetics, and plant utility (water and energy) balances. Keys are
stable public API.
"""

from __future__ import annotations

from phenoforge.families import (
    comminution,
    flotation,
    flotation_continuous,
    grinding,
    leaching,
    thickening,
    utility,
)
from phenoforge.families.base import ModelFamily

_ALL: dict[str, ModelFamily] = {
    f.key: f
    for f in (
        *flotation.ALL,
        *flotation_continuous.ALL,
        *comminution.ALL,
        *grinding.ALL,
        *thickening.ALL,
        *leaching.ALL,
        *utility.ALL,
    )
}

PROCESSES: tuple[str, ...] = ("flotation", "comminution", "thickening", "leaching", "utility")


def list_families(process: str | None = None) -> tuple[ModelFamily, ...]:
    """All families, optionally filtered by process."""
    fams = tuple(_ALL.values())
    if process is None:
        return fams
    return tuple(f for f in fams if f.process == process)


def get_family(key: str) -> ModelFamily:
    try:
        return _ALL[key]
    except KeyError as exc:
        known = ", ".join(sorted(_ALL))
        raise KeyError(f"unknown family '{key}'; known: {known}") from exc
