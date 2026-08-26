"""Family registry: every family the bank currently ships, addressable by key.

The bank grows vertically with the Fragua build (flotation + comminution energy laws
first; thickening, leaching, utility and PBM families land with their cases). Keys are
stable public API.
"""

from __future__ import annotations

from phenoforge.families import comminution, flotation
from phenoforge.families.base import ModelFamily

_ALL: dict[str, ModelFamily] = {
    f.key: f for f in (*flotation.ALL, *comminution.ALL)
}


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
