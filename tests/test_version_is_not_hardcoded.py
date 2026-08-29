"""The package's self-reported version must equal its distribution metadata.

A hardcoded literal in __init__.py drifted through two releases: 0.5.0 and 0.5.1
both reported "0.04.002". That is not cosmetic. The consuming product stamps
this string into EVERY baked manifest as the engine that produced the result, so
a whole matrix of artifacts would have recorded an engine version that never
ran, and the reproducibility claim the artifacts exist to support would be false.

The fix is derivation rather than discipline: __version__ reads the installed
distribution metadata, whose single source is pyproject.toml. These tests pin
that it stays derived.
"""

from __future__ import annotations

import re
from importlib.metadata import version as dist_version
from pathlib import Path

import phenoforge


def test_version_matches_the_installed_distribution() -> None:
    assert phenoforge.__version__ == dist_version("phenoforge")


def test_version_is_not_a_hardcoded_literal_in_the_source() -> None:
    """Guard the mechanism, not just today's value: a future edit that pins the
    string again would pass the test above while reintroducing the drift."""
    src = Path(phenoforge.__file__).read_text(encoding="utf-8")
    literal = re.search(r'^__version__\s*=\s*["\']', src, flags=re.M)
    assert literal is None, (
        "__version__ is assigned a string literal in __init__.py; it must be derived "
        "from importlib.metadata so it cannot drift from pyproject.toml"
    )


def test_version_looks_like_a_release() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", phenoforge.__version__), phenoforge.__version__
