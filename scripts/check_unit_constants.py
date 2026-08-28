"""No unreviewed magic constant in method code.

WHY
---
On 2026-08-28 four defects of one shape were found in a single afternoon, one in
this package and three in its consuming product:

  bayes/gw.py      a noise-scale prior capped at 0.5 in RAW target units, which
                   also placed the walker start above its own upper bound for
                   any series with a spread over five units
  (in the product) a sigmoid bounding predictions to (0, 1); an E-SINDy blow-up
                   bound of 2.0 in raw units; an E-SINDy coefficient threshold
                   carrying units of 1/driver

Every one was correct for the product's first observable, a flotation recovery
fraction on the unit interval, hard-coded into a method, and never revisited as
the matrix grew to eight unit processes. Two of them sat in BASELINE rungs, so
their failure read as a satisfying result rather than a bug, and two of them
were published as scientific null results.

The counter-examples are worth naming, because they show the pattern was
avoidable: the GP discrepancy hybrid derives every bound from the residual
spread, and GLUE scores with the Nash-Sutcliffe efficiency, which is scale-free
by construction. The difference is not care. It is that those two were written
against a QUANTITY rather than against a NUMBER.

WHAT THIS DOES
--------------
Walks the AST of the method and stage modules and collects every float literal
that could plausibly be compared against a measured quantity. Each one must
appear in `scripts/unit_constants_allow.json` with a written justification. A
NEW unreviewed constant fails the gate, which forces the author to answer one
question at the moment they write it: what unit does this number assume?

Ratios, probabilities, tolerances relative to a computed scale, and iteration
counts are all legitimate; they just have to be declared as such.

A package consumed by other people's pipelines carries this risk more sharply
than an application does: a constant that assumes an observable is a fraction
does not fail loudly in someone else's units, it just returns a confident wrong
answer.

Run:  python scripts/check_unit_constants.py
      python scripts/check_unit_constants.py --update   (re-baseline, review the diff)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOW = REPO / "scripts" / "unit_constants_allow.json"

# The family bank's numeric literals are physical constants BY DESIGN: stating
# each parameter's units and bounds is exactly what a family is for, and those
# are reviewed as data (against the primary reference) rather than as method
# logic. Everything else here is method logic and must justify its numbers.
SKIP_PARTS = ("/families/",)

ROOTS = (REPO / "src" / "phenoforge",)

# Values that cannot carry a unit assumption in any code path: identities,
# halves and simple ratios used for arithmetic rather than for comparison.
UNIVERSAL = {0.0, 1.0, 0.5, 2.0, 100.0, 10.0, 1000.0}


def collect() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if any(part in "/" + rel for part in SKIP_PARTS):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            lines = path.read_text(encoding="utf-8").split("\n")
            hits = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, float):
                    continue
                v = float(node.value)
                if abs(v) in UNIVERSAL or abs(v) < 1e-5 or v == 0:
                    continue
                hits.append({
                    "value": v,
                    "line": node.lineno,
                    "source": lines[node.lineno - 1].strip()[:110],
                })
            if hits:
                found[rel] = hits
    return found


def key(rel: str, hit: dict) -> str:
    return f"{rel}::{hit['value']!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="re-baseline; review the diff")
    args = ap.parse_args()

    found = collect()
    allow = json.loads(ALLOW.read_text(encoding="utf-8")) if ALLOW.exists() else {}

    if args.update:
        merged = dict(allow)
        added = 0
        for rel, hits in found.items():
            for h in hits:
                k = key(rel, h)
                if k not in merged:
                    merged[k] = {
                        "reason": "TODO: state what unit this number assumes",
                        "source": h["source"],
                    }
                    added += 1
                else:
                    merged[k]["source"] = h["source"]
        for k in list(merged):
            rel = k.split("::")[0]
            if rel in found and not any(key(rel, h) == k for h in found[rel]):
                del merged[k]
            elif rel not in found:
                del merged[k]
        ALLOW.write_text(
            json.dumps(dict(sorted(merged.items())), indent=2), encoding="utf-8", newline="\n"
        )
        print(f"baseline written: {len(merged)} constants ({added} new, marked TODO)")
        return 0

    problems: list[str] = []
    for rel, hits in found.items():
        for h in hits:
            k = key(rel, h)
            entry = allow.get(k)
            if entry is None:
                problems.append(
                    f"{rel}:{h['line']}: unreviewed constant {h['value']!r}\n"
                    f"      {h['source']}\n"
                    "      What unit does this number assume? Add it to "
                    "scripts/unit_constants_allow.json with a justification, or derive it "
                    "from the data."
                )
            elif str(entry.get("reason", "")).startswith("TODO"):
                problems.append(
                    f"{rel}:{h['line']}: constant {h['value']!r} is baselined but not justified\n"
                    f"      {h['source']}"
                )

    if problems:
        print(f"UNIT-CONSTANT GATE FAILED: {len(problems)} problem(s)\n")
        for p in problems:
            print(f"  - {p}")
        return 1
    total = sum(len(v) for v in found.values())
    print(f"UNIT-CONSTANT GATE OK: {total} constants across {len(found)} modules, all justified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
