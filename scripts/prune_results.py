#!/usr/bin/env python
"""Drop named operating points from a sweep's results.json so `--resume` redoes them.

Used when an input a subset of points depends on is regenerated -- for example
when the sensitivity weights are re-measured, which invalidates the allocated
points but not the uniform baselines that never used them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--drop", nargs="+", required=True,
                    help="regexes matched against the point name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    p = pathlib.Path(args.results)
    rows = json.load(open(p))
    pats = [re.compile(x) for x in args.drop]
    keep = [r for r in rows if not any(x.match(r["name"]) for x in pats)]
    dropped = [r["name"] for r in rows if r not in keep]
    print(f"{len(rows)} points -> keeping {len(keep)}, dropping {len(dropped)}")
    for n in dropped:
        print(f"  drop {n}")
    if not args.dry_run:
        backup = p.with_suffix(".json.bak")
        backup.write_text(json.dumps(rows, indent=2))
        p.write_text(json.dumps(keep, indent=2))
        print(f"wrote {p} (backup at {backup})")


if __name__ == "__main__":
    main()
