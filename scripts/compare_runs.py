"""Compare finished runs across characters.

One subject proves nothing about a pipeline. This collects the `qa.json` from
several runs and puts them in one table, so the question "does this work on
anything other than the character it was built on" has an answer with numbers
attached rather than an opinion.

  python scripts/compare_runs.py out/05_keyframes out/B_quadruped/05_keyframes ...

Writes docs/report/assets/generality.json and a comparison figure.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

CEILING = 0.03


def load_run(run_dir: str) -> dict:
    path = os.path.join(run_dir, "qa.json")
    if not os.path.exists(path):
        raise SystemExit("No qa.json in " + run_dir + " -- run `pipeline qa` there first.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def row_for(name: str, run_dir: str) -> dict:
    doc = load_run(run_dir)
    report = doc["report"]
    views = report["views"]
    return {
        "character": name,
        "run": os.path.relpath(run_dir, REPO).replace(os.sep, "/"),
        "drift_mean": report["overall"]["drift_mean"],
        "drift_max": report["overall"]["drift_max"],
        "per_view": {v: views[v]["drift_max"] for v in views},
        "worst_view": max(views, key=lambda v: views[v]["drift_max"]),
        "pass": report["gate"]["pass"],
    }


def main(argv) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        print("usage: compare_runs.py <name> <run_dir> [<name> <run_dir> ...]")
        return 2

    rows = [row_for(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]

    views = sorted({v for r in rows for v in r["per_view"]})
    head = ("character".ljust(16) + "mean".rjust(8) + "max".rjust(8)
            + "".join(("max " + v).rjust(11) for v in views)
            + "worst".rjust(8) + "gate".rjust(7))
    print(head)
    print("-" * len(head))
    for r in rows:
        print(r["character"].ljust(16)
              + format(r["drift_mean"], ".4f").rjust(8)
              + format(r["drift_max"], ".4f").rjust(8)
              + "".join(format(r["per_view"].get(v, float("nan")), ".4f").rjust(11) for v in views)
              + r["worst_view"].rjust(8)
              + ("PASS" if r["pass"] else "FAIL").rjust(7))

    out = os.path.join(REPO, "docs", "report", "assets", "generality.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"ceiling": CEILING, "runs": rows}, fh, indent=2)
    print("\nwrote " + os.path.relpath(out, REPO))

    failures = [r["character"] for r in rows if not r["pass"]]
    if failures:
        print("gate failed for: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
