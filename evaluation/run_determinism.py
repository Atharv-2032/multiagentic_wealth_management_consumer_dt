"""
Determinism experiment.

Replaces the selection stage with a language model, runs it repeatedly on
identical input, and measures what changes.

Run from the project root, after run_batch has produced the advisor results:

    python -m evaluation.run_determinism
    python -m evaluation.run_determinism --runs 5
    python -m evaluation.run_determinism --runs 3 C22-TIER0 C24-EXCLUSIVITY

Outputs
-------
    out/evaluation/determinism.json   per-persona detail, every run recorded
    out/evaluation/determinism.csv    one row per persona

What this tests
----------------
The claim that selection under stated constraints is a computation, and that
moving it into a component whose output varies would forfeit the property the
architecture exists to provide.

The model is given exactly what the deterministic selector reads -- the
proposals, the conflicts, the budgets -- and the tier ordering stated in full.
It is run at temperature 0, which is the best case rather than the worst.

Three measurements, and the third is the one that matters:

  self-consistency   does it choose the same set on identical input
  agreement          does it choose what the rules would have chosen
  violations         does it respect the budgets and conflicts it was handed

Inconsistency is a reproducibility problem: the same client could receive
different advice on two runs, which for regulated investment advice is a
compliance matter rather than an inconvenience. A violation is worse. It means
the selector chose something the firm cannot carry out, which is precisely the
outcome the deterministic version makes unrepresentable rather than unlikely.

The deterministic selector's variance is zero by construction and is not
measured. There is nothing to sample.

On generality
--------------
This is run against the advisor's selection stage because that is where a model
looks most plausible -- it is a judgement call over competing options, and a
reader might reasonably ask why it is not one. The same experiment could be run
against the retention feasibility engine or the promote eligibility filter, and
there is no reason to expect a different result; those stages are arithmetic
over thresholds, where a model has even less to contribute and the same sampling
behaviour to offer.
"""

import argparse
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from evaluation.llm_selector import select_with_model, violations

RESULT_DIR = os.path.join("out", "evaluation")
DEFAULT_RUNS = 5


def load_result(name):
    with open(os.path.join(RESULT_DIR, f"{name}.json")) as f:
        return json.load(f)


def deterministic_selection(result):
    return [
        f"{p['track']}/{p['action_id']}"
        for p in result["selection"]["selected"]
    ]


def compare(name, runs):
    """
    Run the model selector `runs` times on one persona's proposal set.

    Every run sees identical input. The deterministic selection is read from
    the advisor's own output rather than recomputed, so the comparison is
    against what the system actually did.
    """
    result = load_result(name)
    proposals = result["filtered"]["passed"]
    conflicts = result["conflicts"]
    budget = result["filtered"]["budget"]

    rules_choice = deterministic_selection(result)

    if not proposals:
        return {
            "persona": name,
            "n_proposals": 0,
            "runs": 0,
            "rules_selection": rules_choice,
            "model_selections": [],
            "distinct_selections": 0,
            "self_consistent": True,
            "first_action_stable": True,
            "agreed_with_rules": 0,
            "runs_with_violations": 0,
            "violation_types": [],
            "skipped": "no proposals to select between",
        }

    selections, all_violations, parse_problems = [], [], []
    for _ in range(runs):
        out = select_with_model(proposals, conflicts, budget)
        selections.append(out["selected"])
        v = violations(out["selected"], proposals, conflicts, budget)
        all_violations.append(v)
        if out["unknown_keys"] or out["unaccounted_proposals"]:
            parse_problems.append({
                "unknown": out["unknown_keys"],
                "unaccounted": out["unaccounted_proposals"],
            })

    # Set-level, because two orderings of the same set are the same decision
    # about what happens to the client.
    as_sets = [frozenset(s) for s in selections]
    distinct = len(set(as_sets))

    firsts = [s[0] if s else None for s in selections]
    agreed = sum(1 for s in as_sets if s == frozenset(rules_choice))
    with_violations = sum(1 for v in all_violations if v)

    return {
        "persona": name,
        "n_proposals": len(proposals),
        "runs": runs,
        "rules_selection": rules_choice,
        "model_selections": selections,
        "distinct_selections": distinct,
        "self_consistent": distinct == 1,
        "first_action_stable": len(set(firsts)) == 1,
        "agreed_with_rules": agreed,
        "runs_with_violations": with_violations,
        "violation_types": sorted({v["type"] for vs in all_violations for v in vs}),
        "violation_detail": [v for vs in all_violations for v in vs],
        "parse_problems": parse_problems,
    }


CSV_FIELDS = [
    "persona", "n_proposals", "runs", "rules_selection",
    "distinct_selections", "self_consistent", "first_action_stable",
    "agreed_with_rules", "runs_with_violations", "violation_types",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("personas", nargs="*")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"selection runs per persona (default {DEFAULT_RUNS})")
    args = parser.parse_args()

    available = sorted(
        f[:-5] for f in os.listdir(RESULT_DIR)
        if f.endswith(".json") and not f.startswith(("coverage", "determinism"))
    )
    names = args.personas or available

    missing = [n for n in names if n not in available]
    if missing:
        print(f"no advisor result for {missing}; run evaluation.run_batch first",
              file=sys.stderr)
        sys.exit(2)

    rows = []
    for name in names:
        row = compare(name, args.runs)
        rows.append(row)

        if row.get("skipped"):
            print(f"  {name:18s} skipped: {row['skipped']}")
            continue

        flag = " " if row["self_consistent"] else "!"
        vflag = "V" if row["runs_with_violations"] else " "
        print(f"{flag}{vflag} {name:18s} distinct={row['distinct_selections']}/"
              f"{row['runs']}  agreed={row['agreed_with_rules']}/{row['runs']}  "
              f"violations={row['runs_with_violations']}/{row['runs']} "
              f"{row['violation_types'] or ''}")

    with open(os.path.join(RESULT_DIR, "determinism.json"), "w") as f:
        json.dump(rows, f, indent=2)

    with open(os.path.join(RESULT_DIR, "determinism.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                k: ("; ".join(map(str, v)) if isinstance(v, list) else v)
                for k, v in r.items() if k in CSV_FIELDS
            })

    # --- summary ----------------------------------------------------------
    scored = [r for r in rows if not r.get("skipped")]
    inconsistent = [r for r in scored if not r["self_consistent"]]
    unstable_first = [r for r in scored if not r["first_action_stable"]]
    violating = [r for r in scored if r["runs_with_violations"]]
    total_runs = sum(r["runs"] for r in scored)
    total_agreed = sum(r["agreed_with_rules"] for r in scored)
    total_violating_runs = sum(r["runs_with_violations"] for r in scored)

    vtypes = {}
    for r in scored:
        for t in r["violation_types"]:
            vtypes[t] = vtypes.get(t, 0) + 1

    print(f"\n{'=' * 68}\nSUMMARY  ({args.runs} runs per persona, temperature 0)\n{'=' * 68}")
    print(f"personas with proposals to select between : {len(scored)}")
    print(f"total selection runs                      : {total_runs}")
    print(f"\ndeterministic selector")
    print(f"    distinct selections per persona       : 1 (by construction)")
    print(f"    constraint violations                 : 0 (by construction)")
    print(f"\nmodel selector")
    print(f"    personas where it disagreed with itself: {len(inconsistent)} "
          f"of {len(scored)}")
    print(f"    personas where the first action varied : {len(unstable_first)} "
          f"of {len(scored)}")
    print(f"    runs matching the deterministic choice : {total_agreed} "
          f"of {total_runs} ({total_agreed / max(total_runs, 1):.0%})")
    print(f"    personas with a constraint violation   : {len(violating)} "
          f"of {len(scored)}")
    print(f"    runs with a constraint violation       : {total_violating_runs} "
          f"of {total_runs} ({total_violating_runs / max(total_runs, 1):.0%})")
    for t, n in sorted(vtypes.items()):
        print(f"        {t:24s} on {n} personas")

    print(f"\nwrote {RESULT_DIR}/determinism.csv and determinism.json")


if __name__ == "__main__":
    main()
