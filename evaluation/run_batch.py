"""
Batch evaluation runner.

Runs every persona through all three tracks and the advisor, and writes the
coverage table: what each persona was built to test, what the system produced,
and what a firm without an arbitration layer would have done instead.

Run from the project root:

    python -m evaluation.run_batch
    python -m evaluation.run_batch --no-cache      # ignore saved model output
    python -m evaluation.run_batch C22-TIER0 C23-OVERLAP

Outputs
-------
    out/evaluation/coverage.json   one record per persona, full detail
    out/evaluation/coverage.csv    one row per persona, for the paper
    out/evaluation/<id>.json       the full advisor result for each persona

What the baseline is
---------------------
"Without CDTA" is each track's top-ranked proposal, all executed. That is what
three independent systems produce: retention ranks its remedies and someone acts
on the first, allocation emits its rebalance, promote emits its best product.
No ordering question arises because no selection happens -- nobody holds all
three at once.

Top-ranked only, not everything a track proposed. Promote emits up to three
products, and those are alternatives for one reviewer rather than a basket to be
funded together; counting all three would inflate the conflict count by treating
alternatives as if each would be executed.

This baseline is a construct, and the paper should say so. It models what
isolated tracks would produce rather than reproducing a real firm's
uncoordinated process. The construct is fair -- each track genuinely cannot see
the others -- but it is a modelling choice and not an observed alternative.

Two numbers, not one
---------------------
Conflicts DETECTED and baseline errors AVOIDED are different quantities, and the
difference matters enough that they are reported separately.

A conflict is detected whenever the advisor finds one. A baseline error is
avoided only where the baseline would actually have committed it -- which
requires both halves of the conflict to be in the baseline set.

Funding exclusivity is the case that separates them. Only the promote track
draws on client cash: allocation funds a rebalance by selling, and a retention
remedy spends the firm's money. So exclusivity only ever arises between promote
proposals, and the baseline executes one product per track. The advisor's check
is real bookkeeping among alternatives it is choosing between, but it is not an
error a firm without arbitration would have made.

That is a limitation of the scoped design worth stating: a system in which
allocation could fund from cash, or retention could ask for a client
contribution, would exercise the check across tracks rather than within one.

Conflating the two numbers is what a reviewer would attack, so the summary
prints both.

On the conditions a model decides
----------------------------------
Diagnosis, fit and the semantic detector are language models. The personas
targeting those conditions are built to produce the outcome, but whether they do
is the experiment rather than a fixture. Those rows record what happened; they
are not assertions about what must happen.
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

from cdta.orchestrator import run
from personas.build_personas import CONDITIONS, LEDGERS

PERSONA_DIR = "personas"
LEDGER_DIR = os.path.join("personas", "ledgers")
OUT_DIR = os.path.join("out", "evaluation")


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------

def baseline(proposals):
    """
    What each track would have done alone: its top-ranked proposal.

    Order within a track is the order the adapters produced, which is the order
    the track itself ranked. Taking the first is taking what that track would
    have put in front of a human.
    """
    top = {}
    for p in proposals:
        top.setdefault(p["track"], p)
    return list(top.values())


def baseline_problems(baseline_set, conflicts):
    """
    Which of the detected conflicts the baseline would actually have walked into.

    A conflict counts only where BOTH of its proposals are in the baseline. A
    conflict whose other half was never going to be executed is not an error the
    baseline commits -- it is the advisor doing bookkeeping among alternatives,
    which is work the baseline never had to do because it never held them all.
    """
    keys = {f"{p['track']}/{p['action_id']}" for p in baseline_set}
    return [c for c in conflicts if all(k in keys for k in c["proposals"])]


# ---------------------------------------------------------------------------
# One persona
# ---------------------------------------------------------------------------

def evaluate(name):
    path = os.path.join(PERSONA_DIR, f"{name}.json")
    with open(path) as f:
        twin = json.load(f)

    ledger = []
    if name in LEDGERS:
        with open(os.path.join(LEDGER_DIR, f"{name}.json")) as f:
            ledger = json.load(f)

    result = run(twin, ledger, quiet=True)

    props = result["proposals"]
    base = baseline(props)
    hit = baseline_problems(base, result["conflicts"])

    condition, expected, kind = CONDITIONS.get(name, ("", "", "unknown"))

    selected = [
        f"{p['track']}/{p['action_id']}" for p in result["selection"]["selected"]
    ]
    rejected = [
        f"{r['proposal']['track']}/{r['proposal']['action_id']}:{r['reason']}"
        for r in result["selection"]["rejected"]
    ]

    return {
        "persona": name,
        "condition": condition,
        "expected": expected,
        "kind": kind,
        # what the tracks produced
        "n_proposals": len(props),
        "tracks_present": sorted({p["track"] for p in props}),
        "track_failures": [f["track"] for f in result["track_failures"]],
        "expected_fraction_at_risk": result["context"].get("expected_fraction_at_risk"),
        "tiers": sorted({p["tier"] for p in props}),
        # without the advisor
        "baseline_actions": [f"{p['track']}/{p['action_id']}" for p in base],
        "baseline_conflicts": [c["type"] for c in hit],
        "baseline_conflict_detail": [c["detail"] for c in hit],
        "baseline_would_misfire": bool(hit),
        # with the advisor
        "conflicts_found": [c["type"] for c in result["conflicts"]],
        "selected": selected,
        "rejected": rejected,
        "sequence_gated": result["sequence"]["gated"],
        "steps": len(result["sequence"]["sequence"]),
        # raw, for the per-persona file
        "_result": result,
    }


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "persona", "condition", "expected", "kind",
    "expected_fraction_at_risk", "n_proposals", "tracks_present",
    "baseline_actions", "baseline_conflicts", "baseline_would_misfire",
    "conflicts_found", "selected", "rejected", "sequence_gated",
    "track_failures",
]

MODEL_CACHES = (
    "diagnoses.json", "strategies.json",
    "fit_evaluations.json", "semantic_conflicts.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("personas", nargs="*",
                        help="persona ids to run; default is all of them")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore saved model output and call the models again")
    args = parser.parse_args()

    names = args.personas or sorted(CONDITIONS)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.no_cache:
        # The model stages cache by client_id. Clearing the caches is the only
        # way to force a fresh call, and it is worth saying so rather than
        # hiding it behind a flag that looks free.
        for cache in MODEL_CACHES:
            p = os.path.join("out", cache)
            if os.path.exists(p):
                os.remove(p)
        print("cleared model caches; every model stage will call the model")

    rows, failures = [], []
    for name in names:
        try:
            row = evaluate(name)
        except Exception as e:  # noqa: BLE001 - recorded, not suppressed
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  {name:18s} FAILED  {type(e).__name__}: {e}", file=sys.stderr)
            continue

        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(row.pop("_result"), f, indent=2, default=str)

        rows.append(row)
        flag = "!" if row["baseline_would_misfire"] else " "
        print(f"{flag} {name:18s} sel={len(row['selected'])} "
              f"rej={len(row['rejected'])} conflicts={row['conflicts_found']}")

    with open(os.path.join(OUT_DIR, "coverage.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "coverage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                k: ("; ".join(map(str, v)) if isinstance(v, list) else v)
                for k, v in r.items() if k in CSV_FIELDS
            })

    # --- summary ----------------------------------------------------------
    detected, avoided = {}, {}
    for r in rows:
        for c in r["conflicts_found"]:
            detected[c] = detected.get(c, 0) + 1
        for c in r["baseline_conflicts"]:
            avoided[c] = avoided.get(c, 0) + 1

    misfires = [r for r in rows if r["baseline_would_misfire"]]

    print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
    print(f"personas run                     : {len(rows)}")
    print(f"\nconflicts DETECTED by the advisor:")
    for t, n in sorted(detected.items()):
        print(f"    {t:26s} {n}")
    print(f"\nbaseline errors AVOIDED          : {len(misfires)} of {len(rows)} "
          f"({len(misfires) / max(len(rows), 1):.0%} of personas)")
    for t, n in sorted(avoided.items()):
        print(f"    {t:26s} {n}")
    if not avoided:
        print("    (none)")

    gated = sum(1 for r in rows if r["sequence_gated"])
    print(f"\nsequenced into two steps         : {gated}")

    if failures:
        print(f"\npersonas that failed to run      : {len(failures)}")
        for name, err in failures:
            print(f"    {name}: {err}")

    print(f"\nwrote {OUT_DIR}/coverage.csv and coverage.json")

    if failures:
        sys.exit(4)


if __name__ == "__main__":
    main()
