"""
Orchestrator.

Runs all three tracks for one client, then the cross-track decision advisor.

    retention  ->|
    allocation ->|--> adapt --> filter --> detect --> select --> sequence
    promote    ->|

Usage, from the project root:
    python -m cdta.orchestrator personas/P-CONFLICT-01.json
    python -m cdta.orchestrator --quiet          # final decision only

Why the tracks are run here rather than reading their saved output
-------------------------------------------------------------------
Each track's own runner prints and discards. Running them from one place is what
makes an end-to-end result reproducible: the advisor sees exactly what the
tracks produced on this twin, in this run, rather than whatever was last written
to disk.

It also makes the isolation visible. Each track is called with the twin and
nothing else. None of them receives another's output, and none is told that the
advisor exists. That property is asserted constantly in the design; here it is
enforced by the call signatures.

On failures
------------
A track that cannot run does not stop the advisor. A client with no dated goal
has no allocation proposal, and one the detector has never seen has no retention
proposal, and in both cases the remaining tracks still have something to say.
What would be wrong is silently treating a crashed track as a track with nothing
to propose, so failures are recorded and reported rather than swallowed.
"""

import argparse
import json
import sys

from dotenv import load_dotenv

# Loaded before the stage modules, which build their model clients from it.
load_dotenv()

from allocation_pipeline.allocation_catalogs import select_portfolio
from allocation_pipeline.allocation_proposal import build_proposal
from allocation_pipeline.planning_fields import derive_planning_fields
from cdta.adapters import (
    build_context,
    from_allocation,
    from_promote,
    from_retention,
)
from cdta.conflicts import detect_conflicts
from cdta.hard_filter import hard_filter
from cdta.selection import select
from cdta.semantic_conflicts import detect_semantic_conflicts
from cdta.sequencing import sequence
from promote_pipeline.candidates import generate_candidates
from promote_pipeline.eligibility_filter import eligible_set
from promote_pipeline.fit_evaluation import evaluate_fit
from promote_pipeline.proposal_builder import build_proposals
from retention_pipeline.detector import ChurnDetector
from retention_pipeline.diagnosis import diagnose
from retention_pipeline.feasibility import feasible_set
from retention_pipeline.strategy import select_strategy


def _dump(label, payload, quiet):
    if quiet:
        return
    print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# The three tracks
# ---------------------------------------------------------------------------

def run_retention(twin, ledger):
    """Detector, diagnosis, feasibility, strategy."""
    detector = ChurnDetector.load()
    detector_output = detector.score(twin)
    causes = diagnose(twin, detector_output)
    feasibility_output = feasible_set(twin, causes, ledger)
    options = select_strategy(twin, causes, feasibility_output)
    return {
        "detector": detector_output,
        "diagnosis": causes,
        "feasibility": feasibility_output,
        "strategy": options,
    }


def run_allocation(twin):
    """Planning fields, portfolio selection, proposal. Rationale is skipped.

    The rationale explains a decision to a human reader and nothing downstream
    consumes it, so the advisor does not need it and paying for the call here
    would be spending on prose no component reads.
    """
    planning = derive_planning_fields(twin)
    portfolio = select_portfolio(
        planning["risk_capacity"], planning["horizon_years"]
    )
    proposal = build_proposal(twin, planning, portfolio)
    return {
        "planning_fields": planning,
        "portfolio": portfolio,
        "proposal": proposal,
    }


def run_promote(twin):
    """Candidates, eligibility, fit, proposals."""
    candidate_output = generate_candidates(twin)
    if not candidate_output["candidates"]:
        return {
            "candidates": candidate_output,
            "eligibility": None,
            "fit": [],
            "proposals": {"client_id": twin["client_id"], "proposals": []},
        }

    eligibility_output = eligible_set(twin, candidate_output)
    ranked = evaluate_fit(twin, eligibility_output)
    proposals = build_proposals(twin, ranked, eligibility_output)
    return {
        "candidates": candidate_output,
        "eligibility": eligibility_output,
        "fit": ranked,
        "proposals": proposals,
    }


def _try(name, fn, failures):
    """
    Run a track, recording rather than raising on failure.

    A track that cannot run for this client is a fact about the client, not a
    reason to abandon the other two. Treating a crash as an empty result would
    hide it, so the error is kept and reported alongside the decision.
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - recorded, not suppressed
        failures.append({"track": name, "error": f"{type(e).__name__}: {e}"})
        return None


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def run(twin, ledger=None, quiet=False):
    ledger = ledger or []
    failures = []

    retention = _try("retention", lambda: run_retention(twin, ledger), failures)
    allocation = _try("allocation", lambda: run_allocation(twin), failures)
    promote = _try("promote", lambda: run_promote(twin), failures)

    _dump("TRACK OUTPUT -- retention", retention, quiet)
    _dump("TRACK OUTPUT -- allocation", allocation, quiet)
    _dump("TRACK OUTPUT -- promote", promote, quiet)

    # --- adapt ------------------------------------------------------------
    # Horizon comes from the allocation track's planning fields, which is the
    # one figure the promote adapter needs and promote itself does not compute.
    # It is read from the twin's goals either way, so this is not promote
    # reading allocation's decision -- both read the same client fact.
    horizon = (
        allocation["planning_fields"]["horizon_years"] if allocation else None
    )

    proposals = []
    if allocation:
        proposals += from_allocation(twin, allocation["proposal"])
    if promote and horizon is not None:
        proposals += from_promote(twin, promote["proposals"], horizon)
    if retention:
        proposals += from_retention(
            twin,
            retention["detector"],
            retention["diagnosis"],
            retention["feasibility"],
            retention["strategy"],
        )

    context = build_context(
        twin,
        retention["diagnosis"] if retention else [],
        retention["feasibility"] if retention else None,
        promote["candidates"] if promote else None,
        detector_output=retention["detector"] if retention else None,
    )

    _dump("1. PROPOSALS", {"context": context, "proposals": proposals}, quiet)

    # --- stage 1 ----------------------------------------------------------
    filtered = hard_filter(proposals, context)
    _dump("2. HARD FILTER", filtered, quiet)

    # --- stage 2 ----------------------------------------------------------
    deterministic = detect_conflicts(filtered)
    _dump("3a. CONFLICTS (rules)", deterministic, quiet)

    semantic = detect_semantic_conflicts(filtered, context, deterministic)
    _dump("3b. CONFLICTS (semantic)", semantic, quiet)

    conflicts = deterministic["conflicts"] + semantic

    # --- stage 3 ----------------------------------------------------------
    selection = select(filtered, conflicts, context)
    _dump("4. SELECTION", selection, quiet)

    # --- stage 4 ----------------------------------------------------------
    sequenced = sequence(selection)
    _dump("5. SEQUENCE", sequenced, quiet)

    return {
        "client_id": twin["client_id"],
        "context": context,
        "proposals": proposals,
        "filtered": filtered,
        "conflicts": conflicts,
        "selection": selection,
        "sequence": sequenced,
        "track_failures": failures,
    }


def _summary(result):
    """The decision, without the working."""
    lines = [f"\n{'=' * 64}\nDECISION -- {result['client_id']}\n{'=' * 64}"]

    efar = result["context"].get("expected_fraction_at_risk")
    if efar is not None:
        lines.append(f"expected fraction at risk: {efar:.1%}")

    for step in result["sequence"]["sequence"]:
        lines.append(f"\nstep {step['step']} ({step['when']})")
        for key in step["proposals"]:
            lines.append(f"    {key}")
        lines.append(f"    -- {step['detail']}")

    rejected = result["selection"]["rejected"]
    if rejected:
        lines.append("\nnot proceeding")
        for r in rejected:
            p = r["proposal"]
            lines.append(f"    {p['track']}/{p['action_id']}  ({r['reason']})")
            lines.append(f"        {r['detail']}")

    if result["track_failures"]:
        lines.append("\ntrack failures")
        for f in result["track_failures"]:
            lines.append(f"    {f['track']}: {f['error']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "persona",
        nargs="?",
        default="personas/test_persona.json",
        help="path to the twin JSON file",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="optional path to a JSON list of prior grant entries",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print the final decision only, without the intermediate stages",
    )
    args = parser.parse_args()

    with open(args.persona) as f:
        twin = json.load(f)

    ledger = []
    if args.ledger:
        with open(args.ledger) as f:
            ledger = json.load(f)

    if not args.quiet:
        print(f"loaded twin {twin['client_id']} from {args.persona}")

    try:
        result = run(twin, ledger, quiet=args.quiet)
    except RuntimeError as e:
        # Raised by the semantic detector after retries fail, or when the API
        # key is absent. A track failure is recorded and does not reach here.
        print(f"\nADVISOR FAILED: {e}", file=sys.stderr)
        sys.exit(3)

    print(_summary(result))

    # A track that crashed means the decision was made on an incomplete set.
    # Worth a non-zero exit so a batch run does not report success.
    if result["track_failures"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
