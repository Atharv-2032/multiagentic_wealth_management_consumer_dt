"""
Allocation pipeline runner.

Chains the four stages end-to-end for a single client:

    planning fields  ->  portfolio selection  ->  proposal  ->  rationale

Reads a twin from a JSON file, runs each stage in order, and prints what came
back.

Usage, from the project root:
    python -m allocation_pipeline.allocation_runner path/to/persona.json
    python -m allocation_pipeline.allocation_runner --write-target

Writing the target back
------------------------
Portfolio selection produces the target allocation the promote track reads from
the twin. Until allocation has run, that field is a hand-set stub.

--write-target updates the persona file in place, replacing the stub with the
selected portfolio and marking derived_from as "allocation". It is opt-in
because it modifies a file the other tracks read, and doing that silently on
every run would make it hard to tell which targets were computed and which were
set by hand.

Nothing here reads another track's output, and the write does not change that.
The target is durable client state, written at an allocation run and read by
whatever runs later; promote never waits for allocation, never triggers it, and
does not know when it last ran.
"""

import argparse
import json
import sys

from dotenv import load_dotenv

# Load .env before importing the stage modules. allocation_rationale also calls
# load_dotenv when it builds its client, and load_dotenv does not overwrite
# variables already set, so loading here first simply wins.
load_dotenv()

from allocation_pipeline.allocation_catalogs import (
    MODEL_PORTFOLIOS,
    TOLERANCE_BAND,
    select_portfolio,
)
from allocation_pipeline.allocation_proposal import build_proposal
from allocation_pipeline.allocation_rationale import write_rationale
from allocation_pipeline.planning_fields import derive_planning_fields


def _dump(label, payload):
    """Print a section header and the payload as indented JSON."""
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(json.dumps(payload, indent=2, default=str))


def run(twin, skip_rationale=False):
    """
    Run all four stages. Each stage's input is the previous stage's output plus,
    where needed, the twin. Nothing is hidden.
    """
    # Stage 1 -- planning fields. Deterministic.
    planning_fields = derive_planning_fields(twin)
    _dump("1. PLANNING FIELDS", planning_fields)

    # Stage 2 -- portfolio selection. A lookup, so it prints as one.
    portfolio_name = select_portfolio(
        planning_fields["risk_capacity"], planning_fields["horizon_years"]
    )
    _dump("2. PORTFOLIO SELECTION", {
        "model_portfolio": portfolio_name,
        "weights": MODEL_PORTFOLIOS[portfolio_name],
        "tolerance_band": TOLERANCE_BAND,
        "selected_by": (
            f"{planning_fields['risk_capacity']} capacity, "
            f"{planning_fields['horizon_band']} horizon "
            f"({planning_fields['horizon_years']}y)"
        ),
    })

    # Stage 3 -- proposal. Deterministic. Returns a decision either way: a null
    # action carries its reason and the figures behind it.
    proposal = build_proposal(twin, planning_fields, portfolio_name)
    _dump("3. PROPOSAL", proposal)

    # Stage 4 -- rationale. Explains a decision already made. Cached to disk
    # keyed by client_id, so reruns are free after the first successful call.
    rationale = None
    if not skip_rationale:
        rationale = write_rationale(planning_fields, proposal)
        print(f"\n{'=' * 60}\n4. RATIONALE\n{'=' * 60}")
        print(rationale["text"])
        if rationale["unaccounted_numbers"]:
            # Not an error. Figures in the prose that were not in the input --
            # possibly a year or a count, possibly an invention. Surfaced rather
            # than passed on silently.
            print(
                f"\n[check] figures not found in the input: "
                f"{rationale['unaccounted_numbers']}",
                file=sys.stderr,
            )

    return {
        "planning_fields": planning_fields,
        "portfolio_name": portfolio_name,
        "proposal": proposal,
        "rationale": rationale,
    }


def write_target_to_twin(path, twin, portfolio_name):
    """
    Replace the twin's target_allocation with the selected portfolio.

    derived_from becomes "allocation", which is what distinguishes a computed
    target from a hand-set stub. Anything else already on the twin is left
    alone; only this block is replaced.
    """
    twin["target_allocation"] = {
        "model_portfolio": portfolio_name,
        "weights": MODEL_PORTFOLIOS[portfolio_name],
        "tolerance_band": TOLERANCE_BAND,
        "derived_from": "allocation",
    }

    with open(path, "w") as f:
        json.dump(twin, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "persona",
        nargs="?",
        default="personas/test_persona.json",
        help="path to the twin JSON file (default: personas/test_persona.json)",
    )
    parser.add_argument(
        "--write-target",
        action="store_true",
        help="write the selected portfolio back to the twin as "
             "target_allocation, replacing any stub",
    )
    parser.add_argument(
        "--skip-rationale",
        action="store_true",
        help="run the deterministic stages only, without calling the model",
    )
    args = parser.parse_args()

    with open(args.persona) as f:
        twin = json.load(f)

    print(f"loaded twin {twin['client_id']} from {args.persona}")

    try:
        result = run(twin, skip_rationale=args.skip_rationale)
    except ValueError as e:
        # Raised by the planning stage: no dated goal, or a risk capacity
        # outside the vocabulary. Failing loudly is correct -- guessing a
        # horizon would mean selecting a portfolio from nothing.
        print(f"\nPLANNING FAILED: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        # Raised by the rationale stage after retries fail, or when the API key
        # is absent.
        print(f"\nMODEL STAGE FAILED: {e}", file=sys.stderr)
        sys.exit(3)

    if args.write_target:
        # Read what was there before overwriting, so the line printed says which
        # kind of target is being replaced rather than assuming a stub.
        existing = twin.get("target_allocation") or {}
        previous_source = existing.get("derived_from", "absent")
        previous_name = existing.get("model_portfolio", "none")

        write_target_to_twin(args.persona, twin, result["portfolio_name"])

        print(
            f"\nwrote target_allocation to {args.persona}: "
            f"{result['portfolio_name']} (derived_from: allocation)\n"
            f"  replaced: {previous_name} (derived_from: {previous_source})"
        )


if __name__ == "__main__":
    main()