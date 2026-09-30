"""
Promote pipeline runner.

Chains the four stages end-to-end for a single client:

    candidates  ->  eligibility  ->  fit evaluation  ->  proposals

Reads a twin from a JSON file, runs each stage in order, and prints what came
back. Nothing more -- this is for watching the pipeline behave on one client
before scaling to the 25 personas.

Usage, from the project root:
    python -m promote_pipeline.promote_runner path/to/persona.json
    python -m promote_pipeline.promote_runner   # defaults to personas/test_persona.json

There is no ledger argument. Nothing in this track has a cooldown or a spend
history to check, so there is no per-client state to pass in. The trigger is
assumed to have fired: every client handed to this runner is treated as one
whose twin has just changed materially, which is what the event-driven path
would produce anyway.
"""

import argparse
import json
import sys

from dotenv import load_dotenv

# Load .env before importing the stage modules. fit_evaluation also calls
# load_dotenv when it builds its client, and load_dotenv does not overwrite
# variables already set, so loading here first simply wins.
load_dotenv()

from promote_pipeline.candidates import generate_candidates
from promote_pipeline.eligibility_filter import eligible_set
from promote_pipeline.fit_evaluation import evaluate_fit
from promote_pipeline.proposal_builder import build_proposals


def _dump(label, payload):
    """Print a section header and the payload as indented JSON."""
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(json.dumps(payload, indent=2, default=str))


def run(twin):
    """
    Run all four stages. Each stage's input is the previous stage's output plus,
    where needed, the twin. Nothing is hidden.
    """
    # Stage 1 -- candidate generation. Deterministic.
    candidate_output = generate_candidates(twin)
    _dump("1. CANDIDATES", candidate_output)

    if not candidate_output["candidates"]:
        # The client is inside tolerance on every asset class. There is nothing
        # to propose, and this is a normal outcome rather than a failure.
        print("\nNo candidates: client is within tolerance on every class.")
        return {
            "candidates": candidate_output,
            "eligibility": None,
            "fit": [],
            "proposals": {"client_id": twin["client_id"], "proposals": []},
        }

    # Stage 2 -- eligibility filter. Deterministic.
    eligibility_output = eligible_set(twin, candidate_output)
    _dump("2. ELIGIBILITY", eligibility_output)

    # Stage 3 -- fit evaluation. The Gemini call is cached to disk keyed by
    # client_id, so reruns are free after the first successful call. Returns an
    # empty list without calling the model if nothing survived eligibility.
    ranked = evaluate_fit(twin, eligibility_output)
    _dump("3. FIT EVALUATION", ranked)

    # Stage 4 -- proposal builder. Deterministic.
    proposals = build_proposals(twin, ranked, eligibility_output)
    _dump("4. PROPOSALS", proposals)

    return {
        "candidates": candidate_output,
        "eligibility": eligibility_output,
        "fit": ranked,
        "proposals": proposals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "persona",
        nargs="?",
        default="personas/test_persona.json",
        help="path to the twin JSON file (default: personas/test_persona.json)",
    )
    args = parser.parse_args()

    with open(args.persona) as f:
        twin = json.load(f)

    print(f"loaded twin {twin['client_id']} from {args.persona}")

    try:
        run(twin)
    except KeyError as e:
        # Most likely a twin missing target_allocation, which candidate
        # generation needs and nothing defaults. Failing loudly is correct: a
        # missing target should not be quietly treated as no gap.
        print(f"\nTWIN FIELD MISSING: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        # Raised by fit evaluation after retries fail, or when the API key is
        # absent. The message names the client and the underlying error.
        print(f"\nMODEL STAGE FAILED: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()