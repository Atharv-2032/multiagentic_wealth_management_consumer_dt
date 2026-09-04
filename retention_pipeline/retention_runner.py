"""
Retention pipeline runner.

Chains the four stages end-to-end for a single client:

    detector  ->  diagnosis  ->  feasibility  ->  strategy

Reads a twin from a JSON file, runs each stage in order, and prints what came
back. Nothing more -- this is for watching the pipeline behave on one client
before scaling to the 25 personas.

Usage:
    python runner.py path/to/persona.json
    python runner.py                          # defaults to personas/test_persona.json

Ledger is empty by default. When you build the 25-persona ledgers later, this
runner takes an optional --ledger flag pointing at a JSON list of grant entries
for the client.
"""

import argparse
import json
import sys

from retention_pipeline.detector import ChurnDetector
from retention_pipeline.diagnosis import diagnose
from retention_pipeline.feasibility import feasible_set
from retention_pipeline.strategy import select_strategy


def _dump(label, payload):
    """Print a section header and the payload as indented JSON."""
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(json.dumps(payload, indent=2, default=str))


def run(twin, ledger):
    """
    Run all four stages. Each stage's input is the previous stage's output
    plus, where needed, the twin and the ledger. Nothing is hidden.
    """
    detector = ChurnDetector.load()

    # Stage 1 -- detector.
    detector_output = detector.score(twin)
    _dump("1. DETECTOR", detector_output)

    # Stage 2 -- diagnosis. The Gemini call is cached to disk keyed by
    # client_id, so reruns are free after the first successful call.
    causes = diagnose(twin, detector_output)
    _dump("2. DIAGNOSIS", causes)

    # Stage 3 -- feasibility. Deterministic; no cache needed.
    feasibility_output = feasible_set(twin, causes, ledger)
    _dump("3. FEASIBILITY", feasibility_output)

    # Stage 4 -- strategy. Also cached to disk keyed by client_id.
    options = select_strategy(twin, causes, feasibility_output)
    _dump("4. STRATEGY", options)

    return {
        "detector": detector_output,
        "diagnosis": causes,
        "feasibility": feasibility_output,
        "strategy": options,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "persona",
        nargs="?",
        default="personas/test_persona.json",
        help="path to the twin JSON file (default: personas/test_persona.json)",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="optional path to a JSON list of prior grant entries "
             "(default: empty ledger)",
    )
    args = parser.parse_args()

    with open(args.persona) as f:
        twin = json.load(f)

    if args.ledger:
        with open(args.ledger) as f:
            ledger = json.load(f)
    else:
        ledger = []

    print(f"loaded twin {twin['client_id']} from {args.persona}")
    print(f"ledger: {len(ledger)} prior grant entries")

    try:
        run(twin, ledger)
    except AssertionError as e:
        # The feasibility engine's guard rail. If it fires, the ledger created a
        # grant that violated the spend cap when it was issued -- fix the ledger,
        # not the engine.
        print(f"\nFEASIBILITY GUARD RAIL: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        # Raised by diagnosis or strategy after retries fail. The message names
        # the client and the underlying error.
        print(f"\nMODEL STAGE FAILED: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()