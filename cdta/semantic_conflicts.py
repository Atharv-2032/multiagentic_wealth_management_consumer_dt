"""
Stage 2b -- semantic contradiction detection.

The only language model in the advisor. It asserts conflicts that no rule
anticipated, and it decides nothing.

What it is for
---------------
Stage 2a catches what is arithmetic: two proposals reaching for the same money,
a purchase into a class a rebalance already fills. Some incompatibilities are
not arithmetic at all.

The canonical one: asking a client to put fifty thousand dollars into a new
product is incoherent when that client has been diagnosed as fee-sensitive and
at risk of leaving. Nothing sums wrongly. Both proposals are individually valid
and both passed their own track's filter. The incompatibility only appears if
you read the diagnosis and understand what it implies about how an approach will
land.

No rule anticipates that, and writing one would mean enumerating the pairings in
advance -- which is the thing a language model is actually better at than a
lookup table.

Three bounds on what it may do
-------------------------------
It asserts conflicts and never clears one. A conflict found by stage 2a is a
fact about arithmetic, and a model disagreeing with arithmetic is a model being
wrong. The deterministic constraints are passed in so the model does not
re-report them, not so it can revise them.

Every assertion must cite the fields it rests on. An assertion naming
`addresses` and `diagnosis` is checkable by whoever reviews it; one resting on
an impression is not. Citation is validated, not requested.

It emits constraints, not decisions. It never says which proposal survives.
Selection under stated constraints is a computation, and moving that into a
component whose output varies between runs would forfeit the property the whole
architecture exists to provide.

What it reads
--------------
The passed proposals and the client context -- diagnosis, budget state, gaps.
It does not read the rationale prose any track generated. Prose is where a
model would find whatever it went looking for, and a conflict that cannot be
stated over structured fields is one the advisor has no basis to assert.
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai

MODEL = "gemini-3.7-flash"
MAX_CONFLICTS = 3
MAX_RETRIES = 1

CACHE_PATH = os.path.join("out", "semantic_conflicts.json")

CONFLICT_SEMANTIC = "semantic_contradiction"

# Fields an assertion may cite. A citation outside this set is either invented
# or points at prose, and both are rejected.
CITABLE_FIELDS = {
    "track", "action_id", "amount", "cost", "cost_borne_by",
    "funding_source", "asset_class", "value", "addresses",
    "diagnosis", "idle_cash", "firm_spend_remaining", "gaps",
    "portfolio_value", "expected_fraction_at_risk", "tier",
}


SYSTEM_PROMPT = """\
You are the contradiction detector in a wealth management system. Three \
independent components have each proposed an action for the same client. None of \
them could see what the others proposed. Your task is to identify pairs that are \
incompatible for reasons no rule would have caught.

## What you are not doing

**You are not choosing.** Do not say which proposal should proceed, which is \
better, or which should be dropped. A later component decides that, under \
constraints you supply. Your output is an observation, not a recommendation.

**You are not re-checking arithmetic.** Conflicts over money and over overlapping \
asset classes have already been found by rules and are listed in your input. Do \
not restate them, do not extend them, and do not disagree with them. They are \
computed facts.

**You are not evaluating quality.** Whether a proposal is well-chosen for this \
client is its own track's judgement and was made before you saw it. A proposal \
you find unconvincing is not thereby in conflict with anything.

## What you are looking for

Pairs where each proposal is individually sound, nothing sums wrongly, and yet \
carrying out both would be incoherent.

The clearest kind: one proposal's action is undermined by what the diagnosis says \
about the client. Asking a client to commit more money is incoherent when the \
diagnosis says they are fee-sensitive and moving assets out. Proposing a \
discovery conversation to understand a client is undermined by simultaneously \
selling them something, because the conversation stops being exploratory.

Other kinds exist. What they share is that the incompatibility lives in what the \
actions mean together, not in what they cost.

## What is not a contradiction

**Two proposals simply both being expensive.** That is a budget question and \
budgets are enforced elsewhere.

**One proposal being better than another.** Ranking is not conflict.

**Two proposals addressing different problems.** A client can have more than one \
thing wrong with them, and fixing two things at once is normal.

**A proposal you would not have made.** You are not reviewing the tracks.

Be willing to return nothing. Most clients produce no semantic contradiction, and \
an empty result is the common correct answer. Manufacturing one to appear useful \
is the failure this component is most prone to.

## What you receive

**Proposals** -- each with its track, action, amount, cost, who bears it, funding \
source, asset class, value, and what it addresses.

**Client context** -- the churn diagnosis with confidence levels, the firm's \
remaining spend allowance, the client's idle cash, and their allocation gaps.

**Conflicts already found** -- by rules, over funding and asset class overlap. \
Yours must be different from these.

## Output

Return JSON only, no prose before or after:

{
  "conflicts": [
    {
      "proposals": ["<track>/<action_id>", "<track>/<action_id>"],
      "fields": ["<the structured fields this rests on>"],
      "detail": "<one or two sentences: what makes these two incoherent together, naming what you read>"
    }
  ]
}

Each entry names exactly two proposals, identified as track/action_id exactly as \
given. `fields` must list the fields your assertion rests on, drawn only from the \
fields present in the proposals and context you were given -- an assertion you \
cannot ground in a named field is one you should not make. Return at most three. \
Return `{"conflicts": []}` where there are none.
"""


# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------

def build_user_message(passed, context, deterministic_conflicts):
    """
    Assemble what the model reads.

    source_detail is stripped from every proposal. It holds each track's
    rationale prose, and prose is where a model finds whatever it went looking
    for -- a conflict that cannot be stated over structured fields is one the
    advisor has no basis to assert.

    The deterministic conflicts are included so the model does not re-report
    them as its own findings. They are labelled as already found rather than
    presented as evidence.
    """
    proposals = [
        {k: v for k, v in p.items() if k != "source_detail"}
        for p in passed
    ]

    return (
        f"## Proposals\n{json.dumps(proposals, indent=2)}\n\n"
        f"## Client context\n{json.dumps(context, indent=2)}\n\n"
        f"## Conflicts already found by rules\n"
        f"{json.dumps(deterministic_conflicts, indent=2)}\n\n"
        f"Identify any semantic contradictions among these proposals."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def parse_response(text, passed, deterministic_conflicts):
    """
    Parse and validate.

    Four checks, each enforcing one of the bounds on this component:

      - the named proposals exist, so an assertion cannot be about something
        that was never proposed
      - exactly two are named, since a contradiction is between a pair
      - the cited fields are real, so the claim is checkable
      - the pair is not one the rules already found, so the model cannot
        restate arithmetic as its own finding or quietly contradict it
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)
    conflicts = data.get("conflicts", [])

    if len(conflicts) > MAX_CONFLICTS:
        raise ValueError(
            f"{len(conflicts)} conflicts returned, maximum is {MAX_CONFLICTS}"
        )

    valid_keys = {f"{p['track']}/{p['action_id']}" for p in passed}
    already = {frozenset(c["proposals"]) for c in deterministic_conflicts}

    out = []
    for c in conflicts:
        pair = c.get("proposals", [])
        if len(pair) != 2:
            raise ValueError(f"conflict names {len(pair)} proposals, expected 2")

        for key in pair:
            if key not in valid_keys:
                raise ValueError(f"unknown proposal {key!r}")

        if pair[0] == pair[1]:
            raise ValueError(f"conflict names {pair[0]!r} against itself")

        # A pair the rules already found is not a new assertion. Rejecting it
        # rather than dropping it keeps the boundary visible: the model is not
        # permitted to re-litigate arithmetic, and a response that tries to is
        # a response that misread its instructions.
        if frozenset(pair) in already:
            raise ValueError(
                f"conflict between {pair[0]} and {pair[1]} was already found "
                f"by a rule"
            )

        fields = c.get("fields", [])
        if not fields:
            raise ValueError(f"no fields cited for {pair[0]} / {pair[1]}")
        for f in fields:
            if f not in CITABLE_FIELDS:
                raise ValueError(
                    f"cited field {f!r} is not one the advisor reads"
                )

        if not c.get("detail"):
            raise ValueError(f"no detail given for {pair[0]} / {pair[1]}")

        out.append({
            "type": CONFLICT_SEMANTIC,
            "proposals": list(pair),
            "fields": list(fields),
            "detail": c["detail"],
            # Deliberately absent: which proposal is redundant, or which should
            # survive. A semantic contradiction is symmetric as far as this
            # component is concerned -- it reports that two things cannot both
            # happen, and selection decides which does.
        })

    return out


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env at the project root "
                "or export it in the environment."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _call_once(user_message):
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=2000,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def detect_semantic_conflicts(filter_output, context, deterministic_output,
                              use_cache=True):
    """
    Run the semantic detector for one client.

    Returns an empty list without calling the model where there is nothing to
    compare: a single proposal cannot contradict anything, and spending on a
    model to establish that would be paying for an answer arithmetic already
    gives.
    """
    client_id = context["client_id"]
    passed = filter_output["passed"]
    deterministic = deterministic_output["conflicts"]

    if len(passed) < 2:
        return []

    cache = _load_cache() if use_cache else {}
    if client_id in cache:
        return cache[client_id]

    user_message = build_user_message(passed, context, deterministic)

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        raw = _call_once(user_message)
        try:
            conflicts = parse_response(raw, passed, deterministic)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            time.sleep(1)
    else:
        raise RuntimeError(
            f"semantic conflict detection failed for {client_id} after "
            f"{MAX_RETRIES + 1} attempt(s): {last_error}"
        )

    if use_cache:
        cache[client_id] = conflicts
        _save_cache(cache)

    return conflicts


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
