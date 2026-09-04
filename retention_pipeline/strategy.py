"""
Strategy selection -- stage three of the retention pipeline.

Receives a diagnosis and the set of remedies the client may be offered, and
returns a ranked shortlist of what to do.

Why a language model here
--------------------------
The judgement is a fit question: given these reasons and these permitted
actions, which action best addresses the reasons, and is it worth its cost for
this client. That weighs several things that do not reduce to a formula --
how well a remedy matches a cause, how strong the evidence for that cause was,
how much the relationship is worth relative to what the action costs.

What it does NOT do
--------------------
It does not decide what is permitted. The option set arrives already narrowed by
the feasibility engine, so an ineligible, unaffordable, or on-cooldown remedy is
not something to be avoided here -- it is not present. The model chooses among
options that are all, by construction, allowed.

It also does not make the final call. The output is a ranked shortlist, not a
decision. Cross-track arbitration reconciles this track's shortlist against the
allocation and promote tracks before anything reaches a human.
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-3.7-flash"
MAX_OPTIONS = 3
MAX_RETRIES = 1  # one retry on a malformed/out-of-set response, then raise

CACHE_PATH = os.path.join("out", "strategies.json")


SYSTEM_PROMPT = """\
You are the strategy component of a retention system in a wealth management firm. \
A client has been flagged as at risk of moving assets away, another component has \
diagnosed the likely reasons, and a policy engine has determined which actions the \
firm is permitted to take for this client right now.

Your task is to rank the permitted actions, best first, and say why.

## What is already settled, and not yours to revisit

**What the client's problem is.** The diagnosis is an input. Do not re-diagnose, \
and do not argue with it. If you think the evidence points elsewhere, that is not \
actionable here -- work with the causes you are given.

**What the firm is allowed to do.** Every action in the permitted list has already \
passed eligibility, budget, cooldown, and diagnostic-certainty checks. You do not \
need to consider whether an action is affordable or allowed. It is.

The corollary matters more: **you may only return actions from the permitted list.** \
Do not propose an action that is absent from it, do not suggest an alternative, and \
do not remark that some other action would have been better. An action missing from \
the list was withheld deliberately, by a rule, for a reason you are not shown. \
Treating its absence as an oversight would be a mistake.

## The evidence you receive

**Diagnosis** -- up to three causes, ranked best-supported first, each with a \
confidence level (`high`, `medium`, `low`), the specific evidence cited for it, and \
brief reasoning. The first cause is the primary one.

**Permitted actions** -- what the firm may offer. Each carries an identifier, a \
label, and its cost to the firm. The fee concession is different: instead of a \
single cost it carries a `discount_range` and a matching `cost_range`, because its \
size is variable. See the section on choosing a discount below.

**Client profile** -- portfolio value, the annual fee revenue this client generates, \
tenure with the firm, age, holdings, and stated goals. This is what tells you \
whether an action's cost is proportionate.

## The actions and what each is for

**`fee_concession`** -- a temporary reduction in the fee the client pays.
Addresses `fee_sensitivity`. It removes the specific irritant of paying above the \
going rate. It does nothing for a client whose complaint is performance, and \
discounting a service the client does not value buys nothing.

**`portfolio_review`** -- an advisor-led review of the portfolio and its positioning.
Addresses `performance_dissatisfaction`. It is a substantive response to \
underperformance: an explanation, and usually a change. It is not a retention \
gesture and should not be used as one.

**`discovery_contact`** -- an advisor conversation whose purpose is to find out what \
is going on.
Addresses `competitor_consolidation` and `insufficient_evidence` -- the cases where \
something is clearly wrong but the reason is not visible in the data. Its value is \
information, not repair. It is the correct action when acting on a guess would be \
worse than asking.

**`accept_loss`** -- take no action.
Always permitted, and a legitimate choice rather than a failure. Correct in two \
distinct situations, worth keeping separate in your mind:

- The outflow is not a problem. A retiring client spending down their portfolio as \
planned is doing what the portfolio was for. Intervening would be an error.
- The outflow is a problem, but not one worth solving. If the permitted actions cost \
more than the relationship is worth, or the client is leaving for a reason no \
available action touches, accepting it is the disciplined answer.

## How to rank

Work in this order.

**First, fit to the primary cause.** The action that addresses the best-supported \
cause should normally rank first. This is the main consideration, and it will settle \
most cases on its own.

**Then, proportionality.** Compare the cost of the action to the annual fee revenue \
this client generates. A $150 review is trivial for a client generating $18,000 a \
year and significant for one generating $900. Cost is a tiebreaker between actions \
of similar fit, not a reason to prefer a poor fit over a good one.

**Then, the secondary causes.** Where a client has more than one cause and more than \
one well-fitting action is permitted, the second and third ranks are where the \
remaining causes get addressed.

Two things to be careful about.

**Confidence should moderate spending, not fit.** A cause diagnosed at `medium` \
confidence is a weaker basis for spending money than one at `high`. Where you are \
choosing between a priced action against a weakly-supported cause and a cheaper one, \
say so in your reasoning.

**Do not rank an action highly because it is available.** The permitted list is not \
a set of suggestions. If the only well-matched action for this client's cause is \
absent from the list, then the honest ranking may well put `accept_loss` first, and \
you should say plainly that no permitted action addresses the diagnosed problem.

## Choosing a discount, if the fee concession is permitted

The concession arrives with a `discount_range`: a floor and a ceiling, both already \
checked against policy and the client's remaining budget. Any value in that range is \
allowed, so choose the value that fits, not the largest available.

Scale it to how strong the fee case is. A client whose fee sits far above the book \
average, with high confidence on `fee_sensitivity`, warrants a figure toward the top \
of the range. A marginal or medium-confidence case warrants the floor. Do not default \
to the ceiling; a larger concession than the situation calls for spends the client's \
own future allowance for no additional effect.

Give the discount as a decimal -- 0.15 means a 15 percent reduction. It must fall \
within the range you were given, inclusive.

## Output

Return JSON only, no prose before or after:

{
  "options": [
    {
      "remedy_id": "<an identifier from the permitted list>",
      "addresses": ["<the cause or causes from the diagnosis this action responds to>"],
      "discount": <decimal within discount_range, for fee_concession only; omit otherwise>,
      "reasoning": "<two or three sentences: why this action for this cause, and why at this rank>"
    }
  ]
}

Rank best first. Return at most three options, and fewer if fewer are permitted or \
fewer are worth offering. Every `remedy_id` must appear in the permitted list, and \
no action may appear twice.

For `accept_loss`, set `addresses` to the causes it is a response to, or to an empty \
list where the point is that no permitted action addresses them.
"""


# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------

def build_user_message(twin, diagnosis, feasibility_output):
    """
    Assemble the client's evidence for strategy selection.

    Three inputs, kept distinct in the message so the model does not confuse
    what a diagnostic component concluded with what a policy engine permitted.

    Only the permitted list is passed. The rejected list is audit trail --
    exclusions explained after the fact, retrievable by compliance or human
    review, but not something the model should reason about. The budget block
    and touch budget are also withheld: the prompt states that every permitted
    action is already affordable, so showing spend cap arithmetic invites
    exactly the second-guessing the system prompt forbids.

    Client profile is trimmed to the fields proportionality actually turns on --
    fee revenue, portfolio value, tenure, age -- rather than the full twin.
    The model does not need holdings or goals to rank remedies; those matter
    upstream, in diagnosis.
    """
    profile = {
        "annual_fee_revenue": twin["annual_fee_revenue"],
        "portfolio_value": sum(h["value"] for h in twin["holdings"]),
        "tenure_years": twin["tenure_years"],
        "age": twin["age"],
    }

    return (
        f"## Client profile\n{json.dumps(profile, indent=2)}\n\n"
        f"## Diagnosis\n{json.dumps(diagnosis, indent=2)}\n\n"
        f"## Permitted actions\n{json.dumps(feasibility_output['permitted'], indent=2)}\n\n"
        f"Rank the permitted actions for this client."
    )


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
#
# Validation is checked against the permitted set, not against the catalog. That
# is the point of the stage division: an action outside the permitted set is not
# a poor choice to be scored down, it is an invalid response. Same for a discount
# outside the range -- the range was computed against policy and the client's
# remaining budget, so a value beyond it is not a judgement call, it is a
# violation, and it is caught here rather than reaching a human as a well-argued
# recommendation for something the firm has already ruled out.


def parse_response(text, permitted):
    """
    Parse and validate. A malformed response, a remedy outside the permitted
    set, a repeated remedy, or a fee concession discount outside the given
    range is a failure to surface, not something to silently repair.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)

    options = data.get("options", [])
    if not options:
        raise ValueError("no options returned")
    if len(options) > MAX_OPTIONS:
        raise ValueError(f"{len(options)} options returned, maximum is {MAX_OPTIONS}")

    permitted_by_id = {p["remedy_id"]: p for p in permitted}

    seen = set()
    for o in options:
        rid = o.get("remedy_id")
        if rid not in permitted_by_id:
            raise ValueError(f"remedy_id {rid!r} is not in the permitted list")
        if rid in seen:
            raise ValueError(f"remedy_id {rid!r} appears more than once")
        seen.add(rid)

        if not o.get("reasoning"):
            raise ValueError(f"no reasoning given for {rid}")

        # A discount is required for fee_concession, and only fee_concession,
        # and it must fall inside the range the feasibility engine issued. That
        # range was computed against policy and the client's remaining budget,
        # so a value outside it is a hard violation, not a judgement call.
        permitted_option = permitted_by_id[rid]
        if "discount_range" in permitted_option:
            discount = o.get("discount")
            if discount is None:
                raise ValueError(f"{rid} requires a discount")
            lo, hi = permitted_option["discount_range"]
            if not (lo <= discount <= hi):
                raise ValueError(
                    f"discount {discount} for {rid} is outside the permitted "
                    f"range [{lo}, {hi}]"
                )
        else:
            if "discount" in o and o["discount"] is not None:
                raise ValueError(f"{rid} does not take a discount")

    return options


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
#
# One retry on a malformed or invalid response, then raise. Temperature is 0 --
# not full determinism, sampling still varies -- but it reduces run-to-run
# variation, which matters given the paper claims reproducibility for the
# deterministic stages and is honest about the model stages' variability.

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY from the environment
    return _client


def _call_once(user_message):
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=1500,
        ),
    )
    return response.text


def select_strategy(twin, diagnosis, feasibility_output, use_cache=True):
    """
    Run strategy selection for one client. Cached to disk keyed by client_id, so
    reruns of the pipeline while building downstream stages don't re-spend on
    clients already decided.
    """
    client_id = twin["client_id"]
    cache = _load_cache() if use_cache else {}

    if client_id in cache:
        return cache[client_id]

    user_message = build_user_message(twin, diagnosis, feasibility_output)
    permitted = feasibility_output["permitted"]

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        raw = _call_once(user_message)
        try:
            options = parse_response(raw, permitted)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            time.sleep(1)
    else:
        raise RuntimeError(
            f"strategy selection failed for {client_id} after "
            f"{MAX_RETRIES + 1} attempt(s): {last_error}"
        )

    if use_cache:
        cache[client_id] = options
        _save_cache(cache)

    return options


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)