"""
Rationale -- stage four of the allocation track.

Generates the advisor-facing explanation of what the track decided and why.

What it does NOT do
--------------------
It does not participate in the decision. The portfolio was selected by a lookup,
and the accept-or-reject outcome was settled by arithmetic over cost and
benefit, both upstream and both deterministic. This stage receives a completed
decision and writes prose about it.

That ordering is the point. This is the only track producing regulated
investment advice rather than a business action, and a model that could shade
the recommendation while explaining it would put sampling variance inside
advice. Here the model can only describe.

Client-facing output is out of scope. The advisor version states figures --
realised gains, estimated cost, the benefit assumption behind the decision --
that a client-facing document would need to frame very differently, and writing
both well is two pieces of work rather than one.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai

MODEL = "gemini-3.7-flash"
MAX_RETRIES = 1

CACHE_PATH = os.path.join("out", "allocation_rationales.json")


SYSTEM_PROMPT = """\
You are the rationale component of a wealth management system. A client's target \
portfolio has been selected and a decision has been made about whether to rebalance \
toward it. Your task is to explain that decision to the advisor who will review it.

## Your role

The decision is made. You are not being asked whether it was right, and you are not \
being asked what else might have been done. A planning profile was derived from the \
client's twin, a target portfolio was selected by a documented mapping from that \
profile, and the cost of correcting the client's drift was weighed against its \
estimated benefit. All of that is settled and arrives as input.

Write the explanation an advisor needs to understand and defend it.

Three things follow from that.

**Do not hedge the decision.** Not "the system suggests this may be appropriate" or \
"you may wish to consider". State what was decided and why.

**Do not introduce alternatives.** Do not observe that a different portfolio might \
suit, that the client could rebalance partially, or that waiting is an option. If an \
alternative were on the table it would have arrived as input.

**Do not invent figures.** Every number in your explanation must appear in the input. \
Do not round, restate approximately, annualise, or compute anything new. Where you \
need a figure you were not given, describe it in words instead.

## The two outcomes

**A rebalance was proposed.** Open by saying what is being done. Then explain the \
drift that motivates it, what the trades achieve, and the planning picture that makes \
the target right for this client.

Explain what the correction costs, and why it costs that. The reason is usually more \
useful than the figure. A correction that realises nothing because the sale comes out \
of a sheltered account is a materially different proposition from one that costs the \
same by coincidence, and an advisor needs to know which they are looking at. Where the \
cost is zero, say why it is zero -- do not simply report that it is.

**No action was taken.** This is a result, not a failure, and it deserves the same \
quality of explanation. There are two distinct reasons and they should not be blurred:

- *Within tolerance* -- the portfolio is close enough to target that correcting it \
would be churn. Nothing is wrong.
- *Cost exceeds benefit* -- the portfolio has drifted materially, and correcting it \
would realise a tax bill larger than the drift is worth. Something is wrong and it is \
deliberately not being fixed. Say so plainly; an advisor reading this needs to know \
the drift exists and why it is being tolerated.

## What you receive

**Planning profile** -- horizon, risk capacity, liquidity need, contribution capacity, \
tax position. The picture the target was selected from.

**Target portfolio** -- the model portfolio selected, and its weights.

**Decision** -- drift per asset class and its magnitude, the action taken or not taken, \
and where a cost was computed: the gain that correcting would realise, the estimated \
cost, and the estimated benefit.

## On the benefit figure

The benefit is an assumption, not a measurement. It is the value of holding a \
correctly aligned portfolio, derived from a stated annual rate applied over a capped \
horizon. Where the decision turned on it -- particularly where cost exceeded benefit -- \
say that the comparison rests on that assumption. Do not present it as a fact about \
this client's expected return, and do not restate the rate or the formula.

## Form

Three to five short paragraphs of continuous prose. No headings, no bullet points, no \
bold. An advisor reads this before a client meeting, not on a dashboard.

Write figures the way a person writes them, not the way they arrive. Money takes a \
currency symbol and thousands separators -- $500,000, not 500000. Drop a trailing \
decimal on a whole amount: $3,900, not $3,900.0, and nothing at all rather than $0.0. \
Weights are percentages -- 70% against a 45% target, not 0.7 against 0.45. A drift \
magnitude is a percentage of the portfolio. Never reproduce a raw decimal from the \
input.

Open with the decision. Not the client's profile, not the task, not a summary of what \
you were given -- the first sentence says what is being done or not done. The planning \
picture belongs in the explanation that follows, as support for the decision rather \
than as a preamble to it.

Write plainly. Name the figures that matter and leave out the ones that do not. Do not \
close with a summary of what you just said.

Return the explanation only. No preamble, no JSON, no commentary about the request.
"""


# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------

def build_user_message(planning_fields, proposal):
    """
    Assemble the completed decision.

    Everything here is settled. There is no permitted set to narrow and no
    option the model chooses between, so the message is simply the decision and
    the reasoning material behind it.

    Asset classes inside the tolerance band are left in the drift block rather
    than filtered out. The advisor may reasonably want to know that a class was
    looked at and found close enough, and the prompt already tells the model
    which figures matter.

    realised_gain_detail is forwarded, not just the total. Without it a zero cost
    can only be reported, never explained: the detail is what shows the sale came
    out of a sheltered holding and left the taxable position untouched, which is
    the fact that distinguishes a genuinely free correction from one that happens
    to be cheap.
    """
    payload = {
        "planning_profile": {
            k: v for k, v in planning_fields.items() if k != "client_id"
        },
        "target_portfolio": {
            "name": proposal["model_portfolio"],
            "weights": proposal["target_weights"],
        },
        "decision": {
            "action": proposal["action"],
            "reason": proposal.get("reason"),
            "detail": proposal.get("detail"),
            "drift_magnitude": proposal["drift_magnitude"],
            "drift_by_class": proposal["drift"],
            "portfolio_value": proposal["portfolio_value"],
        },
    }

    for key in (
        "realised_gain",
        "realised_gain_detail",
        "cost",
        "benefit",
        "trades",
        "net_benefit",
    ):
        if key in proposal:
            payload["decision"][key] = proposal[key]

    return (
        f"{json.dumps(payload, indent=2)}\n\n"
        f"Explain this decision to the reviewing advisor."
    )


# ---------------------------------------------------------------------------
# Number check
# ---------------------------------------------------------------------------
#
# The only failure mode here that matters is a fabricated figure. Prose cannot be
# validated against a vocabulary the way the other model stages are, but a number
# that appears in the explanation and nowhere in the input is checkable, and it
# is the error that would reach a client.
#
# The check is deliberately loose. It accepts any number present in the input at
# any scale, and it ignores small integers, which appear in ordinary prose
# ("three asset classes") far more often than they carry meaning. It is a guard
# against invention, not a proof of correctness.

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Below this, a bare number in prose is far more likely to be a word than a
# figure. Percentages are checked separately, so this does not let a wrong
# weight through.
_TRIVIAL_MAX = 12


def _numbers_in(text):
    found = set()
    for match in _NUMBER.findall(text):
        cleaned = match.replace(",", "")
        try:
            found.add(round(float(cleaned), 2))
        except ValueError:
            continue
    return found


def _permitted_numbers(payload_text):
    """
    Every figure the model was given, plus the forms it may legitimately appear
    in. A weight of 0.45 is correctly written as 45 percent, and a value of
    125000.0 as 125,000 -- neither is an invention.
    """
    permitted = set()
    for value in _numbers_in(payload_text):
        permitted.add(value)
        permitted.add(round(value * 100, 2))     # weight written as a percentage
        permitted.add(round(abs(value), 2))      # sign dropped in prose
        permitted.add(round(abs(value) * 100, 2))
        if value == int(value):
            permitted.add(float(int(value)))
    return permitted


def check_numbers(text, user_message):
    """
    Return the figures in the explanation that were not in the input.

    Empty result means nothing was invented. A non-empty result is not proof of
    an error -- the model may have written a year, or a count -- but it is worth
    surfacing rather than passing on silently.
    """
    permitted = _permitted_numbers(user_message)
    invented = []
    for value in sorted(_numbers_in(text)):
        if abs(value) <= _TRIVIAL_MAX:
            continue
        if value in permitted:
            continue
        invented.append(value)
    return invented


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
            max_output_tokens=2500,
            # This stage explains a decision already made. There is nothing to
            # reason through, and on Gemini 3.x thinking tokens come out of the
            # same output budget as the prose -- which truncated the explanation
            # mid-sentence before this was set.
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def write_rationale(planning_fields, proposal, use_cache=True):
    """
    Generate the advisor-facing explanation.

    Returns the text and any figures the check could not account for. A failed
    check is reported rather than raised: unlike a product outside a permitted
    set, an unaccounted number is a warning about prose and not a structurally
    invalid result, and discarding an otherwise sound explanation over a stray
    year would be the wrong trade.
    """
    client_id = proposal["client_id"]
    cache = _load_cache() if use_cache else {}

    if client_id in cache:
        return cache[client_id]

    user_message = build_user_message(planning_fields, proposal)

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            text = _call_once(user_message).strip()
            if not text:
                raise ValueError("empty rationale returned")
            break
        except ValueError as e:
            last_error = e
            time.sleep(1)
    else:
        raise RuntimeError(
            f"rationale failed for {client_id} after {MAX_RETRIES + 1} "
            f"attempt(s): {last_error}"
        )

    result = {
        "text": text,
        "unaccounted_numbers": check_numbers(text, user_message),
    }

    if use_cache:
        cache[client_id] = result
        _save_cache(cache)

    return result


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)