"""
Fit evaluation -- stage three of the promote track.

Receives a client, the products they may be offered, and their allocation
position, and returns a ranked shortlist of what to recommend.

Why a language model here
--------------------------
The judgement is a fit question: given this client's position and these
permitted products, which best serves them. That weighs several things which do
not reduce to a formula -- how well a product's asset class answers the gap, how
much its tax treatment matters at this client's bracket, whether a stated goal
is served, and whether the case is strong enough to be worth making at all.

What it does NOT do
--------------------
It does not decide what may be offered. The option set arrives already narrowed
by the eligibility filter, so an unsuitable, unholdable or unaffordable product
is not something to be avoided here -- it is not present.

It does not decide how much to invest. That is arithmetic over the gap, the
available funding and the product minimum, and it is computed by the proposal
builder. A model asked for an amount would produce a number nobody could
justify.

It does not make the final call. The output is a ranked shortlist, reconciled
against the retention and allocation tracks by the cross-track advisor before
anything reaches a human.

On the absence of a deterministic half
---------------------------------------
The design describes fit evaluation as deterministic checks followed by model
judgement. The deterministic checks turned out to be already done: candidate
generation computes the allocation position and the eligibility filter applies
the hard constraints, and both are upstream. Adding a third deterministic stage
to match the diagram would have been building to the picture rather than to a
need, so fit evaluation is a single model stage.
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai

from promote_pipeline.promote_catalogs import FIT_CAUSES

MODEL = "gemini-3.7-flash"
MAX_OPTIONS = 3
MAX_RETRIES = 1  # one retry on a malformed/out-of-set response, then raise

CACHE_PATH = os.path.join("out", "fit_evaluations.json")


SYSTEM_PROMPT = """\
You are the fit evaluation component of a wealth management system. A client's \
circumstances have changed, a deterministic scan has identified where their \
portfolio sits below its target allocation, and a policy engine has determined \
which products the firm may offer them.

Your task is to rank those products, best first, and say why.

## What is already settled, and not yours to revisit

**Where the client is short.** The gap against target allocation is computed, not \
inferred. Do not recompute it or argue with it.

**What the firm may offer.** Every product in the permitted list has already passed \
suitability, account type and minimum investment checks. You do not need to \
consider whether a product is allowed. It is.

The corollary matters more: **you may only return products from the permitted list.** \
Do not propose a product absent from it, do not suggest alternatives, and do not \
remark that some other product would have suited better. A product missing from the \
list was withheld by a rule, for a reason you are not shown. Treating its absence as \
an oversight would be a mistake.

**How much to invest.** Not your decision. Amounts are computed downstream from the \
gap, the available funding and the product minimum. Rank the products; say nothing \
about size.

## The evidence you receive

**Client profile** -- age, tenure, portfolio value, marginal tax bracket, risk \
capacity, idle cash, and stated goals.

**Allocation position** -- for each asset class: current weight, target weight, the \
gap between them, the dollar value of that gap, and whether the class is under \
target, over target, or within tolerance. A negative gap weight means under target; \
the gap value is the dollars that would close it.

**Permitted products** -- each with its asset class, minimum investment, risk level, \
tax treatment, the accounts it can be held in, and the gap in its asset class.

## The causes a recommendation may address

Every product you rank must be tied to at least one of these. Do not invent others.

**`allocation_gap`**
The client's holdings in this asset class sit below target, outside tolerance. The \
most common and usually the strongest reason: the product moves the portfolio toward \
a target that was set for this client.

**`cash_drag`**
The client is holding cash they do not need. Check the allocation position before \
claiming this: it holds only where the cash class is at or above its target, which \
means there is surplus to deploy. A client sitting below their cash target is not \
suffering cash drag -- they are short of cash, and their idle balance is doing the \
job it is meant to do. An idle cash figure on its own is not evidence of anything.

**`tax_inefficiency`**
The client's marginal bracket makes a tax-advantaged vehicle materially better than \
an otherwise comparable one. This is a comparative claim: it holds when two products \
answer the same gap and one treats income better at this client's bracket. At a low \
bracket it does not hold, and asserting it anyway is wrong.

**`goal_unmatched`**
A stated goal has no vehicle serving it -- an education goal with a near-term date, a \
retirement horizon with nothing matched to it. Use this only where the goal is stated \
in the profile, not where you have inferred one.

**`insufficient_fit`**
Nothing in the permitted set is a good answer for this client. A legitimate finding \
rather than a failure, and the right one when the permitted products are technically \
eligible but none meaningfully serves the client's position. Return it alone, without \
ranking products alongside it.

## How to rank

Work in this order.

**First, does the product answer the gap.** A product in the asset class the client is \
most short of should normally rank first. This is the main consideration and settles \
most cases.

**Then, the secondary characteristics.** Where more than one permitted product \
answers the same gap -- which is common, since eligibility filters by suitability \
rather than by preference -- the ranking turns on tax treatment at this client's \
bracket, on which accounts the product can be held in, and on how the product's \
characteristics sit against the client's horizon and goals. This is where most of \
your actual work is.

**Then, breadth.** Where the client is short in more than one asset class and \
products are permitted for each, the lower ranks are where the remaining gaps get \
addressed. Do not fill all three ranks with near-identical products from one class \
when a second gap is unserved.

Two things to be careful about.

**A tax argument must be specific.** "Tax-efficient" is not a reason. If you claim \
`tax_inefficiency`, name the bracket and say what the treatment does at that bracket. \
If the client's bracket is low enough that the advantage is immaterial, do not claim \
it.

**Do not rank a product highly because it is available.** The permitted list is not a \
set of suggestions. Three products may be eligible and only one worth recommending; \
in that case return one. Returning fewer options than the maximum is a normal outcome, \
not an incomplete answer.

## Output

Return JSON only, no prose before or after:

{
  "ranked": [
    {
      "product_id": "<an identifier from the permitted list>",
      "addresses": ["<one or more of the causes listed above>"],
      "reasoning": "<two or three sentences: why this product for this client, and why at this rank. Name the figures you are relying on.>"
    }
  ]
}

Rank best first. Return at most three products, and fewer where fewer are worth \
recommending. Every `product_id` must appear in the permitted list, and no product may \
appear twice. Where nothing is worth recommending, return a single entry whose \
`addresses` is `["insufficient_fit"]` and whose `product_id` is the closest permitted \
product, with reasoning explaining why it still falls short.
"""


# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------

def build_user_message(twin, eligibility_output):
    """
    Assemble the client's evidence for fit evaluation.

    Three sections, kept distinct so the model does not confuse a fact about
    the client with a decision a policy engine already made.

    Only the permitted list is passed. The rejected list is audit trail --
    retrievable by compliance or human review, but showing it here would invite
    the model to second-guess exclusions the system prompt tells it are not open
    to revision.

    The full gaps map is passed, not only the classes that produced candidates.
    A client over target in one class and short in another is in a different
    position from one merely short, and the ranking may reasonably turn on that.
    The over-target classes are context; they are not candidates and the prompt
    does not invite recommendations against them.

    Client profile is trimmed to the fields the ranking actually turns on. The
    holdings list is left out: the gaps map already expresses the client's
    position by asset class, and the raw list would only invite recomputation of
    a figure the prompt states is settled.
    """
    profile = {
        "age": twin["age"],
        "tenure_years": twin["tenure_years"],
        "portfolio_value": sum(h["value"] for h in twin["holdings"]),
        "tax_bracket": twin["tax_bracket"],
        "risk_capacity": twin["risk_capacity"],
        "idle_cash": twin["idle_cash"],
        "goals": twin["goals"],
    }

    return (
        f"## Client profile\n{json.dumps(profile, indent=2)}\n\n"
        f"## Allocation position\n"
        f"{json.dumps(eligibility_output['gaps'], indent=2)}\n\n"
        f"## Permitted products\n"
        f"{json.dumps(eligibility_output['permitted'], indent=2)}\n\n"
        f"Rank the permitted products for this client."
    )


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
#
# Validated against the permitted set, not the catalog. That is the point of the
# stage division: a product outside the permitted set is not a poor choice to be
# scored down, it is an invalid response, and it is caught here rather than
# reaching a human as a well-argued recommendation for something the firm has
# already ruled out.


def parse_response(text, permitted):
    """
    Parse and validate. A malformed response, a product outside the permitted
    set, a repeated product, or a cause outside the taxonomy is a failure to
    surface, not something to silently repair.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)

    ranked = data.get("ranked", [])
    if not ranked:
        raise ValueError("no products returned")
    if len(ranked) > MAX_OPTIONS:
        raise ValueError(f"{len(ranked)} returned, maximum is {MAX_OPTIONS}")

    permitted_ids = {p["product_id"] for p in permitted}

    seen = set()
    for r in ranked:
        pid = r.get("product_id")
        if pid not in permitted_ids:
            raise ValueError(f"product_id {pid!r} is not in the permitted list")
        if pid in seen:
            raise ValueError(f"product_id {pid!r} appears more than once")
        seen.add(pid)

        causes = r.get("addresses", [])
        if not causes:
            raise ValueError(f"no causes given for {pid}")
        for c in causes:
            if c not in FIT_CAUSES:
                raise ValueError(f"unknown cause {c!r} for {pid}")

        # insufficient_fit means nothing fits, so it cannot sit alongside a
        # ranked recommendation, and it cannot be one of several causes for a
        # product being recommended. Either the set answers the client or it
        # does not.
        if "insufficient_fit" in causes:
            if len(ranked) > 1:
                raise ValueError(
                    "insufficient_fit returned alongside other products"
                )
            if len(causes) > 1:
                raise ValueError(
                    "insufficient_fit returned alongside other causes"
                )

        if not r.get("reasoning"):
            raise ValueError(f"no reasoning given for {pid}")

    return ranked


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
    """
    Build the client once and reuse it.

    load_dotenv() is called here rather than at import time so that reading the
    file is tied to actually needing a key. Importing this module for its prompt
    or its validation -- which the tests do -- then has no side effect on the
    environment.

    It is also safe to call when a runner has already loaded the same file:
    load_dotenv does not overwrite variables that are already set, so an
    entry point that loads .env first still wins.
    """
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
            max_output_tokens=3000,
            # Thinking tokens come out of the same output budget as the response
            # on Gemini 3.x. With six permitted products the JSON was being cut
            # off mid-string, which surfaced as a parse failure rather than as a
            # truncation. The ranking judgement does not need a reasoning pass:
            # the option set arrives already narrowed and the comparison is over
            # a handful of products.
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def evaluate_fit(twin, eligibility_output, use_cache=True):
    """
    Run fit evaluation for one client. Cached to disk keyed by client_id, so
    reruns of the pipeline while building downstream stages don't re-spend on
    clients already evaluated.
    """
    client_id = twin["client_id"]
    cache = _load_cache() if use_cache else {}

    if client_id in cache:
        return cache[client_id]

    permitted = eligibility_output["permitted"]
    if not permitted:
        # Nothing survived eligibility. There is no ranking to make, and calling
        # a model to say so would be spending on a question already answered.
        return []

    user_message = build_user_message(twin, eligibility_output)

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        raw = _call_once(user_message)
        try:
            ranked = parse_response(raw, permitted)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            time.sleep(1)
    else:
        raise RuntimeError(
            f"fit evaluation failed for {client_id} after "
            f"{MAX_RETRIES + 1} attempt(s): {last_error}"
        )

    if use_cache:
        cache[client_id] = ranked
        _save_cache(cache)

    return ranked


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)