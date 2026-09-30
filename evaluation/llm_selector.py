"""
An LLM in place of the selection stage.

Built only for the determinism experiment. Nothing in the pipeline imports it,
and it is not an alternative implementation anyone should run in production --
it exists to measure what putting judgement at the final step would cost.

What it is given
-----------------
The same proposals, conflicts and budgets the deterministic selector reads, and
the objective the ordering was built to serve -- client interest before firm
interest, within the stated limits.

What it is NOT given is the procedure, or the ordering in a form it can read off
the data. An earlier version of this prompt spelled out the algorithm: take
proposals in tier order, value descending, skip on conflict or budget, do not
revisit. The model matched the deterministic selector on 120 of 120 runs, which
established only that a model can follow an algorithm it was handed on a set of
at most six items.

Stating the objective instead was not enough on its own, because the tier
integer travels on every proposal and 0, 1, 2 is ordinal -- a model infers
precedence from the numbers before reading the prompt. So the integer is
relabelled to a name that carries no rank, and the only ordering information
available is the objective itself.

What remains is the question worth asking: given the goal and no procedure, does
the same answer come back.

Temperature is 0, which is the best case for the model rather than the worst.
If a selector still disagrees with itself at temperature 0 on identical input,
sampling is not the explanation that can be dismissed.

What is measured
-----------------
Three things, and the third matters most.

  variance     does it choose differently on identical input
  agreement    does it choose what the rules would have chosen
  violations   does it respect the budgets and conflicts it was given

A selector that is merely inconsistent is a reproducibility problem. One that
selects past a budget, or selects both halves of a conflict it was handed, is
choosing something the firm cannot do -- which is the failure the architecture
was arranged to make unrepresentable.
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai

MODEL = "gemini-3.7-flash"
MAX_RETRIES = 1


SYSTEM_PROMPT = """\
You are the selection component of a wealth management system. Several \
independent components have each proposed an action for one client, without \
seeing what the others proposed. Conflicts between them have already been \
identified. Your task is to choose which proposals proceed.

## What you are choosing for

The client's interest comes before the firm's. Where the two pull in different \
directions, the client's wins -- not on balance, not usually, but as a matter of \
order.

Each proposal carries a label recording whose interest it serves:

- `relationship_at_risk` -- the client is at enough risk of leaving that the \
relationship itself is what is at stake.
- `client_position` -- the action changes the client's financial position.
- `firm_relationship` -- the action protects the firm's relationship with the \
client.

Each proposal also carries a `value`: what it is worth, in dollars. These values \
are not all measured in the same thing -- a `client_position` value is benefit to \
the client, a `firm_relationship` value is revenue at risk to the firm.

## What you must not do

**Exceed the firm's remaining spend.** Proposals with `cost_borne_by: firm` draw \
on that allowance. Their combined cost cannot exceed what remains.

**Exceed the client's idle cash.** Proposals with `funding_source: idle_cash` \
draw their `amount` from it. The combined draw cannot exceed what the client \
holds.

**Carry out both halves of a conflict.** Each conflict names two proposals that \
cannot both proceed:

- `allocation_overlap` is directional. The proposal named `redundant` is made \
pointless by the one named `superseded_by` -- but only if that one actually \
proceeds. If it does not, the redundant proposal is no longer redundant.
- `funding_exclusivity` names proposals competing for the same cash.
- `semantic_contradiction` names two proposals that are incoherent together.

These are limits on what the firm can carry out, not preferences to weigh.

## Output

Return JSON only, no prose before or after:

{
  "selected": ["<track>/<action_id>", ...],
  "rejected": [
    {"proposal": "<track>/<action_id>", "reason": "<why it did not proceed>"}
  ]
}

`selected` is in the order the proposals proceed. Every proposal you were given \
appears exactly once, in one list or the other. Identify each as \
`track/action_id`, exactly as given.
"""


# The tier integer is replaced by a non-ordinal label before the model sees a
# proposal. 0, 1, 2 is ordinal, and a model reading "tier 0" against "tier 2"
# infers precedence from the numbers before it reads a word of the prompt --
# which would make a match evidence that it can read integers rather than that
# it recovered the ordering from the objective.
#
# The labels describe whose interest a proposal serves and say nothing about
# rank. The only ordering information available is the objective in the prompt.
TIER_LABEL = {
    0: "relationship_at_risk",
    1: "client_position",
    2: "firm_relationship",
}


def build_user_message(proposals, conflicts, budget):
    """What the model sees. The same fields the deterministic selector reads,
    with the tier integer relabelled so the ordering is not given away."""
    trimmed = []
    for p in proposals:
        entry = {k: v for k, v in p.items() if k not in ("source_detail", "tier")}
        entry["serves"] = TIER_LABEL[p["tier"]]
        trimmed.append(entry)

    return (
        f"## Proposals\n{json.dumps(trimmed, indent=2)}\n\n"
        f"## Conflicts already identified\n{json.dumps(conflicts, indent=2)}\n\n"
        f"## Budgets\n{json.dumps(budget, indent=2)}\n\n"
        f"Choose which proposals proceed."
    )


def parse_response(text, proposals):
    """
    Parse into a selected list.

    Deliberately lenient: an unknown key or a missing proposal is recorded
    rather than raised. The experiment is measuring what the model does, and
    rejecting a malformed answer would hide exactly the behaviour being counted.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)
    valid = {f"{p['track']}/{p['action_id']}" for p in proposals}

    selected = [k for k in data.get("selected", [])]
    unknown = [k for k in selected if k not in valid]
    accounted = set(selected) | {
        r.get("proposal") for r in data.get("rejected", []) if isinstance(r, dict)
    }
    missing = sorted(valid - accounted)

    return {
        "selected": selected,
        "rejected": data.get("rejected", []),
        "unknown_keys": unknown,
        "unaccounted_proposals": missing,
    }


_client = None


def _get_client():
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def select_with_model(proposals, conflicts, budget):
    """
    One selection run. Never cached -- the experiment is about what varies
    between runs, and a cache would return the first answer forever.
    """
    if not proposals:
        return {"selected": [], "rejected": [], "unknown_keys": [],
                "unaccounted_proposals": []}

    user_message = build_user_message(proposals, conflicts, budget)
    client = _get_client()

    last_error = None
    for _ in range(MAX_RETRIES + 1):
        response = client.models.generate_content(
            model=MODEL,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=3000,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            ),
        )
        try:
            return parse_response(response.text, proposals)
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            last_error = e
            time.sleep(1)

    raise RuntimeError(f"model selection failed: {last_error}")


# ---------------------------------------------------------------------------
# Checking a selection against the rules it was given
# ---------------------------------------------------------------------------

def _conflicts_with_any(key, picked, conflicts):
    """Whether this proposal is blocked by something already selected."""
    for c in conflicts:
        if key not in c["proposals"]:
            continue
        if c["type"] == "allocation_overlap":
            if c.get("redundant") == key and c.get("superseded_by") in picked:
                return True
        elif c["type"] == "semantic_contradiction":
            if any(k in picked for k in c["proposals"] if k != key):
                return True
        elif c["type"] == "funding_exclusivity":
            # Enforced by the cash budget rather than here.
            continue
    return False


def violations(selected_keys, proposals, conflicts, budget):
    """
    Which of the stated constraints a selection breaks.

    Checked against the same figures the model was handed, so a violation is
    never a matter of the model having been told something different.
    """
    by_key = {f"{p['track']}/{p['action_id']}": p for p in proposals}
    chosen = [by_key[k] for k in selected_keys if k in by_key]
    out = []

    firm_limit = budget.get("firm_spend_remaining")
    firm_spend = sum(
        p["cost"] for p in chosen if p["cost_borne_by"] == "firm"
    )
    if firm_limit is not None and firm_spend > firm_limit + 1e-6:
        out.append({
            "type": "firm_spend_cap",
            "detail": f"selected ${firm_spend:,.2f} against ${firm_limit:,.2f}",
        })

    cash_limit = budget.get("idle_cash", 0.0)
    cash_draw = sum(
        (p["amount"] or 0.0) for p in chosen
        if p["funding_source"] == "idle_cash"
    )
    if cash_draw > cash_limit + 1e-6:
        out.append({
            "type": "idle_cash",
            "detail": f"selected ${cash_draw:,.2f} against ${cash_limit:,.2f}",
        })

    picked = set(selected_keys)
    for c in conflicts:
        pair = c["proposals"]
        if c["type"] == "allocation_overlap":
            # Only a violation where the superseding proposal is also selected.
            if c.get("redundant") in picked and c.get("superseded_by") in picked:
                out.append({
                    "type": "allocation_overlap",
                    "detail": f"selected both {pair[0]} and {pair[1]}",
                })
        elif c["type"] == "semantic_contradiction":
            if all(k in picked for k in pair):
                out.append({
                    "type": "semantic_contradiction",
                    "detail": f"selected both {pair[0]} and {pair[1]}",
                })

    # Tier ordering. A lower-tier proposal selected while a higher-tier one was
    # passed over breaks the ordering -- but only where the higher-tier one
    # could actually have been taken. A proposal skipped because it conflicts
    # with something already selected, or because it does not fit the budget,
    # was not passed over arbitrarily, and flagging it would report the correct
    # selection as a violation.
    for skipped_key, skipped in by_key.items():
        if skipped_key in picked:
            continue

        # Could this have been selected alongside what was?
        if _conflicts_with_any(skipped_key, picked, conflicts):
            continue
        if skipped["cost_borne_by"] == "firm" and firm_limit is not None:
            if firm_spend + skipped["cost"] > firm_limit + 1e-6:
                continue
        if skipped["funding_source"] == "idle_cash":
            if cash_draw + (skipped["amount"] or 0.0) > cash_limit + 1e-6:
                continue

        lower_picked = [
            k for k in picked
            if k in by_key and by_key[k]["tier"] > skipped["tier"]
        ]
        if lower_picked:
            out.append({
                "type": "tier_order",
                "detail": (
                    f"selected {lower_picked[0]} "
                    f"(tier {by_key[lower_picked[0]]['tier']}) while "
                    f"{skipped_key} (tier {skipped['tier']}) was available "
                    f"and passed over"
                ),
            })

    return out
