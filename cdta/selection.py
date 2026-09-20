"""
Stage 3 -- selection.

Decides which proposals actually proceed, given what survived the filter, the
conflicts found, and the budgets.

This is where the final answer comes from. Passing stage 1 meant a proposal was
viable alone; reaching here means it competes.

Lexicographic, not blended
---------------------------
Proposals are ordered by tier first and value second, and the second never
outweighs the first. A tier 1 proposal worth a few hundred dollars proceeds
ahead of a tier 2 proposal worth tens of thousands.

The alternative is a weighted score -- some fraction of client benefit plus some
fraction of firm value -- and it needs an exchange rate between quantities
measured in different units. Any rate would have to be defended without
effectiveness priors the system does not have. Lexicographic ordering sidesteps
it: the trade is never made, so the rate never has to be stated.

This is also how the fiduciary ordering binds. Client-interest proposals sit
above firm-interest ones and the ordering is enforced by code, which produces a
claim a firm can demonstrate rather than one it can only assert was followed.

Budgets are hard
-----------------
A proposal that would breach a budget is not selected. Not deferred, not
partially funded, not accepted with a warning. The firm's spend allowance and
the client's idle cash are limits, and a selection that exceeded them would be
proposing something the firm cannot do.

The two are tracked separately and never summed, for the reason stated in the
envelope: they are different people's money.

On rejections
--------------
Every proposal that does not proceed carries why. A selection that returned only
its winners would answer "what are we doing" and not "why not the other thing",
and the second is the question an audit asks.
"""

from cdta.conflicts import CONFLICT_EXCLUSIVITY, CONFLICT_REDUNDANCY
from cdta.envelope import (
    COST_BORNE_FIRM,
    FUNDING_IDLE_CASH,
    TIER_LABELS,
)

SELECTED = "selected"
REJECTED = "rejected"


def _key(proposal):
    return f"{proposal['track']}/{proposal['action_id']}"


def _order(proposals):
    """
    Tier ascending, then value descending.

    Tier 0 first, then 1, then 2. Within a tier the larger value goes first,
    and value is only ever compared against another value from the same tier --
    which is what keeps firm revenue and client benefit from being weighed
    against each other.
    """
    return sorted(proposals, key=lambda p: (p["tier"], -p["value"]))


def _blocking_conflicts(proposal, selected_keys, conflicts):
    """
    Reasons this proposal cannot join what has already been selected.

    Each conflict type blocks differently, and the difference is not cosmetic:

    redundancy is asymmetric and directional. It blocks the redundant proposal
    only where the superseding one was actually selected. If the rebalance lost
    -- on budget, or on an earlier conflict -- then nothing is closing the gap,
    and the purchase that looked redundant is now the only thing addressing it.
    Dropping it on the strength of a conflict whose other half never happened
    would leave the client with neither.

    semantic conflicts are symmetric. Neither side is named as the one to go,
    so whichever reaches selection first under the tier ordering proceeds, and
    the other is blocked. That is not a judgement about which is better; it is
    the ordering doing the work, which is where the decision is supposed to sit.

    exclusivity is handled by the cash budget rather than here. It is a
    statement that a group of proposals cannot all be funded, and the budget
    check enforces exactly that, proposal by proposal, as the cash is consumed.
    """
    blocks = []
    key = _key(proposal)

    for c in conflicts:
        if key not in c["proposals"]:
            continue

        if c["type"] == CONFLICT_REDUNDANCY:
            if c.get("redundant") != key:
                continue
            superseding = c.get("superseded_by")
            if superseding in selected_keys:
                blocks.append({
                    "reason": CONFLICT_REDUNDANCY,
                    "detail": c["detail"],
                    "with": superseding,
                })
            continue

        if c["type"] == CONFLICT_EXCLUSIVITY:
            continue

        # Semantic, and anything else a future detector adds. Symmetric: block
        # if the other side of the pair is already in.
        other = [k for k in c["proposals"] if k != key]
        for o in other:
            if o in selected_keys:
                blocks.append({
                    "reason": c["type"],
                    "detail": c["detail"],
                    "with": o,
                })

    return blocks


def select(filter_output, conflicts, context):
    """
    Walk the ordered proposals and accept what fits.

    One pass, in tier order. A proposal is accepted if it breaches no budget and
    conflicts with nothing already accepted; otherwise it is rejected with the
    reason recorded.

    A single pass is correct rather than merely simple. Reconsidering a rejected
    proposal later would mean a proposal's fate depending on what came after it,
    and the ordering exists precisely so that it does not.
    """
    budget = filter_output["budget"]

    firm_remaining = budget.get("firm_spend_remaining")
    cash_remaining = budget.get("idle_cash", 0.0)

    selected, rejected = [], []
    selected_keys = set()

    for proposal in _order(filter_output["passed"]):
        key = _key(proposal)

        # --- conflicts with what is already in -------------------------------
        blocks = _blocking_conflicts(proposal, selected_keys, conflicts)
        if blocks:
            rejected.append({
                "proposal": proposal,
                "outcome": REJECTED,
                "reason": blocks[0]["reason"],
                "detail": blocks[0]["detail"],
                "conflicts_with": blocks[0]["with"],
            })
            continue

        # --- firm spend ------------------------------------------------------
        firm_cost = (
            proposal["cost"]
            if proposal["cost_borne_by"] == COST_BORNE_FIRM
            else 0.0
        )
        if firm_remaining is not None and firm_cost > firm_remaining:
            rejected.append({
                "proposal": proposal,
                "outcome": REJECTED,
                "reason": "firm_spend_cap",
                "detail": (
                    f"costs ${firm_cost:,.2f}, ${firm_remaining:,.2f} "
                    f"remaining after earlier selections"
                ),
            })
            continue

        # --- client idle cash ------------------------------------------------
        # Drawn down as proposals are accepted, which is what enforces an
        # exclusivity group: the first proposal in the group takes the cash and
        # the rest find it gone.
        draw = (
            proposal["amount"] or 0.0
            if proposal["funding_source"] == FUNDING_IDLE_CASH
            else 0.0
        )
        if draw > cash_remaining:
            rejected.append({
                "proposal": proposal,
                "outcome": REJECTED,
                "reason": "idle_cash",
                "detail": (
                    f"draws ${draw:,.2f}, ${cash_remaining:,.2f} of idle cash "
                    f"remaining after earlier selections"
                ),
            })
            continue

        # --- accepted --------------------------------------------------------
        if firm_remaining is not None:
            firm_remaining = round(firm_remaining - firm_cost, 2)
        cash_remaining = round(cash_remaining - draw, 2)

        selected.append(proposal)
        selected_keys.add(key)

    return {
        "client_id": context["client_id"],
        "selected": selected,
        "rejected": rejected,
        "tiers_present": sorted({p["tier"] for p in filter_output["passed"]}),
        "tier_labels": {
            str(t): TIER_LABELS[t]
            for t in sorted({p["tier"] for p in filter_output["passed"]})
        },
        "budget_after": {
            "firm_spend_remaining": firm_remaining,
            "idle_cash_remaining": cash_remaining,
        },
    }
