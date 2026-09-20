"""
Stage 4 -- sequencing.

Orders the selected proposals for execution.

Priority is not the same as order
----------------------------------
Selection already ranked the proposals: tier first, value second. That answers
which matter most, and where nothing constrains the order it is a perfectly good
execution order too.

What it does not answer is which must happen FIRST. Those come apart in one
case here, and the case is worth the stage.

Where a client is at risk of leaving badly enough that retention took tier 0,
the conversation has to precede anything that moves their money. Restructuring
a portfolio for someone who may not hold it much longer is premature, and the
conversation may change what the right action is -- a client explaining that
they are consolidating for a reason the firm can address is a different client
from the one the tracks modelled.

So the rebalance is not deprioritised. It was selected and it proceeds. It is
gated: it waits for a conversation that may change its terms.

That is the whole of this stage. It does not re-rank, does not drop anything,
and does not revisit selection. A proposal that reaches here is happening; the
only question is when.

What was considered and left out
---------------------------------
Ordering by reversibility -- a phone call before an irreversible sale -- is a
sensible default and produces the same order the rule above already gives on
every client currently modelled. A rule that never changes an outcome is a rule
that cannot be demonstrated, so it is named here rather than implemented.
"""

from cdta.envelope import TIER_RETENTION_DOMINANT

IMMEDIATE = "immediate"
GATED = "gated_on_conversation"


def _key(proposal):
    return f"{proposal['track']}/{proposal['action_id']}"


def _moves_money(proposal):
    """
    Whether carrying this out changes the client's positions.

    The test is the amount, not the cost. A purchase funded entirely from idle
    cash costs nothing and still moves money, and it is the movement that ought
    to wait for the conversation.
    """
    return bool(proposal.get("amount"))


def sequence(selection_output):
    """
    Assign each selected proposal to a step, in execution order.

    Two steps at most. A tier 0 retention proposal, where one was selected,
    goes first and alone; everything that moves money follows it. Where no tier
    0 proposal was selected there is nothing to gate on and the whole set is
    one step, in the order selection produced.
    """
    selected = selection_output["selected"]

    gating = [
        p for p in selected
        if p["tier"] == TIER_RETENTION_DOMINANT and not _moves_money(p)
    ]

    if not gating:
        return {
            "client_id": selection_output["client_id"],
            "sequence": [
                {
                    "step": 1,
                    "when": IMMEDIATE,
                    "proposals": [_key(p) for p in selected],
                    "detail": "no precedence constraint; selection order stands",
                }
            ] if selected else [],
            "gated": False,
        }

    # Everything that is not the gating conversation waits behind it. Proposals
    # that move no money -- a second retention remedy, say -- are not gated:
    # nothing about them is changed by what the conversation turns up.
    gating_keys = {_key(p) for p in gating}
    waiting = [p for p in selected
               if _key(p) not in gating_keys and _moves_money(p)]
    alongside = [p for p in selected
                 if _key(p) not in gating_keys and not _moves_money(p)]

    steps = [{
        "step": 1,
        "when": IMMEDIATE,
        "proposals": [_key(p) for p in gating + alongside],
        "detail": (
            "client is at risk above the dominance threshold; the conversation "
            "precedes anything that moves their money"
        ),
    }]

    if waiting:
        steps.append({
            "step": 2,
            "when": GATED,
            "proposals": [_key(p) for p in waiting],
            "detail": (
                f"held until {', '.join(sorted(gating_keys))} has taken place, "
                f"which may change what is appropriate"
            ),
        })

    return {
        "client_id": selection_output["client_id"],
        "sequence": steps,
        "gated": bool(waiting),
    }