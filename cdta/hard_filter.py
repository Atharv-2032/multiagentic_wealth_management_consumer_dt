"""
Stage 1 -- hard filter.

Judges each proposal alone and returns those that survive, with the budget
picture the later stages work against.

What passing means
-------------------
Surviving stage 1 is not a decision to act. It means a proposal is viable
considered by itself, and may go forward to conflict detection and selection,
where it can still lose -- to a conflict, or simply by ranking below something
else. Nothing here compares proposals to each other.

Why this stage should rarely fire
----------------------------------
Every proposal arriving has already passed its own track's filter. The
feasibility engine enforced the firm spend cap per remedy, promote's amount is
bounded by available funding at construction, and the allocation builder gates
itself on cost against benefit.

So this stage is a guard rail in the sense the assertion in feasibility.py is a
guard rail: it catches a track emitting something it should not have. An
elimination here means an upstream component is wrong, not that a client was
close to a limit. Firing on no proposals is the expected outcome.

What it does that no track could
---------------------------------
The combined budget. Each track approved its own costs against a cap it could
see, but nothing checked their sum. Two retention remedies individually within
the firm's allowance can exceed it together, and two proposals each drawing on
idle cash can each be affordable while jointly impossible.

That sum is computed here and carried forward rather than acted on, because
resolving an overrun means choosing which proposal goes -- and choosing is
selection's job, not a filter's.

On the two caps
----------------
The firm cap and the client limit are separate and never summed. A fee
concession is revenue the firm gives up; a sale cost or a realised tax bill is
the client's money. Adding them would produce a number describing nobody's
budget.

The client side has exactly one limit that is not invented: idle cash. A ceiling
on client cost as a fraction of the portfolio would be a second contestable
parameter alongside the rebalancing premium, and would have to be defended.
Idle cash is a real quantity already on the twin, and the constraint is only
that more of it cannot be spent than exists.

On what was removed
--------------------
The design gives this stage four outcomes: eliminate, defer, escalate, pass.
Two are gone.

Escalate has no trigger: nothing in the scoped system produces a proposal that
must reach a human regardless of how it ranks.

Defer is a genuine loss and worth stating plainly. It is what would make the
advisor stateful -- a proposal blocked by a cooldown is early rather than
invalid, and eliminating it means the track regenerates it next cycle. It is
absent because this version runs once per client, so nothing deferred would ever
be reconsidered, and an outcome that can be expressed but never reached is the
kind of field this codebase removes elsewhere.
"""

from cdta.envelope import (
    COST_BORNE_CLIENT,
    COST_BORNE_FIRM,
    FUNDING_IDLE_CASH,
)

PASS = "pass"
ELIMINATE = "eliminate"


def _firm_cost(proposal):
    return proposal["cost"] if proposal["cost_borne_by"] == COST_BORNE_FIRM else 0.0


def _client_cost(proposal):
    return proposal["cost"] if proposal["cost_borne_by"] == COST_BORNE_CLIENT else 0.0


def _idle_cash_draw(proposal):
    """
    How much of the client's idle cash this proposal spends.

    The amount, not the cost. A purchase funded entirely from idle cash costs
    nothing and still consumes the cash, which is the whole reason two
    zero-cost proposals can be in contention.
    """
    if proposal["funding_source"] != FUNDING_IDLE_CASH:
        return 0.0
    return proposal["amount"] or 0.0


def hard_filter(proposals, context):
    """
    Split proposals into passed and eliminated, and compute the budget picture.

    Eliminations are single-proposal breaches only. A proposal that fits on its
    own survives even where the set it belongs to does not: that overrun is
    reported in the budget block for selection to resolve, because resolving it
    means deciding which proposal goes.
    """
    firm_remaining = context.get("firm_spend_remaining")
    idle_cash = context.get("idle_cash", 0.0)

    passed, eliminated = [], []

    for proposal in proposals:
        firm_cost = _firm_cost(proposal)

        # Firm spend. Checked only where the firm's remaining allowance is
        # known -- a client the retention track never ran on has no budget
        # block, and treating an absent figure as zero would eliminate
        # proposals for a cap nobody computed.
        if firm_remaining is not None and firm_cost > firm_remaining:
            eliminated.append({
                "proposal": proposal,
                "outcome": ELIMINATE,
                "reason": "firm_spend_cap",
                "detail": (
                    f"costs ${firm_cost:,.2f} against ${firm_remaining:,.2f} "
                    f"remaining"
                ),
            })
            continue

        # Idle cash. A proposal drawing more cash than the client holds should
        # be unreachable: the promote builder assigns requires_sale precisely
        # when idle cash falls short. Reaching this branch means that builder
        # is wrong.
        draw = _idle_cash_draw(proposal)
        if draw > idle_cash:
            eliminated.append({
                "proposal": proposal,
                "outcome": ELIMINATE,
                "reason": "idle_cash",
                "detail": (
                    f"draws ${draw:,.2f} from ${idle_cash:,.2f} of idle cash"
                ),
            })
            continue

        passed.append(proposal)

    # The combined picture. Computed over survivors, since an eliminated
    # proposal is not going to be selected and should not count against what
    # remains.
    firm_committed = round(sum(_firm_cost(p) for p in passed), 2)
    client_committed = round(sum(_client_cost(p) for p in passed), 2)
    cash_committed = round(sum(_idle_cash_draw(p) for p in passed), 2)

    budget = {
        "firm_spend_cap": context.get("firm_spend_cap"),
        "firm_spend_remaining": firm_remaining,
        "firm_committed_if_all_selected": firm_committed,
        # True where the survivors cannot all be funded together. Not an error
        # and not resolved here: it is the constraint selection works under.
        "firm_overcommitted": (
            firm_remaining is not None and firm_committed > firm_remaining
        ),
        "idle_cash": idle_cash,
        "idle_cash_committed_if_all_selected": cash_committed,
        "idle_cash_overcommitted": cash_committed > idle_cash,
        # Carried for the record rather than capped. The client's cost has no
        # limit in this version beyond what idle cash imposes, and inventing
        # one would mean defending a number.
        "client_cost_if_all_selected": client_committed,
    }

    return {
        "client_id": context["client_id"],
        "passed": passed,
        "eliminated": eliminated,
        "budget": budget,
    }