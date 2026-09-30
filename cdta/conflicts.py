"""
Stage 2a -- deterministic conflict detection.

Finds the conflicts that are arithmetic over structured fields, and emits them
as constraints. Resolves nothing.

Why detection and resolution are separate
------------------------------------------
A detector that also decided which proposal survives would be making the
selection decision early, in a component whose job is to notice things. Keeping
them apart means every conflict is visible as a record before anything acts on
it, and selection works from a stated set of constraints rather than from
whatever a detector happened to do.

It also matters for what comes next. The language model in stage 2b asserts
conflicts but may never clear one found here, and that rule only means something
if the conflicts found here are recorded rather than already applied.

Why these two detectors and not more
-------------------------------------
Both are arithmetic. A model adds nothing to either and could get them wrong --
one is a sum against a balance, the other is a string comparison on asset class.
Anything requiring an arithmetic answer belongs in this file; anything requiring
a reading of the diagnosis belongs in the next one.

The two conflicts have different shapes, which is worth keeping in mind when
reading the constraints they emit:

  exclusivity   two proposals want the same money. Either may proceed; not
                both. Symmetric -- neither is wrong.
  redundancy    one proposal does what another already does. Asymmetric --
                the smaller is made pointless by the larger, and which is
                which is determined, not chosen.
"""

from cdta.envelope import FUNDING_IDLE_CASH

CONFLICT_EXCLUSIVITY = "funding_exclusivity"
CONFLICT_REDUNDANCY = "allocation_overlap"


def _key(proposal):
    """How a proposal is named inside a constraint."""
    return f"{proposal['track']}/{proposal['action_id']}"


def detect_funding_exclusivity(passed, budget):
    """
    Proposals drawing on the same idle cash.

    Raised only where the draws exceed what the client holds. Two proposals
    against a balance that covers both are not in conflict, however similar they
    look -- the client can simply do both.

    Only idle cash is checked. A rebalance funded by selling draws on the
    over-weight positions being trimmed, which is a different pot, and a
    retention remedy moves no client money at all. So this conflict is always
    between promote proposals, or between a promote proposal and anything else
    that reaches for cash.

    The constraint names every proposal in the group rather than singling one
    out. Which of them proceeds is a ranking question, and ranking happens in
    selection.
    """
    drawers = [
        p for p in passed
        if p["funding_source"] == FUNDING_IDLE_CASH and p["amount"]
    ]
    if len(drawers) < 2:
        return []

    idle_cash = budget.get("idle_cash", 0.0)
    total = sum(p["amount"] for p in drawers)
    if total <= idle_cash:
        return []

    return [{
        "type": CONFLICT_EXCLUSIVITY,
        "proposals": [_key(p) for p in drawers],
        "fields": ["funding_source", "amount"],
        "detail": (
            f"{len(drawers)} proposals draw ${total:,.0f} from "
            f"${idle_cash:,.0f} of idle cash"
        ),
        # Any subset fitting inside the balance is permissible. Selection picks
        # which, under the tier ordering.
        "limit": idle_cash,
    }]


def detect_allocation_overlap(passed):
    """
    A promote purchase into an asset class an allocation rebalance already
    fills.

    The rebalance moves the class to target. A purchase into the same class is
    then buying toward a gap that is already being closed, and doing both would
    overshoot -- the client ends up over-weight in what they were short of.

    The rule is a class comparison and nothing more. Allocation's purchase is
    necessarily the larger, since it closes the whole gap while a promote
    proposal closes a slice of it, so there is no threshold to set and no
    partial-overlap arithmetic to do. Same class, rebalance present, promote
    redundant.

    Asymmetric, unlike exclusivity. The rebalance is not harmed by the purchase
    existing; the purchase is made pointless by the rebalance. The constraint
    records which is which rather than leaving selection to infer it from
    ranking, because ranking would give the same answer for the wrong reason --
    allocation's value figure is computed over the whole portfolio's drift and
    will exceed a single gap's value on almost every client.

    Where allocation emitted nothing -- either inside tolerance or rejected on
    cost -- there is no rebalance, the gap stays open, and a promote purchase
    into it is the only thing addressing it. Absence is the signal.
    """
    rebalances = [p for p in passed if p["track"] == "allocation"]
    if not rebalances:
        return []

    # A rebalance's asset_class is the largest class it buys into; the full
    # trade list travels in source_detail. Both are read, since a rebalance
    # buying into several classes makes each of them redundant to fill.
    filled = set()
    for r in rebalances:
        if r["asset_class"]:
            filled.add(r["asset_class"])
        for trade in r.get("source_detail", {}).get("trades", []):
            if trade.get("direction") == "buy":
                filled.add(trade["asset_class"])

    conflicts = []
    for p in passed:
        if p["track"] != "promote":
            continue
        if p["asset_class"] not in filled:
            continue

        rebalance = rebalances[0]
        conflicts.append({
            "type": CONFLICT_REDUNDANCY,
            "proposals": [_key(rebalance), _key(p)],
            "fields": ["asset_class", "amount"],
            "detail": (
                f"{_key(rebalance)} already buys into {p['asset_class']}; "
                f"{_key(p)} would add ${p['amount']:,.0f} to the same class"
            ),
            # Named rather than left to ranking: the redundancy is a fact about
            # what the proposals do, not about which scores higher.
            "redundant": _key(p),
            "superseded_by": _key(rebalance),
        })

    return conflicts


def detect_conflicts(filter_output):
    """
    Run both detectors and return their constraints.

    Order is fixed and the detectors are independent -- neither reads the
    other's output. A proposal can appear in both an exclusivity group and a
    redundancy pair, and selection has to satisfy both.
    """
    passed = filter_output["passed"]
    budget = filter_output["budget"]

    conflicts = (
        detect_funding_exclusivity(passed, budget)
        + detect_allocation_overlap(passed)
    )

    return {
        "client_id": filter_output["client_id"],
        "conflicts": conflicts,
        "deterministic_count": len(conflicts),
    }