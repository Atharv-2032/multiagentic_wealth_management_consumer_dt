"""
Proposal builder -- stage four of the promote track.

Takes the ranked products and turns each into a proposal: how much, funded from
where, at what cost. Deterministic throughout.

Why this is not a model stage
------------------------------
Every quantity here has a defensible answer computed from figures already
established. The amount is bounded by the gap, the available funding and the
product minimum. The funding source follows from whether idle cash covers the
minimum. The cost of a sale is a stated rate applied to the amount sold. A model
asked for any of these would produce a number nobody could justify, and the
ranking it produced upstream is unaffected by them.

On funding source
------------------
The category is what the cross-track advisor reads to detect exclusivity. Two
proposals drawing on the same idle cash are in conflict however well each is
justified alone, and that check is arithmetic over this label rather than a
matter of interpretation. The field therefore has to discriminate, which is the
reason for the preference order below.

Idle cash is preferred, and a sale is proposed only where idle cash cannot meet
the product minimum. Three consequences follow, all intended:

  - Most proposals are partial fills. Gaps are typically far larger than idle
    cash, and closing the gap is not this track's job -- putting available money
    somewhere sensible is.
  - Selling to close an allocation gap is a rebalance, and rebalancing belongs
    to the allocation track. A promote track that routinely proposed sales would
    be duplicating that work, leaving the advisor to reconcile an overlap this
    system created rather than a genuine conflict.
  - Idle cash becomes the scarce resource both tracks may want, which is exactly
    the contention the advisor exists to resolve.

new_contribution is not in the vocabulary. Nothing in the twin indicates a
client intends to add money, and inferring capacity from income less spending
would be proposing to spend money the client has not offered.
"""

from promote_catalogs import PRODUCTS_BY_ID

# Applied to the amount that must be raised by selling. A stated rate standing
# for commissions, spreads and the tax drag of realising gains. Not estimated
# from anything -- the twin carries unrealized_gain per holding, but which lots
# would actually be sold is a decision this track does not make. The figure is
# here so a proposal carries a cost at all, and it is stated rather than
# computed so that nobody mistakes it for a tax calculation.
SALE_COST_RATE = 0.01

FUNDING_IDLE_CASH = "idle_cash"
FUNDING_REQUIRES_SALE = "requires_sale"


def _amount_for(product, gap_value, idle_cash):
    """
    How much to put into this product.

    Bounded above by the gap -- there is no case for investing past the target
    that generated the recommendation. Bounded below by the product minimum,
    which eligibility has already confirmed is reachable.

    Within those bounds, idle cash sets the amount. Returns None where even the
    gap cannot support the minimum, which eligibility should have caught; it is
    checked again because a caller can pass a permitted list from a stale run.
    """
    minimum = product["min_investment"]

    if gap_value < minimum:
        return None

    return min(gap_value, max(idle_cash, minimum))


def build_proposals(twin, ranked, eligibility_output):
    """
    Turn ranked products into proposals.

    Rank order is preserved exactly as fit evaluation produced it. This stage
    adds quantities; it does not reconsider the ranking, and a proposal that
    turns out expensive does not move down.

    Each proposal is costed independently against the client's full idle cash.
    They are alternatives presented to one reviewer, not a basket to be funded
    together, and the advisor is what decides which single proposal proceeds.
    """
    idle_cash = twin.get("idle_cash", 0)
    permitted = {p["product_id"]: p for p in eligibility_output["permitted"]}

    proposals = []
    for rank, entry in enumerate(ranked, start=1):
        pid = entry["product_id"]
        product = PRODUCTS_BY_ID[pid]
        permitted_entry = permitted[pid]
        gap_value = permitted_entry["gap_value"]

        amount = _amount_for(product, gap_value, idle_cash)
        if amount is None:
            continue

        # Idle cash first. A sale is proposed only where idle cash cannot reach
        # the product minimum, which keeps the funding label informative and
        # keeps rebalancing with the track that owns it.
        from_cash = min(amount, idle_cash)
        from_sale = round(amount - from_cash, 2)

        if from_sale > 0:
            funding_source = FUNDING_REQUIRES_SALE
            cost = round(from_sale * SALE_COST_RATE, 2)
        else:
            funding_source = FUNDING_IDLE_CASH
            cost = 0.0

        proposals.append({
            "rank": rank,
            "product_id": pid,
            "label": product["label"],
            "asset_class": product["asset_class"],
            "amount": round(amount, 2),
            "funding_source": funding_source,
            "funding_breakdown": {
                "from_idle_cash": round(from_cash, 2),
                "from_sale": from_sale,
            },
            "cost": cost,
            "account": permitted_entry["eligible_accounts"][0],
            "addresses": entry["addresses"],
            "rationale": entry["reasoning"],
            "gap_context": {
                "gap_value": gap_value,
                "gap_closed": round(amount, 2),
                "gap_remaining": round(gap_value - amount, 2),
            },
        })

    return {
        "client_id": twin["client_id"],
        "proposals": proposals,
        "idle_cash_available": idle_cash,
    }