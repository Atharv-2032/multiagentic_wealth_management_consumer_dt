"""
Track adapters.

Convert each track's native output into the common envelope. One function per
track, each reading only what that track emits.

Why adapters rather than changing the tracks
---------------------------------------------
The tracks are isolated by design, and none knows the advisor exists. Making
each emit the envelope directly would put knowledge of the arbitration layer
inside components that are supposed to be unaware of it, and would mean a change
to the envelope rippling into three pipelines.

Keeping the conversion here also puts every assumption about how a track's
output maps onto the envelope in one file, where the mapping can be read as one
thing rather than reconstructed from three.

What the adapters do NOT do
----------------------------
They do not rank, filter, or drop proposals on merit. An adapter converts, and
converts everything convertible. The one exception is a proposal representing
no action at all, which carries nothing to arbitrate: allocation's null outcome
and retention's accept_loss contribute no envelope, because the advisor has
nothing to decide about doing nothing.
"""

from allocation_pipeline.allocation_catalogs import (
    ANNUAL_REBALANCE_PREMIUM,
    HORIZON_BENEFIT_CAP_YEARS,
)
from cdta.envelope import (
    COST_BORNE_CLIENT,
    COST_BORNE_FIRM,
    FUNDING_IDLE_CASH,
    FUNDING_NONE,
    FUNDING_REQUIRES_SALE,
    make_context,
    make_proposal,
    validate,
)

# Retention proposals that represent doing nothing. They carry no cost, no
# money movement and no action, so there is nothing for the advisor to weigh
# them against.
NULL_REMEDIES = {"accept_loss"}


def expected_fraction_at_risk(detector_output):
    """
    The share of the client's assets the detector expects to lose.

    Probability times severity. The detector emits both separately because
    neither alone ranks clients usefully -- a near-certain loss of a sliver is
    not the same problem as a coin-flip on most of an account -- and their
    product is the figure that combines them.

    Returns None where the detector did not run, which leaves retention at its
    default tier rather than promoting it on a figure nobody computed.
    """
    if not detector_output:
        return None
    probability = detector_output.get("probability")
    severity = detector_output.get("severity")
    if probability is None or severity is None:
        return None
    return round(probability * severity, 4)


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def from_retention(twin, detector_output, diagnosis, feasibility_output,
                   strategy_options):
    """
    Convert the retention track's ranked remedies.

    Value is expected revenue at risk -- probability times severity times the
    annual fee revenue the client generates. That is what the firm stands to
    lose, which is the right quantity for a tier 2 proposal: retention protects
    the relationship, and the relationship's worth is what it earns.

    The same figure is attached to every remedy for a client, because it
    measures the client's risk rather than any particular remedy's effect. How
    much a given remedy reduces that risk is an effectiveness question, and the
    system has no effectiveness priors -- inventing a per-remedy multiplier
    would be asserting exactly what cannot be shown on synthetic data.

    Cost is borne by the firm. A fee concession is revenue foregone and a
    discovery call is advisor time; both are the firm's money, which is why the
    firm cap and the client cap are kept apart.
    """
    revenue_at_risk = detector_output.get("revenue_at_risk", 0.0)
    permitted = {p["remedy_id"]: p for p in feasibility_output["permitted"]}

    # Decides whether retention takes tier 0. Passed to every proposal from this
    # track because it is a property of the client, so all of them move together
    # -- a client cannot be at risk for one remedy and not another.
    at_risk = expected_fraction_at_risk(detector_output)

    proposals = []
    for option in strategy_options:
        remedy_id = option["remedy_id"]

        if remedy_id in NULL_REMEDIES:
            continue

        permitted_entry = permitted[remedy_id]

        # A fixed-cost remedy carries its cost directly. The fee concession
        # carries a range instead, and the strategy stage chose a discount
        # inside it -- so the cost is recomputed from that choice rather than
        # taken from either end of the range.
        if "cost" in permitted_entry:
            cost = permitted_entry["cost"]
        else:
            discount = option["discount"]
            months = permitted_entry["duration_months"]
            cost = twin["annual_fee_revenue"] * discount * (months / 12.0)

        proposals.append(validate(make_proposal(
            client_id=twin["client_id"],
            track="retention",
            action_id=remedy_id,
            cost=cost,
            cost_borne_by=COST_BORNE_FIRM,
            value=revenue_at_risk,
            addresses=option.get("addresses", []),
            # Retention moves no client money. A concession reduces what the
            # firm charges; it does not draw on the client's cash or require a
            # sale, so it can never contend for the same money as promote or
            # allocation.
            amount=None,
            funding_source=FUNDING_NONE,
            asset_class=None,
            expected_fraction_at_risk=at_risk,
            source_detail={
                "label": permitted_entry.get("label"),
                "discount": option.get("discount"),
                "reasoning": option.get("reasoning"),
                "churn_probability": detector_output.get("probability"),
                "severity": detector_output.get("severity"),
            },
        )))

    return proposals


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------

def promote_value(gap_value, horizon_years):
    """
    What filling part of an allocation gap is worth.

    Deliberately the same formula and the same parameter the allocation track
    uses for its own benefit figure, applied to the slice of the gap this
    purchase closes rather than to the whole portfolio's drift.

    Reusing the parameter is the point. Promote has no native value figure, and
    inventing one -- a projected return uplift, say -- would add a second
    contestable assumption on top of the one the system already carries. One
    stated parameter, varied and reported once, is a defensible position; two
    are an argument.

    This is the value of closing that part of the gap, not the value of the
    product. The distinction matters if anyone reads it as a claim about the
    instrument.
    """
    years = min(horizon_years, HORIZON_BENEFIT_CAP_YEARS)
    return round(gap_value * ANNUAL_REBALANCE_PREMIUM * years, 2)


def from_promote(twin, promote_proposals, horizon_years):
    """
    Convert the promote track's proposals.

    Cost is borne by the client: a sale cost is charged against their money, not
    the firm's. Where funding comes from idle cash the cost is zero, but the
    funding source still matters -- it is what the exclusivity check reads.

    amount is the purchase, and the exclusivity check compares it against idle
    cash rather than the cost. Two proposals each costing nothing can still be
    in conflict if they both spend the same cash.
    """
    proposals = []
    for p in promote_proposals.get("proposals", []):
        gap_value = p["gap_context"]["gap_value"]

        proposals.append(validate(make_proposal(
            client_id=twin["client_id"],
            track="promote",
            action_id=p["product_id"],
            cost=p["cost"],
            cost_borne_by=COST_BORNE_CLIENT,
            value=promote_value(p["gap_context"]["gap_closed"], horizon_years),
            addresses=p.get("addresses", []),
            amount=p["amount"],
            funding_source=p["funding_source"],
            asset_class=p["asset_class"],
            source_detail={
                "label": p.get("label"),
                "rank": p.get("rank"),
                "account": p.get("account"),
                "funding_breakdown": p.get("funding_breakdown"),
                "gap_context": p.get("gap_context"),
                "rationale": p.get("rationale"),
            },
        )))

    return proposals


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

def from_allocation(twin, allocation_proposal):
    """
    Convert the allocation track's rebalance, if there is one.

    A null outcome produces no envelope. The advisor reads that absence as the
    gap staying open, which is what it needs to know: if nothing is closing the
    gap, a promote purchase into the same class is not redundant. Emitting a
    "we did nothing" proposal would mean the advisor ranking an action against a
    non-action, which is not a comparison.

    One envelope for the whole rebalance rather than one per trade. The trades
    are a single decision -- selling equity to buy fixed income is not two
    proposals that could be accepted separately -- and splitting them would let
    the advisor approve half a rebalance.

    asset_class is the class being bought into, since that is what the overlap
    check compares against promote. Where a rebalance buys into several, the
    largest purchase stands for it; the full set stays in source_detail.
    """
    if allocation_proposal.get("action") != "rebalance":
        return []

    trades = allocation_proposal.get("trades", [])
    buys = [t for t in trades if t["direction"] == "buy"]
    sells = [t for t in trades if t["direction"] == "sell"]

    largest_buy = max(buys, key=lambda t: t["amount"], default=None)
    total_buy = sum(t["amount"] for t in buys)

    # A rebalance is funded by selling, unless it is buying nothing at all.
    # Unlike promote it does not draw on idle cash: the money comes from the
    # over-weight classes being trimmed, which is what makes a rebalance and a
    # product purchase fundable at the same time rather than in contention.
    funding_source = FUNDING_REQUIRES_SALE if sells else FUNDING_NONE
    if not total_buy:
        funding_source = FUNDING_NONE

    return [validate(make_proposal(
        client_id=twin["client_id"],
        track="allocation",
        action_id="rebalance",
        cost=allocation_proposal["cost"],
        cost_borne_by=COST_BORNE_CLIENT,
        value=allocation_proposal["benefit"],
        addresses=["allocation_drift"],
        amount=total_buy or None,
        funding_source=funding_source,
        asset_class=largest_buy["asset_class"] if largest_buy else None,
        source_detail={
            "model_portfolio": allocation_proposal.get("model_portfolio"),
            "drift_magnitude": allocation_proposal.get("drift_magnitude"),
            "trades": trades,
            "realised_gain": allocation_proposal.get("realised_gain"),
            "net_benefit": allocation_proposal.get("net_benefit"),
        },
    ))]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def build_context(twin, diagnosis, feasibility_output, candidate_output,
                  detector_output=None):
    """
    Assemble what the advisor knows about the client.

    Pulled from the tracks' own outputs rather than recomputed, so the advisor
    reasons against the same figures the tracks reasoned against. Recomputing
    the spend cap here, for instance, would create a second source that can
    disagree with the feasibility engine's.

    Any of the three sources may be absent -- a client the retention track never
    ran on has no diagnosis and no budget block, and one inside tolerance on
    every class has no gap map.
    """
    budget = (feasibility_output or {}).get("budget", {})

    return make_context(
        client_id=twin["client_id"],
        diagnosis=diagnosis,
        firm_spend_cap=budget.get("spend_cap"),
        firm_spend_remaining=budget.get("remaining"),
        idle_cash=twin.get("idle_cash", 0),
        gaps=(candidate_output or {}).get("gaps", {}),
        portfolio_value=sum(h["value"] for h in twin["holdings"]),
        expected_fraction_at_risk=expected_fraction_at_risk(detector_output),
    )