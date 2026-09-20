"""
The proposal envelope.

One shape every track is converted into, so the advisor reasons over fields
rather than over three different output formats.

Why the fields are what they are
---------------------------------
The advisor is deterministic except at one bounded point, and it never reads the
rationale prose a track generates. Every conflict it can detect therefore has to
be expressible as a field here. The envelope was designed backwards from the
conflicts, not forwards from what the tracks happen to emit:

    two proposals drawing on the same money   -> funding_source, cost, amount
    promote buying into a gap allocation
      would close anyway                      -> asset_class, amount
    firm spend exceeding the cap              -> cost, cost_borne_by
    a proposal incoherent given the
      diagnosis                               -> addresses, plus the client
                                                 context alongside

A field that serves none of those is not carried. A conflict that cannot be
written as a field cannot be detected, and pretending otherwise would mean the
advisor asserting something it has no basis for.

On tier
--------
Tier follows the track, and the rule underneath it is that tier 1 changes the
client's financial position while tier 2 protects the relationship.

Allocation and promote are tier 1. Retention is tier 2, including the remedies
that look client-serving: a fee concession is money spent to stop someone
leaving, and the firm would not offer it to a contented client. The motive is
retention, and motive is what the tier records.

Where a portfolio is genuinely wrong, allocation is what catches it and
allocation is tier 1. Retention's review responds to the client being unhappy
about performance; allocation's rebalance responds to the performance problem
itself. The client-interest action is the one that fixes the portfolio.

On what is NOT here
--------------------
No touch cost, and no contact cap anywhere in the advisor. Limiting how often a
client is contacted is a behavioural constraint; every constraint this version
applies is monetary and computable. Carrying a field nothing enforces is the
error this codebase avoids elsewhere -- firm revenue is absent from the product
catalog, and the unimplemented touch budget was removed from the feasibility
engine for the same reason.

No defer_until. Deferral is an outcome the advisor assigns, not a property a
track declares about its own proposal, so it belongs to the stage that assigns
it rather than to the envelope.
"""

TRACKS = ["retention", "promote", "allocation"]

# Tier 1 changes the client's financial position; tier 2 protects the
# relationship. Fixed per track, so there is nothing per-proposal to compute and
# nothing to argue about when reading a decision back.
TIER_BY_TRACK = {
    "allocation": 1,
    "promote": 1,
    "retention": 2,
}

TIER_LABELS = {1: "client_interest", 2: "firm_interest"}

# Who pays. The two are not summable and are never summed: a fee concession is
# the firm's money, while a sale cost or a realised tax bill is the client's.
# They are capped separately for that reason.
COST_BORNE_FIRM = "firm"
COST_BORNE_CLIENT = "client"

# Where the money comes from. Shared with the promote track's vocabulary because
# the exclusivity check is arithmetic over this label, and two vocabularies would
# mean translating before comparing.
FUNDING_IDLE_CASH = "idle_cash"
FUNDING_REQUIRES_SALE = "requires_sale"
FUNDING_NONE = "none"

FUNDING_SOURCES = [FUNDING_IDLE_CASH, FUNDING_REQUIRES_SALE, FUNDING_NONE]


REQUIRED_FIELDS = (
    "client_id",
    "track",
    "tier",
    "action_id",
    "amount",
    "cost",
    "cost_borne_by",
    "funding_source",
    "asset_class",
    "value",
    "addresses",
)


def make_proposal(
    client_id,
    track,
    action_id,
    cost,
    cost_borne_by,
    value,
    addresses,
    amount=None,
    funding_source=FUNDING_NONE,
    asset_class=None,
    source_detail=None,
):
    """
    Build one envelope.

    tier is not a parameter. It follows from the track, and letting a caller
    pass it would allow two proposals from the same track to land in different
    tiers -- which is exactly the situational ordering the lexicographic
    selection exists to rule out.

    source_detail carries whatever the originating track wants preserved for
    display and audit. Nothing in the advisor reads it. It is kept separate from
    the fields above so that the line between what is reasoned over and what is
    merely carried stays visible.
    """
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    if cost_borne_by not in (COST_BORNE_FIRM, COST_BORNE_CLIENT):
        raise ValueError(f"unknown cost_borne_by: {cost_borne_by!r}")
    if funding_source not in FUNDING_SOURCES:
        raise ValueError(f"unknown funding_source: {funding_source!r}")
    if cost is None or cost < 0:
        raise ValueError(f"cost must be a non-negative number, got {cost!r}")
    if value is None or value < 0:
        raise ValueError(f"value must be a non-negative number, got {value!r}")

    # A proposal that moves money must say where the money comes from, and one
    # that moves none must not claim a source. Without this the exclusivity
    # check can be defeated by an omission rather than by a genuine absence of
    # conflict.
    if amount and funding_source == FUNDING_NONE:
        raise ValueError(
            f"{track}/{action_id} moves {amount} but declares no funding source"
        )
    if not amount and funding_source != FUNDING_NONE:
        raise ValueError(
            f"{track}/{action_id} moves nothing but declares "
            f"funding source {funding_source!r}"
        )

    return {
        "client_id": client_id,
        "track": track,
        "tier": TIER_BY_TRACK[track],
        "action_id": action_id,
        "amount": amount,
        "cost": round(float(cost), 2),
        "cost_borne_by": cost_borne_by,
        "funding_source": funding_source,
        "asset_class": asset_class,
        "value": round(float(value), 2),
        "addresses": list(addresses),
        "source_detail": source_detail or {},
    }


def make_context(client_id, diagnosis, firm_spend_cap, firm_spend_remaining,
                 idle_cash, gaps, portfolio_value):
    """
    What the advisor knows about the client, as distinct from any one proposal.

    The diagnosis lives here rather than on each envelope because "fee-sensitive
    and at risk of leaving" is a fact about the client. Copying it onto every
    proposal would invite a detector to treat two copies as two pieces of
    evidence.

    This is also the only place the semantic detector reads client state from,
    which keeps the boundary clear: proposals are what is being decided between,
    context is what the decision is made against.
    """
    return {
        "client_id": client_id,
        "diagnosis": diagnosis or [],
        "firm_spend_cap": firm_spend_cap,
        "firm_spend_remaining": firm_spend_remaining,
        "idle_cash": idle_cash,
        "gaps": gaps or {},
        "portfolio_value": portfolio_value,
    }


def validate(proposal):
    """
    Confirm an envelope carries every field the advisor reads.

    Called on adapter output rather than trusted, because an adapter that
    silently omits a field would produce a proposal the conflict detectors skip
    without saying so -- a missing funding_source reads as no conflict rather
    than as a broken proposal.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in proposal]
    if missing:
        raise ValueError(
            f"proposal from {proposal.get('track')!r} is missing "
            f"{missing}"
        )
    return proposal