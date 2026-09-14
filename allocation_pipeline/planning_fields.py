"""
Planning fields -- stage one of the allocation track.

Derives the structured planning picture the rest of the track works from:
horizon, risk capacity, liquidity need, contribution capacity and tax position.

Why this is not a model stage
------------------------------
The design placed a language model here, on the grounds that a twin's raw
contents are not planning inputs -- transaction histories, life-event signals,
unstructured notes -- and that turning them into a planning picture is messy
inference over heterogeneous evidence.

That argument holds for the twin the design imagined. It does not hold for the
twin that was built, which carries structured fields throughout and no
unstructured content at all. Every planning field here is a read or a bucketing
of something already recorded: risk capacity is an inferred field the twin
already carries, horizon comes from stated goals, tax position is a bucketing of
the marginal rate, contribution capacity is income less spending.

Putting a model in front of that would not be inference over messy evidence. It
would be a model restating structured fields, with sampling variance attached,
in the one track that produces regulated investment advice. The honest version
is arithmetic, and the limitation to record is that this twin is more structured
than the design assumed.

What this costs
----------------
The design's `constraints` field is not produced. Nothing in the twin records a
client restriction -- no excluded sectors, no concentrated position that cannot
be sold, no held-away assets. Emitting an empty list on every client would
imply a capability that does not exist, so the field is absent and named here
instead.
"""

from datetime import date

from allocation_pipeline.allocation_catalogs import (
    CONTRIBUTION_CAPACITIES,
    LIQUIDITY_NEEDS,
    RISK_CAPACITIES,
    TAX_POSITIONS,
    horizon_band,
)

CURRENT_YEAR = date(2026, 9, 1).year

# Tax bracket cut points. Stated rather than derived: they stand for "is a
# tax-advantaged vehicle worth reaching for", which is a judgement about
# materiality rather than a fact about the rate schedule.
TAX_MODERATE_FLOOR = 0.22
TAX_HIGH_FLOOR = 0.32

# Surplus as a fraction of income. A client saving a fifth of what they earn can
# fund a recommendation from new money; one saving nothing cannot.
CONTRIBUTION_SUBSTANTIAL_FLOOR = 0.20

# Goal proximity driving liquidity need, in years.
LIQUIDITY_HIGH_MAX_YEARS = 3
LIQUIDITY_MODERATE_MAX_YEARS = 8


def _years_to_goal(goal, age):
    """
    Years until a goal comes due.

    Two goal shapes are recorded: an age to reach, and a calendar year. Anything
    else is not dated and cannot contribute a horizon.
    """
    if "target_age" in goal:
        return goal["target_age"] - age
    if "target_year" in goal:
        return goal["target_year"] - CURRENT_YEAR
    return None


def horizon_years(twin):
    """
    Years to the furthest stated goal.

    Furthest rather than nearest, and the choice matters: a client with college
    in six years and retirement in eighteen lands in different portfolios
    depending on which is taken. The furthest goal is used because the target is
    a single allocation for the whole portfolio, and sizing the whole portfolio
    to the nearest goal would hold decades of retirement money at a horizon that
    belongs to one funding need.

    The proper answer is goal-based sub-portfolios, each with its own horizon and
    its own target. That is out of scope, and taking the furthest goal is the
    less wrong of the two available approximations -- but it does mean a
    near-term goal is under-served by the allocation, and that is a limitation
    rather than a design choice.
    """
    age = twin["age"]
    years = [_years_to_goal(g, age) for g in twin.get("goals", [])]
    years = [y for y in years if y is not None]

    if not years:
        raise ValueError(
            f"client {twin['client_id']}: no dated goal, so no horizon. "
            f"A target allocation cannot be selected without one."
        )

    return max(years)


def liquidity_need(twin):
    """
    How much of the portfolio may be called on soon.

    Driven by the nearest dated goal, not the furthest. The furthest goal sets
    the allocation; the nearest sets how much of it needs to stay reachable.

    This field does not affect portfolio selection. Letting it downshift the
    result would add a second dimension of judgement to what is otherwise a
    lookup, for no gain the scoped system demonstrates. It is read by the
    rationale stage, where a near-term funding need is worth explaining even
    when it did not change the recommendation.
    """
    age = twin["age"]
    years = [_years_to_goal(g, age) for g in twin.get("goals", [])]
    years = [y for y in years if y is not None]

    if not years:
        return "low"

    nearest = min(years)
    if nearest <= LIQUIDITY_HIGH_MAX_YEARS:
        return "high"
    if nearest <= LIQUIDITY_MODERATE_MAX_YEARS:
        return "moderate"
    return "low"


def contribution_capacity(twin):
    """
    Whether the client can fund from new money.

    Income less spending, as a fraction of income. Capacity, not intention --
    nothing in the twin records what a client means to contribute, and this
    figure should not be read as money they have offered.
    """
    income = twin.get("annual_income", 0)
    spending = twin.get("monthly_spending", 0) * 12
    surplus = income - spending

    if surplus <= 0 or income <= 0:
        return "none"
    if surplus / income >= CONTRIBUTION_SUBSTANTIAL_FLOOR:
        return "substantial"
    return "modest"


def tax_position(twin):
    """Marginal rate, bucketed."""
    bracket = twin.get("tax_bracket", 0)
    if bracket >= TAX_HIGH_FLOOR:
        return "high"
    if bracket >= TAX_MODERATE_FLOOR:
        return "moderate"
    return "low"


def risk_capacity(twin):
    """
    Read from the twin, not re-derived.

    The twin already carries risk capacity as an inferred field, distinct from
    the stated tolerance on file. Inferring it a second time here would produce
    a second source that can disagree with the first, and nothing in this track
    is better placed to make that inference than whatever produced the twin.
    """
    capacity = twin.get("risk_capacity")
    if capacity not in RISK_CAPACITIES:
        raise ValueError(
            f"client {twin['client_id']}: risk_capacity is {capacity!r}, "
            f"which is not one of {RISK_CAPACITIES}"
        )
    return capacity


def derive_planning_fields(twin):
    """
    The full planning picture. Deterministic: same twin, same fields, every run.
    """
    years = horizon_years(twin)

    fields = {
        "client_id": twin["client_id"],
        "horizon_years": years,
        "horizon_band": horizon_band(years),
        "risk_capacity": risk_capacity(twin),
        "liquidity_need": liquidity_need(twin),
        "contribution_capacity": contribution_capacity(twin),
        "tax_position": tax_position(twin),
    }

    # Vocabulary check. These are consumed downstream by name, so a value
    # outside the stated set is a fault to surface rather than pass on.
    assert fields["liquidity_need"] in LIQUIDITY_NEEDS
    assert fields["contribution_capacity"] in CONTRIBUTION_CAPACITIES
    assert fields["tax_position"] in TAX_POSITIONS

    return fields