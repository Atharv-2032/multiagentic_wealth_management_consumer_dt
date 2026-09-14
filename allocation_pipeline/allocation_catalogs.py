"""
Allocation track catalogs.

Three fixed structures: the sanctioned model portfolios, the mapping from
planning fields to one of them, and the parameters governing whether a
correction is worth making.

On stipulated weights
----------------------
The portfolio weights encode an ordering that is not controversial -- equity
share rises with horizon and with risk capacity -- but the specific numbers
describe no real firm's models. They are stipulated for plausibility, in the
same sense as the churn label weights in the data generator, and the paper
should say so. What the architecture demonstrates does not depend on them being
any particular firm's.

On mapping to models rather than optimising
--------------------------------------------
Asset allocation is a solved numerical problem with decades of established
method. A language model asked for percentages produces plausible numbers that
are not optimal under any stated objective, cannot be reproduced, and cannot be
defended; sampling means the same client can receive different allocations on
two runs, which for regulated investment advice is a compliance problem rather
than an inconvenience.

Mapping to pre-approved models is also what firms do. The models are documented
and consistently applied, which is a compliance mechanism as much as an
investment one.
"""

# ---------------------------------------------------------------------------
# Model portfolios
# ---------------------------------------------------------------------------
#
# Asset class keys match the twin's holdings vocabulary, so drift is a
# dictionary comparison rather than a mapping table.

MODEL_PORTFOLIOS = {
    "capital_preservation": {
        "us_equity": 0.15,
        "intl_equity": 0.05,
        "fixed_income": 0.55,
        "alternatives": 0.00,
        "cash": 0.25,
    },
    "conservative_income": {
        "us_equity": 0.25,
        "intl_equity": 0.10,
        "fixed_income": 0.55,
        "alternatives": 0.05,
        "cash": 0.05,
    },
    "balanced": {
        "us_equity": 0.35,
        "intl_equity": 0.15,
        "fixed_income": 0.40,
        "alternatives": 0.05,
        "cash": 0.05,
    },
    "moderate_growth": {
        "us_equity": 0.45,
        "intl_equity": 0.20,
        "fixed_income": 0.25,
        "alternatives": 0.05,
        "cash": 0.05,
    },
    "growth": {
        "us_equity": 0.55,
        "intl_equity": 0.25,
        "fixed_income": 0.12,
        "alternatives": 0.05,
        "cash": 0.03,
    },
}

# One band for every portfolio. A per-portfolio band would mean two clients with
# identical drift receiving different accept/reject outcomes because of which
# cell they landed in. Defensible in practice, but it muddies what the proposal
# builder demonstrates, so drift magnitude does the work instead.
TOLERANCE_BAND = 0.05


# ---------------------------------------------------------------------------
# Planning field vocabularies
# ---------------------------------------------------------------------------
#
# The profiler emits six fields. Two of them select the portfolio; the rest are
# read by the rationale stage, which explains a recommendation and for which
# liquidity need, contribution capacity and constraints are all relevant even
# though none of them changes the allocation.

RISK_CAPACITIES = ["conservative", "moderate", "aggressive"]
HORIZON_BANDS = ["short", "medium", "long"]     # <5y, 5-15y, >15y
LIQUIDITY_NEEDS = ["low", "moderate", "high"]
CONTRIBUTION_CAPACITIES = ["none", "modest", "substantial"]
TAX_POSITIONS = ["low", "moderate", "high"]

HORIZON_SHORT_MAX_YEARS = 5
HORIZON_MEDIUM_MAX_YEARS = 15


def horizon_band(years):
    """Years to the primary goal, bucketed. Boundaries are inclusive below."""
    if years < HORIZON_SHORT_MAX_YEARS:
        return "short"
    if years <= HORIZON_MEDIUM_MAX_YEARS:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
#
# Risk capacity and horizon only. Tax position governs where holdings sit rather
# than what the mix should be, contribution capacity does not change the target,
# and liquidity need is deliberately not applied -- letting it downshift the
# result would turn a lookup into a rule with a second dimension of judgement,
# for no gain the scoped system demonstrates.

PORTFOLIO_BY_PROFILE = {
    ("conservative", "short"):  "capital_preservation",
    ("conservative", "medium"): "conservative_income",
    ("conservative", "long"):   "conservative_income",

    ("moderate", "short"):      "conservative_income",
    ("moderate", "medium"):     "balanced",
    ("moderate", "long"):       "moderate_growth",

    ("aggressive", "short"):    "balanced",
    ("aggressive", "medium"):   "moderate_growth",
    ("aggressive", "long"):     "growth",
}


def select_portfolio(risk_capacity, horizon_years):
    """
    Deterministic lookup. Raises rather than defaulting: a planning field
    outside the vocabulary means the profiler produced something invalid, and
    quietly substituting a portfolio would hide that behind advice.
    """
    if risk_capacity not in RISK_CAPACITIES:
        raise ValueError(f"unknown risk capacity: {risk_capacity!r}")

    band = horizon_band(horizon_years)
    return PORTFOLIO_BY_PROFILE[(risk_capacity, band)]


# ---------------------------------------------------------------------------
# Cost and benefit parameters
# ---------------------------------------------------------------------------
#
# These decide whether a correction happens at all, and they are stated here
# rather than buried in the builder so they can be varied and reported. The
# accept/reject boundary is a function of them, and a sensitivity table across
# their plausible range is a stronger result than asserting any single value.

# Applied to the aggregate taxable gain that correcting drift would realise.
# A flat rate standing for commissions, spreads and tax drag together. The full
# specification of this stage is a constrained optimisation over tax lots; this
# is the threshold approximation, and the substitution is sound because the
# builder's role is defined by what it outputs -- a decision, a delta and a
# cost -- not by how precisely it computes them.
REALISATION_COST_RATE = 0.18

# The annual value of holding a correctly aligned portfolio, as a fraction of
# portfolio value. Drawn from the rebalancing-premium literature, where
# estimates commonly fall between 0.2% and 0.5% a year and are contested: several
# papers argue the premium is near zero or negative in trending markets. This is
# the one genuinely contestable parameter in the track. Cost is exact; benefit
# is a modelling assumption, and it is what decides the accept-or-reject outcome.
ANNUAL_REBALANCE_PREMIUM = 0.003

# Benefit accrues yearly, so a correction held longer is worth more. Multiplying
# by the full horizon would make benefit so large that no tax cost could exceed
# it and the gate would stop rejecting anything. The cap stands in for two real
# effects: benefit collected decades out is worth less today than the raw year
# count implies, and a target revised twice yearly will not survive untouched
# for thirty years anyway.
HORIZON_BENEFIT_CAP_YEARS = 10