"""
Proposal builder -- stage three of the allocation track.

The selection stage returns a destination. This stage decides whether to travel,
and how. It computes the drift between current holdings and target, estimates
what correcting it would cost, and either emits a delta with its cost or
concludes that no action is warranted.

Why the null action needs its own stage
----------------------------------------
The selection stage has no knowledge of what anything costs. Left to its output
alone the system would recommend rebalancing whenever drift exists, including
cases where correcting a few percentage points realises a tax bill worth many
times the risk reduction gained. Deciding not to act is a real output here, not
the absence of one, and it is returned with the figures that produced it.

On precision
-------------
The full specification of this stage is a constrained optimisation over tax lots
-- minimising realised tax and transaction costs plus a penalty on remaining
drift, subject to wash-sale windows, holding periods, account cash balances and
position limits. What is implemented is a threshold approximation.

The substitution is sound because the builder's role in the architecture is
defined by what it outputs -- a decision, a delta, and a cost -- not by how
precisely it computes them. Its precision is not what the paper demonstrates.
"""

from collections import defaultdict

from allocation_pipeline.allocation_catalogs import (
    ANNUAL_REBALANCE_PREMIUM,
    HORIZON_BENEFIT_CAP_YEARS,
    MODEL_PORTFOLIOS,
    REALISATION_COST_RATE,
    TOLERANCE_BAND,
)


def _current_weights(twin):
    total = sum(h["value"] for h in twin["holdings"])
    if total == 0:
        return {}, 0.0
    by_class = defaultdict(float)
    for h in twin["holdings"]:
        by_class[h["asset_class"]] += h["value"]
    return dict(by_class), total


def compute_drift(twin, target_weights):
    """
    Per-class drift, and a single magnitude.

    The magnitude is the half-sum of absolute deviations, which reads as the
    fraction of the portfolio sitting in the wrong place. The full sum would
    double-count: every dollar over target in one class is a dollar under target
    in another, so halving gives the share that would actually have to move.
    """
    current_values, total = _current_weights(twin)

    per_class = {}
    absolute_sum = 0.0
    for asset_class, target_weight in target_weights.items():
        current_value = current_values.get(asset_class, 0.0)
        current_weight = current_value / total if total else 0.0
        deviation = current_weight - target_weight
        absolute_sum += abs(deviation)

        per_class[asset_class] = {
            "current_weight": round(current_weight, 4),
            "target_weight": target_weight,
            "deviation": round(deviation, 4),
            "current_value": round(current_value, 2),
            "target_value": round(target_weight * total, 2),
            "delta_value": round((target_weight - current_weight) * total, 2),
        }

    return per_class, round(absolute_sum / 2, 4), total


def estimate_realised_gain(twin, per_class):
    """
    The taxable gain correcting the drift would realise.

    Only over-target classes are sold, and only taxable accounts realise a gain
    at all -- selling inside a tax-deferred account triggers nothing.

    Sales are drawn from sheltered accounts first, and reach a taxable holding
    only once the sheltered holding in that class is exhausted. This is what a
    tax-aware firm does, and it is the difference between a rebalance that costs
    something and one that costs nothing: a client holding the same over-weight
    position across both account types can usually be corrected without
    realising a gain at all.

    Where a taxable holding is reached, the gain is taken in proportion to what
    is sold from it. A client selling a third of a taxable position realises a
    third of its gain, not all of it -- charging the full gain would tax shares
    that never move, and the difference is large enough to flip the
    accept/reject outcome, so the approximation is not a rounding matter.

    Which specific lots are sold is a decision this track does not make.
    Proportional treatment assumes a position is sold evenly across its cost
    basis, which is what a firm without lot-level instruction would broadly do.
    """
    sheltered_value = defaultdict(float)
    taxable_value = defaultdict(float)
    taxable_gain = defaultdict(float)
    for h in twin["holdings"]:
        asset_class = h["asset_class"]
        if h["account_type"] == "taxable":
            taxable_value[asset_class] += h["value"]
            taxable_gain[asset_class] += h.get("unrealized_gain", 0)
        else:
            sheltered_value[asset_class] += h["value"]

    realised = 0.0
    detail = {}
    for asset_class, drift in per_class.items():
        if drift["delta_value"] >= 0:
            continue  # under target or on target: nothing sold here

        to_sell = -drift["delta_value"]

        # Sheltered first. Selling inside a tax-deferred or tax-exempt account
        # realises nothing, so it absorbs as much of the sale as it can hold.
        from_sheltered = min(to_sell, sheltered_value.get(asset_class, 0.0))
        from_taxable = to_sell - from_sheltered

        held_taxable = taxable_value.get(asset_class, 0.0)
        if from_taxable <= 0 or held_taxable <= 0:
            detail[asset_class] = {
                "to_sell": round(to_sell, 2),
                "from_sheltered": round(from_sheltered, 2),
                "from_taxable": 0.0,
                "realised_gain": 0.0,
            }
            continue

        from_taxable = min(from_taxable, held_taxable)
        fraction = from_taxable / held_taxable
        gain = taxable_gain.get(asset_class, 0.0) * fraction

        realised += gain
        detail[asset_class] = {
            "to_sell": round(to_sell, 2),
            "from_sheltered": round(from_sheltered, 2),
            "from_taxable": round(from_taxable, 2),
            "taxable_holding": round(held_taxable, 2),
            "fraction_of_taxable_sold": round(fraction, 4),
            "realised_gain": round(gain, 2),
        }

    return round(realised, 2), detail


def estimate_benefit(portfolio_value, drift_magnitude, horizon_years):
    """
    The monetary value of holding a correctly aligned portfolio.

    Cost is exact; this is not. It is the one genuinely contestable figure in
    the track, and it is what decides the accept-or-reject outcome, so the
    parameters behind it are stated in the catalog rather than buried here and
    a sensitivity table across their plausible range is a stronger result than
    asserting any single value.
    """
    years = min(horizon_years, HORIZON_BENEFIT_CAP_YEARS)
    return round(
        portfolio_value * drift_magnitude * ANNUAL_REBALANCE_PREMIUM * years, 2
    )


def build_proposal(twin, planning_fields, portfolio_name):
    """
    Decide whether to rebalance, and return the decision either way.

    A null action is returned as a structured result carrying its reason and the
    figures behind it, not as None. "Why did this client not receive a
    rebalance" is then answerable with numbers -- drift was inside the band, or
    correcting it cost more than it was worth -- rather than with silence.
    """
    target_weights = MODEL_PORTFOLIOS[portfolio_name]
    per_class, drift_magnitude, portfolio_value = compute_drift(twin, target_weights)

    base = {
        "client_id": twin["client_id"],
        "model_portfolio": portfolio_name,
        "target_weights": target_weights,
        "drift": per_class,
        "drift_magnitude": drift_magnitude,
        "portfolio_value": round(portfolio_value, 2),
    }

    # --- inside the band ---------------------------------------------------
    # No class is far enough from target to be worth correcting. Checked on the
    # per-class deviation rather than the magnitude: a portfolio can accumulate
    # a material magnitude from many small deviations that are each within
    # tolerance, and correcting those is churn.
    worst = max((abs(d["deviation"]) for d in per_class.values()), default=0.0)
    if worst <= TOLERANCE_BAND:
        return {
            **base,
            "action": None,
            "reason": "within_tolerance",
            "detail": (
                f"largest deviation {worst:.1%}, band is "
                f"{TOLERANCE_BAND:.0%}"
            ),
        }

    # --- cost and benefit --------------------------------------------------
    realised_gain, gain_detail = estimate_realised_gain(twin, per_class)
    cost = round(realised_gain * REALISATION_COST_RATE, 2)
    benefit = estimate_benefit(
        portfolio_value, drift_magnitude, planning_fields["horizon_years"]
    )

    economics = {
        "realised_gain": realised_gain,
        "realised_gain_detail": gain_detail,
        "cost": cost,
        "benefit": benefit,
        "horizon_years_applied": min(
            planning_fields["horizon_years"], HORIZON_BENEFIT_CAP_YEARS
        ),
    }

    if cost > benefit:
        return {
            **base,
            **economics,
            "action": None,
            "reason": "cost_exceeds_benefit",
            "detail": (
                f"correcting costs ${cost:,.0f} against ${benefit:,.0f} of "
                f"benefit"
            ),
        }

    # --- proposed ----------------------------------------------------------
    # The delta is expressed per class rather than per holding. Which specific
    # positions move, and in which accounts, is an execution decision this track
    # does not make.
    trades = [
        {
            "asset_class": asset_class,
            "direction": "buy" if d["delta_value"] > 0 else "sell",
            "amount": round(abs(d["delta_value"]), 2),
            "from_weight": d["current_weight"],
            "to_weight": d["target_weight"],
        }
        for asset_class, d in per_class.items()
        if abs(d["deviation"]) > TOLERANCE_BAND
    ]

    return {
        **base,
        **economics,
        "action": "rebalance",
        "trades": sorted(trades, key=lambda t: -t["amount"]),
        "net_benefit": round(benefit - cost, 2),
    }