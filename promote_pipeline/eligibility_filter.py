"""
Eligibility filter -- stage two of the promote track.

A hard filter on what may legally and operationally be offered. Takes the
candidate pairs from the scan and returns those that pass, with the rest
recorded and their rejection reason kept.

This is the same pattern as the retention track's feasibility engine, and it
exists for the same reason. If an ineligible product reaches the fit stage, a
model can rank it highly and produce a well-reasoned recommendation for
something that cannot be sold. Filtering first makes that outcome
unrepresentable rather than merely unlikely.

It also keeps two questions apart that are easy to conflate. Eligibility asks
whether an offer is possible. Fit asks whether it is wise. Only the second is a
judgement, and only the second needs a model.

Not checked here
-----------------
Whether the client already holds the product. The twin records holdings by asset
class, not by instrument, so there is no product identity to compare against.
Adding a held_products field to the twin would make the check possible. With
candidates restricted to under-target classes the omission is mild -- the client
is short of the class either way -- but it is an omission, not a decision.
"""

from promote_catalogs import PRODUCTS_BY_ID, RISK_LEVELS


def _available_accounts(twin):
    """Account types the client actually holds something in."""
    return {h["account_type"] for h in twin["holdings"]}


def _fundable_amount(twin, gap_value):
    """
    What could be put into this asset class.

    Idle cash plus the gap, where the gap stands in for money that could be
    freed by trimming an over-weight class. That second term is an assumption
    rather than a fact -- it presumes a sale the client has not agreed to -- and
    it is here because without it no product requiring a sale could ever pass
    eligibility, which would make the requires_sale funding category unreachable.

    The assumption is bounded: it only ever admits a product for consideration.
    Whether a sale is actually appropriate is decided by the proposal builder,
    which assigns the funding source, and by the advisor who reviews it.
    """
    return twin.get("idle_cash", 0) + max(gap_value, 0)


def _risk_permitted(client_capacity, product_risk):
    """
    Ceiling comparison, not exact match. A client whose capacity is moderate may
    hold conservative products; the capacity is an upper bound on risk, not a
    target to hit.
    """
    if client_capacity not in RISK_LEVELS or product_risk not in RISK_LEVELS:
        return False
    return RISK_LEVELS.index(product_risk) <= RISK_LEVELS.index(client_capacity)


def eligible_set(twin, candidate_output):
    """
    Split candidates into permitted and rejected.

    Every rejection carries a reason and the detail behind it. That record is
    what makes "why was this product not offered" answerable with a fact rather
    than a description of how a model weighed things, and documented
    consideration of reasonable alternatives is close to what the regulatory
    care obligation asks a firm to produce.
    """
    accounts = _available_accounts(twin)
    capacity = twin.get("risk_capacity")

    permitted, rejected = [], []

    for candidate in candidate_output["candidates"]:
        product = PRODUCTS_BY_ID[candidate["product_id"]]
        pid = product["id"]

        # --- risk suitability ---------------------------------------------
        # Checked first because it is the only failure here that is about the
        # client rather than the mechanics of the offer.
        if not _risk_permitted(capacity, product["risk_level"]):
            rejected.append({
                "product_id": pid,
                "reason": "risk_level",
                "detail": (
                    f"{product['risk_level']} product, "
                    f"client capacity is {capacity}"
                ),
            })
            continue

        # --- account type --------------------------------------------------
        usable = accounts & set(product["account_types"])
        if not usable:
            rejected.append({
                "product_id": pid,
                "reason": "account_type",
                "detail": (
                    f"held in {sorted(product['account_types'])}, "
                    f"client has {sorted(accounts)}"
                ),
            })
            continue

        # --- minimum investment -------------------------------------------
        fundable = _fundable_amount(twin, candidate["gap_value"])
        if product["min_investment"] > fundable:
            rejected.append({
                "product_id": pid,
                "reason": "min_investment",
                "detail": (
                    f"minimum ${product['min_investment']:,}, "
                    f"${fundable:,.0f} fundable"
                ),
            })
            continue

        permitted.append({
            "product_id": pid,
            "label": product["label"],
            "asset_class": product["asset_class"],
            "min_investment": product["min_investment"],
            "risk_level": product["risk_level"],
            "tax_treatment": product["tax_treatment"],
            "eligible_accounts": sorted(usable),
            "gap_weight": candidate["gap_weight"],
            "gap_value": candidate["gap_value"],
        })

    return {
        "client_id": twin["client_id"],
        "permitted": permitted,
        "rejected": rejected,
        "gaps": candidate_output["gaps"],
    }