"""
Candidate generation -- stage one of the promote track.

A material change to the twin triggers a scan of the product catalog. This
component performs the scan and returns the client-product pairs worth
considering.

Trigger detection is out of scope. The runner treats every client it is handed
as one whose twin has just changed materially, which is what the event-driven
path would produce anyway. Nothing downstream depends on which change fired.

What counts as a candidate
---------------------------
A product whose asset class the client holds below target, outside the tolerance
band. Under-target only, and strictly outside the band.

Both restrictions are deliberate. A class the client is over-weight in has no
gap to fill, so adding to it is not a candidate rather than an ineligible one --
recording "we considered adding US equity to a client already twenty-five points
over target" as a documented alternative would be noise, not diligence.

A class inside the tolerance band is, by the definition of the band, close
enough. Generating candidates there would leave the fit model arguing against a
tolerance the system itself set, and would make every class a candidate for
every client, which is the opposite of narrowing the option set before reasoning
over it.

Idle cash sitting uninvested is a real concern and is deliberately not handled
here. It is not an allocation gap; it is a funding question, and it belongs to
the proposal builder and the `cash_drag` cause rather than to a scan over asset
class weights.
"""

from collections import defaultdict

from promote_catalogs import PRODUCTS


def current_weights(twin):
    """
    Aggregate holdings into asset class weights.

    Returns weights and the portfolio total, since downstream needs both -- a
    gap is easier to read as a percentage and easier to act on as a dollar
    figure.
    """
    total = sum(h["value"] for h in twin["holdings"])
    if total == 0:
        return {}, 0.0

    by_class = defaultdict(float)
    for h in twin["holdings"]:
        by_class[h["asset_class"]] += h["value"]

    return {k: v / total for k, v in by_class.items()}, total


def allocation_gaps(twin):
    """
    Gap per asset class against the twin's target allocation.

    The target is derived twin state, not another track's output. Both the
    allocation track and this one read it from the twin; neither reads the
    other. That is what keeps the tracks isolated, which is the property the
    arbitration layer depends on.

    Negative gap means under target.
    """
    target = twin["target_allocation"]
    weights = target["weights"]
    band = target["tolerance_band"]

    current, total = current_weights(twin)

    gaps = {}
    for asset_class, target_weight in weights.items():
        cur = current.get(asset_class, 0.0)
        gap = cur - target_weight

        if gap < -band:
            status = "under"
        elif gap > band:
            status = "over"
        else:
            status = "in_band"

        gaps[asset_class] = {
            "current_weight": round(cur, 4),
            "target_weight": target_weight,
            "gap_weight": round(gap, 4),
            "gap_value": round(-gap * total, 2),  # positive = dollars needed
            "status": status,
        }

    return gaps


def generate_candidates(twin):
    """
    Return candidate client-product pairs.

    Every product whose asset class is under target, outside the band. No
    eligibility checking and no ranking -- those are the next two stages. This
    answers only which pairs are worth looking at.
    """
    gaps = allocation_gaps(twin)

    candidates = []
    for product in PRODUCTS:
        asset_class = product["asset_class"]
        gap = gaps.get(asset_class)

        if gap is None:
            # The product's asset class is not in this client's target
            # allocation at all, so there is no target to be short of.
            continue

        if gap["status"] != "under":
            continue

        candidates.append({
            "product_id": product["id"],
            "label": product["label"],
            "asset_class": asset_class,
            "gap_weight": gap["gap_weight"],
            "gap_value": gap["gap_value"],
        })

    return {
        "client_id": twin["client_id"],
        "candidates": candidates,
        "gaps": gaps,
    }