"""
Feasibility engine.

Answers one question: what may this client be offered right now?

Not what is appropriate — that is a judgement, made downstream by the strategy
decider. This component answers what is permitted, which is a fact about policy
and spend history and is computed the same way every time.

The division matters for auditability. Asked why a client was not offered a fee
waiver, the system returns a number — the remaining allowance was $181 and the
concession cost $1,785 — rather than a description of how a model weighed things.

It also matters structurally. Because the option set is narrowed before the model
sees it, an infeasible proposal is not an error to catch downstream; it is
unrepresentable. Every constraint that could otherwise be reasoned around in a
prompt is applied here instead.
"""

from datetime import date, datetime

from catalogs import (
    FEE_CONCESSION_MAX,
    FEE_CONCESSION_MIN,
    FEE_CONCESSION_MONTHS,
    REMEDIES,
    fee_concession_cost,
    spend_cap,
)

TODAY = date(2026, 9, 1)

# A diagnosis below high confidence on the primary cause withdraws the priced
# remedies. Spending against an uncertain diagnosis is guessing, and a fee
# concession offered for the wrong reason is worse than no action at all.
#
# competitor_consolidation withdraws them regardless of confidence: it names a
# behaviour, not a motive. Even a confident diagnosis leaves the reason unknown,
# and the appropriate response to not knowing is to find out.
DISCOVERY_ONLY_CAUSES = {"competitor_consolidation", "insufficient_evidence"}


def _months_since(iso_date):
    d = datetime.strptime(iso_date, "%Y-%m-%d").date()
    return (TODAY.year - d.year) * 12 + (TODAY.month - d.month)


def _spent_in_last_year(ledger, client_id):
    total = 0.0
    for e in ledger:
        if e["client_id"] != client_id:
            continue
        if e["status"] == "declined":
            continue
        if _months_since(e["granted_date"]) < 12:
            total += e["cost"]
    return total


def _on_cooldown(ledger, client_id, remedy):
    if remedy["cooldown_months"] == 0:
        return None
    for e in ledger:
        if e["client_id"] != client_id or e["remedy_id"] != remedy["id"]:
            continue
        if e["status"] == "declined":
            continue
        elapsed = _months_since(e["granted_date"])
        if elapsed < remedy["cooldown_months"]:
            return remedy["cooldown_months"] - elapsed
    return None


def feasible_set(twin, causes, ledger):
    """
    Return the remedies this client may be offered, with costs and any bounds.

    Every exclusion carries a reason. The rejected list is not discarded: it is
    the audit record, and it is what makes "why was this not offered" answerable.

    causes is the diagnosis output. Only the primary cause affects feasibility,
    and only by withdrawing the priced remedies when the diagnosis is uncertain.
    """
    revenue = twin["annual_fee_revenue"]
    cap = spend_cap(revenue)
    spent = _spent_in_last_year(ledger, twin["client_id"])
    remaining = round(cap - spent, 2)

    # Guard rail, not a fix. Every past grant in a client's ledger should have
    # been affordable under the cap at the time it was made, so remaining should
    # never go negative. If it does, something upstream — most likely the
    # persona/ledger generator — created a grant that violated the cap when it
    # was issued. That is a data bug, and it should be surfaced loudly here
    # rather than silently clamped and hidden inside a "rejected: spend_cap"
    # entry that looks like ordinary behaviour.
    assert remaining >= 0, (
        f"client {twin['client_id']}: remaining spend cap is {remaining}, "
        f"which means a past grant exceeded the cap when it was issued. "
        f"Fix the ledger/persona generation, not this engine."
    )

    primary = causes[0]
    discovery_only = (
        primary["cause"] in DISCOVERY_ONLY_CAUSES
        or primary["confidence"] != "high"
    )

    permitted, rejected = [], []

    for remedy in REMEDIES:
        rid = remedy["id"]

        if remedy.get("always_feasible"):
            permitted.append({
                "remedy_id": rid,
                "label": remedy["label"],
                "cost": 0.0,
                "touch_cost": remedy["touch_cost"],
            })
            continue

        # Uncertain diagnosis withdraws everything except discovery and the null
        # action. Applied before cost so the logged reason is the real one.
        if discovery_only and rid != "discovery_contact":
            rejected.append({
                "remedy_id": rid,
                "reason": "diagnosis_uncertain",
                "detail": (
                    f"primary cause {primary['cause']} at "
                    f"{primary['confidence']} confidence"
                ),
            })
            continue

        if twin["tenure_years"] < remedy["min_tenure_years"]:
            rejected.append({
                "remedy_id": rid,
                "reason": "tenure",
                "detail": (
                    f"{twin['tenure_years']}y with the firm, "
                    f"{remedy['min_tenure_years']}y required"
                ),
            })
            continue

        months_left = _on_cooldown(ledger, twin["client_id"], remedy)
        if months_left is not None:
            rejected.append({
                "remedy_id": rid,
                "reason": "cooldown",
                "detail": f"{months_left} months remaining",
            })
            continue

        if remedy["cost_type"] == "fixed":
            cost = remedy["cost"]
            if cost > remaining:
                rejected.append({
                    "remedy_id": rid,
                    "reason": "spend_cap",
                    "detail": f"costs ${cost:,.0f}, ${remaining:,.0f} remaining",
                })
                continue
            permitted.append({
                "remedy_id": rid,
                "label": remedy["label"],
                "cost": cost,
                "touch_cost": remedy["touch_cost"],
            })

        else:
            # Variable cost. The engine returns the permitted range; the strategy
            # decider chooses the remedy type and policy fixes the magnitude, so
            # the spend cap is never enforced against a number a model invented.
            floor = fee_concession_cost(revenue, FEE_CONCESSION_MIN)
            if floor > remaining:
                rejected.append({
                    "remedy_id": rid,
                    "reason": "spend_cap",
                    "detail": (
                        f"minimum concession costs ${floor:,.0f}, "
                        f"${remaining:,.0f} remaining"
                    ),
                })
                continue

            affordable_max = min(
                FEE_CONCESSION_MAX,
                remaining / (revenue * FEE_CONCESSION_MONTHS / 12.0),
            )
            permitted.append({
                "remedy_id": rid,
                "label": remedy["label"],
                "cost_range": [
                    floor,
                    fee_concession_cost(revenue, affordable_max),
                ],
                "discount_range": [
                    FEE_CONCESSION_MIN,
                    round(affordable_max, 4),
                ],
                "duration_months": FEE_CONCESSION_MONTHS,
                "touch_cost": remedy["touch_cost"],
            })

    return {
        "client_id": twin["client_id"],
        "permitted": permitted,
        "rejected": rejected,
        "budget": {
            "annual_fee_revenue": revenue,
            "spend_cap": cap,
            "spent_last_12m": round(spent, 2),
            "remaining": remaining,
        },
        # Not a filter — it bounds the combination rather than invalidating any
        # single option, so it travels with the set for the advisor to enforce.
        "touch_budget_remaining": 1,
    }