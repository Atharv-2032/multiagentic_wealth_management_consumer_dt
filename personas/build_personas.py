"""
Persona generator.

Writes the 25 evaluation personas and the two ledgers they need.

Run once from the project root:

    python -m personas.build_personas

Each persona targets one condition in the evaluation set. The conditions are
listed in CONDITIONS below, alongside what the system is expected to produce.

Why a generator rather than 25 hand-written files
--------------------------------------------------
The target_allocation block on each twin has to match what the allocation track
would select for that client. Hand-writing it means hand-computing a lookup and
keeping 25 copies in step with the model portfolio table. Here it is computed
from the twin by the same functions the allocation track uses, so a change to
the portfolios or the mapping grid propagates on the next run rather than
silently leaving the personas describing a system that no longer exists.

It is also the provenance record. Every figure in every persona is visible here
as an argument, which is what makes "we constructed a client with a 19-point
fixed income shortfall" checkable rather than asserted.
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

from allocation_pipeline.allocation_catalogs import (
    MODEL_PORTFOLIOS,
    TOLERANCE_BAND,
    select_portfolio,
)
from allocation_pipeline.planning_fields import derive_planning_fields


def H(ac, v, at="taxable", g=0):
    return {"asset_class": ac, "value": v, "account_type": at, "unrealized_gain": g}


def flow(amt, date, direction, dest):
    return {"amount": amt, "date": date, "direction": direction,
            "destination_type": dest}


def merchant(amts, dates):
    return [flow(a, d, "outflow", "merchant") for a, d in zip(amts, dates)]


def competitor(amts, dates):
    return [flow(a, d, "outflow", "competitor_institution")
            for a, d in zip(amts, dates)]


def inflow(amts, dates):
    return [flow(a, d, "inflow", "own_account") for a, d in zip(amts, dates)]


DATES = ["2025-10-14", "2025-11-20", "2025-12-19", "2026-02-03", "2026-03-22",
         "2026-05-11", "2026-07-06", "2026-08-18"]

FIELD_ORDER = [
    "client_id", "age", "tenure_years", "dependents", "annual_fee_revenue",
    "holdings", "trailing_return_12m", "benchmark_return_12m", "annual_income",
    "monthly_spending", "investable_assets", "idle_cash", "tax_bracket",
    "stated_risk_tolerance", "risk_capacity", "drawdown_behavior", "goals",
    "target_allocation", "recent_flows",
]


def twin(cid, age, tenure, rev, holdings, tr, br, income, spend, idle, bracket,
         tol, cap, dd, goals, flows, target=None):
    """
    Build one twin.

    target_allocation is computed from the twin unless passed explicitly. The
    exception is a client with no dated goal, who cannot be given a horizon and
    therefore cannot have a portfolio selected -- there the block is supplied,
    standing for a target written at an earlier review before the goal was
    removed.
    """
    t = {"client_id": cid, "age": age, "tenure_years": tenure, "dependents": [],
         "annual_fee_revenue": rev, "holdings": holdings,
         "trailing_return_12m": tr, "benchmark_return_12m": br,
         "annual_income": income, "monthly_spending": spend,
         "investable_assets": sum(h["value"] for h in holdings),
         "idle_cash": idle, "tax_bracket": bracket,
         "stated_risk_tolerance": tol, "risk_capacity": cap,
         "drawdown_behavior": dd, "goals": goals, "recent_flows": flows}

    if target is not None:
        t["target_allocation"] = target
    else:
        pf = derive_planning_fields(t)
        name = select_portfolio(pf["risk_capacity"], pf["horizon_years"])
        t["target_allocation"] = {
            "model_portfolio": name,
            "weights": MODEL_PORTFOLIOS[name],
            "tolerance_band": TOLERANCE_BAND,
            "derived_from": "allocation",
        }

    return {k: t[k] for k in FIELD_ORDER}


D = DATES
RET = lambda a: [{"type": "retirement", "target_age": a}]
AGG = "aggressive"

PERSONAS = {}

# C01 competitor_consolidation, high confidence
PERSONAS["C01-COMPETITOR"] = twin("C01-COMPETITOR", 52, 9.0, 10000,
  [H("us_equity", 300000, "taxable", 80000), H("us_equity", 220000, "tax_deferred"),
   H("intl_equity", 150000, "taxable", 22000), H("fixed_income", 80000, "tax_deferred"),
   H("cash", 50000)],
  0.071, 0.079, 210000, 11000, 50000, 0.35, "moderate", "moderate", "held_through", RET(65),
  merchant([12000, 9000, 11000], [D[0], D[3], D[5]])
  + competitor([60000, 70000, 50000], [D[1], D[4], D[6]]))

# C02 performance_dissatisfaction, high confidence
PERSONAS["C02-PERFORMANCE"] = twin("C02-PERFORMANCE", 44, 7.0, 5800,
  [H("us_equity", 280000, "tax_deferred"), H("us_equity", 120000, "taxable", 30000),
   H("intl_equity", 90000, "tax_deferred"), H("fixed_income", 70000, "tax_deferred"),
   H("cash", 20000)],
  0.021, 0.084, 155000, 7200, 20000, 0.24, "moderate", "moderate", "held_through", RET(65),
  merchant([9000, 7500, 8200], [D[1], D[4], D[6]]))

# C03 fee_sensitivity, high confidence
PERSONAS["C03-FEE"] = twin("C03-FEE", 49, 8.0, 9100,
  [H("us_equity", 300000, "tax_deferred"), H("us_equity", 90000, "taxable", 18000),
   H("intl_equity", 130000, "tax_deferred"), H("fixed_income", 110000, "tax_deferred"),
   H("cash", 20000)],
  0.074, 0.081, 180000, 8300, 20000, 0.32, "moderate", "moderate", "held_through", RET(65),
  merchant([11000, 9500, 10500], [D[1], D[4], D[6]]))

# C04 planned_drawdown
PERSONAS["C04-DRAWDOWN"] = twin("C04-DRAWDOWN", 69, 16.0, 9600,
  [H("us_equity", 300000, "taxable", 110000), H("intl_equity", 110000, "taxable", 30000),
   H("fixed_income", 500000, "taxable", 9000), H("cash", 90000)],
  0.061, 0.066, 92000, 9200, 90000, 0.24, "conservative", "conservative", "held_through",
  [{"type": "income", "target_age": 78}],
  merchant([34000, 29000, 36000, 31000, 33000], [D[0], D[2], D[3], D[5], D[7]]))

# C05 insufficient_evidence
PERSONAS["C05-INSUFFICIENT"] = twin("C05-INSUFFICIENT", 47, 6.5, 4800,
  [H("us_equity", 210000, "taxable", 45000), H("us_equity", 140000, "tax_deferred"),
   H("intl_equity", 65000, "taxable", 8000), H("intl_equity", 40000, "tax_deferred"),
   H("fixed_income", 30000, "taxable", 500), H("cash", 15000)],
  0.078, 0.082, 165000, 7800, 15000, 0.32, "moderate", "moderate", "held_through",
  RET(65) + [{"type": "college", "target_year": 2032}],
  merchant([3500, 4200, 2800, 6500, 3200, 5100, 4400], D[:7]))

# C06 diagnosis uncertain -- ambiguous performance, medium confidence expected
PERSONAS["C06-UNCERTAIN"] = twin("C06-UNCERTAIN", 46, 5.0, 6200,
  [H("us_equity", 260000, "tax_deferred"), H("us_equity", 80000, "taxable", 16000),
   H("intl_equity", 100000, "tax_deferred"), H("fixed_income", 90000, "tax_deferred"),
   H("cash", 25000)],
  0.058, 0.083, 150000, 7000, 25000, 0.24, "moderate", "moderate", "held_through", RET(65),
  merchant([8000, 6500], [D[2], D[5]]) + inflow([1200, 1200], [D[0], D[3]]))

# C07 tenure gate -- fee sensitive but under one year
PERSONAS["C07-TENURE"] = twin("C07-TENURE", 41, 0.6, 8800,
  [H("us_equity", 290000, "tax_deferred"), H("us_equity", 70000, "taxable", 9000),
   H("intl_equity", 120000, "tax_deferred"), H("fixed_income", 95000, "tax_deferred"),
   H("cash", 25000)],
  0.073, 0.080, 170000, 7900, 25000, 0.32, "moderate", "moderate", "held_through", RET(65),
  merchant([9500, 8800], [D[2], D[5]]))

# C08 spend cap blocks the concession -- needs prior spend in the ledger
PERSONAS["C08-SPENDCAP"] = twin("C08-SPENDCAP", 50, 6.0, 8000,
  [H("us_equity", 250000, "tax_deferred"), H("us_equity", 70000, "taxable", 12000),
   H("intl_equity", 110000, "tax_deferred"), H("fixed_income", 85000, "tax_deferred"),
   H("cash", 25000)],
  0.072, 0.079, 160000, 7400, 25000, 0.32, "moderate", "moderate", "held_through", RET(65),
  merchant([8500, 9000], [D[2], D[5]]))

# C09 accept_loss ranked first -- small relationship, nothing worth spending on
PERSONAS["C09-ACCEPTLOSS"] = twin("C09-ACCEPTLOSS", 71, 12.0, 900,
  [H("us_equity", 30000, "taxable", 4000), H("fixed_income", 48000, "taxable", 600),
   H("cash", 12000)],
  0.052, 0.058, 42000, 3100, 12000, 0.12, "conservative", "conservative", "held_through",
  [{"type": "income", "target_age": 80}],
  merchant([4200, 3800, 4500], [D[1], D[4], D[6]]))

# C10 discount chosen inside a budget-narrowed range -- needs prior spend
PERSONAS["C10-NARROWED"] = twin("C10-NARROWED", 48, 7.5, 12000,
  [H("us_equity", 380000, "tax_deferred"), H("us_equity", 110000, "taxable", 21000),
   H("intl_equity", 160000, "tax_deferred"), H("fixed_income", 130000, "tax_deferred"),
   H("cash", 30000)],
  0.075, 0.082, 195000, 9000, 30000, 0.35, "moderate", "moderate", "held_through", RET(65),
  merchant([12500, 11000], [D[2], D[5]]))

# C11 within_tolerance -- portfolio already aligned
PERSONAS["C11-INTOLERANCE"] = twin("C11-INTOLERANCE", 62, 21.0, 5400,
  [H("us_equity", 92000, "tax_deferred"), H("intl_equity", 30000, "tax_deferred"),
   H("fixed_income", 200000, "tax_deferred"), H("fixed_income", 130000, "taxable", 3000),
   H("cash", 148000)],
  0.049, 0.054, 120000, 6500, 148000, 0.22, "conservative", "conservative", "held_through",
  RET(65),
  inflow([2000] * 4, [D[0], D[2], D[4], D[6]]) + merchant([7000, 5500], [D[3], D[7]]))

# C12 cost_exceeds_benefit -- appreciated equity, all taxable
PERSONAS["C12-COSTEXCEEDS"] = twin("C12-COSTEXCEEDS", 67, 15.0, 10200,
  [H("us_equity", 800000, "taxable", 480000), H("intl_equity", 150000, "taxable", 60000),
   H("fixed_income", 200000, "taxable", 4000), H("cash", 50000)],
  0.072, 0.078, 96000, 9500, 50000, 0.24, "conservative", "conservative", "held_through",
  [{"type": "income", "target_age": 74}],
  merchant([32000, 28000, 35000, 30000], [D[0], D[2], D[4], D[6]]))

# C13 zero-cost rebalance -- the sale comes out of sheltered accounts
PERSONAS["C13-ZEROCOST"] = twin("C13-ZEROCOST", 47, 6.5, 6500,
  [H("us_equity", 210000, "taxable", 45000), H("us_equity", 140000, "tax_deferred"),
   H("intl_equity", 65000, "taxable", 8000), H("intl_equity", 40000, "tax_deferred"),
   H("fixed_income", 30000, "taxable", 500), H("cash", 15000)],
  0.079, 0.083, 165000, 7800, 15000, 0.32, "moderate", "moderate", "held_through", RET(65),
  merchant([3500, 4200, 2800], [D[0], D[3], D[6]]))

# C14 rebalance with a real taxable cost, still accepted
PERSONAS["C14-PAIDREBAL"] = twin("C14-PAIDREBAL", 38, 4.0, 7200,
  [H("us_equity", 430000, "taxable", 38000), H("intl_equity", 60000, "taxable", 6000),
   H("fixed_income", 40000, "taxable", 800), H("cash", 70000)],
  0.093, 0.097, 175000, 7600, 70000, 0.24, "moderate", "moderate", "bought_more", RET(65),
  inflow([2200] * 3, [D[0], D[3], D[6]]) + merchant([6400], [D[5]]))

# C15 no dated goal -- planning fails loudly, other tracks still run
PERSONAS["C15-NOGOAL"] = twin("C15-NOGOAL", 55, 11.0, 7400,
  [H("us_equity", 340000, "tax_deferred"), H("us_equity", 90000, "taxable", 20000),
   H("intl_equity", 120000, "tax_deferred"), H("fixed_income", 60000, "tax_deferred"),
   H("cash", 40000)],
  0.070, 0.077, 168000, 7700, 40000, 0.32, "moderate", "moderate", "held_through",
  [{"type": "legacy"}],
  merchant([9800, 8600], [D[2], D[5]]),
  target={"model_portfolio": "balanced",
          "weights": {"us_equity": 0.35, "intl_equity": 0.15, "fixed_income": 0.40,
                      "alternatives": 0.05, "cash": 0.05},
          "tolerance_band": 0.05, "derived_from": "allocation"})

# C16 no candidates -- shortfalls only in classes the band cannot call under
PERSONAS["C16-NOCANDIDATES"] = twin("C16-NOCANDIDATES", 30, 3.0, 2000,
  [H("us_equity", 100000, "taxable", 18000), H("us_equity", 40000, "tax_deferred"),
   H("intl_equity", 50000, "tax_deferred"), H("fixed_income", 24000, "tax_deferred"),
   H("cash", 6000)],
  0.104, 0.099, 130000, 5400, 6000, 0.24, AGG, AGG, "bought_more", RET(65),
  inflow([1800] * 3, [D[0], D[3], D[6]]))

# C17 eligibility rejects on risk level
PERSONAS["C17-RISKREJECT"] = twin("C17-RISKREJECT", 64, 18.0, 6000,
  [H("us_equity", 120000, "tax_deferred"), H("intl_equity", 34000, "tax_deferred"),
   H("fixed_income", 240000, "tax_deferred"), H("cash", 106000)],
  0.046, 0.051, 105000, 5800, 106000, 0.22, "conservative", "conservative", "held_through",
  RET(68),
  inflow([1500, 1500], [D[1], D[5]]) + merchant([5200], [D[3]]))

# C18 eligibility rejects on minimum investment
PERSONAS["C18-MINREJECT"] = twin("C18-MINREJECT", 31, 2.5, 1100,
  [H("us_equity", 68000, "taxable", 9000), H("intl_equity", 26000, "tax_deferred"),
   H("fixed_income", 5000, "tax_deferred"), H("cash", 1000)],
  0.108, 0.101, 98000, 4600, 1000, 0.22, AGG, AGG, "bought_more", RET(65),
  inflow([1400, 1400], [D[1], D[5]]))

# C19 insufficient_fit -- constructed to elicit the cause, and does not
#
# The intent was a client for whom nothing in the catalog is a real answer. It
# is not one: the client is short on fixed income and four conservative bond
# funds are permitted, which do serve that gap. The low tax bracket only makes
# the municipal fund's exemption worthless, which bears on ranking rather than
# on fit.
#
# Kept because the negative result is worth reporting. insufficient_fit may be
# close to unreachable by construction: eligibility already removes the products
# whose presence would make it the correct answer, so whatever survives is
# usually a reasonable fill for its class.
PERSONAS["C19-NOFIT"] = twin("C19-NOFIT", 34, 3.5, 1500,
  [H("us_equity", 84000, "taxable", 11000), H("intl_equity", 30000, "tax_deferred"),
   H("fixed_income", 8000, "tax_deferred"), H("cash", 28000)],
  0.099, 0.096, 112000, 5100, 28000, 0.12, AGG, AGG, "bought_more", RET(65),
  inflow([1600, 1600], [D[2], D[6]]))

# C20 breadth -- short in two asset classes at once
PERSONAS["C20-BREADTH"] = twin("C20-BREADTH", 29, 3.0, 1620,
  [H("us_equity", 86000, "taxable", 14000), H("us_equity", 40000, "tax_deferred"),
   H("fixed_income", 9000, "tax_deferred"), H("cash", 45000)],
  0.112, 0.104, 145000, 6200, 45000, 0.24, AGG, AGG, "bought_more",
  RET(65) + [{"type": "home_purchase", "target_year": 2031}],
  inflow([2500] * 4, [D[0], D[2], D[4], D[6]]) + merchant([3800, 2900], [D[3], D[7]]))

# C21 requires_sale funding -- minimum exceeds idle cash
PERSONAS["C21-REQSALE"] = twin("C21-REQSALE", 45, 9.0, 5600,
  [H("us_equity", 300000, "taxable", 52000), H("intl_equity", 97000, "taxable", 12000),
   H("fixed_income", 30000, "taxable", 400), H("cash", 3000)],
  0.081, 0.086, 158000, 8900, 3000, 0.32, "moderate", "moderate", "held_through", RET(65),
  merchant([7800, 6900], [D[2], D[6]]))

# C22 tier 0 -- retention dominates a larger rebalance
PERSONAS["C22-TIER0"] = twin("C22-TIER0", 54, 10.0, 11000,
  [H("us_equity", 420000, "tax_deferred"), H("us_equity", 160000, "taxable", 30000),
   H("intl_equity", 140000, "tax_deferred"), H("fixed_income", 60000, "tax_deferred"),
   H("cash", 20000)],
  0.068, 0.075, 190000, 9800, 20000, 0.32, "moderate", "moderate", "held_through", RET(65),
  merchant([10000, 9000], [D[0], D[5]])
  + competitor([90000, 80000, 70000], [D[1], D[3], D[6]]))

# C23 allocation_overlap -- promote buys into a class the rebalance already fills
PERSONAS["C23-OVERLAP"] = twin("C23-OVERLAP", 43, 8.0, 7000,
  [H("us_equity", 300000, "tax_deferred"), H("us_equity", 120000, "taxable", 26000),
   H("intl_equity", 110000, "tax_deferred"), H("fixed_income", 25000, "tax_deferred"),
   H("cash", 45000)],
  0.086, 0.090, 172000, 8100, 45000, 0.32, "moderate", "moderate", "held_through", RET(65),
  inflow([1800, 1800], [D[1], D[5]]) + merchant([7400], [D[3]]))

# C24 funding_exclusivity -- two purchases reach for the same idle cash
PERSONAS["C24-EXCLUSIVITY"] = twin("C24-EXCLUSIVITY", 40, 7.0, 6800,
  [H("us_equity", 470000, "taxable", 210000), H("intl_equity", 20000, "taxable", 3000),
   H("fixed_income", 25000, "taxable", 400), H("cash", 20000)],
  0.088, 0.094, 168000, 8200, 20000, 0.32, "moderate", "moderate", "held_through", RET(65),
  merchant([8100, 7300], [D[2], D[6]]))

# C25 semantic contradiction -- selling to a fee-sensitive client who is leaving
PERSONAS["C25-SEMANTIC"] = twin("C25-SEMANTIC", 51, 8.5, 13500,
  [H("us_equity", 360000, "tax_deferred"), H("us_equity", 130000, "taxable", 24000),
   H("intl_equity", 150000, "tax_deferred"), H("fixed_income", 45000, "tax_deferred"),
   H("cash", 60000)],
  0.069, 0.077, 205000, 10200, 60000, 0.35, "moderate", "moderate", "held_through", RET(65),
  merchant([11000, 9500], [D[0], D[5]]) + competitor([85000, 75000], [D[2], D[6]]))


# ---------------------------------------------------------------------------
# What each persona is for
# ---------------------------------------------------------------------------
#
# The condition it targets, and what the system is expected to produce. Five of
# these depend on a language model and cannot be guaranteed by construction --
# the inputs are built to produce the outcome, and what the model actually does
# is the experiment rather than a fixture.

CONDITIONS = {
 "C01-COMPETITOR":   ("competitor_consolidation at high confidence",
                      "diagnosis names it; priced remedies withdrawn; discovery only", "model"),
 "C02-PERFORMANCE":  ("performance_dissatisfaction at high confidence",
                      "diagnosis names it; portfolio_review permitted", "model"),
 "C03-FEE":          ("fee_sensitivity at high confidence",
                      "diagnosis names it; fee_concession permitted", "model"),
 "C04-DRAWDOWN":     ("planned_drawdown -- spending, not churn",
                      "diagnosis names it; no retention spend warranted", "model"),
 "C05-INSUFFICIENT": ("insufficient_evidence",
                      "diagnosis declines to invent a cause; discovery only", "model"),
 "C06-UNCERTAIN":    ("diagnosis below high confidence",
                      "feasibility withdraws both priced remedies", "deterministic"),
 "C07-TENURE":       ("tenure gate",
                      "fee_concession rejected: 0.6y with the firm, 1.0y required", "deterministic"),
 "C08-SPENDCAP":     ("spend cap binds (needs ledger)",
                      "fee_concession rejected: $500 remaining, $800 floor", "deterministic"),
 "C09-ACCEPTLOSS":   ("accept_loss ranked first",
                      "no permitted action worth its cost on a $900 relationship", "model"),
 "C10-NARROWED":     ("discount range narrowed by prior spend (needs ledger)",
                      "range [0.10, 0.12] rather than the default [0.10, 0.15]", "deterministic"),
 "C11-INTOLERANCE":  ("allocation within tolerance",
                      "null: largest deviation 0.3%, band 5%; promote finds nothing", "deterministic"),
 "C12-COSTEXCEEDS":  ("allocation cost exceeds benefit",
                      "null: $56,160 cost against $11,131 benefit", "deterministic"),
 "C13-ZEROCOST":     ("rebalance at zero cost",
                      "sheltered accounts absorb the sale; cost $0, benefit $3,900", "deterministic"),
 "C14-PAIDREBAL":    ("rebalance with a real cost, still accepted",
                      "cost $2,545 against $5,999 benefit", "deterministic"),
 "C15-NOGOAL":       ("no dated goal",
                      "planning raises; allocation recorded as a track failure", "deterministic"),
 "C16-NOCANDIDATES": ("promote finds no candidates",
                      "shortfalls only in classes the band cannot call under", "deterministic"),
 "C17-RISKREJECT":   ("eligibility rejects on risk level",
                      "high_yield_bond_fund rejected: aggressive vs conservative", "deterministic"),
 "C18-MINREJECT":    ("eligibility rejects on minimum investment",
                      "municipal and high_yield rejected: $8,000 fundable", "deterministic"),
 "C19-NOFIT":        ("insufficient_fit",
                      "four bond funds eligible for a client short on fixed income; "
                      "the cause is expected NOT to fire, since the permitted "
                      "products do serve the gap", "model"),
 "C20-BREADTH":      ("ranks spread across two asset classes",
                      "six products permitted in fixed_income and intl_equity", "deterministic"),
 "C21-REQSALE":      ("requires_sale funding",
                      "minimum $10,000 exceeds $3,000 idle cash; $7,000 from a sale", "deterministic"),
 "C22-TIER0":        ("retention dominates above the risk threshold",
                      "efar 0.474; discovery selected before the rebalance", "deterministic"),
 "C23-OVERLAP":      ("allocation_overlap redundancy",
                      "rebalance fills fixed_income; promote purchases dropped", "deterministic"),
 "C24-EXCLUSIVITY":  ("funding_exclusivity",
                      "three purchases draw $60,000 from $20,000; one survives", "deterministic"),
 "C25-SEMANTIC":     ("semantic contradiction",
                      "a product sale to a client diagnosed fee-sensitive and leaving", "model"),
}

# Prior spend. Only two conditions are reachable with a ledger, because with an
# empty one the concession floor (10% of revenue) always sits below the cap
# (15%), so the spend cap cannot bind and the range is always [0.10, 0.15].
LEDGERS = {
    "C08-SPENDCAP": [{
        "client_id": "C08-SPENDCAP", "remedy_id": "portfolio_review",
        "cost": 700.0, "granted_date": "2026-03-10",
        "status": "granted", "track": "retention",
    }],
    "C10-NARROWED": [{
        "client_id": "C10-NARROWED", "remedy_id": "portfolio_review",
        "cost": 360.0, "granted_date": "2026-02-15",
        "status": "granted", "track": "retention",
    }],
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ledger_dir = os.path.join(here, "ledgers")
    os.makedirs(ledger_dir, exist_ok=True)

    for name, t in PERSONAS.items():
        with open(os.path.join(here, f"{name}.json"), "w") as f:
            json.dump(t, f, indent=2)
            f.write("\n")

    for name, entries in LEDGERS.items():
        with open(os.path.join(ledger_dir, f"{name}.json"), "w") as f:
            json.dump(entries, f, indent=2)
            f.write("\n")

    print(f"wrote {len(PERSONAS)} personas and {len(LEDGERS)} ledgers to {here}")

    missing = set(PERSONAS) - set(CONDITIONS)
    if missing:
        print(f"  WARNING: no condition recorded for {sorted(missing)}")


if __name__ == "__main__":
    main()
