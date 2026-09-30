"""
Catalogs.

Static reference data for the retention track, plus a ledger schema. These are
data, not logic. Three things here are read elsewhere:

  - REMEDIES and the fee concession constants, by the retention feasibility
    engine, which uses them with the ledger to decide what a given client may
    be offered.
  - CAUSES, by retention diagnosis, as the fixed taxonomy a diagnosis must
    draw from.
  - LEDGER_ENTRY_SCHEMA, as the shape of a spend record.

The PRODUCTS list below is superseded and read by nothing; see the note above
it.

Nothing here makes a judgement. Whether a remedy is PERMITTED is a fact about
policy and spend history; whether it is APPROPRIATE is decided downstream by the
strategy decider. Keeping those apart is what makes an audit question answerable
with a number rather than a description of how a model weighed things.
"""

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

# Retention spend on any one client is capped as a fraction of what that client
# earns the firm annually. Without this the system would spend more retaining a
# client than the relationship is worth, and "accept the loss" would never fire.
#
# A consequence worth stating rather than hiding: small clients get caps below
# the cost of every priced remedy, so they route to the null action automatically.
# That is the cap working, not a gap.
SPEND_CAP_FRACTION = 0.15

# Fee concessions are expressed as a percentage of the client's CURRENT fee, not
# in basis points. A flat basis-point cut is proportionally more generous to the
# client already paying least — 10bp off 0.50 percent is a fifth of their fee,
# off 1.00 percent it is a tenth — which is backwards, since the higher-fee
# client is the one more likely to be fee-sensitive.
FEE_CONCESSION_MIN = 0.10
FEE_CONCESSION_MAX = 0.30
FEE_CONCESSION_MONTHS = 12


# ---------------------------------------------------------------------------
# Remedy catalog
# ---------------------------------------------------------------------------
#
# Each remedy maps to a diagnosed cause. No two causes share a remedy and no
# cause is left without one, which is the collapse/split test applied: if two
# causes always produced the same remedy they should be merged, and if one cause
# produced three they should be split.
#
#   fee_sensitivity           -> fee_concession
#   performance_dissatisfaction -> portfolio_review
#   competitor_consolidation  -> discovery_contact
#   insufficient_evidence     -> discovery_contact
#   planned_drawdown          -> accept_loss
#
# competitor_consolidation routes to discovery rather than to a remedy of its
# own because it names a BEHAVIOUR, not a reason. The twin shows the client
# leaving; it does not show why. Spending against an unknown motive is guessing,
# so the correct action is to acquire the missing information first.

REMEDIES = [
    {
        "id": "fee_concession",
        "label": "Fee concession",
        "addresses": ["fee_sensitivity"],
        "cost_type": "variable",
        # Cost is annual_fee_revenue * discount * (months / 12). The engine
        # returns the permitted discount range; the strategy decider chooses the
        # remedy type and policy fixes the magnitude, so the moral-hazard rule is
        # never enforced against a number a model invented.
        "discount_min": FEE_CONCESSION_MIN,
        "discount_max": FEE_CONCESSION_MAX,
        "duration_months": FEE_CONCESSION_MONTHS,
        "cooldown_months": 24,
        "touch_cost": 1,
        "min_tenure_years": 1.0,
        "notes": (
            "Long cooldown by design. Repeated discounting becomes a permanent "
            "price cut and teaches clients that threatening to leave produces one."
        ),
    },
    {
        "id": "portfolio_review",
        "label": "Portfolio review",
        "addresses": ["performance_dissatisfaction"],
        "cost_type": "fixed",
        "cost": 150.0,          # advisor and analyst time
        "cooldown_months": 6,
        "touch_cost": 1,
        "min_tenure_years": 0.0,
        "notes": "Addresses returns disappointing expectations, not price.",
    },
    {
        "id": "discovery_contact",
        "label": "Discovery contact",
        "addresses": ["competitor_consolidation", "insufficient_evidence"],
        "cost_type": "fixed",
        "cost": 80.0,           # advisor time only
        "cooldown_months": 3,
        "touch_cost": 1,
        "min_tenure_years": 0.0,
        "notes": (
            "The only action whose purpose is reducing uncertainty rather than "
            "changing an outcome. Exists because diagnosis is allowed to admit "
            "ignorance, and because spending against an unknown cause is guessing."
        ),
    },
    {
        "id": "accept_loss",
        "label": "Accept the loss",
        "addresses": ["planned_drawdown", "*"],
        "cost_type": "fixed",
        "cost": 0.0,
        "cooldown_months": 0,
        "touch_cost": 0,
        "min_tenure_years": 0.0,
        "always_feasible": True,
        "notes": (
            "Always in the candidate set. A pipeline that must always propose an "
            "intervention will spend on clients whose expected retained value "
            "does not justify it."
        ),
    },
]

REMEDIES_BY_ID = {r["id"]: r for r in REMEDIES}

# Cause taxonomy, kept beside the remedies because it was derived from them.
CAUSES = [
    "competitor_consolidation",
    "performance_dissatisfaction",
    "fee_sensitivity",
    "planned_drawdown",
    "insufficient_evidence",
]


# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------
#
# SUPERSEDED. Nothing imports PRODUCTS or PRODUCTS_BY_ID from this module. The
# live product catalog is promote_pipeline/promote_catalogs.py, which carries a
# different field set -- min_investment, account_types, risk_level,
# tax_treatment -- and is what candidates.py, eligibility_filter.py and
# proposal_builder.py all read.
#
# Minimums are spread deliberately from zero upward. If every client cleared
# every minimum the eligibility filter would never reject anything and the
# infeasibility guarantee would be untested.

PRODUCTS = [
    # --- no minimum, broad access -----------------------------------------
    {"id": "core_us_index", "label": "Core US equity index fund",
     "asset_class": "us_equity", "minimum": 0, "expense_ratio": 0.0003,
     "liquidity": "daily", "proprietary": False},

    {"id": "core_intl_index", "label": "International equity index fund",
     "asset_class": "intl_equity", "minimum": 0, "expense_ratio": 0.0007,
     "liquidity": "daily", "proprietary": False},

    {"id": "core_bond_index", "label": "Aggregate bond index fund",
     "asset_class": "fixed_income", "minimum": 0, "expense_ratio": 0.0004,
     "liquidity": "daily", "proprietary": False},

    {"id": "short_duration_bond", "label": "Short-duration bond fund",
     "asset_class": "fixed_income", "minimum": 0, "expense_ratio": 0.0012,
     "liquidity": "daily", "proprietary": False},

    {"id": "muni_bond_fund", "label": "Municipal bond fund",
     "asset_class": "fixed_income", "minimum": 0, "expense_ratio": 0.0009,
     "liquidity": "daily", "proprietary": False,
     "notes": "Only worthwhile at higher marginal rates; that is a fit question."},

    # --- modest minimums ---------------------------------------------------
    {"id": "dividend_equity", "label": "Dividend-focused equity fund",
     "asset_class": "us_equity", "minimum": 25_000, "expense_ratio": 0.0035,
     "liquidity": "daily", "proprietary": False},

    {"id": "emerging_markets", "label": "Emerging markets equity fund",
     "asset_class": "intl_equity", "minimum": 25_000, "expense_ratio": 0.0068,
     "liquidity": "daily", "proprietary": False},

    {"id": "house_balanced", "label": "House balanced model portfolio",
     "asset_class": "us_equity", "minimum": 50_000, "expense_ratio": 0.0055,
     "liquidity": "daily", "proprietary": True},

    {"id": "tips_fund", "label": "Inflation-protected securities fund",
     "asset_class": "fixed_income", "minimum": 50_000, "expense_ratio": 0.0011,
     "liquidity": "daily", "proprietary": False},

    # --- separately managed accounts ---------------------------------------
    {"id": "direct_index_us", "label": "Direct indexing, US large cap",
     "asset_class": "us_equity", "minimum": 250_000, "expense_ratio": 0.0025,
     "liquidity": "daily", "proprietary": False,
     "notes": "Holds constituents directly, enabling per-lot loss harvesting. "
              "Worth most in a large taxable account at a high marginal rate."},

    {"id": "sma_muni_ladder", "label": "Municipal bond SMA ladder",
     "asset_class": "fixed_income", "minimum": 250_000, "expense_ratio": 0.0030,
     "liquidity": "monthly", "proprietary": False},

    {"id": "house_tax_managed", "label": "House tax-managed equity SMA",
     "asset_class": "us_equity", "minimum": 500_000, "expense_ratio": 0.0060,
     "liquidity": "monthly", "proprietary": True},

    # --- alternatives ------------------------------------------------------
    {"id": "real_assets_fund", "label": "Real assets fund",
     "asset_class": "alternatives", "minimum": 100_000, "expense_ratio": 0.0090,
     "liquidity": "quarterly", "proprietary": False},

    {"id": "private_credit", "label": "Private credit fund",
     "asset_class": "alternatives", "minimum": 500_000, "expense_ratio": 0.0150,
     "liquidity": "locked_5y", "proprietary": False},

    {"id": "private_equity", "label": "Private equity fund",
     "asset_class": "alternatives", "minimum": 1_000_000, "expense_ratio": 0.0200,
     "liquidity": "locked_7y", "proprietary": False},
]

PRODUCTS_BY_ID = {p["id"]: p for p in PRODUCTS}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
#
# What has already been spent on a client and when. The feasibility engine reads
# it to apply cooldowns and the spend cap; implementation writes to it.
#
# It is deliberately client-level rather than track-level: a rule relating
# retention spend to spend elsewhere cannot be enforced by a component that sees
# only one track.

LEDGER_ENTRY_SCHEMA = {
    "client_id": "str",
    "remedy_id": "str",
    "track": "str",              # retention | promote | allocation
    "cost": "float",             # dollars actually committed
    "magnitude": "float | None",  # discount fraction for variable remedies
    "granted_date": "YYYY-MM-DD",
    "status": "str",             # granted | expired | declined
}


def empty_ledger():
    """A ledger with no history. Most clients start here."""
    return []


def spend_cap(annual_fee_revenue):
    """Total the firm will spend retaining this client over a rolling year."""
    return round(annual_fee_revenue * SPEND_CAP_FRACTION, 2)


def fee_concession_cost(annual_fee_revenue, discount):
    """Cost of a concession at a given discount, over the standard duration."""
    return round(
        annual_fee_revenue * discount * (FEE_CONCESSION_MONTHS / 12.0), 2
    )


if __name__ == "__main__":
    print(f"{len(REMEDIES)} remedies, {len(PRODUCTS)} products")
    print(f"\nspend caps at {SPEND_CAP_FRACTION:.0%} of annual fee revenue:")
    for rev in (775, 1_500, 3_300, 8_000, 21_000, 60_000):
        cap = spend_cap(rev)
        lo = fee_concession_cost(rev, FEE_CONCESSION_MIN)
        affordable = [r["label"] for r in REMEDIES
                      if r["cost_type"] == "fixed" and r["cost"] <= cap]
        if lo <= cap:
            affordable.append("Fee concession")
        print(f"  revenue ${rev:>6,}  cap ${cap:>8,.0f}  "
              f"min concession ${lo:>7,.0f}  -> {len(affordable)} feasible")

    print("\nproduct minimums:")
    for p in sorted(PRODUCTS, key=lambda x: x["minimum"]):
        print(f"  ${p['minimum']:>9,}  {p['label']}")