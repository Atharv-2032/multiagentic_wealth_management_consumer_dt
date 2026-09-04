"""
Promote track catalogs.

Three fixed vocabularies the promote track reads: the product catalog, the
funding source categories, and the fit cause taxonomy.

On what is NOT here
--------------------
No firm revenue per product. Benefit characterisation is out of scope for this
version, and its removal takes conflict-of-interest detection out of the
fiduciary filter with it. Recording that as a stated limitation is honest;
carrying a revenue field that nothing checks would imply a safeguard that does
not exist.

No investor qualification field. With a single client tier there is nothing for
it to discriminate.

Every remaining product attribute is read by something downstream. asset_class
matches against the twin's target allocation, min_investment is the eligibility
filter's main constraint, account_types is an operational restriction,
risk_level is checked against the client's risk capacity, and tax_treatment is
context for the fit model rather than an enforced rule.
"""

# ---------------------------------------------------------------------------
# Funding sources
# ---------------------------------------------------------------------------
#
# A fixed vocabulary, not free text, because the cross-track advisor reads this
# field to detect exclusivity. Two proposals drawing on the same idle cash are
# in conflict however well each is justified on its own, and that check is
# arithmetic over this label.

FUNDING_SOURCES = ["new_contribution", "idle_cash", "requires_sale"]


# ---------------------------------------------------------------------------
# Fit causes
# ---------------------------------------------------------------------------
#
# Why a product fits a client who does not hold it. Same role as the retention
# track's cause taxonomy, and the same reason for fixing it: a downstream
# component can only rely on a vocabulary that cannot grow at inference time.

FIT_CAUSES = [
    "allocation_gap",       # holdings sit under target for this asset class
    "cash_drag",            # idle cash uninvested against a stated horizon
    "tax_inefficiency",     # bracket makes a tax-advantaged vehicle materially better
    "concentration_risk",   # position concentrated enough that diversifying helps
    "goal_unmatched",       # a stated goal has no vehicle serving it
    "insufficient_fit",     # nothing in the permitted set is a good fit
]


# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------
#
# asset_class values match the twin's holdings vocabulary, so gap matching is a
# dictionary lookup rather than a mapping table.

RISK_LEVELS = ["conservative", "moderate", "aggressive"]

PRODUCTS = [
    # --- fixed income ------------------------------------------------------
    {
        "id": "municipal_bond_fund",
        "label": "Municipal bond fund",
        "asset_class": "fixed_income",
        "min_investment": 10_000,
        "account_types": ["taxable"],
        "risk_level": "conservative",
        "tax_treatment": "federally_tax_exempt_income",
    },
    {
        "id": "core_bond_fund",
        "label": "Core aggregate bond fund",
        "asset_class": "fixed_income",
        "min_investment": 5_000,
        "account_types": ["taxable", "tax_deferred", "tax_exempt"],
        "risk_level": "conservative",
        "tax_treatment": "ordinary_income",
    },
    {
        "id": "short_duration_treasury",
        "label": "Short duration treasury fund",
        "asset_class": "fixed_income",
        "min_investment": 5_000,
        "account_types": ["taxable", "tax_deferred", "tax_exempt"],
        "risk_level": "conservative",
        "tax_treatment": "state_tax_exempt_income",
    },
    {
        "id": "high_yield_bond_fund",
        "label": "High yield bond fund",
        "asset_class": "fixed_income",
        "min_investment": 25_000,
        "account_types": ["tax_deferred", "tax_exempt"],
        "risk_level": "aggressive",
        "tax_treatment": "ordinary_income",
    },

    # --- US equity ---------------------------------------------------------
    {
        "id": "us_total_market_index",
        "label": "US total market index fund",
        "asset_class": "us_equity",
        "min_investment": 2_500,
        "account_types": ["taxable", "tax_deferred", "tax_exempt"],
        "risk_level": "moderate",
        "tax_treatment": "qualified_dividends",
    },
    {
        "id": "tax_managed_us_equity",
        "label": "Tax-managed US equity fund",
        "asset_class": "us_equity",
        "min_investment": 50_000,
        "account_types": ["taxable"],
        "risk_level": "moderate",
        "tax_treatment": "loss_harvested",
    },
    {
        "id": "us_dividend_income",
        "label": "US dividend income fund",
        "asset_class": "us_equity",
        "min_investment": 10_000,
        "account_types": ["taxable", "tax_deferred", "tax_exempt"],
        "risk_level": "moderate",
        "tax_treatment": "qualified_dividends",
    },

    # --- international equity ---------------------------------------------
    {
        "id": "intl_developed_index",
        "label": "International developed markets index fund",
        "asset_class": "intl_equity",
        "min_investment": 2_500,
        "account_types": ["taxable", "tax_deferred", "tax_exempt"],
        "risk_level": "moderate",
        "tax_treatment": "qualified_dividends",
    },
    {
        "id": "emerging_markets_fund",
        "label": "Emerging markets equity fund",
        "asset_class": "intl_equity",
        "min_investment": 10_000,
        "account_types": ["taxable", "tax_deferred", "tax_exempt"],
        "risk_level": "aggressive",
        "tax_treatment": "qualified_dividends",
    },

    # --- alternatives ------------------------------------------------------
    {
        "id": "real_estate_income_fund",
        "label": "Real estate income fund",
        "asset_class": "alternatives",
        "min_investment": 25_000,
        "account_types": ["taxable", "tax_deferred"],
        "risk_level": "moderate",
        "tax_treatment": "ordinary_income",
    },
    {
        "id": "managed_futures_fund",
        "label": "Managed futures fund",
        "asset_class": "alternatives",
        "min_investment": 100_000,
        "account_types": ["taxable", "tax_deferred"],
        "risk_level": "aggressive",
        "tax_treatment": "mixed_capital_gains",
    },

    # --- cash --------------------------------------------------------------
    {
        "id": "money_market_fund",
        "label": "Government money market fund",
        "asset_class": "cash",
        "min_investment": 1_000,
        "account_types": ["taxable", "tax_deferred", "tax_exempt"],
        "risk_level": "conservative",
        "tax_treatment": "ordinary_income",
    },
]


PRODUCTS_BY_ID = {p["id"]: p for p in PRODUCTS}