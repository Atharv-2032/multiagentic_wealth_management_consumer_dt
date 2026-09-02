"""
Base client generator.

Produces everything in the twin schema EXCEPT recent_flows and the churn label.
Those depend on the label rule and are generated in a later step.

Generation order matters. Each step reads what came before, so that nothing is
sampled independently once something upstream constrains it:

    age -> dependents -> goals -> income -> spending -> portfolio size
        -> asset allocation -> account split -> unrealised gains
        -> benchmark return -> actual return -> fee revenue -> risk fields
"""

import random
from datetime import date

# ---------------------------------------------------------------- parameters

AGE_MIN, AGE_MAX = 25, 75

# Log-normal portfolio sizing. Chosen so the bulk of clients land between
# roughly $150k and $1.5M, with a thin tail above.
PORTFOLIO_LOG_MEAN = 13.6
PORTFOLIO_LOG_SD = 1.05
PORTFOLIO_MIN, PORTFOLIO_MAX = 50_000, 10_000_000

# Market returns for the trailing 12-month window. Fixed globally so that every
# client's benchmark is computed against the same market, which is what makes
# relative performance comparable across clients.
MARKET_RETURNS = {
    "us_equity": 0.110,
    "intl_equity": 0.070,
    "fixed_income": 0.020,
    "alternatives": 0.050,
    "cash": 0.040,
}

# Tiered fee schedule. Breakpoints matter: a flat percentage would make fee load
# identical for every client and remove it as a predictive signal.
FEE_TIERS = [
    (250_000, 0.0110),
    (1_000_000, 0.0085),
    (3_000_000, 0.0065),
    (float("inf"), 0.0050),
]

TAX_BRACKETS = [0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

ASSET_CLASSES = ["us_equity", "intl_equity", "fixed_income", "alternatives", "cash"]

CURRENT_YEAR = 2026


# ---------------------------------------------------------------- helpers

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _round_to(x, step):
    return int(round(x / step) * step)


# ---------------------------------------------------------------- steps

def gen_age(rng):
    return rng.randint(AGE_MIN, AGE_MAX)


def gen_dependents(rng, age):
    """
    Dependents must be plausible for the client's age. A 34-year-old cannot have
    a 28-year-old child. Parents are assumed to have been at least 22 at birth,
    which caps the eldest dependent's age at (age - 22).
    """
    max_child_age = age - 22
    if max_child_age < 0:
        return []

    if age < 30:
        n = rng.choices([0, 1], weights=[0.75, 0.25])[0]
    elif age < 45:
        n = rng.choices([0, 1, 2, 3], weights=[0.20, 0.35, 0.35, 0.10])[0]
    elif age < 60:
        n = rng.choices([0, 1, 2, 3], weights=[0.10, 0.40, 0.40, 0.10])[0]
    else:
        # Older clients usually have adult children who are no longer dependents.
        n = rng.choices([0, 1, 2], weights=[0.70, 0.20, 0.10])[0]

    # Dependent age is drawn from a window that shifts upward with the client's
    # age. A uniform draw across the whole permitted range would make a 70-year-old
    # as likely to have a 3-year-old as a 24-year-old, which in turn gives older
    # clients near-dated education goals and shortens their horizon unrealistically.
    upper = min(max_child_age, 25)
    lower = max(0, age - 52)
    if lower > upper:
        lower = upper
    return [{"age": rng.randint(lower, upper)} for _ in range(n)]


def gen_goals(rng, age, dependents, portfolio_value, monthly_spending):
    """
    Goals derive from age and dependents. Never sampled independently, or you get
    a 72-year-old saving for a retirement that already happened, or a parent of
    two teenagers with no education goal at all.
    """
    goals = []

    # Education: one goal per dependent still under 18, dated to when they turn 18.
    for d in dependents:
        if d["age"] < 18:
            years_out = 18 - d["age"]
            goals.append({
                "label": "education",
                "target_amount": _round_to(rng.uniform(60_000, 280_000), 5_000),
                "target_date": f"{CURRENT_YEAR + years_out}-08-01",
            })

    # Retirement: only if they have not already retired.
    retirement_age = rng.randint(62, 70)
    if age < retirement_age:
        years_out = retirement_age - age
        # Target derives from the lifestyle being maintained, not from current
        # holdings. Deriving it from the portfolio would conflate what a client
        # needs with what they happen to have, so an under-saved client would
        # look on track. Retirement spending is taken at 80 percent of current,
        # multiplied by 25 to 30 years of drawdown.
        annual_retirement_spend = monthly_spending * 12 * 0.8
        target = annual_retirement_spend * rng.uniform(25, 30)
        goals.append({
            "label": "retirement",
            "target_amount": _round_to(target, 25_000),
            "target_date": f"{CURRENT_YEAR + years_out}-06-01",
        })

    # Optional property goal, more likely for younger clients.
    if age < 55 and rng.random() < 0.30:
        goals.append({
            "label": "property",
            "target_amount": _round_to(rng.uniform(80_000, 500_000), 10_000),
            "target_date": f"{CURRENT_YEAR + rng.randint(2, 8)}-04-01",
        })

    # Every client needs at least one goal for the profiler to work with.
    if not goals:
        goals.append({
            "label": "wealth_preservation",
            "target_amount": _round_to(portfolio_value * rng.uniform(1.1, 1.4), 25_000),
            "target_date": f"{CURRENT_YEAR + rng.randint(5, 15)}-01-01",
        })

    return goals


def gen_income(rng, age):
    """Income peaks in mid-career and tapers after typical retirement age."""
    if age < 35:
        base = rng.lognormvariate(11.35, 0.45)
    elif age < 50:
        base = rng.lognormvariate(11.75, 0.55)
    elif age < 65:
        base = rng.lognormvariate(11.85, 0.60)
    else:
        base = rng.lognormvariate(11.00, 0.55)

    # Retired clients get a lower floor. Income and wealth decouple in retirement:
    # a client drawing down a substantial portfolio may report modest income, and
    # a working-age floor would compress a third of this band against the clamp.
    floor = 30_000 if age >= 65 else 45_000
    return _round_to(_clamp(base, floor, 1_200_000), 1_000)


def gen_spending(rng, income):
    """Spending as a fraction of income. Must leave a surplus for most clients."""
    frac = _clamp(rng.gauss(0.62, 0.13), 0.30, 0.95)
    return _round_to(income * frac / 12, 100)


def gen_portfolio_value(rng, age, income):
    """
    Wealth accumulates, so portfolio size correlates with both age and income
    rather than being drawn freely.
    """
    base = rng.lognormvariate(PORTFOLIO_LOG_MEAN, PORTFOLIO_LOG_SD)
    age_factor = 0.35 + (age - AGE_MIN) / (AGE_MAX - AGE_MIN) * 1.5
    income_factor = _clamp(income / 150_000, 0.4, 3.0)
    value = base * age_factor * income_factor * 0.55
    return _round_to(_clamp(value, PORTFOLIO_MIN, PORTFOLIO_MAX), 1_000)


def gen_holding_weights(rng, age, risk_capacity):
    """
    Plausible asset-class weights for this client's actual holdings.

    This is NOT a target allocation. The allocation engine computes targets at
    runtime from the profiler's output; nothing about the target is stored on the
    twin, or the engine would be reading back its own earlier output and the
    pipeline would compute nothing.

    What this produces is simply a portfolio that hangs together — an older, risk
    averse client is not sitting at 95 percent equity — with enough spread that
    some clients happen to sit near whatever the engine later selects and others
    clearly do not. Drift is discovered at runtime rather than planted here.
    """
    # Equity share broadly falls with age and rises with risk capacity, with
    # enough noise that the relationship is loose rather than mechanical.
    equity = 1.10 - (age / 100.0)
    equity += {"low": -0.18, "moderate": 0.0, "high": 0.15}[risk_capacity]
    equity += rng.gauss(0.0, 0.12)
    equity = _clamp(equity, 0.15, 0.95)

    intl_share = rng.uniform(0.15, 0.40)
    alt = rng.uniform(0.0, 0.12) if risk_capacity != "low" else rng.uniform(0.0, 0.03)
    cash = rng.uniform(0.01, 0.10)
    fixed = _clamp(1.0 - equity - alt - cash, 0.01, 0.80)

    total = equity + alt + cash + fixed
    return {
        "us_equity": equity * (1 - intl_share) / total,
        "intl_equity": equity * intl_share / total,
        "fixed_income": fixed / total,
        "alternatives": alt / total,
        "cash": cash / total,
    }


def gen_holdings(rng, portfolio_value, weights, age, tenure_years):
    """
    Split the portfolio across asset classes and account types.

    Sheltered accounts carry no unrealised gain, since gains are not tracked there.
    Gains in taxable scale with tenure: longer holding means more accumulated gain.
    """

    # Share of the portfolio held in sheltered accounts. Older clients have had
    # longer to accumulate retirement balances.
    deferred_share = _clamp(rng.gauss(0.30 + (age - 40) * 0.004, 0.12), 0.0, 0.60)
    exempt_share = _clamp(rng.gauss(0.08, 0.05), 0.0, 0.25)
    taxable_share = _clamp(1.0 - deferred_share - exempt_share, 0.15, 1.0)
    norm = taxable_share + deferred_share + exempt_share
    taxable_share, deferred_share, exempt_share = (
        taxable_share / norm, deferred_share / norm, exempt_share / norm
    )

    # Unrealised gain as a fraction of position value. Roughly 6 percent per year
    # held, but with a floor well above zero: a Gaussian centred near zero for
    # short-tenure clients would be clamped to zero on a third of draws, leaving
    # the allocation builder with no tax cost to reason about at all.
    gain_ratio = _clamp(rng.gauss(0.05 + 0.055 * tenure_years, 0.08), 0.02, 0.65)

    holdings = []
    for ac, w in weights.items():
        ac_value = portfolio_value * w
        if ac_value < 1_000:
            continue

        for acct, share in (
            ("taxable", taxable_share),
            ("tax_deferred", deferred_share),
            ("tax_exempt", exempt_share),
        ):
            v = ac_value * share
            if v < 1_000:
                continue
            # Cash is held in taxable only; sheltered cash is not modelled.
            if ac == "cash" and acct != "taxable":
                continue
            gain = 0
            if acct == "taxable" and ac != "cash":
                gain = _round_to(v * gain_ratio * rng.uniform(0.7, 1.3), 100)
            holdings.append({
                "asset_class": ac,
                "value": _round_to(v, 500),
                "account_type": acct,
                "unrealized_gain": gain,
            })

    return holdings


def compute_benchmark(holdings):
    """
    The benchmark is the return a low-cost index portfolio at THIS client's asset
    mix would have delivered. Derived from holdings rather than sampled, so it is
    coherent with the portfolio by construction.
    """
    total = sum(h["value"] for h in holdings)
    if total == 0:
        return 0.0
    return sum(h["value"] * MARKET_RETURNS[h["asset_class"]] for h in holdings) / total


def compute_fee_revenue(portfolio_value):
    """Tiered fee schedule, so fee load varies genuinely across clients."""
    remaining, fee, lower = portfolio_value, 0.0, 0
    for upper, rate in FEE_TIERS:
        band = min(remaining, upper - lower)
        if band <= 0:
            break
        fee += band * rate
        remaining -= band
        lower = upper
    return round(fee, 2)


def gen_actual_return(rng, benchmark, fee_rate, underperform):
    """
    Actual return is the benchmark less fees, plus a dispersion term.

    A small negative spread is normal and roughly equals the fee. Only a spread
    well beyond the fee should read as performance dissatisfaction, so
    underperformance is planted deliberately rather than left to chance.
    """
    if underperform:
        drag = rng.uniform(0.025, 0.055)
    else:
        drag = rng.gauss(0.0, 0.010)
    return round(benchmark - fee_rate - drag, 4)


def gen_risk_fields(rng):
    """
    Stated tolerance and drawdown behaviour are drawn independently, so they
    disagree for a meaningful minority. Capacity is then derived: stated
    tolerance, stepped down one level where the client sold heavily.

    This is the twin's only derived field, and the concrete instance of revealed
    versus stated preference.
    """
    stated = rng.choices(["low", "moderate", "high"], weights=[0.25, 0.45, 0.30])[0]
    behavior = rng.choices(["held", "trimmed", "sold_heavily"], weights=[0.45, 0.25, 0.30])[0]

    order = ["low", "moderate", "high"]
    idx = order.index(stated)
    if behavior == "sold_heavily":
        idx = max(0, idx - 1)
    capacity = order[idx]

    return stated, behavior, capacity


# ---------------------------------------------------------------- assembly

def generate_base_client(rng, client_id, underperform=None, holding_weights=None):
    """
    Produce one client, everything except recent_flows and the churn label.

    underperform can be forced for designed scenarios. holding_weights can be
    supplied directly when a designed client needs to sit exactly on whatever the
    allocation engine would select for them: run the engine once during
    generation, pass its weights in here, and discard the target afterwards. The
    client is then on-target by construction without the target being stored.
    """
    age = gen_age(rng)
    tenure_years = round(_clamp(rng.expovariate(1 / 6.0), 0.3, min(age - 22, 30)), 1)

    dependents = gen_dependents(rng, age)
    income = gen_income(rng, age)
    spending = gen_spending(rng, income)
    portfolio_value = gen_portfolio_value(rng, age, income)
    goals = gen_goals(rng, age, dependents, portfolio_value, spending)

    stated, behavior, capacity = gen_risk_fields(rng)

    if holding_weights is None:
        holding_weights = gen_holding_weights(rng, age, capacity)
    holdings = gen_holdings(rng, portfolio_value, holding_weights, age, tenure_years)

    actual_value = sum(h["value"] for h in holdings)
    fee_revenue = compute_fee_revenue(actual_value)
    fee_rate = fee_revenue / actual_value if actual_value else 0.0

    benchmark = round(compute_benchmark(holdings), 4)
    if underperform is None:
        underperform = rng.random() < 0.22
    actual_return = gen_actual_return(rng, benchmark, fee_rate, underperform)

    cash_held = sum(h["value"] for h in holdings if h["asset_class"] == "cash")
    idle_cash = _round_to(cash_held * rng.uniform(0.35, 0.95), 500)

    # Investable assets is what the client could deploy. Since external_accounts
    # was removed from the schema, anything above the liquid holdings the twin
    # contains would be money that appears nowhere else, so this is taken as the
    # taxable portion exactly.
    investable_assets = _round_to(
        sum(h["value"] for h in holdings if h["account_type"] == "taxable"), 1_000
    )

    tax_bracket = rng.choices(
        TAX_BRACKETS,
        weights=[0.10, 0.20, 0.22, 0.24, 0.16, 0.08] if income > 150_000
        else [0.30, 0.32, 0.20, 0.12, 0.04, 0.02],
    )[0]

    return {
        "client_id": client_id,
        "age": age,
        "tenure_years": tenure_years,
        "dependents": dependents,
        "annual_fee_revenue": fee_revenue,
        "holdings": holdings,
        "investable_assets": investable_assets,
        "idle_cash": idle_cash,
        "annual_income": income,
        "monthly_spending": spending,
        "tax_bracket": tax_bracket,
        "goals": goals,
        "stated_risk_tolerance": stated,
        "risk_capacity": capacity,
        "drawdown_behavior": behavior,
        "trailing_return_12m": actual_return,
        "benchmark_return_12m": benchmark,
        # attached by later steps
        "recent_flows": [],
        "_meta": {
            "holding_weights": {k: round(v, 4) for k, v in holding_weights.items()},
            "underperform": underperform,
            "portfolio_value": actual_value,
            "fee_rate": round(fee_rate, 5),
        },
    }


if __name__ == "__main__":
    rng = random.Random(42)
    c = generate_base_client(rng, "C-00001")
    import json
    print(json.dumps(c, indent=2))




    
