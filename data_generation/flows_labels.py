"""
Flows and churn labelling.

Flows are generated BEFORE the label, not after. Deciding who churns and then
planting matching outflows would make the label a direct restatement of what was
planted, and the detector would be reading back a stamp rather than estimating
anything. Instead:

    latent dissatisfaction  ->  flow behaviour  ->  observable features  ->  score

Dissatisfaction is unobservable. It is driven by fee load, performance spread and
drawdown behaviour, and it influences how a client moves money. The detector never
sees it; it sees only the flows and the other twin fields, and has to infer.

The churn label is a threshold on a score computed from OBSERVABLE features plus
noise. The weights are stipulated, not estimated. They encode an ordering the
churn literature broadly supports: transactional signals of exit dominate,
performance and fee sensitivity matter but act more slowly, and tenure is
protective. The magnitudes are chosen for plausibility and are not claimed to
describe any real population.
"""

import math

CURRENT_YEAR = 2026
TARGET_CHURN_RATE = 0.13

# Weights over observable features. Ordering matters; the exact magnitudes do not.
W_COMPETITOR_SHARE = 0.34   # money landing at a named competitor is the most direct evidence
W_OUTFLOW_INTENSITY = 0.26  # magnitude relative to portfolio size
W_PERF_SPREAD = 0.15        # underperformance beyond what fees explain
W_FEE_EXCESS = 0.13         # fee load above the book average
W_CONTRIB_STOPPED = 0.07    # contributions drying up
W_SOLD_HEAVILY = 0.05       # panic-sold in the last drawdown

TENURE_PROTECTION = 0.10    # subtracted, scaled by tenure

NOISE_SD = 0.24             # sets the detector's achievable ceiling

BOOK_AVG_FEE_RATE = 0.0092  # reference point for fee load excess


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _round_to(x, step):
    return int(round(x / step) * step)


# ---------------------------------------------------------------- latent state

def compute_dissatisfaction(client):
    """
    Latent, unobservable, and never written to the twin. This is what actually
    drives a client's behaviour; the detector has to infer it from what that
    behaviour leaves behind.
    """
    m = client["_meta"]
    fee_excess = _clamp((m["fee_rate"] - BOOK_AVG_FEE_RATE) / 0.004, -1.0, 1.5)
    spread = client["benchmark_return_12m"] - client["trailing_return_12m"]
    perf_excess = _clamp((spread - m["fee_rate"]) / 0.04, -0.5, 1.5)
    sold = 1.0 if client["drawdown_behavior"] == "sold_heavily" else 0.0

    d = 0.35 * fee_excess + 0.45 * perf_excess + 0.20 * sold
    return _clamp(d, 0.0, 1.0)


# ---------------------------------------------------------------- flows

def gen_flows(rng, client, dissatisfaction):
    """
    Twelve months of material transactions. Roughly 8 to 15 per client — the
    signal, not a full bank feed.

    Contributions reflect surplus. Ordinary spending outflows occur for everyone.
    Transfers to a competitor appear only where dissatisfaction is high enough to
    have prompted them, and their size and count scale with it.
    """
    flows = []
    portfolio = client["_meta"]["portfolio_value"]
    surplus = client["annual_income"] - client["monthly_spending"] * 12

    # --- contributions -----------------------------------------------------
    # A dissatisfied client is likely to stop contributing before they move money.
    contributes = surplus > 6_000 and rng.random() > (0.15 + 0.55 * dissatisfaction)
    if contributes:
        monthly = _round_to(surplus * rng.uniform(0.25, 0.55) / 12, 100)
        stop_month = 12
        if dissatisfaction > 0.5 and rng.random() < 0.5:
            stop_month = rng.randint(4, 10)  # contributions dry up part way through
        for month in range(stop_month):
            if monthly < 200:
                continue
            flows.append({
                "amount": monthly,
                "date": _month_date(rng, month),
                "direction": "inflow",
                "destination_type": "own_account",
            })

    # --- ordinary spending outflows ---------------------------------------
    for _ in range(rng.randint(2, 6)):
        amt = _round_to(portfolio * rng.uniform(0.004, 0.045), 500)
        if amt < 500:
            continue
        flows.append({
            "amount": amt,
            "date": _month_date(rng, rng.randint(0, 11)),
            "direction": "outflow",
            "destination_type": "merchant",
        })

    # --- transfers to a competitor ----------------------------------------
    # The strongest single signal, but deliberately not a clean marker. A
    # baseline share of contented clients also move money elsewhere — diversifying
    # custodians, consolidating a spouse's accounts, chasing a promotional rate —
    # and plenty of departing clients never show a visible transfer inside the
    # window, because the decision precedes the paperwork.
    #
    # Without that overlap, the presence of a competitor flow would be close to a
    # label stamp and the detector would separate the classes almost perfectly,
    # which no real churn model does.
    p_transfer = 0.12 + 0.55 * dissatisfaction
    if rng.random() < p_transfer:
        n = 1 + int(dissatisfaction * rng.uniform(0.5, 2.5))
        for i in range(n):
            frac = rng.uniform(0.02, 0.06 + 0.22 * dissatisfaction)
            amt = _round_to(portfolio * frac, 500)
            if amt < 1_000:
                continue
            flows.append({
                "amount": amt,
                "date": _month_date(rng, rng.randint(3, 11)),
                "direction": "outflow",
                "destination_type": "competitor_institution",
            })

    # --- unattributed outflows --------------------------------------------
    # Aggregation coverage is partial in practice. These support the
    # insufficient-evidence outcome in diagnosis.
    if rng.random() < 0.30:
        amt = _round_to(portfolio * rng.uniform(0.005, 0.06), 500)
        if amt >= 1_000:
            flows.append({
                "amount": amt,
                "date": _month_date(rng, rng.randint(0, 11)),
                "direction": "outflow",
                "destination_type": "unknown",
            })

    flows.sort(key=lambda f: f["date"])
    return flows


def _month_date(rng, month_index):
    """month_index 0 is twelve months ago, 11 is last month."""
    total = (CURRENT_YEAR - 1) * 12 + 8 + month_index  # window ends around Aug 2026
    year, month = divmod(total, 12)
    if month == 0:
        year, month = year - 1, 12
    return f"{year:04d}-{month:02d}-{rng.randint(1, 28):02d}"


# ---------------------------------------------------------------- features

def extract_features(client):
    """
    What the detector actually consumes. A model cannot take a variable-length
    list of transactions, so the flow list is reduced to scalars here.
    """
    m = client["_meta"]
    portfolio = m["portfolio_value"]
    flows = client["recent_flows"]

    outflows = [f for f in flows if f["direction"] == "outflow"]
    inflows = [f for f in flows if f["direction"] == "inflow"]

    total_out = sum(f["amount"] for f in outflows)
    total_in = sum(f["amount"] for f in inflows)
    comp_out = sum(f["amount"] for f in outflows
                   if f["destination_type"] == "competitor_institution")

    # Recent half of the window versus the earlier half, for trend.
    recent = [f for f in outflows if f["date"] >= f"{CURRENT_YEAR}-03"]
    recent_out = sum(f["amount"] for f in recent)

    spread = client["benchmark_return_12m"] - client["trailing_return_12m"]

    return {
        "outflow_intensity": round(total_out / portfolio, 4) if portfolio else 0.0,
        "competitor_share": round(comp_out / total_out, 4) if total_out else 0.0,
        "competitor_intensity": round(comp_out / portfolio, 4) if portfolio else 0.0,
        "net_flow_ratio": round((total_in - total_out) / portfolio, 4) if portfolio else 0.0,
        "recent_outflow_share": round(recent_out / total_out, 4) if total_out else 0.0,
        "contributions_stopped": 1 if total_in == 0 else 0,
        "perf_spread_excess": round(spread - m["fee_rate"], 4),
        "fee_rate": m["fee_rate"],
        "fee_excess": round(m["fee_rate"] - BOOK_AVG_FEE_RATE, 5),
        "tenure_years": client["tenure_years"],
        "sold_heavily": 1 if client["drawdown_behavior"] == "sold_heavily" else 0,
        "portfolio_value": portfolio,
        "age": client["age"],
    }


# ---------------------------------------------------------------- label

def churn_score(rng, feats):
    """
    Score over observable features, plus noise.

    Noise is what stops the label being a deterministic function of the features.
    Without it a model recovers the rule almost perfectly and the exercise is
    transparently circular; with it there is a ceiling well below perfect
    separation, which is what a real churn model faces.
    """
    comp = _clamp(feats["competitor_share"], 0.0, 1.0)
    intensity = _clamp(feats["outflow_intensity"] / 0.35, 0.0, 1.0)
    perf = _clamp(feats["perf_spread_excess"] / 0.045, 0.0, 1.0)
    fee = _clamp(feats["fee_excess"] / 0.004, 0.0, 1.0)
    stopped = feats["contributions_stopped"]
    sold = feats["sold_heavily"]
    tenure = _clamp(feats["tenure_years"] / 20.0, 0.0, 1.0)

    s = (W_COMPETITOR_SHARE * comp
         + W_OUTFLOW_INTENSITY * intensity
         + W_PERF_SPREAD * perf
         + W_FEE_EXCESS * fee
         + W_CONTRIB_STOPPED * stopped
         + W_SOLD_HEAVILY * sold
         - TENURE_PROTECTION * tenure)

    # One interaction: a high fee is tolerable when returns are strong and becomes
    # salient when they are not, so the pair is worse than the sum of its parts.
    s += 0.08 * fee * perf

    return s + rng.gauss(0.0, NOISE_SD)


def gen_severity(rng, feats, churned):
    """
    Expected fraction of assets at risk. Severity is graded, not binary: partial
    attrition is more common than full account closure.
    """
    if not churned:
        return round(_clamp(rng.gauss(0.04, 0.03), 0.0, 0.15), 3)
    base = 0.25 + 0.5 * _clamp(feats["competitor_share"], 0.0, 1.0)
    return round(_clamp(rng.gauss(base, 0.18), 0.05, 1.0), 3)


# ---------------------------------------------------------------- assembly

def attach_flows_and_label(rng, client, threshold=None):
    """
    Attach flows, extract features, and score. Returns (client, features, score).
    The threshold is calibrated across the whole population in a second pass, so
    it is passed in rather than fixed here.
    """
    d = compute_dissatisfaction(client)
    client["recent_flows"] = gen_flows(rng, client, d)
    feats = extract_features(client)
    score = churn_score(rng, feats)

    client["_meta"]["dissatisfaction"] = round(d, 3)
    client["_meta"]["churn_score"] = round(score, 4)

    if threshold is not None:
        churned = score >= threshold
        client["_meta"]["churned"] = churned
        client["_meta"]["severity"] = gen_severity(rng, feats, churned)

    return client, feats, score


def calibrate_threshold(scores, target_rate=TARGET_CHURN_RATE):
    """Pick the cut that produces the intended base rate across the population."""
    s = sorted(scores, reverse=True)
    idx = int(len(s) * target_rate)
    return s[min(idx, len(s) - 1)]