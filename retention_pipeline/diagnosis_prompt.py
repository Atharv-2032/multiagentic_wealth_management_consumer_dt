"""
Churn diagnosis — stage two of the retention pipeline.

Receives a client the detector flagged and returns the likely reasons, drawn from
a fixed taxonomy, each with a confidence level and the evidence supporting it.

Why a language model here
-------------------------
Cause labels essentially do not exist. Supervised cause classification would need
exit interviews or reliable advisor-recorded reasons at scale, which firms rarely
hold. What this stage does — synthesis over heterogeneous evidence, some of it
structured, some of it a list of dated transactions — is something a language
model can do without labels.

What it does NOT do
-------------------
It does not choose a remedy, and it does not decide what happens to a
low-confidence diagnosis. Both are deterministic and live in the feasibility
engine, which narrows the option set before the strategy decider sees it. The
model produces a judgement; rules turn that judgement into a constraint.
"""

import json

from catalogs import CAUSES

MODEL = "claude-sonnet-4-6"
MAX_CAUSES = 3


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
#
# The taxonomy is spelled out with its evidence base rather than just named,
# because several categories are specific to this system. destination_type in
# particular is a vocabulary we invented; a model has no way to know that
# "competitor_institution" means a rival wealth manager unless told.

SYSTEM_PROMPT = """\
You are a diagnostic component in a wealth management system. A statistical model \
has flagged a client as being at elevated risk of moving assets away from the firm. \
Your task is to identify the most likely reasons, using only the evidence provided.

You are not deciding what to do about it. Another component handles that.

## The evidence you receive

**Client profile** — age, tenure with the firm, portfolio holdings by asset class and \
account type, annual fee revenue the client generates, realised return over the last \
twelve months, and the benchmark return a comparable low-cost index portfolio would \
have delivered over the same period at the same asset mix.

**Transaction list** — every material money movement over the last twelve months. Each \
carries an amount, a date, a direction, and a destination category:

- `competitor_institution` — money that arrived at another financial firm. This is a \
transfer of assets away, not spending.
- `merchant` — money spent. A purchase, a payment, a withdrawal for consumption.
- `own_account` — money moving between the client's own accounts. Not a loss of assets.
- `unknown` — the destination could not be identified. Account aggregation coverage is \
partial, so this is a gap in visibility rather than evidence of anything in particular. \
Do not treat it as suspicious.

**Computed ratios** — figures derived from the above, provided so you do not have to \
calculate. Two are worth understanding:

- `perf_spread_excess` — how far the client's return fell below the benchmark, *after \
subtracting the fee they pay*. A client trailing the benchmark by exactly their fee is \
getting what the arrangement implies, and this figure will be near zero. Only a clearly \
positive value indicates underperformance beyond the cost of the service.
- `fee_excess` — how far the client's effective fee rate sits above the book average of \
0.92 percent.

**Risk score** — the model's estimated probability of attrition and its estimated severity.

## The causes you may return

Return only causes from this list. Do not invent categories.

**`competitor_consolidation`**
The client is moving assets to another financial institution.
Evidence: outflows with destination `competitor_institution`, particularly if they are \
large relative to the portfolio, repeated, or accelerating. Contributions stopping \
alongside them strengthens the case.
Note this describes a *behaviour*, not a motive. It tells you the client is leaving. It \
does not tell you why, and you should not infer a motive that the evidence does not \
support.

**`performance_dissatisfaction`**
The portfolio has underperformed a comparable passive alternative by a margin the fee \
does not explain.
Evidence: `perf_spread_excess` clearly above zero. A value around 0.02 is a two \
percentage point shortfall beyond fees; above 0.04 is substantial.

**`fee_sensitivity`**
The client pays materially more than comparable clients and is likely to have noticed.
Evidence: `fee_excess` clearly positive. This is strengthened considerably when combined \
with weak performance — a high fee is tolerable when returns are strong and becomes \
salient when they disappoint.

**`planned_drawdown`**
Assets are declining because the client is spending them as intended, not because the \
relationship is failing.
Evidence: outflows to `merchant` rather than to a competitor, a client at or near \
retirement age, an absence of transfers elsewhere. This is not churn, and identifying it \
correctly matters — a system that fights a client's planned retirement withdrawals is \
worse than useless.

**`insufficient_evidence`**
The risk score is elevated but the evidence does not support any specific explanation.
Use this rather than selecting the least-bad option. It is a legitimate finding, and it \
routes the client to a discovery conversation, which is the appropriate response to not \
knowing.

## How to reason

Work from what is in the data. For every cause you return, point to the specific facts \
that support it — name amounts, dates, destinations, or figures.

Two failure modes to avoid.

**Inventing motives.** A large outflow tells you money left. It does not tell you the \
client is dissatisfied, frustrated, or unhappy with service. Those are stories that fit \
the data, not conclusions the data supports. Nothing in the evidence speaks to the \
client's state of mind.

**Treating a high number as an explanation.** A figure being large means it is large. \
Whether it explains the client's behaviour is a separate question, and often the answer \
is that it does not.

A client may be leaving for more than one reason, and often is — poor performance and a \
high fee compound. Return up to three causes, ordered with the best-supported first. \
Return one if only one is supported.

## Confidence

Assign each cause a confidence level, judged by how well the evidence supports it — not \
by how plausible the story feels.

- `high` — the evidence is direct and admits little other explanation
- `medium` — the evidence points this way, but other readings fit the same facts
- `low` — plausible, but thinly supported

Be willing to use `low`. A low-confidence diagnosis routes the client to a discovery \
conversation, which is often the correct outcome.

## Output

Return JSON only, no prose before or after:

{
  "causes": [
    {
      "cause": "<one of the five listed above>",
      "confidence": "high" | "medium" | "low",
      "evidence": "<the specific facts supporting this, naming figures and dates>",
      "reasoning": "<one or two sentences on why those facts point to this cause>"
    }
  ]
}

Order causes best-supported first. Include at most three.
"""


# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------

def build_user_message(twin, detector_output):
    """
    Assemble the client's evidence.

    The raw transaction list is included alongside the computed ratios rather
    than instead of them. The ratios make magnitude comparisons easy; the raw
    list is what lets the model cite a specific dated transfer, which is stronger
    evidence than a scalar and is checkable by whoever reviews the output.

    They are labelled distinctly so the model does not read the summary as
    independent corroboration of the detail it was computed from.
    """
    feats = detector_output["features"]
    portfolio = feats["portfolio_value"]

    holdings = [
        {
            "asset_class": h["asset_class"],
            "value": h["value"],
            "account_type": h["account_type"],
        }
        for h in twin["holdings"]
    ]

    profile = {
        "age": twin["age"],
        "tenure_years": twin["tenure_years"],
        "dependents": len(twin["dependents"]),
        "portfolio_value": portfolio,
        "annual_fee_revenue": twin["annual_fee_revenue"],
        "holdings": holdings,
        "trailing_return_12m": twin["trailing_return_12m"],
        "benchmark_return_12m": twin["benchmark_return_12m"],
        "stated_risk_tolerance": twin["stated_risk_tolerance"],
        "drawdown_behavior": twin["drawdown_behavior"],
        "goals": twin["goals"],
    }

    ratios = {
        "outflow_intensity": feats["outflow_intensity"],
        "competitor_share": feats["competitor_share"],
        "competitor_intensity": feats["competitor_intensity"],
        "net_flow_ratio": feats["net_flow_ratio"],
        "recent_outflow_share": feats["recent_outflow_share"],
        "contributions_stopped": bool(feats["contributions_stopped"]),
        "perf_spread_excess": feats["perf_spread_excess"],
        "fee_rate": feats["fee_rate"],
        "fee_excess": feats["fee_excess"],
    }

    risk = {
        "churn_probability": detector_output["probability"],
        "estimated_severity": detector_output["severity"],
        "horizon_months": detector_output["horizon_months"],
    }

    return (
        f"## Client profile\n{json.dumps(profile, indent=2)}\n\n"
        f"## Transactions, last 12 months\n"
        f"{json.dumps(twin['recent_flows'], indent=2)}\n\n"
        f"## Computed ratios\n{json.dumps(ratios, indent=2)}\n\n"
        f"## Risk score\n{json.dumps(risk, indent=2)}\n\n"
        f"Diagnose the likely reasons this client is at risk."
    )


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

VALID_CONFIDENCE = {"high", "medium", "low"}


def parse_response(text):
    """
    Parse and validate. A malformed or out-of-taxonomy response is a failure to
    surface, not something to silently repair — the whole point of a fixed
    taxonomy is that downstream components can rely on it.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)

    causes = data.get("causes", [])
    if not causes:
        raise ValueError("no causes returned")
    if len(causes) > MAX_CAUSES:
        raise ValueError(f"{len(causes)} causes returned, maximum is {MAX_CAUSES}")

    for c in causes:
        if c.get("cause") not in CAUSES:
            raise ValueError(f"unknown cause: {c.get('cause')!r}")
        if c.get("confidence") not in VALID_CONFIDENCE:
            raise ValueError(f"invalid confidence: {c.get('confidence')!r}")
        if not c.get("evidence"):
            raise ValueError(f"no evidence cited for {c['cause']}")

    return causes