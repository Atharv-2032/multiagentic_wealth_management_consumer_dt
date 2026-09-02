"""
Generation driver.

Produces the detector's training set: 10,000 clients with flows, extracted
features and churn labels.

Runs in two passes, because the label threshold cannot be known until every
client has been scored. Pass one generates and scores; pass two calibrates the
cut to the intended base rate and applies it.

Outputs
-------
    out/clients_train.jsonl   full twins, one JSON object per line, no labels
    out/features_train.csv    extracted features plus label and severity
    out/generation_meta.json  seed, parameters, resulting distributions

The twin file carries no label. Generation state — latent dissatisfaction, the
churn flag, whether underperformance was planted — lives only in the feature
file and the meta sidecar. Leaving it on the twin would put the answer inside
the pipeline's own input.
"""

import csv
import json
import os
import random
import statistics as st

from base_client import generate_base_client
from flows_labels import (
    attach_flows_and_label,
    calibrate_threshold,
    extract_features,
    gen_severity,
    TARGET_CHURN_RATE,
    NOISE_SD,
    W_COMPETITOR_SHARE,
    W_OUTFLOW_INTENSITY,
    W_PERF_SPREAD,
    W_FEE_EXCESS,
    W_CONTRIB_STOPPED,
    W_SOLD_HEAVILY,
    TENURE_PROTECTION,
)

SEED = 2026
N_CLIENTS = 10_000
OUT_DIR = "out"

FEATURE_KEYS = [
    "outflow_intensity",
    "competitor_share",
    "competitor_intensity",
    "net_flow_ratio",
    "recent_outflow_share",
    "contributions_stopped",
    "perf_spread_excess",
    "fee_rate",
    "fee_excess",
    "tenure_years",
    "sold_heavily",
    "portfolio_value",
    "age",
]


def _q(values, p):
    s = sorted(values)
    return s[min(int(p * len(s)), len(s) - 1)]


def generate_population(seed=SEED, n=N_CLIENTS):
    """Pass one: build every client and score it. No labels yet."""
    rng = random.Random(seed)
    clients, features, scores = [], [], []

    for i in range(n):
        client = generate_base_client(rng, f"C-{i:05d}")
        client, feats, score = attach_flows_and_label(rng, client)
        clients.append(client)
        features.append(feats)
        scores.append(score)

    return rng, clients, features, scores


def apply_labels(rng, clients, features, scores):
    """Pass two: calibrate the cut to the target base rate, then label."""
    threshold = calibrate_threshold(scores, TARGET_CHURN_RATE)

    labels, severities = [], []
    for client, feats, score in zip(clients, features, scores):
        churned = score >= threshold
        severity = gen_severity(rng, feats, churned)
        labels.append(1 if churned else 0)
        severities.append(severity)
        # kept in _meta for inspection during generation, stripped before writing
        client["_meta"]["churned"] = churned
        client["_meta"]["severity"] = severity

    return threshold, labels, severities


def write_twins(clients, path):
    """
    One twin per line. _meta is dropped: it holds generation state including the
    churn label, and the pipeline must not see it.
    """
    with open(path, "w") as f:
        for client in clients:
            twin = {k: v for k, v in client.items() if k != "_meta"}
            f.write(json.dumps(twin) + "\n")


def write_features(clients, features, labels, severities, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id"] + FEATURE_KEYS + ["churned", "severity"])
        for client, feats, label, sev in zip(clients, features, labels, severities):
            writer.writerow(
                [client["client_id"]]
                + [feats[k] for k in FEATURE_KEYS]
                + [label, sev]
            )


def summarise(clients, features, labels, severities, threshold):
    n = len(clients)
    churners = [i for i, y in enumerate(labels) if y == 1]

    portfolio = [c["_meta"]["portfolio_value"] for c in clients]
    fee_rate = [c["_meta"]["fee_rate"] for c in clients]
    n_flows = [len(c["recent_flows"]) for c in clients]
    comp_share = [f["competitor_share"] for f in features]
    sev_churn = [severities[i] for i in churners]

    return {
        "seed": SEED,
        "n_clients": n,
        "churn": {
            "target_rate": TARGET_CHURN_RATE,
            "actual_rate": round(len(churners) / n, 4),
            "threshold": round(threshold, 4),
            "n_churners": len(churners),
        },
        "label_rule": {
            "note": (
                "Weights are stipulated, not estimated. They encode an ordering "
                "the churn literature broadly supports; the magnitudes are chosen "
                "for plausibility and describe no real population."
            ),
            "weights": {
                "competitor_share": W_COMPETITOR_SHARE,
                "outflow_intensity": W_OUTFLOW_INTENSITY,
                "perf_spread_excess": W_PERF_SPREAD,
                "fee_excess": W_FEE_EXCESS,
                "contributions_stopped": W_CONTRIB_STOPPED,
                "sold_heavily": W_SOLD_HEAVILY,
                "tenure_protection": -TENURE_PROTECTION,
                "fee_x_perf_interaction": 0.08,
            },
            "noise_sd": NOISE_SD,
        },
        "distributions": {
            "portfolio_value": {
                "p10": _q(portfolio, 0.10),
                "median": int(st.median(portfolio)),
                "p90": _q(portfolio, 0.90),
            },
            "fee_rate": {
                "p10": round(_q(fee_rate, 0.10), 5),
                "median": round(st.median(fee_rate), 5),
                "p90": round(_q(fee_rate, 0.90), 5),
            },
            "flows_per_client": {
                "p10": _q(n_flows, 0.10),
                "median": int(st.median(n_flows)),
                "p90": _q(n_flows, 0.90),
            },
            "severity_of_churners": {
                "p10": round(_q(sev_churn, 0.10), 3),
                "median": round(st.median(sev_churn), 3),
                "p90": round(_q(sev_churn, 0.90), 3),
            },
            "pct_no_competitor_flow": round(
                100 * sum(1 for v in comp_share if v == 0) / n, 1
            ),
            "pct_churners_without_competitor_flow": round(
                100
                * sum(1 for i in churners if features[i]["competitor_share"] == 0)
                / max(len(churners), 1),
                1,
            ),
        },
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rng, clients, features, scores = generate_population()
    threshold, labels, severities = apply_labels(rng, clients, features, scores)

    write_twins(clients, os.path.join(OUT_DIR, "clients_train.jsonl"))
    write_features(
        clients, features, labels, severities,
        os.path.join(OUT_DIR, "features_train.csv"),
    )

    meta = summarise(clients, features, labels, severities, threshold)
    with open(os.path.join(OUT_DIR, "generation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(clients)} clients to {OUT_DIR}/")
    print(f"  churn rate {meta['churn']['actual_rate']:.1%} "
          f"({meta['churn']['n_churners']} churners), "
          f"threshold {meta['churn']['threshold']}")
    d = meta["distributions"]
    print(f"  portfolio  p10 {d['portfolio_value']['p10']:,}  "
          f"median {d['portfolio_value']['median']:,}  "
          f"p90 {d['portfolio_value']['p90']:,}")
    print(f"  churners with no competitor flow: "
          f"{d['pct_churners_without_competitor_flow']}%")


if __name__ == "__main__":
    main()