from data_generation.flows_labels import extract_features
import csv
import json
import os
import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURE_KEYS = [
    "outflow_intensity", "competitor_share", "competitor_intensity",
    "net_flow_ratio", "recent_outflow_share", "contributions_stopped",
    "perf_spread_excess", "fee_rate", "fee_excess",
    "tenure_years", "sold_heavily", "portfolio_value", "age",
]

HORIZON_MONTHS = 12
MODEL_PATH = "out/detector.pkl"

def load_training_data(features_csv):
    X, y = [], []
    with open(features_csv) as f:
        for row in csv.DictReader(f):
            X.append([float(row[k]) for k in FEATURE_KEYS])
            y.append(int(row["churned"]))
    return np.array(X), np.array(y)

def train(features_csv, seed = 1, test_size = 0.3):
    X, y = load_training_data(features_csv)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        random_state=seed,
    )

    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_tr, y_tr)

    return model, X_te, y_te


def evaluate(model, X_te, y_te):
    p_te = model.predict_proba(X_te)[:, 1]
    return {
        "n_test": int(len(y_te)),
        "base_rate": round(float(y_te.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "pr_auc": round(float(average_precision_score(y_te, p_te)), 4),
        "brier": round(float(brier_score_loss(y_te, p_te)), 4),
        "calibration": calibration_table(y_te, p_te),
        "horizon_months": HORIZON_MONTHS,
    }

def calibration_table(y_true, p_pred, bins=5):
    edges = np.quantile(p_pred, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_pred > lo) & (p_pred <= hi)
        if m.sum() == 0:
            continue
        table.append({
            "n": int(m.sum()),
            "mean_predicted": round(float(p_pred[m].mean()), 4),
            "observed_rate": round(float(y_true[m].mean()), 4),
        })
    return table

class ChurnDetector: 

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def estimate_severity(feats, probability):
        """
        Expected fraction of assets at risk, conditional on churn.

        Deliberately a stated function rather than a second trained model.
        Severity in the generated data is driven almost entirely by how much has
        already moved to a competitor, so a model would recover exactly that and
        present it as a finding.
        """
        comp = min(max(feats["competitor_share"], 0.0), 1.0)
        already_out = min(max(feats["outflow_intensity"], 0.0), 1.0)

        base = 0.25 + 0.45 * comp + 0.15 * already_out
        # A higher probability tends to accompany a more advanced departure.
        base += 0.10 * min(max(probability - 0.5, 0.0), 0.5) * 2
        return round(min(max(base, 0.05), 1.0), 3)

    def score(self, twin):
        feats = extract_features(twin)
        vector = [feats[k] for k in FEATURE_KEYS]
        prob = float(self.model.predict_proba([vector])[0, 1])
        severity = self.estimate_severity(feats, prob)

        portfolio = feats["portfolio_value"]
        revenue = twin["annual_fee_revenue"]

        return {
            "client_id": twin["client_id"],
            "probability": round(prob, 4),
            "severity": severity,
            "horizon_months": HORIZON_MONTHS,
            "assets_at_risk": round(portfolio * severity * prob, 2),
            "revenue_at_risk": round(revenue * severity * prob, 2),
            "features": feats,
            "attributions": None,
        }
    def score_batch(self, twins):
        return [self.score(t) for t in twins]
def main():
    features_csv = "out/features_train.csv"

    model, X_te, y_te = train(features_csv)
    metrics = evaluate(model, X_te, y_te)

    detector = ChurnDetector(model)
    detector.save()

    with open("out/detector_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"tested on {metrics['n_test']:,}, base rate {metrics['base_rate']:.1%}")
    print(f"  ROC-AUC {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC  {metrics['pr_auc']:.3f}  "
          f"(lift {metrics['pr_auc']/metrics['base_rate']:.1f}x)")
    print(f"  Brier   {metrics['brier']:.4f}")
    print("\ncalibration")
    print("  predicted   observed    n")
    for row in metrics["calibration"]:
        print(f"    {row['mean_predicted']:.3f}       "
              f"{row['observed_rate']:.3f}    {row['n']:>4}")


if __name__ == "__main__":
    main()