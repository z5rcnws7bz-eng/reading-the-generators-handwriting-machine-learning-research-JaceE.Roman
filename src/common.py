"""Shared data loading, feature engineering and CV utilities for PS S6E11 (fraud, ROC-AUC)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OOF = ROOT / "oof"
OOF.mkdir(exist_ok=True)
TARGET = "is_fraud"
N_SPLITS = 5
SEED = 42

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
BASE = NUMS + CATS + BOOLS


def load():
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    for df in (tr, te):
        for c in BOOLS:
            df[c] = df[c].astype(int)
    y = tr.pop(TARGET).astype(int).values
    return tr, te, y


def folds(y):
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def add_features(tr, te):
    """Domain + Playground-style features computed on train+test jointly (unsupervised only)."""
    full = pd.concat([tr, te], ignore_index=True)
    f = pd.DataFrame(index=full.index)
    f["amt_to_bal"] = full.amount_usd / (full.account_balance_usd + 1)
    f["log_amt"] = np.log1p(full.amount_usd)
    f["log_bal"] = np.log1p(full.account_balance_usd)
    f["amt_cents"] = (full.amount_usd * 100 % 100).round()
    f["amt_round"] = (f.amt_cents == 0).astype(int)
    f["txn_per_hour"] = full.txn_count_last_24h / (full.hours_since_last_txn + 0.1)
    f["risk_x_vel"] = full.merchant_risk_score * full.velocity_score
    f["n_flags"] = full[BOOLS].sum(1)
    f["night"] = full.time_of_day_hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    f["weekend"] = (full.day_of_week >= 5).astype(int)
    f["card_age_ratio"] = full.card_age_months / (full.customer_age * 12)
    f["dist_per_hr"] = full.distance_from_home_km / (full.hours_since_last_txn + 0.1)
    # frequency encodings (synthetic-data trick: rarity of exact values carries signal)
    for c in NUMS + CATS:
        f[f"cnt_{c}"] = full[c].map(full[c].value_counts())
    # behaviour relative to merchant category / auth
    for g in ["merchant_category", "auth_method"]:
        for c in ["amount_usd", "velocity_score", "merchant_risk_score"]:
            f[f"{c}_rel_{g}"] = full[c] / (full.groupby(g)[c].transform("median") + 1e-6)
    f["cat_auth"] = full.merchant_category + "|" + full.auth_method
    f["cat_channel"] = full.merchant_category + "|" + full.channel
    f["auth_device"] = full.auth_method + "|" + full.device_type
    out = pd.concat([full, f], axis=1)
    return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)


def save(name, oof, test_pred, y):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, oof)
    np.save(OOF / f"{name}_oof.npy", oof)
    np.save(OOF / f"{name}_test.npy", test_pred)
    print(f"[{name}] OOF AUC = {auc:.5f}", flush=True)
    return auc
