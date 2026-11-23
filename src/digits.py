"""Digit-forensics features (CTGAN discretisation artefacts) + leak-free in-fold target encoding."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from common import NUMS

CONT = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
        "velocity_score", "merchant_risk_score"]


def digit_features(df):
    f = pd.DataFrame(index=df.index)
    for c in CONT:
        v = np.round(df[c].values * 100).astype(np.int64)
        f[f"{c}_d_cents"] = v % 10
        f[f"{c}_d_tenths"] = v // 10 % 10
        f[f"{c}_d_ones"] = v // 100 % 10
        f[f"{c}_d_tens"] = v // 1000 % 10
        f[f"{c}_d_last2"] = v % 100
        f[f"{c}_ndec"] = np.where(v % 100 == 0, 0, np.where(v % 10 == 0, 1, 2))
    return f


def te_keys(df):
    """High-cardinality keys to target-encode."""
    k = pd.DataFrame(index=df.index)
    for c in NUMS:
        k[f"te_{c}"] = df[c].astype(str)
    for c in CONT:
        v = np.round(df[c].values * 100).astype(np.int64)
        k[f"te_{c}_last2"] = (v % 100).astype(str)
        k[f"te_{c}_last3"] = (v % 1000).astype(str)
    return k


def _enc(key_fit, y_fit, key_apply, prior, m=30):
    g = pd.DataFrame({"k": key_fit.values, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
    enc = (g["sum"] + m * prior) / (g["count"] + m)
    return key_apply.map(enc).fillna(prior).values


def target_encode(K_tr, y_tr, K_apply_list, seed=0):
    """Returns (train TE via inner 5-fold, [TE for each apply frame fitted on all of K_tr])."""
    prior = y_tr.mean()
    out_tr = pd.DataFrame(index=K_tr.index, columns=K_tr.columns, dtype=float)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for a, b in skf.split(K_tr, y_tr):
        for c in K_tr.columns:
            out_tr.iloc[b, out_tr.columns.get_loc(c)] = _enc(K_tr[c].iloc[a], y_tr[a], K_tr[c].iloc[b], prior)
    outs = []
    for K in K_apply_list:
        o = pd.DataFrame(index=K.index)
        for c in K_tr.columns:
            o[c] = _enc(K_tr[c], y_tr, K[c], prior)
        outs.append(o)
    return out_tr, outs
