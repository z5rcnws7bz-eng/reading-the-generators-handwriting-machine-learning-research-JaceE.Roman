"""Generator-aware features (GReaT / GPT-2 written rows):
- positional digits of each number (row-wise)
- keys for in-fold target encoding: exact value, integer part, decimal part (+ optional GPT-2 token keys)
- neighbour-value frequencies (no target): how 'lonely' an exact value is
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

C2 = ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km"]  # 2 decimals
C1 = ["velocity_score", "merchant_risk_score"]  # 1 decimal
ND = {**{c: 2 for c in C2}, **{c: 1 for c in C1}}


def digits(df):
    f = {}
    for c in C2 + C1:
        nd = ND[c]
        v = np.round(df[c].to_numpy() * 10 ** nd).astype(np.int64)
        ip = v // 10 ** nd
        f[f"{c}_i0"], f[f"{c}_i1"] = ip % 10, ip // 10 % 10
        f[f"{c}_f1"] = v // 10 ** (nd - 1) % 10
        if nd == 2:
            f[f"{c}_f2"] = v % 10
    for c in ["customer_age", "card_age_months"]:
        f[f"{c}_i0"] = df[c].to_numpy() % 10
    return pd.DataFrame(f, index=df.index).astype("int16")


def keys(df, token_table=None):
    k = {}
    for c in ["merchant_risk_score", "velocity_score", "customer_age", "card_age_months", "txn_count_last_24h",
              "hours_since_last_txn", "amount_usd", "distance_from_home_km", "account_balance_usd"]:
        k[c] = df[c].to_numpy()
    for c in C2 + C1:
        nd = ND[c]
        v = np.round(df[c].to_numpy() * 10 ** nd).astype(np.int64)
        k[f"{c}_ip"], k[f"{c}_dp"] = v // 10 ** nd, v % 10 ** nd
    if token_table is not None:
        for c in ["account_balance_usd", "amount_usd"]:
            ip = df[c].to_numpy().astype(np.int64)
            dp = np.round(df[c].to_numpy() * 100).astype(np.int64) % 100
            k[f"{c}_tk1"], k[f"{c}_tk2"] = token_table[ip, 0], token_table[ip, 1]
            k[f"{c}_tk2dp"] = token_table[ip, 1] * 100 + dp
    return pd.DataFrame(k, index=df.index)


def target_encode(Ktr, ytr, Kothers, m=20.0, seed=42):
    """In-fold TE: training rows via inner 5-fold, other frames from all training rows."""
    yt = ytr.astype(np.float64)
    inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(yt)), yt))
    out_tr, outs = {}, [dict() for _ in Kothers]
    for c in Ktr.columns:
        codes, _ = pd.factorize(np.concatenate([Ktr[c].to_numpy()] + [K[c].to_numpy() for K in Kothers]))
        n = codes.max() + 1
        ctr = codes[:len(Ktr)]
        te = np.empty(len(Ktr))
        for a, b in inner:
            s = np.bincount(ctr[a], weights=yt[a], minlength=n); cnt = np.bincount(ctr[a], minlength=n)
            te[b] = (s[ctr[b]] + m * yt[a].mean()) / (cnt[ctr[b]] + m)
        out_tr[f"te_{c}"] = te
        s = np.bincount(ctr, weights=yt, minlength=n); cnt = np.bincount(ctr, minlength=n)
        full = (s + m * yt.mean()) / (cnt + m)
        off = len(Ktr)
        for j, K in enumerate(Kothers):
            outs[j][f"te_{c}"] = full[codes[off:off + len(K)]]; off += len(K)
    return pd.DataFrame(out_tr), [pd.DataFrame(o) for o in outs]


def neighbour_freq(Dtr, Dothers, ks=(1, 5)):
    """Frequency of the exact value vs its nearest distinct neighbours (reference = training rows)."""
    res = [dict() for _ in range(1 + len(Dothers))]
    for c in C2 + C1:
        g = lambda d: np.round(d[c].to_numpy() * 10 ** ND[c]).astype(np.int64)
        gf = g(Dtr)
        u, cnt = np.unique(gf, return_counts=True)
        cn = np.r_[0.0, np.cumsum(cnt)]
        for i, d in enumerate([Dtr] + Dothers):
            gq, self_ = (gf, 1.0) if i == 0 else (g(d), 0.0)
            lo, hi = np.searchsorted(u, gq, "left"), np.searchsorted(u, gq, "right")
            own, n = cn[hi] - cn[lo] - self_, len(gf) - self_
            res[i][f"nf_{c}_own"] = own / n
            for k in ks:
                a, b = np.maximum(lo - k, 0), np.minimum(hi + k, len(u))
                nb = (cn[lo] - cn[a] + cn[b] - cn[hi]) / np.maximum((lo - a) + (b - hi), 1)
                res[i][f"nf_{c}_nb{k}"] = nb / n
                res[i][f"nf_{c}_sp{k}"] = np.log((own + 1) / (nb + 1))
            big = float(u[-1] - u[0] + 1)
            res[i][f"nf_{c}_gp"] = np.where(lo > 0, gq - u[np.maximum(lo - 1, 0)], big).astype(float)
            res[i][f"nf_{c}_gn"] = np.where(hi < len(u), u[np.minimum(hi, len(u) - 1)] - gq, big).astype(float)
    return [pd.DataFrame(r) for r in res]
