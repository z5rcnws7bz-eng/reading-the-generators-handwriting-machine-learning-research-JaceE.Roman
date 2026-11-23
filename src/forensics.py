"""Forensic evidence: does the fraud rate vary across generator units (decimal endings, BPE tokens)
more than chance allows? Funnel test with real vs shuffled labels, competition vs original data."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from common import load, BOOLS, ROOT, DATA

tr, te, y = load()
o = pd.read_csv(DATA / "original/credit_card_fraud_2026.csv")
yo = o.is_fraud.values
TOK = np.load(ROOT / "data/gpt2_int_tokens.npy")
rng = np.random.default_rng(0)


def units(df):
    u = {}
    for c, nd in [("amount_usd", 2), ("account_balance_usd", 2), ("hours_since_last_txn", 2),
                  ("distance_from_home_km", 2), ("velocity_score", 1), ("merchant_risk_score", 1)]:
        v = np.round(df[c].to_numpy() * 10 ** nd).astype(np.int64)
        u[f"{c} | decimal part"] = v % 10 ** nd
    for c in ["account_balance_usd", "amount_usd"]:
        ip = np.minimum(df[c].to_numpy().astype(np.int64), len(TOK) - 1)
        u[f"{c} | 2nd BPE token"] = np.where(TOK[ip, 1] >= 0, TOK[ip, 1], -1)
    return u


def funnel(key, lab, min_n):
    g = pd.DataFrame({"k": key, "y": lab}).query("k >= 0").groupby("k").y.agg(["sum", "count"])
    g = g[g["count"] >= min_n]
    p0 = lab.mean()
    out = np.abs(g["sum"] / g["count"] - p0) > 2 * np.sqrt(p0 * (1 - p0) / g["count"])
    ct = np.c_[g["count"] - g["sum"], g["sum"]]
    p = chi2_contingency(ct)[1] if len(g) > 1 and ct[:, 1].sum() > 0 else np.nan
    return len(g), float(out.mean()), float(p)


rows = []
for data_name, df, lab, min_n in [("competition", tr, y, 100), ("original", o, yo, 30)]:
    for name, key in units(df).items():
        n, real, p = funnel(key, lab, min_n)
        shuf = np.mean([funnel(key, rng.permutation(lab), min_n)[1] for _ in range(5)])
        rows.append(dict(data=data_name, unit=name, groups=n, outside_2sd_real=round(real, 3),
                         outside_2sd_shuffled=round(shuf, 3), chi2_p=p))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.to_string(index=False))
r.to_csv(ROOT / "results" / "results_forensics.csv", index=False)
# how many integer tokens per value (balance, amount)
for c in ["account_balance_usd", "amount_usd"]:
    n = TOK[np.minimum(tr[c].to_numpy().astype(np.int64), len(TOK) - 1), 2]
    print(c, "share of rows whose integer part is 2+ BPE tokens:", round(float((n >= 2).mean()), 3))
