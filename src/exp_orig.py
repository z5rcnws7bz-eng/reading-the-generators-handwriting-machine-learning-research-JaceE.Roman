"""Quick experiments: ways of using the original dataset, scored with fast LightGBM 5-fold CV."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from common import load, folds, BASE, CATS, BOOLS, NUMS, DATA

tr, te, y = load()
o = pd.read_csv(DATA / "original/credit_card_fraud_2026.csv")
for c in BOOLS: o[c] = o[c].astype(int)
yo = o.is_fraud.values
prior = yo.mean()


def orig_feats(df):
    f = pd.DataFrame(index=df.index)
    # smoothed per-value fraud rate in the original (for continuous cols: on rounded bins)
    for c in NUMS + CATS:
        key_o, key_d = o[c], df[c]
        if c in ("amount_usd", "account_balance_usd", "distance_from_home_km", "hours_since_last_txn"):
            q = np.unique(np.quantile(o[c], np.linspace(0, 1, 41)))
            key_o, key_d = np.digitize(o[c], q), np.digitize(df[c], q)
        g = pd.DataFrame({"k": key_o, "y": yo}).groupby("k").y.agg(["sum", "count"])
        te_ = (g["sum"] + 20 * prior) / (g["count"] + 20)
        f[f"ote_{c}"] = pd.Series(key_d).map(te_).fillna(prior).values
    return f


lrm = make_pipeline(make_column_transformer((StandardScaler(), NUMS + BOOLS),
                                            (OneHotEncoder(handle_unknown="ignore"), CATS)),
                    LogisticRegression(C=1, max_iter=3000)).fit(o[BASE], yo)


def prep(df, mode):
    X = df[BASE].copy()
    if mode in ("ote", "all"):
        X = pd.concat([X, orig_feats(df)], axis=1)
    if mode in ("olr", "all"):
        X["orig_lr_logit"] = lrm.decision_function(df[BASE])
    for c in CATS: X[c] = pd.Categorical(X[c], categories=sorted(o[c].unique()))
    return X


fl = folds(y)
for mode in sys.argv[1:]:
    rows = mode.endswith("+rows"); m_ = mode.replace("+rows", "")
    X = prep(tr, m_); Xo = prep(o, m_); Xo["is_orig"] = 1; X["is_orig"] = 0
    oof = np.zeros(len(y))
    for a, b in fl:
        Xa, ya = X.iloc[a], y[a]
        if rows:
            Xa = pd.concat([Xa, Xo]); ya = np.r_[ya, yo]
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                               verbose=-1, n_jobs=6)
        m.fit(Xa, ya, eval_set=[(X.iloc[b], y[b])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    print(f"{mode:12s} AUC {roc_auc_score(y, oof):.5f}", flush=True)
