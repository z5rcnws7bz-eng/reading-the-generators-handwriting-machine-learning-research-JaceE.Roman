"""Second-generation models with digit features + in-fold target encoding.
Usage: python src/train2.py <lgb|xgb|cat> [--te] [--lr 0.03] [--tag name]"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import load, folds, save, BASE, CATS, SEED
from digits import digit_features, te_keys, target_encode

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--te", action="store_true")
ap.add_argument("--lr", type=float, default=0.03); ap.add_argument("--tag", default="")
ap.add_argument("--nodig", action="store_true")
A = ap.parse_args()
name = f"{A.model}_dig{'_te' if A.te else ''}{A.tag}"

tr, te, y = load()
X = tr[BASE].copy(); XT = te[BASE].copy()
if not A.nodig:
    X = pd.concat([X, digit_features(tr)], axis=1); XT = pd.concat([XT, digit_features(te)], axis=1)
dcat = [c for c in X.columns if c.endswith(("_d_cents", "_d_tenths", "_d_ones", "_d_tens"))]
for c in CATS + dcat:
    dt = pd.CategoricalDtype(sorted(set(X[c]) | set(XT[c])))
    X[c] = X[c].astype(dt); XT[c] = XT[c].astype(dt)
Ktr, Kte = (te_keys(tr), te_keys(te)) if A.te else (None, None)

oof = np.zeros(len(y)); tp = np.zeros(len(te)); t0 = time.time()
for k, (a, b) in enumerate(folds(y)):
    Xa, Xb, Xt = X.iloc[a], X.iloc[b], XT
    if A.te:
        ta, (tb, tt) = target_encode(Ktr.iloc[a].reset_index(drop=True), y[a],
                                     [Ktr.iloc[b].reset_index(drop=True), Kte], seed=k)
        Xa = pd.concat([Xa.reset_index(drop=True), ta.astype(float)], axis=1)
        Xb = pd.concat([Xb.reset_index(drop=True), tb], axis=1)
        Xt = pd.concat([XT.reset_index(drop=True), tt], axis=1)
    ya, yb = y[a], y[b]
    if A.model == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=8000, learning_rate=A.lr, num_leaves=31, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5,
                               reg_alpha=1, cat_smooth=20, max_cat_to_onehot=12, random_state=SEED + k,
                               verbose=-1, n_jobs=10)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        pb, pt, it = m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], m.best_iteration_
    elif A.model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=8000, learning_rate=A.lr, max_depth=6, min_child_weight=5,
                              subsample=0.8, colsample_bytree=0.4, reg_lambda=5, reg_alpha=1,
                              tree_method="hist", enable_categorical=True, max_cat_to_onehot=12,
                              eval_metric="auc", early_stopping_rounds=300, random_state=SEED + k, n_jobs=10)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        pb, pt, it = m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], m.best_iteration
    elif A.model == "cat":
        from catboost import CatBoostClassifier, Pool
        cc = CATS + dcat
        conv = lambda D: D.assign(**{c: D[c].astype(str) for c in cc})
        Xa, Xb, Xt = conv(Xa), conv(Xb), conv(Xt)
        m = CatBoostClassifier(iterations=8000, learning_rate=max(A.lr, 0.05), depth=6, l2_leaf_reg=5,
                               eval_metric="AUC", od_type="Iter", od_wait=300, random_seed=SEED + k,
                               thread_count=10, verbose=0)
        m.fit(Pool(Xa, ya, cat_features=cc), eval_set=Pool(Xb, yb, cat_features=cc), use_best_model=True)
        pb, pt, it = m.predict_proba(Xb)[:, 1], m.predict_proba(Pool(Xt, cat_features=cc))[:, 1], m.get_best_iteration()
    oof[b] = pb; tp += pt / 5
    print(f"  fold {k} auc={roc_auc_score(yb, pb):.5f} it={it} t={time.time()-t0:.0f}s", flush=True)
save(name, oof, tp, y)
