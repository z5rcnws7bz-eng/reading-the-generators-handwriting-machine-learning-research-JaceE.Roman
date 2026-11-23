"""Train one model family with 5-fold CV; saves OOF + test predictions to oof/.
Usage: python src/train.py <model> [--fe]
models: lgb, xgb, cat, cat_allcat, lr
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load, folds, add_features, save, BASE, CATS, NUMS, SEED

model = sys.argv[1]
fe = "--fe" in sys.argv
name = model + ("_fe" if fe else "")

tr, te, y = load()
if fe:
    tr, te = add_features(tr, te)
feats = [c for c in tr.columns if c != "id"]
cat_cols = [c for c in feats if tr[c].dtype == object or str(tr[c].dtype) == "str"]
for c in cat_cols:
    dt = pd.CategoricalDtype(sorted(set(tr[c]) | set(te[c])))
    tr[c] = tr[c].astype(dt); te[c] = te[c].astype(dt)

fl = folds(y)
oof = np.zeros(len(tr)); tp = np.zeros(len(te))
t0 = time.time()
for k, (a, b) in enumerate(fl):
    Xa, Xb, ya, yb = tr.iloc[a][feats], tr.iloc[b][feats], y[a], y[b]
    if model == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.02, num_leaves=31, max_depth=-1,
                               min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=5, reg_alpha=1, cat_smooth=20,
                               max_cat_to_onehot=8, random_state=SEED + k, verbose=-1, n_jobs=10)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        pb = m.predict_proba(Xb)[:, 1]; pt = m.predict_proba(te[feats])[:, 1]; it = m.best_iteration_
    elif model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_depth=6, min_child_weight=5,
                              subsample=0.8, colsample_bytree=0.5, reg_lambda=5, reg_alpha=1,
                              tree_method="hist", enable_categorical=True, max_cat_to_onehot=8,
                              eval_metric="auc", early_stopping_rounds=300, random_state=SEED + k,
                              n_jobs=10)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        pb = m.predict_proba(Xb)[:, 1]; pt = m.predict_proba(te[feats])[:, 1]; it = m.best_iteration
    elif model in ("cat", "cat_allcat"):
        from catboost import CatBoostClassifier, Pool
        Xa, Xb, Xt = Xa.copy(), Xb.copy(), te[feats].copy()
        cc = list(cat_cols)
        if model == "cat_allcat":  # treat every discrete-ish numeric as categorical too (Playground trick)
            for c in NUMS:
                if tr[c].nunique() < 5000:
                    for X in (Xa, Xb, Xt):
                        X[c + "_c"] = X[c].astype(str)
                    cc.append(c + "_c")
        for X in (Xa, Xb, Xt):
            for c in cat_cols:
                X[c] = X[c].astype(str)
        m = CatBoostClassifier(iterations=6000, learning_rate=0.04, depth=6, l2_leaf_reg=5,
                               eval_metric="AUC", od_type="Iter", od_wait=300, random_seed=SEED + k,
                               thread_count=10, verbose=0)
        m.fit(Pool(Xa, ya, cat_features=cc), eval_set=Pool(Xb, yb, cat_features=cc), use_best_model=True)
        pb = m.predict_proba(Xb)[:, 1]; pt = m.predict_proba(Pool(Xt, cat_features=cc))[:, 1]
        it = m.get_best_iteration()
    elif model == "lr":
        from sklearn.pipeline import make_pipeline
        from sklearn.compose import make_column_transformer
        from sklearn.preprocessing import OneHotEncoder, SplineTransformer
        from sklearn.linear_model import LogisticRegression
        num = [c for c in feats if c not in cat_cols]
        ct = make_column_transformer(
            (SplineTransformer(n_knots=8, degree=3, knots="quantile"), num),
            (OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat_cols))
        m = make_pipeline(ct, LogisticRegression(C=0.5, max_iter=2000))
        m.fit(Xa, ya)
        pb = m.predict_proba(Xb)[:, 1]; pt = m.predict_proba(te[feats])[:, 1]; it = 0
    oof[b] = pb; tp += pt / len(fl)
    print(f"  fold {k} auc={roc_auc_score(yb, pb):.5f} it={it} t={time.time()-t0:.0f}s", flush=True)
save(name, oof, tp, y)
