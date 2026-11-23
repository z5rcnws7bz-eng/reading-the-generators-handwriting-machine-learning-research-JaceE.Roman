"""Generator-aware pipeline: digits + in-fold TE (+tokens) + neighbour freq, shallow GBDTs.
Usage: python src/train3.py <lgb|xgb|cat> [--tok] [--nf] [--lr 0.02] [--seeds 1] [--folds 5] [--tag x]"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import load, save, NUMS, CATS, BOOLS, ROOT
from feats3 import digits, keys, target_encode, neighbour_freq

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--tok", action="store_true"); ap.add_argument("--nf", action="store_true")
ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--tag", default="")
A = ap.parse_args()
name = f"g3_{A.model}{'_tok' if A.tok else ''}{'_nf' if A.nf else ''}_f{A.folds}{A.tag}"

tr, te, y = load()
for c in CATS:
    dt = pd.CategoricalDtype(sorted(set(tr[c]) | set(te[c])))
    tr[c], te[c] = tr[c].astype(dt), te[c].astype(dt)
TOK = np.load(ROOT / "data/gpt2_int_tokens.npy") if A.tok else None
F = pd.concat([tr[NUMS + CATS + BOOLS], digits(tr)], axis=1)
FT = pd.concat([te[NUMS + CATS + BOOLS], digits(te)], axis=1)
K, KT = keys(tr, TOK), keys(te, TOK)

fl = list(StratifiedKFold(A.folds, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
oof = np.zeros(len(y)); tp = np.zeros(len(te)); t0 = time.time()
for k, (a, b) in enumerate(fl):
    ta, (tb, tt) = target_encode(K.iloc[a], y[a], [K.iloc[b], KT])
    blocks = [[ta], [tb], [tt]]
    if A.nf:
        na, nb_, nt = neighbour_freq(tr.iloc[a], [tr.iloc[b], te])
        blocks[0].append(na); blocks[1].append(nb_); blocks[2].append(nt)
    Xa = pd.concat([F.iloc[a].reset_index(drop=True)] + blocks[0], axis=1)
    Xb = pd.concat([F.iloc[b].reset_index(drop=True)] + blocks[1], axis=1)
    Xt = pd.concat([FT.reset_index(drop=True)] + blocks[2], axis=1)
    pb = np.zeros(len(b)); pt = np.zeros(len(te))
    for s in range(A.seeds):
        if A.model == "lgb":
            import lightgbm as lgb
            P = dict(objective="binary", metric="auc", learning_rate=A.lr, max_depth=3, num_leaves=3,
                     min_child_samples=1700, feature_fraction=0.74, bagging_fraction=0.85, bagging_freq=1,
                     lambda_l1=0.038, lambda_l2=0.115, min_gain_to_split=1.08, cat_smooth=1.94,
                     min_data_per_group=23, n_jobs=10, verbose=-1, seed=42 + s)
            d = lgb.Dataset(Xa, y[a])
            m = lgb.train(P, d, 30000, valid_sets=[lgb.Dataset(Xb, y[b], reference=d)],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            pb += m.predict(Xb, num_iteration=m.best_iteration); pt += m.predict(Xt, num_iteration=m.best_iteration)
            it = m.best_iteration
        elif A.model == "xgb":
            import xgboost as xgb
            m = xgb.XGBClassifier(n_estimators=30000, learning_rate=A.lr, max_depth=2, min_child_weight=50,
                                  subsample=0.85, colsample_bytree=0.7, reg_lambda=1, tree_method="hist",
                                  enable_categorical=True, max_cat_to_onehot=12, eval_metric="auc",
                                  early_stopping_rounds=300, random_state=42 + s, n_jobs=10)
            m.fit(Xa, y[a], eval_set=[(Xb, y[b])], verbose=False)
            pb += m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1]; it = m.best_iteration
        elif A.model == "cat":
            from catboost import CatBoostClassifier, Pool
            conv = lambda D: D.assign(**{c: D[c].astype(str) for c in CATS})
            m = CatBoostClassifier(iterations=30000, learning_rate=max(A.lr, 0.04), depth=3, l2_leaf_reg=10,
                                   eval_metric="AUC", od_type="Iter", od_wait=400, random_seed=42 + s,
                                   thread_count=10, verbose=0)
            m.fit(Pool(conv(Xa), y[a], cat_features=CATS), eval_set=Pool(conv(Xb), y[b], cat_features=CATS),
                  use_best_model=True)
            pb += m.predict_proba(conv(Xb))[:, 1]; pt += m.predict_proba(Pool(conv(Xt), cat_features=CATS))[:, 1]
            it = m.get_best_iteration()
    oof[b] = pb / A.seeds; tp += pt / A.seeds / len(fl)
    print(f"  fold {k} auc={roc_auc_score(y[b], oof[b]):.5f} it={it} t={time.time()-t0:.0f}s", flush=True)
save(name, oof, tp, y)
