"""Ablation ladder + artefact-only model, same folds, shallow LightGBM (lr 0.05, 1 seed).
Usage: python src/exp3.py <set> [<set> ...]
sets: raw_shallow, digits, digits_te, artefact_only"""
import sys, time, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import load, folds, NUMS, CATS, BOOLS, ROOT, OOF
from feats3 import digits, keys, target_encode, C2, C1

tr, te, y = load()
for c in CATS:
    dt = pd.CategoricalDtype(sorted(set(tr[c]) | set(te[c])))
    tr[c], te[c] = tr[c].astype(dt), te[c].astype(dt)
TOK = np.load(ROOT / "data/gpt2_int_tokens.npy")
D = digits(tr)
K = keys(tr, TOK)
frac_digits = [c for c in D.columns if c.endswith(("_f1", "_f2"))]
art_keys = [c for c in K.columns if c.endswith(("_dp", "_tk2", "_tk2dp"))]
P = dict(objective="binary", metric="auc", learning_rate=0.05, max_depth=3, num_leaves=3, min_child_samples=1700,
         feature_fraction=0.74, bagging_fraction=0.85, bagging_freq=1, lambda_l1=0.038, lambda_l2=0.115,
         min_gain_to_split=1.08, cat_smooth=1.94, min_data_per_group=23, n_jobs=10, verbose=-1, seed=42)

for s in sys.argv[1:]:
    if s == "raw_shallow":
        F, Kc = tr[NUMS + CATS + BOOLS], []
    elif s == "digits":
        F, Kc = pd.concat([tr[NUMS + CATS + BOOLS], D], axis=1), []
    elif s == "digits_te":
        F, Kc = pd.concat([tr[NUMS + CATS + BOOLS], D], axis=1), [c for c in K.columns if not c.endswith(("_tk1", "_tk2", "_tk2dp"))]
    elif s == "artefact_only":  # no raw values, no magnitudes: only fractional digits + decimal/token-tail keys
        F, Kc = D[frac_digits], art_keys
    oof = np.zeros(len(y)); fa = []; t0 = time.time()
    for a, b in folds(y):
        Xa, Xb = F.iloc[a].reset_index(drop=True), F.iloc[b].reset_index(drop=True)
        if Kc:
            ta, (tb,) = target_encode(K[Kc].iloc[a], y[a], [K[Kc].iloc[b]])
            Xa, Xb = pd.concat([Xa, ta], axis=1), pd.concat([Xb, tb], axis=1)
        d = lgb.Dataset(Xa, y[a])
        m = lgb.train(P, d, 30000, valid_sets=[lgb.Dataset(Xb, y[b], reference=d)],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[b] = m.predict(Xb, num_iteration=m.best_iteration); fa.append(roc_auc_score(y[b], oof[b]))
    (OOF.parent / "exp").mkdir(exist_ok=True); np.save(OOF.parent / "exp" / f"{s}_oof.npy", oof)
    r = dict(set=s, auc=roc_auc_score(y, oof), folds=[round(x, 5) for x in fa], n_feats=F.shape[1] + len(Kc), secs=round(time.time() - t0))
    print(json.dumps(r), flush=True)
    with open(ROOT / "results" / "results_ablation.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
