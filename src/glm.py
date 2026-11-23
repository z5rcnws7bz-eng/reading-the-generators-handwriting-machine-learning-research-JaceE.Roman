"""Additive logistic model (GLM) on generator-aware features: logit of in-fold target encodings
(values, integer/decimal parts, GPT-2 tokens), one-hot categories, spline-expanded numerics, neighbour frequencies.
Usage: python src/glm.py [--C 0.5]"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from common import load, folds, save, NUMS, CATS, BOOLS, ROOT
from feats3 import keys, target_encode, neighbour_freq

ap = argparse.ArgumentParser(); ap.add_argument("--C", type=float, default=0.5); ap.add_argument("--folds", type=int, default=5); A = ap.parse_args()
tr, te, y = load()
TOK = np.load(ROOT / "data/gpt2_int_tokens.npy")
K, KT = keys(tr, TOK), keys(te, TOK)
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
oof = np.zeros(len(y)); tp = np.zeros(len(te)); t0 = time.time()
for k, (a, b) in enumerate(StratifiedKFold(A.folds, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    ta, (tb, tt) = target_encode(K.iloc[a], y[a], [K.iloc[b], KT])
    na, nb, nt = neighbour_freq(tr.iloc[a], [tr.iloc[b], te])
    oh = OneHotEncoder(handle_unknown="ignore").fit(tr.iloc[a][CATS])
    sp = SplineTransformer(n_knots=6, degree=3, knots="quantile").fit(tr.iloc[a][NUMS])
    sc = StandardScaler()
    def mat(D, T, N, fit=False):
        dense = np.c_[lg(T.to_numpy()), np.log1p(np.abs(N.to_numpy())) * np.sign(N.to_numpy()), D[BOOLS].to_numpy()]
        dense = sc.fit_transform(dense) if fit else sc.transform(dense)
        return np.c_[dense, sp.transform(D[NUMS]), oh.transform(D[CATS]).toarray()]
    Xa = mat(tr.iloc[a], ta, na, fit=True); Xb = mat(tr.iloc[b], tb, nb); Xt = mat(te, tt, nt)
    m = LogisticRegression(C=A.C, max_iter=3000).fit(Xa, y[a])
    oof[b] = m.predict_proba(Xb)[:, 1]; tp += m.predict_proba(Xt)[:, 1] / A.folds
    print(f"  fold {k} auc={roc_auc_score(y[b], oof[b]):.5f} t={time.time()-t0:.0f}s", flush=True)
save("glm_tok_nf" + ("" if A.folds == 5 else f"_f{A.folds}"), oof, tp, y)
