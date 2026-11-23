"""Hill-climbing blend in logit-rank space with an honest nested-CV estimate. Writes submission.csv.
Usage: python src/ensemble.py [prefix-filter ...]   e.g. python src/ensemble.py g3_"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import OOF, ROOT, load

_, te, y = load()
pref = sys.argv[1:]
names = sorted(p.name[:-8] for p in OOF.glob("*_oof.npy") if not pref or any(q in p.name for q in pref))
L = lambda a: (lambda r: np.log(r / (1 - r)))((rankdata(a) - 0.5) / len(a))
O = np.column_stack([L(np.load(OOF / f"{n}_oof.npy")) for n in names])
T = np.column_stack([L(np.load(OOF / f"{n}_test.npy")) for n in names])
for j, n in enumerate(names):
    print(f"{n:34s} {roc_auc_score(y, O[:, j]):.5f}")


def hill(X, t, max_iter=60):
    sc = [roc_auc_score(t, X[:, j]) for j in range(X.shape[1])]
    w = np.zeros(X.shape[1]); w[int(np.argmax(sc))] = 1; cur = X @ w; best = max(sc)
    for _ in range(max_iter):
        cand = [roc_auc_score(t, cur + X[:, j]) for j in range(X.shape[1])]
        j = int(np.argmax(cand))
        if cand[j] <= best + 1e-7:
            break
        best = cand[j]; cur = cur + X[:, j]; w[j] += 1
    return w / w.sum()


nested = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(O, y):
    nested[b] = O[b] @ hill(O[a], y[a])
w = hill(O, y)
best_single = max(roc_auc_score(y, O[:, j]) for j in range(len(names)))
print(f"best single {best_single:.5f} | nested blend {roc_auc_score(y, nested):.5f} | in-sample {roc_auc_score(y, O @ w):.5f}")
print("weights:", {n: round(x, 3) for n, x in zip(names, w) if x > 0})
pred = rankdata(T @ w) / len(T)
pd.DataFrame({"id": te.id, "is_fraud": pred}).to_csv(ROOT / "submission.csv", index=False)
print("wrote submission.csv")
