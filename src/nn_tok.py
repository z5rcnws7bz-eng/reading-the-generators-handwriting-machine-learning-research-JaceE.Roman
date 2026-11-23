"""Wide & deep neural net over generator units: every number becomes the fields GPT-2 wrote
(exact value, integer part, decimal part, BPE tokens) as learned embeddings, plus quantile-normalised values.
No target encoding inside: a different inductive bias from the GBDTs. Apple-GPU (MPS) training.
Usage: python src/nn_tok.py [--epochs 8] [--seed 0] [--tag x]"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from common import load, folds, save, NUMS, CATS, BOOLS, ROOT
from feats3 import C2, C1, ND

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
ap.add_argument("--dim", type=int, default=12)
ap.add_argument("--minc", type=int, default=5); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--noval", action="store_true"); ap.add_argument("--l2", type=float, default=2e-6)
ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--folds", type=int, default=5)
A = ap.parse_args()
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tr, te, y = load()
full = pd.concat([tr, te], ignore_index=True)
TOK = np.load(ROOT / "data/gpt2_int_tokens.npy")

fields = {}
for c in C2 + C1:
    v = np.round(full[c].to_numpy() * 10 ** ND[c]).astype(np.int64)
    if not A.noval:
        fields[f"{c}_val"] = v
    if not (A.noval and c == "account_balance_usd"):
        fields[f"{c}_ip"] = v // 10 ** ND[c]
    fields[f"{c}_dp"] = v % 10 ** ND[c]
for c in ["account_balance_usd", "amount_usd"]:
    ip = full[c].to_numpy().astype(np.int64)
    fields[f"{c}_tk1"], fields[f"{c}_tk2"] = TOK[ip, 0], TOK[ip, 1]
for c in [n for n in NUMS if n not in C2 + C1] + CATS + BOOLS:
    fields[c] = full[c].astype(str).to_numpy()
codes, offsets, off = [], [], 0
for name, v in fields.items():
    cde, uniq = pd.factorize(v)
    cnt = np.bincount(cde)
    cde = np.where(cnt[cde] >= A.minc, cde + 1, 0)  # rare -> 0 (unknown)
    codes.append(cde + off); offsets.append(off); off += len(uniq) + 1
X_cat = np.stack(codes, 1).astype(np.int64)
X_num = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000,
                            random_state=0).fit_transform(full[NUMS].to_numpy()).astype(np.float32)
n_tr = len(tr)
print(f"{X_cat.shape[1]} embedded fields, vocabulary {off:,}, device {dev}", flush=True)


class Net(nn.Module):
    def __init__(self, vocab, nf, nn_, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.wide = nn.Embedding(vocab, 1)
        nn.init.normal_(self.emb.weight, 0, 0.01); nn.init.zeros_(self.wide.weight)
        self.mlp = nn.Sequential(nn.Linear(nf * d + nn_, 512), nn.SiLU(), nn.Dropout(0.3),
                                 nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        self.edrop = nn.Dropout(A.drop)

    def forward(self, xc, xn):
        e = self.edrop(self.emb(xc)).flatten(1)
        return self.mlp(torch.cat([e, xn], 1)).squeeze(1) + self.wide(xc).sum((1, 2))


def predict(m, idx, Xc, Xn):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 32768):
            j = idx[i:i + 32768]
            out.append(torch.sigmoid(m(Xc[j], Xn[j])).float().cpu().numpy())
    return np.concatenate(out)


Xc = torch.tensor(X_cat, device=dev); Xn = torch.tensor(X_num, device=dev)
yt = torch.tensor(y, dtype=torch.float32, device=dev)
oof = np.zeros(n_tr); tp = np.zeros(len(te)); t0 = time.time()
te_idx = torch.arange(n_tr, n_tr + len(te), device=dev)
for k, (a, b) in enumerate(StratifiedKFold(A.folds, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    torch.manual_seed(A.seed * 100 + k)
    m = Net(off, X_cat.shape[1], X_num.shape[1], A.dim).to(dev)
    opt = torch.optim.AdamW([{"params": m.emb.parameters(), "weight_decay": 0},
                             {"params": m.wide.parameters(), "weight_decay": 0},
                             {"params": m.mlp.parameters(), "weight_decay": 1e-4}], lr=A.lr)
    steps = A.epochs * (len(a) // 4096 + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=steps, pct_start=0.1)
    ia = torch.tensor(a, device=dev); ib = torch.tensor(b, device=dev)
    best, best_pb, best_pt = -1, None, None
    for ep in range(A.epochs):
        m.train(); perm = ia[torch.randperm(len(ia), device=dev)]
        for i in range(0, len(perm), 4096):
            j = perm[i:i + 4096]
            logit = m(Xc[j], Xn[j])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yt[j])
            # L2 on the embeddings actually used (sparse-friendly regularisation)
            loss = loss + A.l2 * (m.emb(Xc[j]).pow(2).sum() + 10 * m.wide(Xc[j]).pow(2).sum()) / len(j)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        pb = predict(m, ib, Xc, Xn); auc = roc_auc_score(y[b], pb)
        if auc > best:
            best, best_pb, best_pt = auc, pb, predict(m, te_idx, Xc, Xn)
        print(f"  fold {k} ep {ep} auc={auc:.5f} t={time.time()-t0:.0f}s", flush=True)
    oof[b] = best_pb; tp += best_pt / A.folds
save(f"nn_tok{A.tag}", oof, tp, y)
