"""Print-quality figures for the ESWA manuscript (Times New Roman, 600 dpi PNG).
Numbers come from results_forensics.csv, results_ablation.jsonl and the saved OOF predictions."""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "fig"
for f in ["Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf"]:
    font_manager.fontManager.addfont(f"/System/Library/Fonts/Supplemental/{f}")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "savefig.dpi": 600, "mathtext.fontset": "stix"})
BLUE, GREY, LIGHT, GREEN, INK = "#1f5fa8", "#9a9a9a", "#d9d9d9", "#4a9a5a", "#222222"

# ---- Figure 1: forensic funnel test -------------------------------------------------------------
r = pd.read_csv(ROOT / "results" / "results_forensics.csv")
nice = {"amount_usd": "Amount", "account_balance_usd": "Account balance", "hours_since_last_txn": "Hours since last txn",
        "distance_from_home_km": "Distance from home", "velocity_score": "Velocity score",
        "merchant_risk_score": "Merchant risk score"}
def label(u):
    col, kind = [s.strip() for s in u.split("|")]
    return f"{nice[col]}: {'decimal part' if 'decimal' in kind else '2nd GPT-2 token'}"
comp = r[r.data == "competition"].set_index("unit"); orig = r[r.data == "original"].set_index("unit")
order = comp.sort_values("outside_2sd_real").index
fig, ax = plt.subplots(figsize=(6.3, 3.4))
for i, u in enumerate(order):
    ax.plot([0, comp.loc[u, "outside_2sd_real"] * 100], [i, i], color=LIGHT, lw=1.2, zorder=1)
    ax.scatter(comp.loc[u, "outside_2sd_shuffled"] * 100, i, s=34, facecolors="white", edgecolors=GREY, lw=1, zorder=3)
    if u in orig.index and not np.isnan(orig.loc[u, "outside_2sd_real"]):
        ax.scatter(orig.loc[u, "outside_2sd_real"] * 100, i, marker="s", s=26, color=GREY, zorder=2)
    ax.scatter(comp.loc[u, "outside_2sd_real"] * 100, i, s=44, color=BLUE, zorder=4)
    ax.text(comp.loc[u, "outside_2sd_real"] * 100 + 1.2, i, f"{comp.loc[u, 'outside_2sd_real'] * 100:.0f}%",
            va="center", fontsize=9, color=INK)
ax.axvline(4.55, color=GREY, ls=(0, (3, 3)), lw=0.8)
ax.text(5.3, len(order) - 0.35, "chance (4.6%)", fontsize=8.5, color="#555555")
ax.set_yticks(range(len(order)), [label(u) for u in order])
ax.set_xlim(0, 57); ax.set_xlabel("Share of value groups with fraud rate outside ±2σ of the overall rate (%)")
ax.tick_params(axis="y", length=0)
h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6, label="Competition data, real labels"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=6, label="Competition data, shuffled labels"),
     plt.Line2D([], [], marker="s", ls="", color=GREY, ms=5, label="Original NumPy data, real labels")]
ax.legend(handles=h, loc="lower right", frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig1_forensics.png"); plt.close(fig)

# ---- Figure 2: pipeline ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.3, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
def box(x, y, w, h, title, lines, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", lw=1.6 if accent else 0.8,
                                ec=BLUE if accent else "#555555", fc="#e8eff8" if accent else "white"))
    ax.text(x + 1.6, y + h - 3.2, title, fontsize=9.5, fontweight="bold", va="top", color=INK)
    for k, s in enumerate(lines):
        ax.text(x + 1.6, y + h - 8.2 - 4.4 * k, s, fontsize=8.3, va="top", color="#333333")
def arrow(x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#555555"))
box(1, 87, 98, 12, "1  Row as the generator wrote it",
    ["24 columns; each number re-read as the text GPT-2 emitted, e.g. 1269.35 → “ 12” “69” “.” “35”"])
box(1, 67, 98, 16, "2  Generator-unit decomposition (row-wise, no labels)",
    ["Positional digits (last two integer digits, every decimal digit); integer part; decimal part",
     "First and second GPT-2 byte-pair token of the integer part (account balance, amount)"])
box(1, 47, 98, 16, "3  Leak-free encoders fitted inside each training fold",
    ["Target encoding of values, parts and tokens (m = 20; inner 5-fold for training rows)",
     "Neighbour-value frequencies: own count, mean count of 1 and 5 nearest values, spike, gaps"])
w = 31.3
for j, (t, ls) in enumerate([("4a  Shallow GBDTs", ["LightGBM: 3 leaves, depth 3", "XGBoost (depth 2), CatBoost", "≥ 1,700 rows per leaf"]),
                             ("4b  Additive GLM", ["Logistic model on logit-TE,", "cubic splines, one-hot,", "neighbour frequencies"]),
                             ("4c  Token-embedding net", ["33 embedded unit fields", "+ wide term; MLP 512–256", "no target encoding; 5 seeds"])]):
    box(1 + j * (w + 2.05), 22.5, w, 22.5, t, ls)
box(1, 0.5, 98, 18, "5  Logit-rank hill-climbing blend",
    ["Weights chosen on four folds and scored on the fifth (nested cross-validation),",
     "never by public-leaderboard feedback"], accent=True)
arrow(50, 87, 83); arrow(50, 67, 63)
for j in range(3):
    cx = 1 + j * (w + 2.05) + w / 2; arrow(cx, 47, 45.2); arrow(cx, 22.5, 18.7)
fig.tight_layout(pad=0.2); fig.savefig(OUT / "fig2_pipeline.png"); plt.close(fig)

# ---- Figure 3: ablation bridge -----------------------------------------------------------------
steps = [("Deep LightGBM, raw features (start)", 0.94674), ("Shallow trees: 3 leaves, depth 3", 0.94801),
         ("+ positional digits", 0.94858), ("+ in-fold TE of values and parts", 0.95013),
         ("+ neighbour-value frequencies", 0.95029), ("+ GPT-2 token keys", 0.95055),
         ("3 seeds, learning rate 0.02", 0.95063), ("Blend with GLM and token network", 0.95099)]
base = steps[0][1]; g = lambda a: (a - base) * 1e4
fig, ax = plt.subplots(figsize=(6.3, 3.5))
n = len(steps)
for i, (name, auc) in enumerate(steps):
    y = n - i
    if i:
        p = g(steps[i - 1][1]); c = g(auc)
        ax.barh(y, c - p, left=p, height=0.62, color=GREEN, alpha=0.75)
        ax.plot([p, p], [y + 0.31, y + 0.69], color=GREY, lw=0.6, ls=":")
        ax.text(c + 0.5, y, f"+{c - p:.1f}", va="center", fontsize=8.5)
    ax.text(59.5, y, f"{auc:.5f}", va="center", ha="right", fontsize=8.5, color="#444444")
tot = g(steps[-1][1])
ax.barh(0, tot, height=0.62, color=GREY)
ax.text(tot + 0.5, 0, f"+{tot:.1f}", va="center", fontsize=8.5, fontweight="bold")
ax.text(59.5, 0, f"{steps[-1][1]:.5f}", va="center", ha="right", fontsize=8.5, fontweight="bold")
ax.text(59.5, n + 0.85, "OOF AUC", ha="right", fontsize=8.5, color="#444444")
ax.set_yticks(list(range(n, 0, -1)) + [0], [s for s, _ in steps] + ["Total gain"])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(0, 60); ax.set_ylim(-0.6, n + 1.2); ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlabel(r"Out-of-fold AUC gain over the start ($\times10^{-4}$)")
ax.tick_params(axis="y", length=0); ax.axvline(0, color="#555555", lw=0.6)
fig.tight_layout(); fig.savefig(OUT / "fig3_ablation.png"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
