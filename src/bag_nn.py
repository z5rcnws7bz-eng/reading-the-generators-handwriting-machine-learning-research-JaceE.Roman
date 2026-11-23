"""Average the seeds of the token-embedding network into one ensemble member.
Usage: python src/bag_nn.py <prefix> <n_seeds> <out_name>
e.g.   python src/bag_nn.py nn_tok_a_s 5 nn_tok_bag5   (averages nn_tok_a_s0 .. nn_tok_a_s4)"""
import sys, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OOF, load, save

prefix, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
_, _, y = load()
names = [f"{prefix}{s}" for s in range(n)]
oof = np.mean([np.load(OOF / f"{k}_oof.npy") for k in names], 0)
test = np.mean([np.load(OOF / f"{k}_test.npy") for k in names], 0)
save(out, oof, test, y)
# keep single seeds out of the blend's candidate pool
(OOF.parent / "exp").mkdir(exist_ok=True)
for k in names:
    for suf in ("_oof.npy", "_test.npy"):
        shutil.move(str(OOF / f"{k}{suf}"), OOF.parent / "exp" / f"{k}{suf}")
