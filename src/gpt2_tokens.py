"""Build the GPT-2 byte-pair token table for integer parts (used by the token features).

Downloads `vocab.json` and `merges.txt` from the Hugging Face `openai-community/gpt2` repository
(Modified MIT licence), checks that GPT-2 splits " 1269.35" into " 12", "69", ".", "35", and writes
data/gpt2_int_tokens.npy: one row per integer 0..max -> [first token id, second token id or -1, n tokens].
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA


def main():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    tdir = DATA / "gpt2_tokenizer"
    tdir.mkdir(parents=True, exist_ok=True)
    for f in ["vocab.json", "merges.txt"]:
        (tdir / f).write_bytes(Path(hf_hub_download("openai-community/gpt2", f)).read_bytes())
    tok = Tokenizer(BPE.from_file(str(tdir / "vocab.json"), str(tdir / "merges.txt")))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    check = [tok.decode([i]) for i in tok.encode(" 1269.35").ids]
    assert check == [" 12", "69", ".", "35"], f"unexpected GPT-2 split: {check}"

    cols = ["account_balance_usd", "amount_usd"]
    mx = int(max(pd.read_csv(DATA / f, usecols=cols).max().max() for f in ["train.csv", "test.csv"])) + 1
    ids = [e.ids for e in tok.encode_batch([" " + str(v) for v in range(mx + 1)])]
    table = np.array([[i[0], i[1] if len(i) > 1 else -1, len(i)] for i in ids], dtype=np.int64)
    np.save(DATA / "gpt2_int_tokens.npy", table)
    print(f"token table {table.shape}; share of integers with 2+ tokens: {(table[:, 2] > 1).mean():.3f}")


if __name__ == "__main__":
    main()
