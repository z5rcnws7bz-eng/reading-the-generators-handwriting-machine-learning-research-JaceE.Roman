# Notices, data terms and compliance

This file sets out what the repository contains, what it deliberately leaves out, the licences of third-party
material, and the conditions for responsible use.

## 1. Licence of this repository

The source code, figures, result tables and manuscript in this repository are released under the
[MIT licence](LICENSE), © 2026 Jace Roman, unless a section below says otherwise.

## 2. Data that are **not** included

| Material | Owner / licence | Status in this repository |
|---|---|---|
| Kaggle Playground Series S6E11 `train.csv`, `test.csv`, `sample_submission.csv` | Kaggle, under the [competition rules](https://www.kaggle.com/competitions/playground-series-s6e11/rules) | **Not redistributed.** Download from Kaggle after accepting the rules. |
| Out-of-fold predictions, test predictions, submission files | Derived from the competition data | **Not redistributed.** Regenerate them with the code. |
| Credit Card Fraud Detection 2026 (uditjain13) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Not included, for consistency; free to download from Kaggle. |
| GPT-2 tokenizer `vocab.json`, `merges.txt` | OpenAI, Modified MIT licence, via Hugging Face `openai-community/gpt2` | Not included; downloaded at run time by `src/gpt2_tokens.py`. |

`results/` contains only aggregate statistics computed by this project (AUC values, shares of value groups),
which do not allow any row of the competition data to be reconstructed.

## 3. Kaggle competition rules

- **No data redistribution.** The repository holds no competition data or row-level predictions.
- **Code sharing during the competition.** Kaggle's rules allow sharing code during a competition only when
  it is made available to all participants, for example on Kaggle's own forums or notebooks; private
  sharing outside a team is not allowed. While the competition is open, this repository should stay
  private, or its code should also be posted publicly on Kaggle. After the competition closes the
  repository can be made public.
- **Prior work is credited.** The GPT-2 token-key and neighbour-frequency ideas build on the public notebook
  [*How GReaT Tokenizes Numbers*](https://www.kaggle.com/code/azizdzhon/how-great-tokenizes-numbers-bpe-token-target)
  by azizdzhon. No code was copied from it; the implementation here was written independently.

Always check the current competition rules yourself; this summary is not legal advice.

## 4. Third-party software

The code depends on, but does not bundle, these open-source packages: NumPy, pandas, SciPy, scikit-learn (BSD-3-Clause);
LightGBM, XGBoost (MIT / Apache 2.0); CatBoost (Apache 2.0); PyTorch (BSD-3-Clause); Hugging Face `tokenizers`
and `huggingface_hub` (Apache 2.0); Matplotlib (PSF-compatible); PyMuPDF (AGPL-3.0, optional, used only to
inspect manuscript pages); docx (npm, MIT). Each is used under its own licence.

## 5. Intended use and limitations

- **Research and education only.** The models were trained and evaluated on *synthetic* data for a public
  benchmark. They have not been validated on real transactions and are not a fraud-detection product.
- **Do not use for decisions about individuals** (for example declining payments, freezing accounts or
  credit decisions) without independent validation on real, representative data, a fairness and bias
  assessment, human oversight, and compliance with the laws that apply to you. Depending on the
  jurisdiction these may include data-protection law (for example the GDPR), consumer-credit and
  anti-discrimination law, payment-industry rules, and AI regulation (for example the EU AI Act).
- **No personal data.** The competition data are synthetic, and the source dataset states that it contains
  no real cardholder, bank or transaction records. This repository processes no personal data.
- **The main finding is about artefacts.** The strongest features exploit how a language model *wrote* the
  synthetic data. They are shortcuts specific to this generator and will not transfer to real fraud.

## 6. Disclosure of AI assistance

The analysis code, experiments, figures and the manuscript draft were produced with the help of Claude
(Anthropic), an AI assistant, under the author's direction. The author reviewed the results and is
responsible for the content. This matches the disclosure in the manuscript.

## 7. Warranty

The software is provided "as is", without warranty of any kind (see [LICENSE](LICENSE)).
