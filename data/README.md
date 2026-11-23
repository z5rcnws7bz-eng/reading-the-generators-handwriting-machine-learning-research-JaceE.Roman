# Data

This folder is intentionally empty in the repository. **No competition data, predictions or
submissions are redistributed here.** Kaggle competition data are governed by the competition
rules, which you accept when you join the competition.

## 1. Competition data (required)

1. Sign in to Kaggle and join
   [Playground Series S6E11 – Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11),
   accepting its rules.
2. Download the data from the **Data** tab, or with the Kaggle CLI:
   `kaggle competitions download -c playground-series-s6e11`
3. Unzip so that this folder contains `train.csv`, `test.csv` and `sample_submission.csv`.

## 2. Original source dataset (optional, for the negative-result experiments)

[Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
by uditjain13 (CC0). Place `credit_card_fraud_2026.csv` in `data/original/`.

## 3. GPT-2 token table (generated)

Run `python src/gpt2_tokens.py`. It downloads `vocab.json` and `merges.txt` from the Hugging Face
`openai-community/gpt2` repository into `data/gpt2_tokenizer/` and writes `data/gpt2_int_tokens.npy`.

Expected layout:

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── original/credit_card_fraud_2026.csv      # optional
├── gpt2_tokenizer/{vocab.json, merges.txt}  # generated
└── gpt2_int_tokens.npy                      # generated
```
