# Contributing

Issues and pull requests are welcome.

- Keep the validation honest: use the fixed split in `src/common.py`, fit every supervised encoding
  inside the training fold, and report out-of-fold AUC. A public-leaderboard score is not evidence.
- Do not add competition data, predictions or submission files (see [NOTICE.md](NOTICE.md)).
- Describe each change with the OOF AUC before and after, and the number of folds that improved.
- By contributing you agree that your contribution is licensed under the [MIT licence](LICENSE).
