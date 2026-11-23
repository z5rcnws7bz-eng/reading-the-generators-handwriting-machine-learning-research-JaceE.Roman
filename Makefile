# One-command reproduction of the paper. Run `make help` for the targets.
PY ?= python
NN  = --noval --minc 50 --lr 1e-3 --l2 1e-4 --drop 0.3 --epochs 6

.PHONY: help setup tokens forensics ablation models blend figures paper all

help:
	@echo "setup      install Python dependencies"
	@echo "tokens     build the GPT-2 integer token table"
	@echo "forensics  funnel test -> results/results_forensics.csv"
	@echo "ablation   ablation ladder + artefact-only model -> results/results_ablation.jsonl"
	@echo "models     train the five ensemble members (OOF + test predictions in oof/)"
	@echo "blend      nested-CV logit-rank blend -> submission.csv"
	@echo "figures    manuscript figures from results/"
	@echo "paper      build the Word manuscript (needs Node.js)"
	@echo "all        tokens forensics ablation models blend figures"

setup:
	$(PY) -m pip install -r requirements.txt

tokens:
	$(PY) src/gpt2_tokens.py

forensics:
	$(PY) src/forensics.py

ablation:
	$(PY) src/exp3.py artefact_only raw_shallow digits digits_te
	$(PY) src/train3.py lgb --nf
	$(PY) src/train3.py lgb --tok --nf

models:
	$(PY) src/train3.py lgb --tok --nf --lr 0.02 --seeds 3 --tag _s3
	$(PY) src/train3.py xgb --tok --nf
	$(PY) src/train3.py cat --nf
	$(PY) src/glm.py
	for s in 0 1 2 3 4; do $(PY) src/nn_tok.py $(NN) --seed $$s --tag _a_s$$s; done
	$(PY) src/bag_nn.py nn_tok_a_s 5 nn_tok_bag5

blend:
	$(PY) src/ensemble.py _f5 glm_tok_nf_oof nn_tok_bag5_oof

figures:
	$(PY) manuscript/make_figures.py

paper:
	cd manuscript && npm install && node build_docx.js

all: tokens forensics ablation models blend figures
