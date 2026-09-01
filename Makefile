.PHONY: help install test lint fmt tokenizer data train train-moe ablate eval generate chat clean smoke

PY ?= .venv/bin/python
CONFIG ?= configs/train/pretrain-300m.yaml

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## create the venv and install everything
	uv venv --python 3.12 .venv || python3 -m venv .venv
	$(PY) -m pip install -e ".[dev,data]"

test:  ## run the full test suite
	$(PY) -m pytest tests/ -q

lint:  ## static checks
	$(PY) -m ruff check slm tests scripts

fmt:  ## autofix lint
	$(PY) -m ruff check --fix slm tests scripts

smoke:  ## end-to-end pipeline on TinyShakespeare (~1 minute, CPU)
	bash scripts/smoke.sh

tokenizer:  ## train the 32k BPE tokenizer
	$(PY) scripts/train_tokenizer.py $(SOURCE) -o data/tokenizer.json --vocab-size 32768

data:  ## tokenise a corpus into shards
	$(PY) scripts/prepare_data.py $(SOURCE) -o data/tokens/train -t data/tokenizer.json

train:  ## train the dense 300M flagship
	$(PY) scripts/train.py $(CONFIG)

train-moe:  ## train the sparse 500M flagship
	$(PY) scripts/train.py configs/train/pretrain-moe-500m.yaml

ablate:  ## run the A1-A10 ablation grid
	$(PY) scripts/run_ablation.py --all

eval:  ## evaluate a checkpoint
	$(PY) scripts/evaluate.py $(CKPT) -t data/tokenizer.json

generate:  ## sample from a checkpoint
	$(PY) scripts/generate.py $(CKPT) -t data/tokenizer.json --prompt "$(PROMPT)"

chat:  ## interactive chat with an SFT checkpoint
	$(PY) scripts/chat.py $(CKPT) -t data/tokenizer.json

finish:  ## gracefully end a run: decay the LR and stop
	@echo "$(STEPS)" > runs/$(RUN)/FINISH && echo "FINISH written to runs/$(RUN)"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
