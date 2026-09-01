#!/usr/bin/env bash
# Full pipeline on TinyShakespeare: tokenizer -> shards -> train -> resume -> sample.
set -euo pipefail
PY="${PYTHON:-.venv/bin/python}"
RAW=data/raw/tinyshakespeare.txt
TOK=data/tokenizer-shakespeare.json
OUT=data/tokens/shakespeare

echo "=== 1/5 corpus ==="
mkdir -p data/raw
[[ -f $RAW ]] || curl -sL -o $RAW \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c $RAW

echo "=== 2/5 tokenizer ==="
$PY scripts/train_tokenizer.py $RAW -o $TOK --vocab-size 4096 2>&1 | tail -2

echo "=== 3/5 shards ==="
rm -rf $OUT
$PY scripts/prepare_data.py $RAW -o $OUT/train -t $TOK \
   --val-dir $OUT/val --val-fraction 0.02 --split-blank --workers 2 \
   --shard-tokens 100000 2>&1 | tail -1

echo "=== 4/5 train, get preempted, resume ==="
rm -rf runs/smoke
# identical command both times: the budget never changes, only the machine dies.
# max_runtime_sec forces the first launch to be cut short mid-run.
BUDGET="train.run_name=smoke train.total_tokens=500000 ckpt.interval=10"
$PY scripts/train.py configs/train/debug.yaml $BUDGET train.max_runtime_sec=4 2>&1 \
  | grep -E "preempt|stopped early|final save" | head -3
echo "--- same command again: must resume, not restart ---"
$PY scripts/train.py configs/train/debug.yaml $BUDGET 2>&1 \
  | grep -E "resume|step +(6[01])/|DONE" | head -4
[[ -f runs/smoke/DONE ]] && echo "DONE marker written: supervisor would stop here"

echo "=== 5/5 sample ==="
$PY scripts/generate.py "$(cat runs/smoke/checkpoints/LATEST | tr -d '\n' | sed 's|^|runs/smoke/checkpoints/|')" \
   -t $TOK --prompt "KING RICHARD:" -n 60 --temperature 0.8 2>&1 | tail -4

echo "=== smoke passed ==="
