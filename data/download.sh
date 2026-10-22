#!/usr/bin/env bash
# Downloads every public dataset used by the experiments into data/raw/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw/squad2 raw/qasper raw/msmarco

# SQuAD 2.0 (CC BY-SA 4.0) - https://rajpurkar.github.io/SQuAD-explorer/
for f in train-v2.0.json dev-v2.0.json; do
  [ -f raw/squad2/$f ] || curl -fL -o raw/squad2/$f https://rajpurkar.github.io/SQuAD-explorer/dataset/$f
done
# Optional cross-check: save the official "Evaluation Script v2.0" linked from the SQuAD website
# as raw/squad2/evaluate-v2.0.py; evaluation/squad_eval.py --official will then run it as well.

# QASPER v0.3 (CC BY 4.0) - https://allenai.org/data/qasper
if [ ! -f raw/qasper/qasper-train-v0.3.json ]; then
  curl -fL -o raw/qasper/qasper-train-dev-v0.3.tgz https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
  curl -fL -o raw/qasper/qasper-test-and-evaluator-v0.3.tgz https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
  tar -xzf raw/qasper/qasper-train-dev-v0.3.tgz -C raw/qasper
  tar -xzf raw/qasper/qasper-test-and-evaluator-v0.3.tgz -C raw/qasper
fi

# MS MARCO passage ranking (re-ranker training) - https://microsoft.github.io/msmarco/
if [ ! -f raw/msmarco/triples.train.small.tsv ]; then
  curl -fL -o raw/msmarco/triples.train.small.tar.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz
  tar -xzf raw/msmarco/triples.train.small.tar.gz -C raw/msmarco
fi

# NLTK sentence tokeniser (Punkt)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

sha256sum raw/squad2/*.json > raw/SHA256SUMS.txt
echo "done"
