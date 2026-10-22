#!/usr/bin/env bash
# Regenerates every table and figure of the manuscript. Needs a CUDA GPU and OPENAI_API_KEY.
# Approximate cost and time are listed in RELEASE.md. Each step can be run on its own.
set -euo pipefail
export PYTHONPATH=src
bash data/download.sh
python data/build_corpus_squad.py                                  # Section 4.1 corpus (stats in corpus_stats.json)
for s in 42 123 456; do python training/train_reranker_msmarco.py --seed $s; done
python experiments/run_squad.py --run_name edugpt_full              # Table 1 (EduGPT-RAG row)
python evaluation/squad_eval.py --run edugpt_full
python experiments/run_ablation.py --table all                      # Tables 2, 3, 4
for b in vanilla_gpt4 dpr_bart colbert_qa realm self_rag; do       # Table 1 baselines
  python experiments/run_baselines.py --baseline $b
  python evaluation/run_hallucination.py --run $b                   # Table 5
  python evaluation/squad_eval.py --run $b
done
echo "FiD and Atlas: see docs/BASELINES.md (export -> official reader -> import)"
python evaluation/significance.py --a edugpt_full --b self_rag     # Section 4.2 tests
python evaluation/significance.py --a edugpt_full --b atlas || true
python experiments/sensitivity.py --study chunk                     # Table 6
python experiments/sensitivity.py --study embedding                 # Table 7
python experiments/sensitivity.py --study tau                       # Tables 9, 10
python experiments/latency_benchmark.py                             # Table 8
for p in paraphrase entity numeric; do                              # contamination probes
  python experiments/contamination_tests.py --probe $p
  python experiments/run_squad.py --run_name contam_$p --questions data/processed/squad2/contamination_$p.jsonl
  python evaluation/squad_eval.py --run contam_$p
done
python evaluation/independent_metric_check.py --run edugpt_full    # Reviewer 1, Comment 2
python tools/export_tables.py                                       # re-run tables vs manuscript
python figures/make_figures.py --source rerun
python tools/check_consistency.py
