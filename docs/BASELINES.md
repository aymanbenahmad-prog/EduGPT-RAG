# Baselines (Table 1)

All baselines answer the same 11,873 SQuAD 2.0 development questions and are scored by the same
code (`evaluation/squad_eval.py`, official SQuAD 2.0 normalisation). Systems that retrieve from
this corpus use the index of Section 4.1. Settings are in `configs/baselines/baselines.yaml`.

| Baseline | How it runs | Checkpoint | Notes |
|---|---|---|---|
| Vanilla GPT-4 | `run_baselines.py --baseline vanilla_gpt4` | OpenAI `gpt-4` | question only, no retrieval |
| BM25 + GPT-3.5 | run `t2_bm25_only` (`run_ablation.py --table 2`) | GPT-3.5-turbo | identical to Table 2 row 1 |
| DPR + BART | `--baseline dpr_bart` | facebook/dpr-*-single-nq-base, facebook/bart-large | top-5 DPR passages from this corpus |
| ColBERT-QA | `--baseline colbert_qa` | colbert-ir/colbertv2.0 (PLAID) | reader not stated in the manuscript; GPT-3.5 CoT reader |
| REALM | `--baseline realm` | google/realm-orqa-nq-openqa | transformers 4.36; retrieves from its own block records |
| FiD | export / official reader / import | FiD `nq_reader_large` | see below |
| Self-RAG | `--baseline self_rag` | selfrag/selfrag_llama2-7b | threshold 0.5; re-implemented short-form inference |
| Atlas | export / official reader / import | Atlas `atlas-large` | see below |

## FiD and Atlas

```bash
python experiments/run_baselines.py --baseline fid --stage export      # writes reader_input.json per seed
# FiD:   python test_reader.py --model_path <nq_reader_large> --eval_data results/raw/fid/seed42/reader_input.json \
#          --n_context 100 --write_results ; copy final_output.txt to results/raw/fid/seed42/reader_predictions.txt
# Atlas: python evaluate.py --reader_model_type google/t5-large-lm-adapt --model_path <atlas-large> \
#          --eval_data results/raw/atlas/seed42/reader_input.json --n_context 100 --write_results
#          (copy the *.jsonl predictions to results/raw/atlas/seed42/reader_predictions.txt)
python experiments/run_baselines.py --baseline fid --stage import
python evaluation/run_hallucination.py --run fid && python evaluation/squad_eval.py --run fid
```

## Hallucination metrics for every baseline

`evaluation/run_hallucination.py --run <name>` scores any baseline's answers with the same NLI
model and thresholds as EduGPT-RAG, so Table 5 can include all eight baselines, Atlas and
REALM included. Citation accuracy needs a cited passage index and is reported only for systems
that produce one (EduGPT-RAG, ColBERT-QA with the CoT reader, Self-RAG with its selected passage).
