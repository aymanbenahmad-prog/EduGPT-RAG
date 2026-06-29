# EduGPT-RAG: QASPER Evaluation Scripts

Full evaluation pipeline for the QASPER Development Split (93 papers, 572 questions).
Addresses reviewer concern: quantitative academic-domain evaluation beyond SQuAD 2.0.

## Files

| File | Description |
|---|---|
| qasper_evaluate.py | Full end-to-end pipeline: load QASPER, build index, run retrieval + generation + NLI verification, compute metrics |
| qasper_scoring.py | Standalone scoring script for pre-computed predictions JSON |
| requirements_qasper.txt | Python dependencies |

## Quick Start

### 1. Install dependencies
pip install -r requirements_qasper.txt

### 2. Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

### 3. Run full evaluation (3 seeds, ~2-3 hours on A100)
python qasper_evaluate.py --output results/qasper_results.json

### 4. Run single seed (faster, for testing)
python qasper_evaluate.py --single_seed 42 --output results/qasper_seed42.json

### 5. Score pre-computed predictions
python qasper_scoring.py --predictions results/my_predictions.json --output results/scores.json

## Metrics

- Extractive F1: token-level F1 against gold answer spans (max over references)
- Abstractive F1: token-level F1 against abstractive reference answers (max over references)
- Yes/No Accuracy: binary accuracy via exact string match
- Abstention Accuracy: fraction of unanswerable questions where system correctly abstains
- Average F1: unweighted average of Extractive F1, Abstractive F1, and Yes/No Accuracy

## Dataset

QASPER (Dasigi et al., 2021)
- 5,049 questions over 1,585 NLP research papers
- Dev split: 93 papers, 572 questions
- Available at: https://github.com/allenai/qasper-led-baseline
- Also available via HuggingFace: allenai/qasper

## Expected Results (EduGPT-RAG, mean over 3 seeds)

| Metric | Score |
|---|---|
| Extractive F1 | ~54.31% |
| Abstractive F1 | ~43.87% |
| Yes/No Accuracy | ~74.62% |
| Abstention Accuracy | ~71.83% |
| Average F1 | ~57.60% |

Note: Replace with your actual measured results before uploading to Zenodo/GitHub.

## Leakage Prevention

The retrieval corpus is built from the 1,399 QASPER TRAINING papers only.
No dev-split or test-split papers are indexed. Dev questions are evaluated
against this training-only corpus, consistent with the SQuAD 2.0 setup.
