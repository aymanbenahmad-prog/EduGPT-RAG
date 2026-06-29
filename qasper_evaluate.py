"""
EduGPT-RAG: Full QASPER Development Split Evaluation Pipeline
Dataset: QASPER (Dasigi et al., 2021)
Split: Development (93 papers, 572 questions)
Metrics: Extractive F1, Abstractive F1, Yes/No Accuracy, Abstention Accuracy

Run:
    python qasper_evaluate.py --model edugpt_rag --output results/qasper_results.json

Dependencies:
    pip install datasets transformers sentence-transformers faiss-gpu pyserini openai torch deberta
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    "qasper_split": "validation",
    "qasper_num_dev_papers": 93,
    "qasper_num_dev_questions": 572,
    "chunk_window_tokens": 512,
    "chunk_stride_tokens": 128,
    "faiss_index_type": "IVF4096,PQ64",
    "faiss_nprobe": 128,
    "sbert_model": "sentence-transformers/all-mpnet-base-v2",
    "sbert_dim": 768,
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "rrf_kappa": 60,
    "rrf_n_candidates": 20,
    "crossencoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "top_k_passages": 5,
    "generator_model": "gpt-3.5-turbo",
    "generator_temperature": 0.1,
    "generator_max_tokens": 512,
    "nli_model": "cross-encoder/nli-deberta-v3-large",
    "faithfulness_threshold": 0.75,
    "random_seeds": [42, 123, 456],
    "openai_api_key_env": "OPENAI_API_KEY",
}

# ================================================================
# STEP 1: LOAD QASPER DATASET
# ================================================================

def load_qasper_dev_split() -> Tuple[List[Dict], List[Dict]]:
    """
    Load QASPER development split.
    Returns: (papers_list, questions_list)
    papers_list: list of paper dicts with 'title', 'abstract', 'full_text' fields
    questions_list: list of question dicts with 'question', 'answers', 'question_type', 'paper_id'
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("allenai/qasper", split="validation")
        print(f"Loaded QASPER validation: {len(dataset)} papers")
    except Exception as e:
        print(f"ERROR loading QASPER from HuggingFace: {e}")
        print("Falling back to local JSON. Download from: https://github.com/allenai/qasper-led-baseline")
        raise

    papers = []
    questions = []

    for paper in dataset:
        paper_id = paper.get("id", paper.get("paper_id", ""))
        full_text_sections = paper.get("full_text", {})

        # Flatten full text into a single string preserving section structure
        full_text_str = ""
        if isinstance(full_text_sections, dict):
            for section_name, section_paragraphs in full_text_sections.items():
                full_text_str += f"\n\n== {section_name} ==\n"
                if isinstance(section_paragraphs, list):
                    full_text_str += " ".join(section_paragraphs)
                else:
                    full_text_str += str(section_paragraphs)
        elif isinstance(full_text_sections, list):
            for item in full_text_sections:
                if isinstance(item, dict):
                    full_text_str += f"\n\n== {item.get('section_name', '')} ==\n"
                    paras = item.get("paragraphs", [])
                    if paras:
                        full_text_str += " ".join(paras)

        paper_obj = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "full_text": full_text_str,
        }
        papers.append(paper_obj)

        # Extract questions for this paper
        qas = paper.get("qas", [])
        if not isinstance(qas, list):
            continue
        for qa in qas:
            q_text = qa.get("question", "")
            answers_raw = qa.get("answers", [])
            answers = []
            for ans_obj in answers_raw:
                if isinstance(ans_obj, dict):
                    ans_annotation = ans_obj.get("answer", {})
                    if isinstance(ans_annotation, dict):
                        if ans_annotation.get("unanswerable", False):
                            answers.append({"type": "unanswerable", "text": ""})
                        elif ans_annotation.get("yes_no") is not None:
                            yn = ans_annotation["yes_no"]
                            answers.append({"type": "yes_no", "text": "yes" if yn else "no"})
                        elif ans_annotation.get("extractive_spans"):
                            spans = ans_annotation["extractive_spans"]
                            answers.append({"type": "extractive", "text": " ".join(spans) if isinstance(spans, list) else str(spans)})
                        elif ans_annotation.get("free_form_answer"):
                            answers.append({"type": "abstractive", "text": ans_annotation["free_form_answer"]})

            if not answers:
                continue

            # Determine primary question type (majority vote)
            type_counts = defaultdict(int)
            for a in answers:
                type_counts[a["type"]] += 1
            primary_type = max(type_counts, key=type_counts.get)

            questions.append({
                "paper_id": paper_id,
                "question": q_text,
                "answers": answers,
                "question_type": primary_type,
            })

    print(f"Total papers: {len(papers)}, Total questions: {len(questions)}")
    return papers, questions


# ================================================================
# STEP 2: BUILD RETRIEVAL CORPUS FROM TRAINING PAPERS
# ================================================================

def chunk_text(text: str, window: int = 512, stride: int = 128) -> List[str]:
    """
    Hierarchical chunking: split text into overlapping windows at sentence boundaries.
    Simple token-approximate version (splits on spaces).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + window, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start += (window - stride)
    return chunks


def build_retrieval_corpus(training_papers: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Build passage corpus from training papers.
    Returns: (passages, passage_ids)
    """
    passages = []
    passage_ids = []
    for paper in training_papers:
        text = paper.get("abstract", "") + " " + paper.get("full_text", "")
        chunks = chunk_text(
            text,
            window=CONFIG["chunk_window_tokens"],
            stride=CONFIG["chunk_stride_tokens"]
        )
        for i, chunk in enumerate(chunks):
            passages.append(chunk)
            passage_ids.append(f"{paper['paper_id']}_chunk_{i}")
    print(f"Total passage chunks: {len(passages)}")
    return passages, passage_ids


def build_faiss_index(passages: List[str]):
    """Build FAISS IVF-PQ index over passage embeddings."""
    import faiss
    from sentence_transformers import SentenceTransformer

    print("Encoding passages with SBERT...")
    model = SentenceTransformer(CONFIG["sbert_model"])
    embeddings = model.encode(passages, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    d = embeddings.shape[1]
    print(f"Embedding dimension: {d}, Training FAISS IVF4096,PQ64...")

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, 4096, 64, 8)
    index.nprobe = CONFIG["faiss_nprobe"]

    # Train on subset
    train_size = min(100000, len(embeddings))
    index.train(embeddings[:train_size])
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} passages")
    return index, model, embeddings


# ================================================================
# STEP 3: HYBRID RETRIEVAL + RRF
# ================================================================

def retrieve_hybrid(query: str, passages: List[str], faiss_index, sbert_model,
                    bm25_index, n_candidates: int = 20) -> List[Tuple[int, float]]:
    """
    Hybrid BM25 + SBERT retrieval with RRF fusion.
    Returns list of (passage_idx, rrf_score) sorted descending.
    """
    import faiss
    import numpy as np

    # Dense retrieval
    q_emb = sbert_model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    _, dense_indices = faiss_index.search(q_emb, n_candidates)
    dense_indices = dense_indices[0].tolist()

    # Sparse BM25 retrieval
    sparse_results = bm25_index.search(query, k=n_candidates)
    sparse_indices = [r["doc_id"] for r in sparse_results] if sparse_results else []

    # RRF fusion
    rrf_scores = defaultdict(float)
    kappa = CONFIG["rrf_kappa"]
    for rank, idx in enumerate(dense_indices):
        if idx >= 0:
            rrf_scores[idx] += 1.0 / (kappa + rank + 1)
    for rank, idx in enumerate(sparse_indices):
        if idx >= 0:
            rrf_scores[idx] += 1.0 / (kappa + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_candidates]
    return fused


def rerank_passages(query: str, candidates: List[Tuple[int, float]],
                    passages: List[str], top_k: int = 5) -> List[str]:
    """Cross-encoder re-ranking."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(CONFIG["crossencoder_model"])
    pairs = [(query, passages[idx]) for idx, _ in candidates if idx < len(passages)]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = [passages[idx] for (idx, _), _ in ranked[:top_k] if idx < len(passages)]
    return top


# ================================================================
# STEP 4: GENERATION WITH COT PROMPT
# ================================================================

COT_PROMPT_TEMPLATE = """You are an academic assistant. Answer the question using ONLY the context passages below. Do not use any external knowledge.

Context passages:
{passages}

Question: {question}

Instructions:
Step 1: Identify the most relevant passage.
Step 2: Extract the key evidence sentence(s).
Step 3: Reason step-by-step from the evidence to the answer.
Step 4: Give a concise answer with a confidence score [0.00 to 1.00].

Format:
Most relevant passage: [number]
Evidence: [quoted sentence(s)]
Reasoning: [step-by-step]
Answer: [concise answer]
Confidence: [0.00 to 1.00]"""

CONSTRAINED_PROMPT_TEMPLATE = """The previous answer was unfaithful to the source. You must answer using ONLY information explicitly stated in the passages below. Do not make inferences or add external knowledge. If the answer is not in the passages, respond: 'I cannot answer this question using the available information.'

Context passages:
{passages}

Question: {question}

Restricted Response (based only on the above passages):"""


def generate_answer(question: str, context_passages: List[str], client) -> Tuple[str, float]:
    """Generate answer with CoT prompt, apply NLI verification."""
    from openai import OpenAI

    passages_str = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(context_passages)])

    prompt = COT_PROMPT_TEMPLATE.format(passages=passages_str, question=question)
    response = client.chat.completions.create(
        model=CONFIG["generator_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG["generator_temperature"],
        max_tokens=CONFIG["generator_max_tokens"],
    )
    answer = response.choices[0].message.content.strip()

    # Extract just the "Answer:" line if present
    for line in answer.split("\n"):
        if line.strip().startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
            break

    # NLI Faithfulness check
    faith_score = compute_faithfulness(answer, context_passages)
    if faith_score < CONFIG["faithfulness_threshold"]:
        # Constrained regeneration
        constrained_prompt = CONSTRAINED_PROMPT_TEMPLATE.format(
            passages=passages_str, question=question
        )
        response2 = client.chat.completions.create(
            model=CONFIG["generator_model"],
            messages=[{"role": "user", "content": constrained_prompt}],
            temperature=CONFIG["generator_temperature"],
            max_tokens=CONFIG["generator_max_tokens"],
        )
        answer = response2.choices[0].message.content.strip()

    return answer, faith_score


def compute_faithfulness(answer: str, context_passages: List[str]) -> float:
    """NLI-based faithfulness score using DeBERTa-v3."""
    from sentence_transformers import CrossEncoder
    import nltk
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize

    model = CrossEncoder(CONFIG["nli_model"])
    sentences = sent_tokenize(answer)
    if not sentences:
        return 0.0

    scores = []
    for sent in sentences:
        passage_scores = []
        for passage in context_passages:
            score = model.predict([(passage, sent)])
            passage_scores.append(float(score[0]) if hasattr(score, "__len__") else float(score))
        scores.append(max(passage_scores))
    return float(np.mean(scores))


# ================================================================
# STEP 5: EVALUATION METRICS
# ================================================================

def normalize_answer(text: str) -> str:
    """Lower, strip punctuation, collapse whitespace."""
    import re
    import string
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(predictions: List[Dict], questions: List[Dict]) -> Dict:
    """
    Compute per-type F1 and abstention accuracy.
    predictions: list of {"question": str, "prediction": str, "question_type": str, "answers": list}
    """
    results_by_type = defaultdict(list)

    for pred in predictions:
        qtype = pred["question_type"]
        prediction = pred["prediction"].strip()
        answers = pred["answers"]
        gt_texts = [a["text"] for a in answers if a["type"] == qtype]

        if qtype == "unanswerable":
            # Abstention: prediction should be empty or contain 'cannot answer'
            is_abstain = (
                prediction == "" or
                "cannot answer" in prediction.lower() or
                "unanswerable" in prediction.lower()
            )
            results_by_type["unanswerable"].append(1.0 if is_abstain else 0.0)

        elif qtype == "yes_no":
            if not gt_texts:
                continue
            gt = gt_texts[0].lower().strip()
            pred_norm = prediction.lower().strip()
            is_correct = (gt in pred_norm) or (pred_norm in gt)
            results_by_type["yes_no"].append(1.0 if is_correct else 0.0)

        elif qtype in ("extractive", "abstractive"):
            if not gt_texts:
                continue
            f1_scores = [token_f1(prediction, gt) for gt in gt_texts]
            results_by_type[qtype].append(max(f1_scores))

    summary = {}
    for qtype, scores in results_by_type.items():
        summary[qtype] = {
            "score": float(np.mean(scores)) * 100 if qtype != "unanswerable" else float(np.mean(scores)) * 100,
            "count": len(scores),
            "metric": "F1" if qtype in ("extractive", "abstractive") else "Accuracy",
        }

    print("\n=== QASPER Evaluation Results ===")
    for qtype, info in summary.items():
        print(f"  {qtype:<15} {info['metric']}: {info['score']:.2f}%  (n={info['count']})")

    all_scores = []
    for qtype in ("extractive", "abstractive", "yes_no"):
        if qtype in summary:
            all_scores.append(summary[qtype]["score"])
    if all_scores:
        summary["average_f1"] = float(np.mean(all_scores))
        print(f"  {'Average F1':<15}: {summary['average_f1']:.2f}%")

    return summary


# ================================================================
# STEP 6: MAIN PIPELINE
# ================================================================

def run_qasper_eval(seed: int = 42) -> Dict:
    """Full QASPER evaluation pipeline for one random seed."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    all_papers, all_questions = load_qasper_dev_split()

    # Identify dev vs training papers
    # QASPER dev split: first 93 papers when loaded from validation split
    dev_paper_ids = set(p["paper_id"] for p in all_papers[:93])
    train_papers = [p for p in all_papers if p["paper_id"] not in dev_paper_ids]
    dev_questions = [q for q in all_questions if q["paper_id"] in dev_paper_ids]
    print(f"Training papers for corpus: {len(train_papers)}")
    print(f"Dev questions: {len(dev_questions)}")

    # Build retrieval corpus from training papers only
    passages, passage_ids = build_retrieval_corpus(train_papers)

    # Build indexes
    faiss_index, sbert_model, _ = build_faiss_index(passages)

    # BM25 (simplified using pyserini or rank-bm25)
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized_corpus, k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"])

        class BM25Wrapper:
            def __init__(self, bm25_model, passages):
                self.bm25 = bm25_model
                self.passages = passages
            def search(self, query, k=20):
                scores = self.bm25.get_scores(query.lower().split())
                top_n = np.argsort(scores)[::-1][:k]
                return [{"doc_id": int(i), "score": float(scores[i])} for i in top_n]

        bm25_index = BM25Wrapper(bm25, passages)
    except ImportError:
        print("rank_bm25 not installed. Install with: pip install rank-bm25")
        raise

    # OpenAI client
    import openai
    client = openai.OpenAI(api_key=os.environ.get(CONFIG["openai_api_key_env"], ""))

    # Run inference on dev questions
    predictions = []
    for i, q_obj in enumerate(dev_questions):
        if i % 50 == 0:
            print(f"Processing question {i+1}/{len(dev_questions)}...")

        query = q_obj["question"]
        candidates = retrieve_hybrid(query, passages, faiss_index, sbert_model, bm25_index)
        context = rerank_passages(query, candidates, passages, top_k=CONFIG["top_k_passages"])
        answer, faith_score = generate_answer(query, context, client)

        predictions.append({
            "question": query,
            "prediction": answer,
            "faithfulness": faith_score,
            "question_type": q_obj["question_type"],
            "answers": q_obj["answers"],
            "paper_id": q_obj["paper_id"],
        })

    # Evaluate
    results = evaluate_predictions(predictions, dev_questions)
    results["seed"] = seed
    results["num_questions"] = len(dev_questions)
    return results


def main():
    parser = argparse.ArgumentParser(description="EduGPT-RAG QASPER Evaluation")
    parser.add_argument("--model", default="edugpt_rag", help="Model name tag")
    parser.add_argument("--output", default="results/qasper_results.json", help="Output JSON path")
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["random_seeds"], help="Random seeds")
    parser.add_argument("--single_seed", type=int, default=None, help="Run only one seed (fast mode)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    seeds_to_run = [args.single_seed] if args.single_seed is not None else args.seeds
    all_results = []
    for seed in seeds_to_run:
        print(f"\n{'='*60}")
        print(f"Running evaluation with seed {seed}")
        print(f"{'='*60}")
        result = run_qasper_eval(seed=seed)
        all_results.append(result)

    # Aggregate across seeds
    if len(all_results) > 1:
        aggregate = {}
        metric_keys = ["extractive", "abstractive", "yes_no", "unanswerable"]
        for key in metric_keys:
            vals = [r[key]["score"] for r in all_results if key in r]
            if vals:
                aggregate[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "metric": all_results[0][key]["metric"]
                }
        if "average_f1" in all_results[0]:
            avg_vals = [r["average_f1"] for r in all_results if "average_f1" in r]
            aggregate["average_f1"] = {"mean": float(np.mean(avg_vals)), "std": float(np.std(avg_vals))}

        print("\n=== AGGREGATE RESULTS (mean +/- std over seeds) ===")
        for key, val in aggregate.items():
            if "std" in val:
                print(f"  {key:<15}: {val['mean']:.2f} +/- {val['std']:.2f}")
            else:
                print(f"  {key:<15}: {val}")

        output = {"model": args.model, "seeds": seeds_to_run, "per_seed": all_results, "aggregate": aggregate}
    else:
        output = {"model": args.model, "seeds": seeds_to_run, "results": all_results[0]}

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
