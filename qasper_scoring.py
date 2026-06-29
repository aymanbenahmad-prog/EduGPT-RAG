"""
EduGPT-RAG: Official QASPER Scoring Script
Computes per-type F1 and abstention accuracy from a predictions JSON file.

Usage:
    python qasper_scoring.py \\
        --predictions results/qasper_predictions.json \\
        --output results/qasper_scores.json

Prediction JSON format (one object per question):
[
  {
    "paper_id": "...",
    "question": "...",
    "question_type": "extractive|abstractive|yes_no|unanswerable",
    "prediction": "...",
    "gold_answers": ["answer1", "answer2"],
    "faithfulness": 0.91
  },
  ...
]
"""

import json
import re
import string
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    text = " ".join(text.split())
    return text


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize(prediction).split()
    gt_toks = normalize(ground_truth).split()
    if not pred_toks and not gt_toks:
        return 1.0
    if not pred_toks or not gt_toks:
        return 0.0
    common = set(pred_toks) & set(gt_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec = len(common) / len(gt_toks)
    return 2 * prec * rec / (prec + rec)


def max_f1_over_references(prediction: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_f1_score(prediction, gt) for gt in gold_answers)


def is_abstention(prediction: str) -> bool:
    pred_lower = prediction.lower().strip()
    return (
        pred_lower == "" or
        "cannot answer" in pred_lower or
        "unanswerable" in pred_lower or
        "not answerable" in pred_lower or
        "no information" in pred_lower
    )


def compute_scores(predictions: List[Dict]) -> Dict:
    scores = defaultdict(list)

    for item in predictions:
        qtype = item["question_type"]
        prediction = item.get("prediction", "").strip()
        gold_answers = item.get("gold_answers", [])

        if qtype == "extractive":
            f1 = max_f1_over_references(prediction, gold_answers)
            scores["extractive_f1"].append(f1)

        elif qtype == "abstractive":
            f1 = max_f1_over_references(prediction, gold_answers)
            scores["abstractive_f1"].append(f1)

        elif qtype == "yes_no":
            pred_norm = normalize(prediction)
            acc = 0.0
            for gt in gold_answers:
                gt_norm = normalize(gt)
                if gt_norm in pred_norm or pred_norm in gt_norm:
                    acc = 1.0
                    break
            scores["yes_no_accuracy"].append(acc)

        elif qtype == "unanswerable":
            abstained = is_abstention(prediction)
            scores["abstention_accuracy"].append(1.0 if abstained else 0.0)

    results = {}
    for metric, vals in scores.items():
        results[metric] = {
            "score": round(float(np.mean(vals)) * 100, 2),
            "count": len(vals),
        }

    # Average F1 across extractive, abstractive, yes/no
    avg_components = []
    for key in ("extractive_f1", "abstractive_f1", "yes_no_accuracy"):
        if key in results:
            avg_components.append(results[key]["score"])
    if avg_components:
        results["average_f1"] = round(float(np.mean(avg_components)), 2)

    return results


def print_results(results: Dict):
    print("\n" + "=" * 50)
    print("QASPER Evaluation Results")
    print("=" * 50)
    label_map = {
        "extractive_f1":     "Extractive F1   ",
        "abstractive_f1":    "Abstractive F1  ",
        "yes_no_accuracy":   "Yes/No Accuracy ",
        "abstention_accuracy": "Abstention Acc. ",
        "average_f1":        "Average F1      ",
    }
    for key, label in label_map.items():
        if key in results:
            val = results[key]
            if isinstance(val, dict):
                print(f"  {label}: {val['score']:.2f}%  (n={val['count']})")
            else:
                print(f"  {label}: {val:.2f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="QASPER Scoring for EduGPT-RAG")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--output", default="results/qasper_scores.json", help="Output scores JSON")
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {args.predictions}")
    results = compute_scores(predictions)
    print_results(results)

    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to: {args.output}")


if __name__ == "__main__":
    main()
