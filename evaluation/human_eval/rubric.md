# Human evaluation protocol (Section 4.5)

The manuscript reports a human evaluation of 200 randomly sampled SQuAD 2.0 development
answers rated by three annotators. **The original annotation files were not archived and are
not part of this package.** This folder provides the protocol and the analysis script so that
the evaluation can be repeated and its data released.

## Sampling
`python evaluation/human_eval/make_sheet.py --runs edugpt_full self_rag --n 200 --seed 0`
draws the same 200 question ids for both systems, shuffles the 400 (question, answer) items,
hides the system name behind an item id and writes `annotation_sheet.csv` plus a private
`key.csv` (kept away from annotators).

## Scales (1 to 3)
**Factual Accuracy**: 1 = incorrect or hallucinated, 2 = partially correct or incomplete,
3 = fully correct.
**Source Faithfulness**: 1 = contains claims not in the shown passages, 2 = mostly supported,
3 = every claim supported by the shown passages.
Annotators see the question, the five context passages and the answer. They do not see the
system identity or any automatic score.

## Analysis
`python evaluation/human_eval/kappa_pearson.py --sheet annotations.csv --key key.csv`
reports per-system means, Fleiss' kappa per scale, Pearson and Spearman correlation between the
mean Source Faithfulness rating and the NLI faithfulness score with 95% bootstrap intervals, and a
paired Wilcoxon signed-rank test between systems.
