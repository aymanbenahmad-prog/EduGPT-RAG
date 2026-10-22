"""Prompts reproduced verbatim from Appendix A.1 and A.2 of the manuscript."""

COT_PROMPT = (
    "System: You are an academic assistant to an expert. Respond by answering questions with the use of "
    "information in the context passages below. Do not use any external knowledge.\n\n"
    "Context passages: {passages}\n\n"
    "Question: {question}\n\n"
    "Instructions: Step 1: Find out which of the passages above is the most relevant to the question. "
    "Step 2: Find the particular sentence(s) of evidence that justify the answer. Step 3: Reason logically, "
    "starting with the evidence, to the answer. Step 4: Short, straightforward answer with a confidence "
    "rating [0.00 to 1.00].\n\n"
    "Format: Most relevant passage: [passage number] Evidence: [quoted evidence sentence(s)] "
    "Reasoning: [step-by-step reasoning] Answer: [concise answer] Confidence: [0.00 to 1.00]."
)

# Used for the "- CoT prompting" ablation: same grounding instruction, no reasoning steps.
DIRECT_PROMPT = (
    "System: You are an academic assistant to an expert. Respond by answering questions with the use of "
    "information in the context passages below. Do not use any external knowledge.\n\n"
    "Context passages: {passages}\n\n"
    "Question: {question}\n\n"
    "Format: Most relevant passage: [passage number] Answer: [concise answer]"
)

ABSTAIN_STRING = "I cannot answer this question using the information available."

CONSTRAINED_PROMPT = (
    "System: The answer you gave earlier was considered unfaithful to the source. You have to respond based "
    "only on the precise information in the context passages below. Do not make inferences, generalisations, "
    "or bring on board any information that is not explicitly stated in the text. In case the answer is not "
    "available in the passages, answer: '" + ABSTAIN_STRING + "'\n\n"
    "Context passages: {passages}\n\n"
    "Question: {question}\n\n"
    "Restricted Response (only based on the above passages):"
)

# Question-only prompt for the Vanilla GPT-4 baseline (no retrieval context).
CLOSED_BOOK_PROMPT = (
    "Answer the question with a short span. If the question cannot be answered, reply exactly: '"
    + ABSTAIN_STRING + "'\n\nQuestion: {question}\nAnswer:"
)


def format_passages(passages):
    return " ".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
