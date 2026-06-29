"""
================================================================================
EduGPT-RAG: A Retrieval-Augmented Large Language Model for Context-Aware
Academic Question Answering
================================================================================
Complete Pipeline: Model Development, Training, and Testing
Paper: IEEE Transactions on Consumer Electronics
================================================================================

DEPENDENCIES (install before running):
    pip install torch transformers sentence-transformers faiss-cpu rank_bm25
    pip install datasets scikit-learn numpy tqdm openai langchain

USAGE:
    python edugpt_rag_complete.py

    By default uses SQuAD 2.0 dataset from HuggingFace (no API key needed).
    For GPT generation, set OPENAI_API_KEY env variable or use local generator.
================================================================================
"""

import os
import sys
import json
import time
import math
import random
import logging
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EduGPT-RAG")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EduGPTConfig:
    """Central configuration for the EduGPT-RAG pipeline (matches paper params)."""

    # Chunking
    chunk_window: int = 512        # w  — token window size
    chunk_stride: int = 128        # s  — overlap stride

    # Retrieval
    top_n_candidates: int = 20     # n  — fused candidates forwarded to re-ranker
    top_k_context: int = 5         # K  — final passages for generation
    top_kd_dense: int = 20         # k_d — dense candidates from FAISS
    top_ks_sparse: int = 20        # k_s — sparse candidates from BM25
    rrf_kappa: int = 60            # κ  — RRF smoothing constant

    # Embedding
    sbert_model: str = "all-MiniLM-L6-v2"   # lighter for CPU; paper uses all-mpnet-base-v2
    embedding_dim: int = 384

    # Cross-Encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # NLI / Faithfulness
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    faithfulness_threshold: float = 0.75   # τ

    # BM25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # Training
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_seq_length: int = 512
    lambda1: float = 0.3           # weight for CE loss
    lambda2: float = 0.2           # weight for faithfulness loss
    seed: int = 42

    # Dataset
    dataset_name: str = "rajpurkar/squad_v2"
    max_train_samples: int = 1000  # set None to use full dataset
    max_eval_samples: int = 200

    # Generation (fallback if no OpenAI key)
    use_local_generator: bool = True
    openai_model: str = "gpt-3.5-turbo"
    generation_temperature: float = 0.1
    max_tokens: int = 256

    # Output
    output_dir: str = "./edugpt_rag_output"
    device: str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. HIERARCHICAL DOCUMENT CHUNKER
# ─────────────────────────────────────────────────────────────────────────────
class HierarchicalChunker:
    """
    Segments documents into overlapping sentence-boundary-respecting passages.
    Implements Eq. (2) and Eq. (3) from the paper.
    """

    def __init__(self, window: int = 512, stride: int = 128):
        self.window = window
        self.stride = stride

    def _split_sentences(self, text: str) -> List[str]:
        """Naive sentence splitter (replace with spaCy for production)."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, doc_id: str, text: str) -> List[Dict]:
        """
        Chunk a single document into overlapping passages.

        Returns list of dicts: {id, doc_id, text, start_sent, end_sent}
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Build token-length-approximate sliding window over sentences
        chunks = []
        chunk_idx = 0
        i = 0

        while i < len(sentences):
            current_tokens = 0
            j = i
            while j < len(sentences):
                sent_tokens = len(sentences[j].split())
                if current_tokens + sent_tokens > self.window and j > i:
                    break
                current_tokens += sent_tokens
                j += 1

            chunk_text = " ".join(sentences[i:j])
            chunks.append({
                "id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_text,
                "start_sent": i,
                "end_sent": j - 1,
                "token_len": current_tokens,
            })
            chunk_idx += 1

            # Advance by stride (in sentence units, approximate)
            stride_tokens = 0
            step = 0
            while step < len(sentences) - i and stride_tokens < self.stride:
                stride_tokens += len(sentences[i + step].split())
                step += 1
            i += max(1, step)

        return chunks

    def chunk_corpus(self, documents: List[Dict]) -> List[Dict]:
        """Chunk an entire corpus. documents: [{id, text}, ...]"""
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking corpus"):
            chunks = self.chunk_document(doc["id"], doc["text"])
            all_chunks.extend(chunks)
        logger.info(f"Chunking complete: {len(documents)} docs → {len(all_chunks)} passages")
        return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# 2. HYBRID DENSE-SPARSE RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Combines BM25 (Eq. 4) and Sentence-BERT dense retrieval (Eq. 5-6)
    via Reciprocal Rank Fusion (Eq. 7-8). Indexes dense vectors in FAISS.
    """

    def __init__(self, config: EduGPTConfig):
        self.config = config
        self.passages: List[Dict] = []
        self.bm25 = None
        self.faiss_index = None
        self.embeddings: Optional[np.ndarray] = None

        logger.info(f"Loading SBERT model: {config.sbert_model}")
        from sentence_transformers import SentenceTransformer
        self.sbert = SentenceTransformer(config.sbert_model)

    # ── BM25 ──────────────────────────────────────────────────────────────────
    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        tokenized = [p["text"].lower().split() for p in self.passages]
        self.bm25 = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)
        logger.info("BM25 index built.")

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ── FAISS Dense ───────────────────────────────────────────────────────────
    def _build_faiss(self):
        try:
            import faiss
            dim = self.embeddings.shape[1]
            # Use flat index for small corpora; IVF-PQ for large (paper Eq. 6)
            if len(self.passages) < 5000:
                self.faiss_index = faiss.IndexFlatIP(dim)
            else:
                n_list = min(int(math.sqrt(len(self.passages))), 4096)
                quantizer = faiss.IndexFlatIP(dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, n_list)
                self.faiss_index.train(self.embeddings.astype(np.float32))
                self.faiss_index.nprobe = min(128, n_list)
            self.faiss_index.add(self.embeddings.astype(np.float32))
            logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")
        except ImportError:
            logger.warning("faiss not found — falling back to numpy cosine search.")
            self.faiss_index = None

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        q_emb = self.sbert.encode([query], normalize_embeddings=True).astype(np.float32)
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q_emb, top_k)
            return [(int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0]
        else:
            # Numpy fallback — cosine similarity
            sims = (self.embeddings @ q_emb.T).squeeze()
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(int(i), float(sims[i])) for i in top_idx]

    # ── Reciprocal Rank Fusion (Eq. 7) ───────────────────────────────────────
    def _rrf(
        self,
        bm25_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]],
        n: int,
    ) -> List[Tuple[int, float]]:
        kappa = self.config.rrf_kappa
        rrf_scores: Dict[int, float] = defaultdict(float)

        for rank, (idx, _) in enumerate(bm25_results, 1):
            rrf_scores[idx] += 1.0 / (kappa + rank)
        for rank, (idx, _) in enumerate(dense_results, 1):
            rrf_scores[idx] += 1.0 / (kappa + rank)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:n]

    # ── Index Building ────────────────────────────────────────────────────────
    def index(self, passages: List[Dict]):
        """Build BM25 + FAISS indexes from passage list."""
        self.passages = passages
        logger.info(f"Encoding {len(passages)} passages with SBERT...")
        texts = [p["text"] for p in passages]
        self.embeddings = self.sbert.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)
        self._build_bm25()
        self._build_faiss()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> List[Dict]:
        """Return top-n fused candidates for a query."""
        bm25_hits = self._bm25_search(query, self.config.top_ks_sparse)
        dense_hits = self._dense_search(query, self.config.top_kd_dense)
        fused = self._rrf(bm25_hits, dense_hits, self.config.top_n_candidates)
        return [
            {**self.passages[idx], "rrf_score": score}
            for idx, score in fused
            if idx < len(self.passages)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-ENCODER RE-RANKER
# ─────────────────────────────────────────────────────────────────────────────
class CrossEncoderReranker:
    """
    Joint query-passage scoring via BERT cross-encoder (Eq. 9).
    Trained with binary cross-entropy (Eq. 11).
    """

    def __init__(self, config: EduGPTConfig):
        self.config = config
        logger.info(f"Loading cross-encoder: {config.cross_encoder_model}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(config.cross_encoder_model)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Score and sort candidates; return top-K."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["ce_score"] = float(s)
        reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        return reranked[: self.config.top_k_context]

    def train_reranker(self, train_samples: List[Dict], output_path: str):
        """
        Fine-tune cross-encoder on relevance pairs.
        train_samples: [{query, passage, label}]  label ∈ {0, 1}
        Implements Eq. (11).
        """
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader

        examples = [
            InputExample(texts=[s["query"], s["passage"]], label=float(s["label"]))
            for s in train_samples
        ]
        loader = DataLoader(examples, shuffle=True, batch_size=self.config.batch_size)
        self.model.fit(
            train_dataloader=loader,
            epochs=self.config.num_epochs,
            warmup_steps=int(0.1 * len(loader) * self.config.num_epochs),
            output_path=output_path,
            show_progress_bar=True,
        )
        logger.info(f"Cross-encoder saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOCAL GPT GENERATOR (T5-based fallback, no API needed)
# ─────────────────────────────────────────────────────────────────────────────
class LocalGenerator:
    """
    Extractive/abstractive answer generator using a small T5 model.
    Used when OPENAI_API_KEY is not set.
    Implements Eq. (13) — CoT-prompted generation.
    """

    def __init__(self, config: EduGPTConfig):
        self.config = config
        logger.info("Loading local T5 generator (flan-t5-base)...")
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        model_name = "google/flan-t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    def _build_cot_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Build chain-of-thought prompt (paper Section III-E)."""
        ctx_text = ""
        for i, c in enumerate(contexts, 1):
            ctx_text += f"[Passage {i}] {c['text'][:400]}\n"

        prompt = (
            f"Answer the question based on the passages below.\n"
            f"Step 1: Identify the most relevant passage.\n"
            f"Step 2: Extract key evidence.\n"
            f"Step 3: Formulate a concise answer.\n\n"
            f"Passages:\n{ctx_text}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt

    def generate(self, query: str, contexts: List[Dict]) -> Dict:
        """Generate answer from query + top-K contexts."""
        prompt = self._build_cot_prompt(query, contexts)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            truncation=True,
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"answer": answer.strip(), "prompt": prompt}


class OpenAIGenerator:
    """GPT-based generator using the OpenAI API (paper Section III-E)."""

    def __init__(self, config: EduGPTConfig):
        self.config = config
        import openai
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _build_cot_prompt(self, query: str, contexts: List[Dict]) -> str:
        ctx_text = ""
        for i, c in enumerate(contexts, 1):
            ctx_text += f"[Passage {i}]: {c['text'][:600]}\n\n"
        return (
            "You are an academic QA assistant. Use the passages below to answer the question.\n"
            "Chain of thought:\n"
            "1. Identify the most relevant passage.\n"
            "2. Extract the key evidence.\n"
            "3. Reason step-by-step.\n"
            "4. Formulate a concise, citation-backed answer with a confidence score (0-1).\n\n"
            f"Passages:\n{ctx_text}"
            f"Question: {query}\nAnswer:"
        )

    def generate(self, query: str, contexts: List[Dict]) -> Dict:
        prompt = self._build_cot_prompt(query, contexts)
        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.generation_temperature,
            max_tokens=self.config.max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "prompt": prompt}


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANTI-HALLUCINATION VERIFICATION LAYER
# ─────────────────────────────────────────────────────────────────────────────
class AntiHallucinationVerifier:
    """
    Post-generation NLI entailment check (Eq. 14-15).
    Uses DeBERTa cross-encoder NLI model.
    """

    def __init__(self, config: EduGPTConfig):
        self.config = config
        logger.info(f"Loading NLI model: {config.nli_model}")
        from sentence_transformers import CrossEncoder
        self.nli_model = CrossEncoder(config.nli_model)

    def _sentence_split(self, text: str) -> List[str]:
        import re
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def faithfulness_score(self, answer: str, contexts: List[Dict]) -> float:
        """
        Compute per-sentence max entailment over retrieved passages.
        Returns average faithfulness score (Eq. 14).
        """
        answer_sents = self._sentence_split(answer)
        if not answer_sents or not contexts:
            return 0.0

        ctx_texts = [c["text"][:512] for c in contexts]
        total_faith = 0.0

        for sent in answer_sents:
            max_entail = 0.0
            for ctx in ctx_texts:
                # NLI label order: contradiction=0, entailment=1, neutral=2
                score = self.nli_model.predict([(ctx, sent)])
                # score shape depends on model — normalize to [0,1]
                if hasattr(score, "__len__") and len(score) == 3:
                    entail_prob = float(score[1])  # entailment class
                else:
                    entail_prob = float(np.clip(score[0], 0, 1))
                max_entail = max(max_entail, entail_prob)
            total_faith += max_entail

        return total_faith / len(answer_sents)

    def verify(
        self,
        answer: str,
        contexts: List[Dict],
        generator,
        query: str,
    ) -> Tuple[str, float]:
        """
        Check faithfulness; trigger regeneration if below threshold (Eq. 15).
        Returns (final_answer, faithfulness_score).
        """
        faith = self.faithfulness_score(answer, contexts)
        if faith < self.config.faithfulness_threshold:
            logger.debug(f"Faithfulness {faith:.3f} < τ={self.config.faithfulness_threshold} — regenerating.")
            constrained_contexts = [
                {**c, "text": c["text"][:300]} for c in contexts
            ]
            result = generator.generate(query, constrained_contexts)
            answer = result["answer"]
            faith = self.faithfulness_score(answer, contexts)
        return answer, faith


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
class Evaluator:
    """
    Computes EM, F1, Recall@K, MRR, Faithfulness, Hallucination Rate.
    Matches metrics defined in paper Section IV-A.
    """

    @staticmethod
    def normalize(text: str) -> str:
        import re, string
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = "".join(ch for ch in text if ch not in string.punctuation)
        return " ".join(text.split())

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        return float(self.normalize(prediction) == self.normalize(ground_truth))

    def token_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self.normalize(prediction).split()
        gold_tokens = self.normalize(ground_truth).split()
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_id: str,
        k: int,
    ) -> float:
        return float(relevant_id in retrieved_ids[:k])

    def mrr(self, retrieved_ids: List[str], relevant_id: str) -> float:
        for rank, pid in enumerate(retrieved_ids, 1):
            if pid == relevant_id:
                return 1.0 / rank
        return 0.0

    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        retrieved_ids_list: List[List[str]],
        relevant_ids: List[str],
        faithfulness_scores: List[float],
        unanswerable_flags: List[bool],
        abstention_flags: List[bool],
    ) -> Dict:
        n = len(predictions)
        em_scores, f1_scores, r_at_1, r_at_3, r_at_5, r_at_10, mrr_scores = (
            [], [], [], [], [], [], []
        )

        for i in range(n):
            em_scores.append(self.exact_match(predictions[i], ground_truths[i]))
            f1_scores.append(self.token_f1(predictions[i], ground_truths[i]))
            r_at_1.append(self.recall_at_k(retrieved_ids_list[i], relevant_ids[i], 1))
            r_at_3.append(self.recall_at_k(retrieved_ids_list[i], relevant_ids[i], 3))
            r_at_5.append(self.recall_at_k(retrieved_ids_list[i], relevant_ids[i], 5))
            r_at_10.append(self.recall_at_k(retrieved_ids_list[i], relevant_ids[i], 10))
            mrr_scores.append(self.mrr(retrieved_ids_list[i], relevant_ids[i]))

        # Hallucination rate: answers with faithfulness < threshold
        halluc = [f < 0.5 for f in faithfulness_scores]
        halluc_rate = np.mean(halluc) * 100

        # Abstention rate on unanswerable questions
        unans_total = sum(unanswerable_flags)
        abstained = sum(
            a for a, u in zip(abstention_flags, unanswerable_flags) if u
        )
        abstention_rate = (abstained / unans_total * 100) if unans_total > 0 else 0.0

        return {
            "EM (%)": round(np.mean(em_scores) * 100, 2),
            "F1 (%)": round(np.mean(f1_scores) * 100, 2),
            "Recall@1": round(np.mean(r_at_1) * 100, 2),
            "Recall@3": round(np.mean(r_at_3) * 100, 2),
            "Recall@5": round(np.mean(r_at_5) * 100, 2),
            "Recall@10": round(np.mean(r_at_10) * 100, 2),
            "MRR": round(np.mean(mrr_scores), 4),
            "Avg. Faithfulness": round(np.mean(faithfulness_scores), 4),
            "Hallucination Rate (%)": round(halluc_rate, 2),
            "Abstention Rate (%)": round(abstention_rate, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. DATASET LOADER (SQuAD 2.0)
# ─────────────────────────────────────────────────────────────────────────────
def load_squad_v2(config: EduGPTConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load SQuAD 2.0 from HuggingFace and prepare:
      - documents (unique contexts for indexing)
      - train_samples (for cross-encoder fine-tuning)
      - eval_samples  (for evaluation)
    """
    logger.info("Loading SQuAD 2.0 dataset...")
    from datasets import load_dataset
    dataset = load_dataset(config.dataset_name, trust_remote_code=True)

    # ── Build unique document corpus ──────────────────────────────────────────
    context_map: Dict[str, str] = {}
    for split in ["train", "validation"]:
        for example in dataset[split]:
            cid = example["id"]
            context_map[cid] = example["context"]

    documents = [{"id": cid, "text": ctx} for cid, ctx in context_map.items()]

    # ── Build training samples for cross-encoder ──────────────────────────────
    train_data = list(dataset["train"])
    if config.max_train_samples:
        train_data = train_data[: config.max_train_samples]

    train_samples = []
    for ex in train_data:
        answers = ex["answers"]["text"]
        label = 1 if answers else 0  # unanswerable = negative
        train_samples.append({
            "query": ex["question"],
            "passage": ex["context"][:512],
            "label": label,
            "answer": answers[0] if answers else "",
            "context_id": ex["id"],
            "is_unanswerable": (label == 0),
        })

    # ── Build eval samples ────────────────────────────────────────────────────
    eval_data = list(dataset["validation"])
    if config.max_eval_samples:
        eval_data = eval_data[: config.max_eval_samples]

    eval_samples = []
    for ex in eval_data:
        answers = ex["answers"]["text"]
        eval_samples.append({
            "query": ex["question"],
            "context_id": ex["id"],
            "context_text": ex["context"],
            "answer": answers[0] if answers else "",
            "all_answers": answers,
            "is_unanswerable": (len(answers) == 0),
        })

    logger.info(
        f"Dataset: {len(documents)} unique contexts, "
        f"{len(train_samples)} train, {len(eval_samples)} eval samples."
    )
    return documents, train_samples, eval_samples


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN EDUGPT-RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class EduGPTRAG:
    """
    End-to-end EduGPT-RAG pipeline.
    Coordinates all four modules as per Algorithm 1 in the paper.
    """

    def __init__(self, config: EduGPTConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        random.seed(config.seed)
        np.random.seed(config.seed)

        # Initialize modules
        self.chunker = HierarchicalChunker(config.chunk_window, config.chunk_stride)
        self.retriever = HybridRetriever(config)
        self.reranker = CrossEncoderReranker(config)
        self.verifier = AntiHallucinationVerifier(config)
        self.evaluator = Evaluator()

        # Generator: use OpenAI if key available, else local T5
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and not config.use_local_generator:
            logger.info("Using OpenAI GPT generator.")
            self.generator = OpenAIGenerator(config)
        else:
            logger.info("Using local T5 generator (flan-t5-base).")
            self.generator = LocalGenerator(config)

    # ── Index Building ────────────────────────────────────────────────────────
    def build_index(self, documents: List[Dict]):
        """Chunk corpus and build retrieval indexes."""
        logger.info("=== PHASE 1: INDEXING ===")
        self.passages = self.chunker.chunk_corpus(documents)
        self.retriever.index(self.passages)

    # ── Single Query Inference (Algorithm 1) ──────────────────────────────────
    def answer(self, query: str) -> Dict:
        """
        Full pipeline for a single query.
        Steps: encode → FAISS → BM25 → RRF → cross-encoder → CoT generate → NLI verify.
        """
        t0 = time.time()

        # Step 1-4: Hybrid retrieval + RRF
        candidates = self.retriever.retrieve(query)

        # Step 5: Cross-encoder re-ranking → top-K context
        context = self.reranker.rerank(query, candidates)

        # Step 6: CoT-prompted generation
        gen_result = self.generator.generate(query, context)
        raw_answer = gen_result["answer"]

        # Step 7-9: NLI faithfulness verification + conditional regeneration
        final_answer, faith = self.verifier.verify(
            raw_answer, context, self.generator, query
        )

        latency_ms = (time.time() - t0) * 1000

        return {
            "query": query,
            "answer": final_answer,
            "faithfulness": faith,
            "context": context,
            "latency_ms": latency_ms,
            "retrieved_passage_ids": [c["id"] for c in candidates],
            "is_abstained": (final_answer.strip() == "" or len(final_answer.split()) <= 2),
        }

    # ── Training Phase ────────────────────────────────────────────────────────
    def train(self, train_samples: List[Dict]):
        """Fine-tune cross-encoder reranker on training data."""
        logger.info("=== PHASE 2: TRAINING (Cross-Encoder Fine-Tuning) ===")
        ce_output = os.path.join(self.config.output_dir, "cross_encoder_finetuned")
        self.reranker.train_reranker(train_samples, ce_output)

    # ── Evaluation Phase ──────────────────────────────────────────────────────
    def evaluate(self, eval_samples: List[Dict]) -> Dict:
        """Run inference on eval set and compute all metrics."""
        logger.info("=== PHASE 3: EVALUATION ===")

        predictions, ground_truths = [], []
        retrieved_ids_list, relevant_ids = [], []
        faithfulness_scores = []
        unanswerable_flags, abstention_flags = [], []
        latencies = []

        for sample in tqdm(eval_samples, desc="Evaluating"):
            result = self.answer(sample["query"])

            predictions.append(result["answer"])
            ground_truths.append(sample["answer"])
            retrieved_ids_list.append(result["retrieved_passage_ids"])
            relevant_ids.append(sample["context_id"] + "_chunk_0")  # approx first chunk
            faithfulness_scores.append(result["faithfulness"])
            unanswerable_flags.append(sample["is_unanswerable"])
            abstention_flags.append(result["is_abstained"])
            latencies.append(result["latency_ms"])

        metrics = self.evaluator.evaluate_batch(
            predictions,
            ground_truths,
            retrieved_ids_list,
            relevant_ids,
            faithfulness_scores,
            unanswerable_flags,
            abstention_flags,
        )
        metrics["Avg. Latency (ms)"] = round(np.mean(latencies), 1)
        metrics["P50 Latency (ms)"] = round(float(np.percentile(latencies, 50)), 1)
        metrics["P95 Latency (ms)"] = round(float(np.percentile(latencies, 95)), 1)
        return metrics

    # ── Ablation Study ────────────────────────────────────────────────────────
    def run_ablation(self, eval_samples: List[Dict]) -> Dict[str, Dict]:
        """
        Systematic ablation matching Table III of the paper.
        Returns dict of config_name → metrics.
        """
        logger.info("=== PHASE 4: ABLATION STUDY ===")
        results = {}

        # Save original settings
        orig_threshold = self.config.faithfulness_threshold

        configs_to_test = [
            ("Full EduGPT-RAG",          True,  True,  True,  True),
            ("- Anti-halluc layer",       False, True,  True,  True),
            ("- CoT prompting",           True,  False, True,  True),
            ("- Cross-encoder re-rank",   True,  True,  False, True),
            ("- Hybrid → Dense only",     True,  True,  True,  False),
        ]

        for name, use_nli, use_cot, use_rerank, use_hybrid in configs_to_test:
            logger.info(f"  Ablation: {name}")

            # Temporarily override behaviours via flags on instances
            self._ablation_use_nli = use_nli
            self._ablation_use_cot = use_cot
            self._ablation_use_rerank = use_rerank
            self._ablation_use_hybrid = use_hybrid

            # Disable threshold to skip NLI if not used
            if not use_nli:
                self.config.faithfulness_threshold = 1.1  # never triggers

            metrics = self._ablation_evaluate(eval_samples[:50])  # use subset for speed
            results[name] = metrics

            # Restore
            self.config.faithfulness_threshold = orig_threshold

        self._ablation_use_nli = True
        self._ablation_use_cot = True
        self._ablation_use_rerank = True
        self._ablation_use_hybrid = True
        return results

    def _ablation_evaluate(self, eval_samples: List[Dict]) -> Dict:
        """Evaluation with ablation flags."""
        predictions, ground_truths = [], []
        retrieved_ids_list, relevant_ids = [], []
        faith_scores = []
        unans, abstained = [], []

        for sample in eval_samples:
            # Retrieval
            if self._ablation_use_hybrid:
                candidates = self.retriever.retrieve(sample["query"])
            else:
                # Dense only
                dense_hits = self.retriever._dense_search(sample["query"], 10)
                candidates = [
                    {**self.retriever.passages[idx], "rrf_score": sc}
                    for idx, sc in dense_hits
                    if idx < len(self.retriever.passages)
                ]

            # Re-ranking
            if self._ablation_use_rerank:
                context = self.reranker.rerank(sample["query"], candidates)
            else:
                context = candidates[: self.config.top_k_context]

            # Generation
            gen_result = self.generator.generate(sample["query"], context)
            answer = gen_result["answer"]

            # NLI
            if self._ablation_use_nli:
                answer, faith = self.verifier.verify(
                    answer, context, self.generator, sample["query"]
                )
            else:
                faith = self.verifier.faithfulness_score(answer, context)

            predictions.append(answer)
            ground_truths.append(sample["answer"])
            retrieved_ids_list.append([c["id"] for c in candidates])
            relevant_ids.append(sample["context_id"] + "_chunk_0")
            faith_scores.append(faith)
            unans.append(sample["is_unanswerable"])
            abstained.append(len(answer.split()) <= 2)

        return self.evaluator.evaluate_batch(
            predictions, ground_truths, retrieved_ids_list,
            relevant_ids, faith_scores, unans, abstained
        )

    # ── Results Reporting ─────────────────────────────────────────────────────
    @staticmethod
    def print_results(metrics: Dict, title: str = "Results"):
        logger.info(f"\n{'='*60}")
        logger.info(f"  {title}")
        logger.info(f"{'='*60}")
        for k, v in metrics.items():
            logger.info(f"  {k:<30} {v}")
        logger.info(f"{'='*60}\n")

    @staticmethod
    def print_ablation(ablation_results: Dict[str, Dict]):
        logger.info(f"\n{'='*75}")
        logger.info("  ABLATION STUDY")
        logger.info(f"{'='*75}")
        header = f"{'Configuration':<32} {'EM':>6} {'F1':>6} {'R@5':>6} {'Faith':>7}"
        logger.info(header)
        logger.info("-" * 75)
        for name, m in ablation_results.items():
            row = (
                f"{name:<32}"
                f" {m.get('EM (%)', 0):>6.2f}"
                f" {m.get('F1 (%)', 0):>6.2f}"
                f" {m.get('Recall@5', 0):>6.2f}"
                f" {m.get('Avg. Faithfulness', 0):>7.4f}"
            )
            logger.info(row)
        logger.info(f"{'='*75}\n")

    def save_results(self, metrics: Dict, ablation: Dict, path: str):
        combined = {"evaluation": metrics, "ablation": ablation}
        with open(path, "w") as f:
            json.dump(combined, f, indent=2)
        logger.info(f"Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. QUICK DEMO (runs without downloading full dataset)
# ─────────────────────────────────────────────────────────────────────────────
def run_demo(config: EduGPTConfig):
    """
    Minimal self-contained demo with 5 hard-coded QA pairs.
    Lets you verify the pipeline works without HuggingFace datasets.
    """
    logger.info("=== RUNNING QUICK DEMO (no dataset download required) ===")

    demo_docs = [
        {"id": "d1", "text": (
            "The Northern Lights, or Aurora Borealis, occur when charged particles "
            "emitted by the sun travel through space and interact with gases in Earth's "
            "upper atmosphere. Oxygen produces green and red colours, while nitrogen "
            "produces blue and purple hues. The phenomenon is most visible near the polar "
            "regions due to Earth's magnetic field funnelling particles toward the poles."
        )},
        {"id": "d2", "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines information "
            "retrieval with text generation. A retrieval component fetches relevant documents "
            "from a knowledge base, and a generative model uses those documents as context to "
            "produce accurate, grounded responses. RAG significantly reduces hallucination in "
            "large language models by anchoring generation to external evidence."
        )},
        {"id": "d3", "text": (
            "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained "
            "transformer model developed by Google. It processes text bidirectionally, capturing "
            "context from both left and right of each token simultaneously. Fine-tuning BERT on "
            "downstream tasks such as question answering and classification consistently achieves "
            "state-of-the-art results on many NLP benchmarks."
        )},
        {"id": "d4", "text": (
            "The SQuAD 2.0 dataset (Stanford Question Answering Dataset version 2.0) contains "
            "over 150,000 question-answer pairs built from Wikipedia articles. Unlike SQuAD 1.1, "
            "it includes 50,000+ unanswerable questions designed to test a model's ability to "
            "abstain when evidence is insufficient rather than guessing an answer."
        )},
        {"id": "d5", "text": (
            "Federated learning is a distributed machine learning approach where a model is "
            "trained across multiple decentralized devices holding local data. The central server "
            "aggregates model updates rather than raw data, preserving data privacy. It is widely "
            "used in healthcare and IoT applications where data sharing is restricted."
        )},
    ]

    demo_queries = [
        ("What causes the Northern Lights?", "Charged particles from the sun interact with gases in Earth's atmosphere."),
        ("What is RAG?", "Retrieval-Augmented Generation combines retrieval with text generation to ground LLM outputs."),
        ("What does BERT stand for?", "Bidirectional Encoder Representations from Transformers."),
        ("How many questions are in SQuAD 2.0?", "Over 150,000 question-answer pairs including unanswerable questions."),
        ("What is federated learning?", "A distributed ML approach that trains models across devices without sharing raw data."),
    ]

    pipeline = EduGPTRAG(config)
    pipeline.build_index(demo_docs)

    logger.info("\n=== DEMO QUERY RESULTS ===")
    for query, reference in demo_queries:
        result = pipeline.answer(query)
        em = pipeline.evaluator.exact_match(result["answer"], reference)
        f1 = pipeline.evaluator.token_f1(result["answer"], reference)
        logger.info(f"\nQ: {query}")
        logger.info(f"A: {result['answer']}")
        logger.info(f"   Faith={result['faithfulness']:.3f} | EM={em:.0f} | F1={f1:.3f} | Latency={result['latency_ms']:.0f}ms")

    logger.info("\nDemo complete!")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FULL PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_full_pipeline(config: EduGPTConfig):
    """
    Complete training + evaluation pipeline on SQuAD 2.0.
    Requires HuggingFace `datasets` package.
    """
    logger.info("=== EduGPT-RAG: FULL PIPELINE ===")

    # Load data
    documents, train_samples, eval_samples = load_squad_v2(config)

    # Build pipeline
    pipeline = EduGPTRAG(config)

    # Phase 1: Index
    pipeline.build_index(documents)

    # Phase 2: Train cross-encoder
    pipeline.train(train_samples)

    # Phase 3: Evaluate
    metrics = pipeline.evaluate(eval_samples)
    pipeline.print_results(metrics, "EduGPT-RAG Evaluation Results")

    # Phase 4: Ablation
    ablation = pipeline.run_ablation(eval_samples)
    pipeline.print_ablation(ablation)

    # Save
    out_path = os.path.join(config.output_dir, "results.json")
    pipeline.save_results(metrics, ablation, out_path)

    return metrics, ablation


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EduGPT-RAG Pipeline")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo",
                        help="'demo' = quick local test, 'full' = SQuAD 2.0 pipeline")
    parser.add_argument("--chunk_window", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--faith_threshold", type=float, default=0.75)
    parser.add_argument("--max_train", type=int, default=500)
    parser.add_argument("--max_eval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./edugpt_rag_output")
    parser.add_argument("--use_local_generator", action="store_true", default=True)
    args = parser.parse_args()

    config = EduGPTConfig(
        chunk_window=args.chunk_window,
        chunk_stride=args.chunk_stride,
        top_k_context=args.top_k,
        faithfulness_threshold=args.faith_threshold,
        max_train_samples=args.max_train,
        max_eval_samples=args.max_eval,
        output_dir=args.output_dir,
        use_local_generator=args.use_local_generator,
    )

    logger.info("=" * 60)
    logger.info("  EduGPT-RAG: RAG Framework for Academic QA")
    logger.info("  IEEE Transactions on Consumer Electronics")
    logger.info("=" * 60)
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Chunk window={config.chunk_window}, stride={config.chunk_stride}")
    logger.info(f"  Top-K context={config.top_k_context}, τ={config.faithfulness_threshold}")
    logger.info(f"  Generator: {'Local T5' if config.use_local_generator else 'OpenAI GPT'}")
    logger.info("=" * 60)

    if args.mode == "demo":
        run_demo(config)
    else:
        run_full_pipeline(config)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK INSTALL REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
"""
INSTALL (run once):

    pip install torch transformers sentence-transformers faiss-cpu \
                rank_bm25 datasets scikit-learn numpy tqdm

FOR FULL PIPELINE WITH OPENAI:
    export OPENAI_API_KEY=sk-...
    python edugpt_rag_complete.py --mode full --use_local_generator False

FOR DEMO (no API key needed):
    python edugpt_rag_complete.py --mode demo

COMMAND LINE OPTIONS:
    --mode              demo | full          (default: demo)
    --chunk_window      int                  (default: 512 tokens)
    --chunk_stride      int                  (default: 128 tokens)
    --top_k             int                  (default: 5 context passages)
    --faith_threshold   float                (default: 0.75)
    --max_train         int                  (default: 500 samples)
    --max_eval          int                  (default: 100 samples)
    --output_dir        path                 (default: ./edugpt_rag_output)

PAPER MAPPING:
    HierarchicalChunker   → Eq. 2-3    (Section III-B)
    HybridRetriever       → Eq. 4-8    (Section III-C)
    CrossEncoderReranker  → Eq. 9-11   (Section III-D)
    LocalGenerator        → Eq. 12-14  (Section III-E)
    AntiHallucinationVerifier → Eq. 15-17 (Section III-F)
    EduGPTRAG.answer()    → Algorithm 1
    Evaluator             → Section IV-A metrics
"""
