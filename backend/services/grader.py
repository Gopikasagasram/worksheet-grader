# =============================================================================
# services/grader.py
# Phase 4 — Grading Logic
# Scores each student answer using a weighted combination of:
#   - Keyword match      (30%)
#   - Semantic similarity(50%)
#   - Structure score    (20%)
# Borderline answers (40-60%) are re-evaluated by Groq Llama for accuracy.
# =============================================================================

from groq import Groq
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ── Groq client initialisation ────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Lazy embedder — same singleton pattern as rag.py ─────────────────────────
_embedder = None


def _get_embedder():
    """Lazily loads sentence-transformers model on first call."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# =============================================================================
# Scoring helpers — each returns a float between 0.0 and 1.0
# =============================================================================

def keyword_score(student_answer: str, keywords: list) -> float:
    """
    Calculates the fraction of required keywords present in the student answer.

    Args:
        student_answer: The student's raw answer text.
        keywords: List of required keyword strings from the answer key.

    Returns:
        float: matched_count / total_keywords (0.0 if keywords list is empty)
    """
    lower   = student_answer.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    return len(matched) / len(keywords) if keywords else 0.0


def semantic_score(student_answer: str, reference_answer: str) -> float:
    """
    Computes cosine similarity between student and reference answer embeddings.
    Captures correct concept even when different words are used (paraphrasing).

    Args:
        student_answer:   Student's answer text.
        reference_answer: Teacher's reference answer from answer key.

    Returns:
        float: Cosine similarity score between 0.0 and 1.0.
    """
    embedder = _get_embedder()
    e1 = embedder.encode([student_answer])
    e2 = embedder.encode([reference_answer])
    return float(cosine_similarity(e1, e2)[0][0])


def structure_score(student_answer: str, reference_answer: str) -> float:
    """
    Measures answer depth by comparing word count ratio.
    Penalises very short answers even if keywords are present.

    Args:
        student_answer:   Student's answer text.
        reference_answer: Teacher's reference answer.

    Returns:
        float: min(student_word_count / reference_word_count, 1.0)
    """
    ref_len = len(reference_answer.split())
    if ref_len == 0:
        return 0.0
    return min(len(student_answer.split()) / ref_len, 1.0)


# =============================================================================
# Main grading function
# =============================================================================

def grade_answer(student_answer: str, answer_key: dict) -> dict:
    """
    Grades a single student answer against the answer key.

    Scoring formula:
        final = (keyword_score * 0.30) + (semantic_score * 0.50) + (structure_score * 0.20)
        score = round(final * max_marks, 1)

    Borderline re-scoring:
        If 0.40 <= final <= 0.60, Groq Llama re-evaluates using the rubric.
        Falls back to computed final if LLM call fails.

    Args:
        student_answer: Extracted answer text from OCR.
        answer_key:     Full answer key dict from ChromaDB.

    Returns:
        dict: {
            score, max_marks,
            keyword_score, semantic_score, structure_score,
            matched_keywords, missed_keywords
        }
    """

    # ── Handle blank answers ──────────────────────────────────────────────────
    if student_answer.strip().upper() == "NO ANSWER":
        return {
            "score":            0,
            "max_marks":        answer_key["max_marks"],
            "keyword_score":    0,
            "semantic_score":   0,
            "structure_score":  0,
            "matched_keywords": [],
            "missed_keywords":  answer_key["keywords"],
        }

    # ── Compute three independent scores ─────────────────────────────────────
    kw_score = keyword_score(student_answer, answer_key["keywords"])
    sem      = semantic_score(student_answer, answer_key["answer"])
    st       = structure_score(student_answer, answer_key["answer"])

    # ── Weighted combination ──────────────────────────────────────────────────
    final = kw_score * 0.30 + sem * 0.50 + st * 0.20

    # ── LLM re-scoring for borderline answers ─────────────────────────────────
    if 0.40 <= final <= 0.60:
        prompt = (
            "A student answered a question. Score their answer from 0.0 to 1.0.\n\n"
            f"Question: {answer_key['question']}\n"
            f"Reference Answer: {answer_key['answer']}\n"
            f"Student Answer: {student_answer}\n"
            f"Rubric: {answer_key['rubric']}\n"
            f"Initial Score: {round(final, 2)}\n\n"
            "Reply with ONLY a single decimal number between 0.0 and 1.0. No explanation."
        )
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            final = float(resp.choices[0].message.content.strip())
        except (ValueError, Exception):
            pass  # keep computed final if LLM call fails

    # ── Build keyword match / miss lists ─────────────────────────────────────
    matched = [kw for kw in answer_key["keywords"] if kw.lower() in student_answer.lower()]
    missed  = [kw for kw in answer_key["keywords"] if kw.lower() not in student_answer.lower()]

    return {
        "score":            round(final * answer_key["max_marks"], 1),
        "max_marks":        answer_key["max_marks"],
        "keyword_score":    round(kw_score, 2),
        "semantic_score":   round(sem, 2),
        "structure_score":  round(st, 2),
        "matched_keywords": matched,
        "missed_keywords":  missed,
    }
