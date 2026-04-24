# =============================================================================
# services/rag.py
# Phase 3 — RAG (Retrieval Augmented Generation)
# Stores teacher-provided answer keys in ChromaDB with embeddings.
# Retrieves by exact question_id first; falls back to semantic search.
# =============================================================================

import chromadb
import json

# ── Lazy embedder — loaded only on first use to avoid slow startup ────────────
_embedder = None


def _get_embedder():
    """
    Lazily loads sentence-transformers model on first call.
    Reuses the same instance for all subsequent calls (singleton pattern).
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── ChromaDB setup — persists to disk at ./chroma_db ─────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(name="answer_keys")


def store_answer_key(data: list) -> dict:
    """
    Embeds and stores a list of answer key items into ChromaDB.
    Uses upsert so re-storing the same question_id safely overwrites old data.

    Args:
        data: List of dicts each containing:
              question_id, question, answer, keywords, rubric, max_marks

    Returns:
        dict: { "status": "stored", "count": <int> }
    """
    embedder = _get_embedder()

    for item in data:
        qid  = item["question_id"]

        # Combine question + answer text for richer embedding
        text = item["question"] + " " + item["answer"]
        emb  = embedder.encode(text).tolist()

        collection.upsert(
            ids=[qid],
            embeddings=[emb],
            documents=[json.dumps(item)],
            metadatas=[{"question_id": qid}],
        )

    return {"status": "stored", "count": len(data)}


def get_answer_key(question_id: str) -> dict:
    """
    Retrieves answer key for a given question.

    Strategy:
        1. Exact ID lookup (fast, deterministic) — used when OCR reads Q1/Q2 correctly.
        2. Semantic search fallback — used when OCR reads "Question 1" instead of "Q1".

    Args:
        question_id: The question identifier string e.g. "Q1"

    Returns:
        dict: Full answer key object, or None if not found.
    """

    # ── Primary: exact lookup by question_id ─────────────────────────────────
    try:
        result = collection.get(ids=[question_id])
        if result["documents"]:
            return json.loads(result["documents"][0])
    except Exception:
        pass  # fall through to semantic search

    # ── Fallback: semantic similarity search ──────────────────────────────────
    embedder = _get_embedder()
    emb      = embedder.encode(question_id).tolist()
    result   = collection.query(query_embeddings=[emb], n_results=1)

    if result["documents"] and result["documents"][0]:
        return json.loads(result["documents"][0][0])

    return None


# ── CLI test entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [{
        "question_id": "Q1",
        "question":    "What is photosynthesis?",
        "answer":      "Plants convert sunlight into glucose using chlorophyll",
        "keywords":    ["sunlight", "glucose", "chlorophyll"],
        "rubric":      "2pts process, 2pts keywords, 1pt structure",
        "max_marks":   5,
    }]
    print(store_answer_key(sample))
    print(get_answer_key("Q1"))
