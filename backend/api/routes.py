# =============================================================================
# api/routes.py
# FastAPI route definitions — all HTTP endpoints for the grading pipeline.
# Routes only handle HTTP concerns (request parsing, response shaping,
# error codes). All business logic lives in services/.
# =============================================================================

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import shutil
import os
import traceback

from services.ocr      import extract_and_segment
from services.rag      import store_answer_key, get_answer_key
from services.grader   import grade_answer
from services.feedback import generate_feedback

router = APIRouter()

# Ensure uploads directory exists at startup
os.makedirs("uploads", exist_ok=True)


# =============================================================================
# Health check
# =============================================================================

@router.get("/health", tags=["System"])
def health():
    """Returns server status. Used by n8n and monitoring to verify uptime."""
    return {"status": "ok"}


# =============================================================================
# OCR endpoint
# =============================================================================

@router.post("/ocr", tags=["OCR"])
async def run_ocr(file: UploadFile = File(...)):
    """
    Accepts a worksheet image and returns extracted Q-A pairs.

    Args:
        file: Uploaded worksheet image (jpg, jpeg, png).

    Returns:
        { "qa_pairs": { "Q1": "answer", "Q2": "answer", ... } }
    """
    path = f"uploads/{file.filename}"
    try:
        # Save uploaded file to disk (Gemini reads from file path)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = extract_and_segment(path)
        return {"qa_pairs": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Answer key endpoints
# =============================================================================

class AnswerKeyItem(BaseModel):
    """Pydantic model for a single answer key entry — validates request body."""
    question_id: str
    question:    str
    answer:      str
    keywords:    List[str]
    rubric:      str
    max_marks:   float


@router.post("/store-answer-key", tags=["Answer Key"])
def store_key(data: List[AnswerKeyItem]):
    """
    Stores a list of answer key items into ChromaDB.

    Args:
        data: JSON array of AnswerKeyItem objects.

    Returns:
        { "status": "stored", "count": <int> }
    """
    try:
        return store_answer_key([item.dict() for item in data])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fetch-answer-key/{question_id}", tags=["Answer Key"])
def fetch_key(question_id: str):
    """
    Retrieves the stored answer key for a specific question.

    Args:
        question_id: e.g. "Q1"

    Returns:
        Full answer key dict, or { "error": "Question not found" }
    """
    result = get_answer_key(question_id)
    return result if result else {"error": "Question not found"}


# =============================================================================
# Grading endpoints
# =============================================================================

class GradeRequest(BaseModel):
    """Pydantic model for single-answer grading request."""
    question_id:    str
    student_answer: str


@router.post("/grade", tags=["Grading"])
def grade(req: GradeRequest):
    """
    Grades a single student answer against its stored answer key.

    Args:
        req: { question_id, student_answer }

    Returns:
        { score, max_marks, keyword_score, semantic_score,
          structure_score, matched_keywords, missed_keywords }
    """
    try:
        key = get_answer_key(req.question_id)
        if not key:
            raise HTTPException(status_code=404, detail="Answer key not found")
        return grade_answer(req.student_answer, key)
    except HTTPException:
        raise  # re-raise 404 without wrapping
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/grade-full-sheet", tags=["Grading"])
async def grade_full_sheet(file: UploadFile = File(...)):
    """
    Full pipeline endpoint — runs OCR, fetches answer keys,
    grades each answer, generates feedback, and returns complete results.

    Args:
        file: Uploaded worksheet image.

    Returns:
        {
          "results": {
            "Q1": {
              student_answer, score, max_marks,
              keyword_score, semantic_score, structure_score,
              matched_keywords, missed_keywords, feedback
            },
            ...
          }
        }
    """

    # ── Save image to disk ────────────────────────────────────────────────────
    path = f"uploads/{file.filename}"
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    # ── OCR: extract Q-A pairs from image ────────────────────────────────────
    try:
        qa_pairs = extract_and_segment(path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    # ── Grade each question independently ────────────────────────────────────
    results = {}

    for qid, student_answer in qa_pairs.items():
        try:
            # Fetch answer key for this question
            key = get_answer_key(qid)
            if not key:
                results[qid] = {"error": f"No answer key found for {qid}"}
                continue

            # Compute score
            grade_result = grade_answer(student_answer, key)

            # Generate natural language feedback
            feedback = generate_feedback(
                question        = key["question"],
                student_answer  = student_answer,
                answer_key      = key["answer"],
                keywords        = key["keywords"],
                missed_keywords = grade_result["missed_keywords"],
                score           = grade_result["score"],
                max_marks       = grade_result["max_marks"],
            )

            # Assemble full result for this question
            results[qid] = {
                "student_answer":   student_answer,
                "score":            grade_result["score"],
                "max_marks":        grade_result["max_marks"],
                "keyword_score":    grade_result["keyword_score"],
                "semantic_score":   grade_result["semantic_score"],
                "structure_score":  grade_result["structure_score"],
                "matched_keywords": grade_result["matched_keywords"],
                "missed_keywords":  grade_result["missed_keywords"],
                "feedback":         feedback,
            }

        except Exception as e:
            # Per-question error does not abort the full sheet
            traceback.print_exc()
            results[qid] = {"error": str(e)}

    return {"results": results}
