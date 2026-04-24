# =============================================================================
# services/feedback.py
# Phase 5 — Feedback Generation
# Uses Gemini to generate short, encouraging, specific teacher-style feedback
# for each student answer, referencing missed keywords and score context.
# =============================================================================

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# ── Gemini client initialisation ──────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_feedback(
    question: str,
    student_answer: str,
    answer_key: str,
    keywords: list,
    missed_keywords: list,
    score: float,
    max_marks: float,
) -> str:
    """
    Generates 2-3 sentence teacher-style feedback for a student answer.

    The prompt passes full grading context (score, missed keywords, reference answer)
    so feedback is specific and not generic.

    Args:
        question:         The original question text.
        student_answer:   The student's extracted answer.
        answer_key:       The reference answer from the answer key.
        keywords:         All required keywords for this question.
        missed_keywords:  Keywords not found in the student answer.
        score:            Computed score for this answer.
        max_marks:        Maximum possible marks for this question.

    Returns:
        str: 2-3 sentence feedback string, or error message if API call fails.
    """

    # ── Build missed keywords string ──────────────────────────────────────────
    missed_str = ", ".join(missed_keywords) if missed_keywords else "None"

    # ── Feedback prompt ───────────────────────────────────────────────────────
    prompt = (
        "You are a helpful teacher giving feedback on a student's answer.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {answer_key}\n"
        f"Student Answer: {student_answer}\n"
        f"Required Keywords: {', '.join(keywords)}\n"
        f"Missing Keywords: {missed_str}\n"
        f"Score: {score} / {max_marks}\n\n"
        "Give the student short, encouraging, specific feedback (2-3 sentences). "
        "Mention what they got right and what concept or keyword they missed. "
        "Do not repeat the full answer. Be concise and helpful."
    )

    # ── Call Gemini API with error fallback ───────────────────────────────────
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        # Return a soft error so the full grading pipeline is not blocked
        return f"Feedback unavailable: {e}"
