# =============================================================================
# services/ocr.py
# Phase 1 + 2 — OCR and Segmentation
# Uses Gemini Vision to read a handwritten worksheet image and extract
# each question-answer pair as a structured dictionary.
# =============================================================================

from google import genai
from google.genai import types
import os
import json
import re
import mimetypes
from dotenv import load_dotenv

load_dotenv()

# ── Gemini client initialisation ──────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def extract_and_segment(image_path: str) -> dict:
    """
    Reads a handwritten worksheet image and returns segmented Q-A pairs.

    Args:
        image_path: Absolute or relative path to the worksheet image.

    Returns:
        dict: { "Q1": "student answer", "Q2": "student answer", ... }

    Raises:
        ValueError: If Gemini returns non-JSON output.
        RuntimeError: If response text cannot be extracted.
    """

    # ── MIME detection — supports jpeg, png, webp, gif ────────────────────────
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        mime_type = "image/jpeg"   # safe fallback for unknown extensions

    # ── Read image bytes from disk ────────────────────────────────────────────
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # ── Prompt — instructs Gemini to return raw JSON only ─────────────────────
    prompt = """
You are an exam worksheet reader.

Look at this handwritten answer sheet carefully.

Extract each question number and its handwritten answer.

Return ONLY a JSON object like this:
{
    "Q1": "student answer here",
    "Q2": "student answer here",
    "Q3": "student answer here"
}

Rules:
- Do not add any explanation
- Do not add markdown or code blocks
- Just return the raw JSON
- If answer is blank write "NO ANSWER"
- Keep the answer exactly as written by student
"""

    # ── Call Gemini Vision API ────────────────────────────────────────────────
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
    )

    # ── Safe response text extractor (handles varying Gemini SDK versions) ────
    def _response_text(resp):
        if hasattr(resp, "text") and isinstance(resp.text, str):
            return resp.text
        if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
            return resp.output_text
        output = getattr(resp, "output", None)
        if isinstance(output, (list, tuple)) and len(output) > 0:
            first   = output[0]
            content = getattr(first, "content", None)
            if isinstance(content, (list, tuple)) and len(content) > 0:
                text = getattr(content[0], "text", None)
                if isinstance(text, str):
                    return text
        try:
            return str(resp)
        except Exception:
            raise RuntimeError("Could not extract text from model response")

    # ── Strip accidental markdown fences and parse JSON ───────────────────────
    raw = _response_text(response).strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned non-JSON output: {raw[:200]}") from e

    return result


# ── CLI test entry point ──────────────────────────────────────────────────────
def test_ocr():
    """Quick CLI test: python ocr.py uploads/sample.png"""
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "uploads/sample.png"
    print(f"Reading image: {image_path}")
    result = extract_and_segment(image_path)
    print("\n--- Extracted Q&A Pairs ---")
    for question, answer in result.items():
        print(f"{question}: {answer}")
    return result


if __name__ == "__main__":
    test_ocr()
