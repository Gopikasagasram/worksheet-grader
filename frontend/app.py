# =============================================================================
# frontend/app.py
# Streamlit UI — two-tab interface for the Worksheet Grader.
#
# Tab 1 — Teacher Setup:
#   Store answer key via JSON paste or file upload.
#
# Tab 2 — Grade Worksheets:
#   Single mode: grade one worksheet, download CSV.
#   Batch mode:  grade multiple worksheets in parallel, download CSV.
#
# This file contains ONLY UI logic and HTTP calls to the FastAPI backend.
# No business logic, no direct imports from backend services.
# =============================================================================

import streamlit as st
import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Backend base URL — change to ngrok URL when using n8n cloud ──────────────
API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Worksheet Grader", page_icon="📝", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*='css'] { font-family: 'Inter', sans-serif; }

.score-card {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px; padding: 1.4rem; color: #fff;
    text-align: center; box-shadow: 0 6px 24px rgba(0,0,0,.4);
}
.score-card h1 { font-size: 2.6rem; margin: 0; }
.score-card p  { margin: .2rem 0 0; opacity: .7; font-size: .9rem; }

.student-header {
    background: linear-gradient(90deg, #1e3a5f, #2d6a9f);
    border-radius: 10px; padding: 0.8rem 1.2rem;
    color: #fff; margin-bottom: 0.5rem;
}
.batch-summary {
    background: #1a1a2e; border-radius: 12px;
    padding: 1rem 1.5rem; margin: 1rem 0;
    border-left: 4px solid #4ade80;
}
</style>
""", unsafe_allow_html=True)

st.title("📝 Scanned Worksheet Grader")
st.caption("Gemini 2.5 Flash (OCR) · Groq Llama 3.3 70B (Grading & Feedback) · ChromaDB (RAG)")

tab1, tab2 = st.tabs(["📚 Teacher Setup", "✅ Grade Worksheets"])


# =============================================================================
# Helper functions — shared between single and batch modes
# =============================================================================

def display_student_results(results: dict) -> tuple:
    """
    Renders per-question result cards for one student.

    Args:
        results: Dict of { question_id: result_dict } from /grade-full-sheet.

    Returns:
        tuple: (total_score, total_max)
    """
    total_score = total_max = 0.0

    for qid, res in results.items():
        # ── Error case — question could not be graded ─────────────────────────
        if "error" in res:
            st.warning(f"**{qid}**: {res['error']}")
            continue

        score, mx = res["score"], res["max_marks"]
        total_score += score
        total_max   += mx
        pct = score / mx if mx else 0

        # Score colour: green ≥ 80%, amber ≥ 60%, red < 60%
        clr = "#22c55e" if pct >= .8 else ("#f59e0b" if pct >= .6 else "#ef4444")

        with st.expander(f"**{qid}** — {score} / {mx} marks", expanded=True):
            left, right = st.columns([3, 2])

            with left:
                # ── Student answer ────────────────────────────────────────────
                st.markdown("**📄 Student Answer:**")
                st.write(res["student_answer"])

                # ── Keywords ──────────────────────────────────────────────────
                st.markdown("**✅ Keywords Matched:**")
                matched_text = "  ·  ".join(res["matched_keywords"])
                st.success(matched_text if matched_text else "None")

                st.markdown("**❌ Keywords Missed:**")
                if res["missed_keywords"]:
                    st.error("  ·  ".join(res["missed_keywords"]))
                else:
                    st.success("All keywords present 🎉")

                # ── Feedback ──────────────────────────────────────────────────
                st.markdown("**💬 Teacher Feedback:**")
                st.info(res["feedback"])

            with right:
                # ── Score card ────────────────────────────────────────────────
                st.markdown(
                    f"<div class='score-card'><p>Score</p>"
                    f"<h1 style='color:{clr}'>{score}/{mx}</h1>"
                    f"<p>{pct*100:.1f}%</p></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(min(pct, 1.0))

                # ── Score breakdown caption ───────────────────────────────────
                st.caption(
                    f"Keyword {res['keyword_score']*100:.0f}%  "
                    f"Semantic {res['semantic_score']*100:.0f}%  "
                    f"Structure {res['structure_score']*100:.0f}%"
                )

    return total_score, total_max


def show_total(total_score: float, total_max: float, num_questions: int):
    """
    Renders total score metrics and a grade-level message.

    Args:
        total_score: Sum of all question scores.
        total_max:Sum of all max marks.
        num_questions:  Number of questions graded.
    """
    if total_max > 0:
        st.divider()
        pct_total = round(total_score / total_max * 100, 1)
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Total Score",  f"{total_score} / {total_max}")
        c2.metric("📈 Percentage",   f"{pct_total}%")
        c3.metric("📋 Questions",    num_questions)

        # ── Grade message ─────────────────────────────────────────────────────
        if pct_total >= 80:
            st.success("🏆 Excellent performance!")
        elif pct_total >= 60:
            st.info("👍 Good work! Keep practising.")
        else:
            st.warning("📚 Needs more revision on key concepts.")


def grade_one_file(uploaded_file) -> tuple:
    """
    Sends one file to FastAPI /grade-full-sheet and returns results.
    Designed to run inside a ThreadPoolExecutor for parallel grading.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        tuple: (filename: str, results: dict)
    """
    try:
        r = requests.post(
            f"{API_URL}/grade-full-sheet",
            files={"file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "image/jpeg",
            )},
            timeout=300,
        )
        if r.status_code == 200:
            return uploaded_file.name, r.json()["results"]
        else:
            return uploaded_file.name, {"error": f"Server {r.status_code}: {r.text}"}
    except requests.ConnectionError:
        return uploaded_file.name, {"error": "FastAPI not reachable"}
    except Exception as e:
        return uploaded_file.name, {"error": str(e)}


def build_csv(all_results: dict) -> str:
    """
    Converts all grading results into a CSV string for download.

    Args:
        all_results: { filename: { qid: result_dict } }

    Returns:
        str: CSV-formatted string.
    """
    rows = []
    for student_file, results in all_results.items():
        student_name = student_file.rsplit(".", 1)[0]  # strip file extension

        # ── Top-level error (entire file failed) ──────────────────────────────
        if isinstance(results, dict) and "error" in results:
            rows.append({
                "Student":          student_name,
                "Question":         "—",
                "Score":            "—",
                "Max Marks":        "—",
                "Percentage":       "—",
                "Matched Keywords": "—",
                "Missed Keywords":  results["error"],
                "Feedback":         "—",
            })
            continue

        # ── Per-question rows ─────────────────────────────────────────────────
        for qid, res in results.items():
            if "error" in res:
                rows.append({
                    "Student":          student_name,
                    "Question":         qid,
                    "Score":            "—",
                    "Max Marks":        "—",
                    "Percentage":       "—",
                    "Matched Keywords": "—",
                    "Missed Keywords":  res["error"],
                    "Feedback":         "—",
                })
            else:
                pct = round(res["score"] / res["max_marks"] * 100, 1) if res["max_marks"] else 0
                rows.append({
                    "Student":          student_name,
                    "Question":         qid,
                    "Score":            res["score"],
                    "Max Marks":        res["max_marks"],
                    "Percentage":       f"{pct}%",
                    "Matched Keywords": ", ".join(res["matched_keywords"]),
                    "Missed Keywords":  ", ".join(res["missed_keywords"]),
                    "Feedback":         res["feedback"],
                })

    return pd.DataFrame(rows).to_csv(index=False)


# =============================================================================
# TAB 1 — Teacher Setup
# =============================================================================
with tab1:
    st.header("Store Answer Key")
    st.info("Paste the teacher's answer key JSON and click **Store**.")

    # ── Default sample answer key shown in text area ──────────────────────────
    default_key = json.dumps([
        {
            "question_id": "Q1",
            "question": "What is photosynthesis?",
            "answer": "Plants convert sunlight into glucose using chlorophyll",
            "keywords": ["sunlight", "glucose", "chlorophyll"],
            "rubric": "2pts process, 2pts keywords, 1pt structure",
            "max_marks": 5,
        },
        {
            "question_id": "Q2",
            "question": "What is Newton's second law?",
            "answer": "Force equals mass times acceleration (F=ma)",
            "keywords": ["force", "mass", "acceleration", "F=ma"],
            "rubric": "3pts formula, 2pts explanation",
            "max_marks": 5,
        },
    ], indent=2)

    # ── Option 1: paste JSON directly ────────────────────────────────────────
    st.subheader("Option 1 — Paste JSON")
    key_input = st.text_area("Answer Key JSON", value=default_key, height=300)

    if st.button("💾 Store Answer Key", use_container_width=True):
        try:
            data = json.loads(key_input)
            r = requests.post(f"{API_URL}/store-answer-key", json=data, timeout=30)
            if r.status_code == 200:
                cnt = r.json().get("count", "?")
                st.success(f"✅ {cnt} question(s) stored successfully!")
            else:
                st.error(f"Server error {r.status_code}: {r.text}")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON — please fix the format.")
        except requests.ConnectionError:
            st.error("❌ FastAPI not reachable. Run: uvicorn main:app --reload")

    st.divider()

    # ── Option 2: upload JSON file ────────────────────────────────────────────
    st.subheader("Option 2 — Upload JSON File")
    uploaded_key = st.file_uploader(
        "Upload answer key (.json)",
        type=["json"],
        key="key_uploader"
    )
    if uploaded_key:
        try:
            data = json.loads(uploaded_key.read())
            st.json(data)
            if st.button("💾 Store Uploaded Key", use_container_width=True):
                r = requests.post(f"{API_URL}/store-answer-key", json=data, timeout=30)
                if r.status_code == 200:
                    cnt = r.json().get("count", "?")
                    st.success(f"✅ {cnt} question(s) stored from file!")
                else:
                    st.error(f"Server error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")


# =============================================================================
# TAB 2 — Grade Worksheets (Single + Batch)
# =============================================================================
with tab2:
    st.header("Grade Student Worksheets")

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Grading mode",
        ["Single worksheet", "Batch (multiple worksheets)"],
        horizontal=True
    )

    st.divider()

    # =========================================================================
    # SINGLE MODE — grade one worksheet
    # =========================================================================
    if mode == "Single worksheet":

        student_name = st.text_input("Student Name (optional)", placeholder="e.g. Karthik")

        uploaded = st.file_uploader(
            "Upload scanned worksheet image",
            type=["jpg", "jpeg", "png"],
            key="single_uploader"
        )

        if uploaded:
            st.image(uploaded, caption="Uploaded worksheet", use_container_width=True)

        if uploaded and st.button("🎯 Grade Worksheet", use_container_width=True):
            with st.spinner("Reading handwriting and grading…"):
                try:
                    r = requests.post(
                        f"{API_URL}/grade-full-sheet",
                        files={"file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            "image/jpeg"
                        )},
                        timeout=300,
                    )
                except requests.ConnectionError:
                    st.error("❌ FastAPI not reachable. Run: uvicorn main:app --reload")
                    st.stop()

            if r.status_code != 200:
                st.error(f"Server error {r.status_code}: {r.text}")
                st.stop()

            # ── Display results ───────────────────────────────────────────────
            label = student_name if student_name else uploaded.name
            st.subheader(f"Results — {label}")

            results = r.json()["results"]
            total_score, total_max = display_student_results(results)
            show_total(total_score, total_max, len(results))

            # ── CSV download ──────────────────────────────────────────────────
            if total_max > 0:
                csv = build_csv({label: results})
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"{label}_results.csv",
                    "text/csv",
                    use_container_width=True,
                )

    # =========================================================================
    # BATCH MODE — grade multiple worksheets in parallel
    # =========================================================================
    else:
        st.info(
            "Upload multiple worksheet images. "
            "Each file name is used as the student identifier. "
            "e.g. `karthik.jpg`, `priya.jpg`"
        )

        uploaded_files = st.file_uploader(
            "Upload scanned worksheet images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        # ── Parallel workers slider ───────────────────────────────────────────
        # Controls how many files are graded simultaneously via ThreadPoolExecutor.
        # Higher = faster, but risks hitting Gemini 15 req/min free tier limit.
        max_workers = st.slider(
            "Parallel workers (higher = faster, more API load)",
            min_value=1,
            max_value=5,
            value=3,
        )

        # ── Preview thumbnails ────────────────────────────────────────────────
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            cols = st.columns(min(len(uploaded_files), 4))
            for i, f in enumerate(uploaded_files):
                with cols[i % 4]:
                    st.image(f.getvalue(), caption=f.name, use_container_width=True)

        if uploaded_files and st.button(
            f"🎯 Grade All {len(uploaded_files)} Worksheet(s)",
            use_container_width=True
        ):
            all_results  = {}
            completed    = 0
            progress_bar = st.progress(0, text="Starting…")
            status_text  = st.empty()

            # ── Parallel grading via ThreadPoolExecutor ───────────────────────
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(grade_one_file, f): f
                    for f in uploaded_files
                }
                for future in as_completed(futures):
                    filename, result = future.result()
                    all_results[filename] = result
                    completed += 1
                    pct = completed / len(uploaded_files)
                    progress_bar.progress(
                        pct,
                        text=f"Graded {completed}/{len(uploaded_files)} — {filename}"
                    )

            progress_bar.progress(1.0, text="✅ All worksheets graded!")
            status_text.success(
                f"✅ Graded {len(uploaded_files)} worksheet(s) successfully!"
            )

            # ── Batch summary table ───────────────────────────────────────────
            st.subheader("📊 Batch Summary")
            summary_rows = []
            for fname, results in all_results.items():
                sname = fname.rsplit(".", 1)[0]
                if isinstance(results, dict) and "error" in results:
                    summary_rows.append({
                        "Student":     sname,
                        "Total Score": "—",
                        "Max Marks":   "—",
                        "Percentage":  "Error",
                        "Status":      results["error"],
                    })
                    continue
                ts = tm = 0.0
                for res in results.values():
                    if "error" not in res:
                        ts += res["score"]
                        tm += res["max_marks"]
                pct_val = round(ts / tm * 100, 1) if tm else 0
                grade   = "🏆 Excellent" if pct_val >= 80 else ("👍 Good" if pct_val >= 60 else "Needs Work")
                summary_rows.append({
                    "Student":     sname,
                    "Total Score": ts,
                    "Max Marks":   tm,
                    "Percentage":  f"{pct_val}%",
                    "Status":      grade,
                })

            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )

            # ── CSV download ──────────────────────────────────────────────────
            csv = build_csv(all_results)
            st.download_button(
                "📥 Download All Results (CSV)",
                csv,
                "batch_results.csv",
                "text/csv",
                use_container_width=True,
            )

            st.divider()

            # ── Per-student detailed results ──────────────────────────────────
            st.subheader("📋 Detailed Results Per Student")

            for fname, results in all_results.items():
                sname = fname.rsplit(".", 1)[0]
                st.markdown(
                    f"<div class='student-header'>👤 {sname}</div>",
                    unsafe_allow_html=True,
                )
                if isinstance(results, dict) and "error" in results:
                    st.error(f"Grading failed: {results['error']}")
                    continue

                total_score, total_max = display_student_results(results)
                show_total(total_score, total_max, len(results))
                st.markdown("---")
