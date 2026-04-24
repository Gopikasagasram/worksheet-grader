# =============================================================================
# backend/main.py
# FastAPI application entry point.
# Responsibilities:
#   - Create the FastAPI app instance
#   - Register CORS middleware
#   - Mount the API router from api/routes.py
# All business logic is delegated to services/.
# All route definitions live in api/routes.py.
# =============================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Worksheet Grader API",
    version="1.0.0",
    description="AI-powered handwritten worksheet grading pipeline.",
)

# ── CORS — allow all origins for local dev and n8n cloud calls ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register all routes from api/routes.py ────────────────────────────────────
app.include_router(router)
