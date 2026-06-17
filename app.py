"""
app.py
------
FastAPI backend for the ConvoRAG chatbot.

Routes:
  GET  /              -> chat UI  (static/index.html)
  GET  /style.css     -> CSS
  GET  /app.js        -> JS
  GET  /api/stats     -> system statistics
  GET  /api/persona   -> persona JSON
  GET  /api/checkpoints -> topic timeline
  POST /api/chat      -> RAG query -> Groq answer
  GET  /api/checkpoint/{id} -> single checkpoint detail
  GET  /docs          -> auto-generated Swagger UI (bonus!)
"""

import json
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ConvoRAG",
    description="Conversation Intelligence – RAG + Persona Chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files at /static/* (fallback)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Lazy globals ───────────────────────────────────────────────────────────────
_persona: Optional[dict]    = None
_checkpoints: Optional[list] = None
_engine = None


def get_persona() -> dict:
    global _persona
    if _persona is None:
        p = DATA_DIR / "persona.json"
        _persona = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _persona


def get_checkpoints() -> list:
    global _checkpoints
    if _checkpoints is None:
        cp = DATA_DIR / "checkpoints.json"
        _checkpoints = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else []
    return _checkpoints


def get_engine():
    global _engine
    if _engine is None:
        from rag_engine import get_engine as _get_engine
        _engine = _get_engine()
        _engine.load()
    return _engine


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: dict


# ── Static file routes ─────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/style.css", include_in_schema=False)
def serve_css():
    return FileResponse(str(STATIC_DIR / "style.css"), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def serve_js():
    return FileResponse(str(STATIC_DIR / "app.js"), media_type="application/javascript")


# ── API routes ─────────────────────────────────────────────────────────────────
@app.get("/api/stats", summary="System statistics")
def api_stats():
    cps     = get_checkpoints()
    persona = get_persona()
    return {
        "total_messages":    persona.get("total_messages_analysed", 0),
        "topic_segments":    sum(1 for c in cps if c["type"] == "topic"),
        "hundred_checkpoints": sum(1 for c in cps if c["type"] == "hundred"),
        "personality_traits": persona.get("personality_traits", []),
        "top_habits":         persona.get("habits", []),
    }


@app.get("/api/persona", summary="Full persona JSON")
def api_persona():
    return get_persona()


@app.get("/api/checkpoints", summary="Topic timeline + 100-msg checkpoints")
def api_checkpoints():
    cps = get_checkpoints()
    topic_cps   = [c for c in cps if c["type"] == "topic"]
    hundred_cps = [c for c in cps if c["type"] == "hundred"]
    return {
        "topic_checkpoints":   topic_cps[:50],
        "hundred_checkpoints": hundred_cps[:20],
        "total_topic":         len(topic_cps),
        "total_hundred":       len(hundred_cps),
    }


@app.get("/api/checkpoint/{cp_id}", summary="Single checkpoint detail")
def api_checkpoint_detail(cp_id: str):
    for cp in get_checkpoints():
        if cp["id"] == cp_id:
            return cp
    raise HTTPException(status_code=404, detail="Checkpoint not found")


@app.post("/api/chat", response_model=ChatResponse, summary="Ask the RAG chatbot")
def api_chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    try:
        engine  = get_engine()
        persona = get_persona()
        result  = engine.answer(query, persona=persona)
        return ChatResponse(
            query=query,
            answer=result["answer"],
            sources=result.get("sources", {}),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import os

    if not (DATA_DIR / "checkpoints.json").exists():
        print("ERROR: Data not built. Run: python build_index.py")
        raise SystemExit(1)

    port = int(os.getenv("PORT", 5000))
    print(f"Starting ConvoRAG on port {port}")
    print(f"API docs at  http://localhost:{port}/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
