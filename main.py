import os
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ingest_pdf, ask_question

app = FastAPI(title="RAG Chatbot API")

# Allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Directories
UPLOAD_DIR = "./uploads"
SESSIONS_DIR = "./sessions"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024


# ─────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────

def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def save_session(session_id: str, pdf_name: str, messages: List[Dict], 
                 created_at: str = None, name: str = None):
    """Save session to JSON file."""
    now = datetime.now().isoformat()
    path = get_session_path(session_id)
    
    # Preserve created_at if updating existing session
    existing = load_session(session_id)
    if existing and created_at is None:
        created_at = existing.get("created_at", now)
    if created_at is None:
        created_at = now
    
    # Preserve name or use existing
    if name is None and existing:
        name = existing.get("name")
    if name is None:
        name = pdf_name.replace(".pdf", "")
    
    session_data = {
        "id": session_id,
        "name": name,
        "pdf_name": pdf_name,
        "created_at": created_at,
        "updated_at": now,
        "messages": messages
    }
    
    with open(path, "w") as f:
        json.dump(session_data, f, indent=2)


def load_session(session_id: str) -> Optional[Dict]:
    """Load session from JSON file."""
    path = get_session_path(session_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def list_sessions() -> List[Dict]:
    """List all saved sessions."""
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(SESSIONS_DIR, filename), "r") as f:
                data = json.load(f)
                sessions.append({
                    "id": data.get("id", ""),
                    "name": data.get("name", data.get("pdf_name", "Untitled")),
                    "pdf_name": data.get("pdf_name", ""),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", []))
                })
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


# ─────────────────────────────────────────────────────────────
# API MODELS
# ─────────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    message: str
    chunks: int
    file_size_mb: float

class ChatRequest(BaseModel):
    question: str
    session_id: str
    chat_history: List[List[str]] = []

class ChatResponse(BaseModel):
    answer: str

class RenameRequest(BaseModel):
    name: str


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.post("/upload", response_model=SessionResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and create a new chat session."""
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    session_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    total_size = 0
    
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                os.remove(file_path)
                raise HTTPException(status_code=413, detail="File too large. Max 500 MB.")
            f.write(chunk)
    
    try:
        num_chunks = ingest_pdf(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    save_session(session_id, file.filename, [])
    
    return SessionResponse(
        session_id=session_id,
        message=f"PDF uploaded and indexed successfully.",
        chunks=num_chunks,
        file_size_mb=round(total_size / (1024 * 1024), 1)
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with PDF and save to session history."""
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if not os.path.exists("./chroma_store"):
        raise HTTPException(status_code=400, detail="No document uploaded yet.")
    
    history_tuples = [
        (pair[0], pair[1])
        for pair in request.chat_history
        if len(pair) == 2
    ]
    
    answer = ask_question(request.question, history_tuples)
    
    session = load_session(request.session_id)
    if session:
        session["messages"].append({"role": "user", "content": request.question})
        session["messages"].append({"role": "assistant", "content": answer})
        save_session(
            request.session_id, 
            session["pdf_name"], 
            session["messages"],
            name=session.get("name")
        )
    
    return ChatResponse(answer=answer)


@app.get("/sessions")
async def get_sessions():
    """Get all saved chat sessions."""
    return {"sessions": list_sessions()}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Load a specific chat session with full history."""
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, request: RenameRequest):
    """Rename a chat session."""
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    save_session(
        session_id,
        session["pdf_name"],
        session["messages"],
        created_at=session.get("created_at"),
        name=request.name
    )
    
    return {"message": "Session renamed.", "name": request.name}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    path = get_session_path(session_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Session not found.")
    os.remove(path)
    return {"message": f"Session deleted."}
