# 🤖 RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot with persistent chat history.  
Upload any PDF → Ask questions → Chat history saved automatically → Resume later.

---

## Features

✅ **Upload PDFs** (up to 500 MB)  
✅ **Ask questions** with context-aware answers  
✅ **Conversation memory** — remembers what you discussed  
✅ **Persistent sessions** — close and resume anytime  
✅ **Multiple sessions** — chat with different PDFs separately  
✅ **Local embeddings** — no API limits for embeddings  
✅ **Fast responses** — powered by Groq (Llama 3.3 70B)

---

## Quick Start

### 1. Get a Groq API Key (Free)
Go to: **https://console.groq.com/keys**  
Create a key and add it to `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
**Option A — Use the startup script:**
```bash
cd rag-chatbot
./start.sh
```

**Option B — Run manually (2 terminals):**

Terminal 1 (FastAPI):
```bash
cd rag-chatbot
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Terminal 2 (Streamlit):
```bash
cd rag-chatbot
streamlit run app.py
```

Open: **http://localhost:8501**

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      UPLOAD PDF                              │
│  PDF → Chunks → HuggingFace Embeddings → ChromaDB           │
│                         ↓                                    │
│              Create new session (saved to disk)              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      ASK QUESTION                            │
│  Question → Vector Search → Top 3 Chunks → Groq LLM → Answer │
│                         ↓                                    │
│            Save Q&A to session (persists on disk)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RESUME LATER                              │
│  Sessions saved in ./sessions/ folder                        │
│  Click any session in sidebar to resume chat                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
rag-chatbot/
├── main.py          # FastAPI backend (upload, chat, sessions)
├── rag.py           # RAG logic (embeddings, retrieval, LLM)
├── app.py           # Streamlit frontend
├── .env             # Groq API key
├── requirements.txt # Dependencies
├── start.sh         # Startup script
├── uploads/         # Uploaded PDFs
├── sessions/        # Saved chat sessions (JSON files)
└── chroma_store/    # Vector embeddings (ChromaDB)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload PDF, create new session |
| `POST` | `/chat` | Ask question (saves to session) |
| `GET` | `/sessions` | List all saved sessions |
| `GET` | `/sessions/{id}` | Load specific session |
| `DELETE` | `/sessions/{id}` | Delete a session |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Web UI |
| Backend | FastAPI | REST API |
| LLM | Groq (Llama 3.3 70B) | Answer generation |
| Embeddings | HuggingFace (local) | Text → vectors |
| Vector Store | ChromaDB | Similarity search |
| Orchestration | LangChain | RAG pipeline |
| Storage | JSON files | Session persistence |

---

## Limits

| Metric | Limit |
|--------|-------|
| PDF size | 500 MB |
| Groq free tier | ~14,000 requests/day |
| Context window | 8,192 tokens |

---

## Troubleshooting

**"No document uploaded yet"**
→ Upload a PDF first, or select a saved session from the sidebar.

**"Connection refused"**
→ FastAPI isn't running. Start it with: `python3 -m uvicorn main:app --reload`

**Slow responses**
→ First question loads the embedding model (takes ~10s). Subsequent questions are fast.

**Rate limit errors**
→ You hit Groq's free tier limit. Wait 60 seconds or get a new API key.

---

## Reset Everything

Delete all saved data and start fresh:
```bash
rm -rf uploads/ sessions/ chroma_store/
```
