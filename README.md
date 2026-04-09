# RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot with persistent chat history.
Upload any PDF, ask questions, and resume conversations anytime.

---

## Features

- Upload PDFs and ask questions instantly
- Conversation memory — remembers what you discussed
- Persistent sessions — close and resume later
- Multiple sessions — chat with different PDFs separately
- Local embeddings — no API limits for vector generation
- Fast responses — powered by Groq (Llama 3.3 70B)

---

## Local Setup

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
```bash
cd rag-chatbot
streamlit run app.py
```

Open: **http://localhost:8501**

---

## Deploy to Streamlit Cloud (Free)

### 1. Fork or use this repo
This repo is already on GitHub at: `https://github.com/nish0753/rag-chat-bot`

### 2. Go to Streamlit Cloud
Visit: **https://share.streamlit.io**
Sign in with GitHub and click **New app**.

### 3. Configure the app
- **Repository:** `nish0753/rag-chat-bot`
- **Branch:** `main`
- **Main file path:** `app.py`

### 4. Add your Groq API key
In your Streamlit Cloud app dashboard, go to **Settings > Secrets** and add:
```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

### 5. Deploy
Click **Deploy**. Your app will be live in about 1-2 minutes.

**Important:** Streamlit Cloud uses ephemeral storage, so uploaded PDFs, vector stores, and sessions persist only while the app is active. If the app restarts, you will need to re-upload PDFs. For persistent storage, consider deploying on Render or Railway with a mounted disk.

---

## How It Works

```
Upload PDF -> Split into chunks -> Embed -> Store in ChromaDB
                                           |
Ask question -> Vector search -> Top chunks -> Groq LLM -> Answer
                                           |
                              Save Q&A to session (JSON file)
```

---

## Folder Structure

```
rag-chatbot/
├── app.py           # Streamlit app (frontend + RAG logic)
├── .env             # Groq API key (local only)
├── requirements.txt # Dependencies
├── uploads/         # Uploaded PDFs
├── sessions/        # Saved chat sessions (JSON files)
└── chroma_store/    # Vector embeddings (ChromaDB)
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace (sentence-transformers) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Storage | JSON files (sessions) |

---

## Troubleshooting

**"Error processing PDF"**
- Make sure your `.env` file has a valid `GROQ_API_KEY`
- First question loads the embedding model (~10s). Subsequent questions are faster.

**Rate limit errors**
- You hit Groq's free tier limit. Wait 60 seconds or upgrade your plan.

**App runs out of memory on Streamlit Cloud**
- Large PDFs or many sessions can exceed the free tier memory limit. Try smaller PDFs or deploy on a platform with more resources.

---

## Reset Everything

Delete all saved data and start fresh:
```bash
rm -rf uploads/ sessions/ chroma_store/
```
