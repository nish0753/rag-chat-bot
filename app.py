import os
import json
import uuid
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain

# ── Configuration ──────────────────────────────────────────────
# Support both .env (local) and Streamlit Secrets (cloud)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Set it in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

CHROMA_DIR = "./chroma_store"
UPLOAD_DIR = "./uploads"
SESSIONS_DIR = "./sessions"
MAX_FILE_SIZE = 500 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ── RAG Components ─────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.3)


def ingest_pdf(file_path: str) -> int:
    """Load a PDF, split into chunks, embed and store in ChromaDB."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    return len(chunks)


def ask_question(question: str, chat_history: List[Tuple[str, str]] = None) -> str:
    """Retrieve relevant chunks and answer using Groq."""
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        rephrase_question=False
    )

    result = qa_chain.invoke({"question": question, "chat_history": chat_history or []})
    return result["answer"]


def generate_quiz(num_questions: int = 15) -> list:
    """Generate a multiple-choice quiz from the PDF content."""
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Retrieve diverse chunks from the entire document
    all_docs = vector_store.get()
    # Sample chunks for breadth
    import random
    indices = random.sample(range(len(all_docs["ids"])), min(num_questions + 5, len(all_docs["ids"])))
    sampled_texts = [all_docs["documents"][i] for i in indices]

    # Build context from sampled chunks
    context = "\n\n---\n\n".join(sampled_texts[:num_questions])

    prompt = f"""Based on the following document context, generate exactly {num_questions} multiple-choice quiz questions. Each question must have 4 options (A, B, C, D) with exactly one correct answer.

Return ONLY valid JSON in this exact format, no markdown, no extra text:
[{{"question":"Question text","options":["Option A","Option B","Option C","Option D"],"correct_index":0,"explanation":"Detailed explanation of why the correct answer is correct and why others are wrong, with reference to the document content."}}]

Rules:
- Questions should cover different topics from the context
- Options should be plausible but only one is correct
- correct_index is 0-based (0=A, 1=B, 2=C, 3=D)
- Explanation should be thorough and educational
- Do not include any text before or after the JSON array

Context:
{context}"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Strip markdown code blocks if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("\n", 1)[0]
        content = content.strip("`").strip()

    import json as _json
    questions = _json.loads(content)
    return questions


def get_question_explanation(question_text: str, correct_answer: str) -> str:
    """Get a detailed explanation for a specific quiz question."""
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    docs = vector_store.similarity_search(question_text, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Based on the following document context, provide a thorough explanation of this quiz question.

Question: {question_text}
Correct Answer: {correct_answer}

Document Context:
{context}

Provide a detailed, educational explanation that covers:
1. Why this answer is correct (with specific references to the context)
2. Key concepts and background information
3. Any related important details a student should know"""

    response = llm.invoke(prompt)
    return response.content


# ── Session Management ─────────────────────────────────────────
def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def save_session(session_id: str, pdf_name: str, messages: list,
                 created_at: str = None, name: str = None):
    now = datetime.now().isoformat()
    path = get_session_path(session_id)

    existing = load_session(session_id)
    if existing and created_at is None:
        created_at = existing.get("created_at", now)
    if created_at is None:
        created_at = now
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


def load_session(session_id: str):
    path = get_session_path(session_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def list_sessions() -> list:
    sessions = []
    if not os.path.exists(SESSIONS_DIR):
        return sessions
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


def rename_session(session_id: str, new_name: str):
    session = load_session(session_id)
    if session:
        save_session(session_id, session["pdf_name"], session["messages"],
                     created_at=session.get("created_at"), name=new_name)


def delete_session(session_id: str):
    path = get_session_path(session_id)
    if os.path.exists(path):
        os.remove(path)


def format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now()
        diff = (now - dt).total_seconds()

        if diff < 60:
            return "Just now"
        elif diff < 3600:
            return f"{int(diff/60)}m ago"
        elif diff < 86400:
            return f"{int(diff/3600)}h ago"
        else:
            return dt.strftime("%d %b")
    except:
        return ""


# ── Streamlit Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a !important; }
    .main { background-color: #0f0f1a !important; }
    .stApp, .main, p, span, div, label, h1, h2, h3, h4, h5, h6 { color: #e2e8f0 !important; }
    .stCaption, [data-testid="stCaption"] { color: #64748b !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid #2d2d4e !important;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stCaption { color: #64748b !important; }
    [data-testid="stChatMessage"] {
        background: #1e1e30 !important; border: 1px solid #2d2d4e !important;
        border-radius: 16px; padding: 16px; margin-bottom: 12px;
    }
    [data-testid="stChatMessage"] * { color: #e2e8f0 !important; background: transparent !important; }
    [data-testid="stChatInput"] {
        border: 1px solid #3d3d5a !important; border-radius: 16px; background: #1a1a2e !important;
    }
    [data-testid="stChatInput"] textarea { background: #1a1a2e !important; color: #e2e8f0 !important; }
    [data-testid="stChatInput"]::placeholder { color: #64748b !important; }
    .stButton button {
        border-radius: 10px; font-weight: 500; transition: all 0.2s;
        background: #1e1e30 !important; border: 1px solid #3d3d5a !important; color: #e2e8f0 !important;
    }
    .stButton button:hover {
        transform: translateY(-1px); box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3); border-color: #6366f1 !important;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important; color: white !important;
    }
    .stButton button[kind="primary"]:hover { box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4); }
    [data-testid="stFileUploader"] {
        border: 2px dashed #3d3d5a !important; border-radius: 12px; background: #1a1a2e !important; padding: 20px;
    }
    [data-testid="stFileUploader"] * { color: #e2e8f0 !important; }
    [data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }
    hr { border-color: #2d2d4e !important; }
    .stTextInput input {
        background: #1a1a2e !important; border: 1px solid #3d3d5a !important;
        color: #e2e8f0 !important; border-radius: 8px;
    }
    .stTextInput input:focus { border-color: #6366f1 !important; }
    .stTextInput label { color: #e2e8f0 !important; }
    .stSpinner > div { border-color: #6366f1 transparent transparent transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = None
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_rename" not in st.session_state:
    st.session_state.show_rename = None
if "editing_name" not in st.session_state:
    st.session_state.editing_name = False
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_show_explanation" not in st.session_state:
    st.session_state.quiz_show_explanation = {}
if "num_quiz_questions" not in st.session_state:
    st.session_state.num_quiz_questions = 15


# ── Sidebar: Sessions ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### Chat Sessions")

    sessions = list_sessions()

    if sessions:
        st.caption(f"{len(sessions)} saved conversation{'s' if len(sessions) > 1 else ''}")
        st.divider()

        for session in sessions:
            is_active = session["id"] == st.session_state.current_session_id

            col1, col2, col3 = st.columns([6, 1, 1])

            with col1:
                btn_label = f"{'> ' if is_active else ''}{session['name'][:25]}{'...' if len(session['name']) > 25 else ''}"
                if st.button(btn_label, key=f"load_{session['id']}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    data = load_session(session["id"])
                    if data:
                        st.session_state.current_session_id = session["id"]
                        st.session_state.current_session_name = session["name"]
                        st.session_state.current_pdf = data["pdf_name"]
                        st.session_state.messages = data["messages"]
                        st.session_state.show_rename = None
                        st.rerun()

            with col2:
                if st.button("Edit", key=f"edit_{session['id']}", help="Rename session"):
                    st.session_state.show_rename = session["id"]
                    st.rerun()

            with col3:
                if st.button("Del", key=f"del_{session['id']}", help="Delete session"):
                    delete_session(session["id"])
                    if st.session_state.current_session_id == session["id"]:
                        st.session_state.current_session_id = None
                        st.session_state.current_session_name = None
                        st.session_state.current_pdf = None
                        st.session_state.messages = []
                    st.rerun()

            st.caption(f"{session['message_count']} messages - {format_time(session['updated_at'])}")

            if st.session_state.show_rename == session["id"]:
                new_name = st.text_input("New name", value=session["name"],
                                         key=f"rename_input_{session['id']}", label_visibility="collapsed")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Save", key=f"save_rename_{session['id']}", use_container_width=True):
                        rename_session(session["id"], new_name)
                        if st.session_state.current_session_id == session["id"]:
                            st.session_state.current_session_name = new_name
                        st.session_state.show_rename = None
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_rename_{session['id']}", use_container_width=True):
                        st.session_state.show_rename = None
                        st.rerun()

            st.divider()
    else:
        st.info("No saved sessions yet. Upload a PDF to start!")
        st.divider()

    if st.button("New Chat", use_container_width=True, type="primary"):
        st.session_state.current_session_id = None
        st.session_state.current_session_name = None
        st.session_state.current_pdf = None
        st.session_state.messages = []
        st.session_state.show_rename = None
        st.session_state.editing_name = False
        st.rerun()


# ── Main Content ───────────────────────────────────────────────
if not st.session_state.current_session_id:
    # -- Welcome Screen --
    st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <h1 style='margin-bottom: 0.5rem;'>RAG Chatbot</h1>
            <p style='font-size: 1.1rem; margin-bottom: 2.5rem; opacity: 0.7;'>
                Upload a PDF and start asking questions
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"],
                                         help="Maximum file size: 500 MB")

        if uploaded_file:
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)

            if file_size > MAX_FILE_SIZE:
                st.error(f"File too large: {file_size_mb:.1f} MB. Maximum: 500 MB")
            else:
                st.info(f"Selected: **{uploaded_file.name}** - {file_size_mb:.1f} MB")

                if st.button("Upload and Start Chatting", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {file_size_mb:.1f} MB..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = os.path.join(UPLOAD_DIR, f"temp_{uploaded_file.name}")
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Ingest PDF
                            num_chunks = ingest_pdf(temp_path)
                            os.remove(temp_path)

                            # Create session
                            session_id = str(uuid.uuid4())[:8]
                            save_session(session_id, uploaded_file.name, [])

                            st.session_state.current_session_id = session_id
                            st.session_state.current_session_name = uploaded_file.name.replace(".pdf", "")
                            st.session_state.current_pdf = uploaded_file.name
                            st.session_state.messages = []
                            st.success(f"Ready! {num_chunks} chunks indexed")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing PDF: {e}")

        st.markdown("""
            <div style='text-align: center; margin-top: 2.5rem; opacity: 0.5; font-size: 0.9rem;'>
                <p>Chat with PDFs - Resume conversations anytime - Rename and organize</p>
            </div>
        """, unsafe_allow_html=True)

else:
    # -- Chat Screen or Quiz --
    col1, col2, col3 = st.columns([5, 1, 1])

    with col1:
        if st.session_state.editing_name:
            new_name = st.text_input("Session name", value=st.session_state.current_session_name,
                                     key="edit_name_input", label_visibility="collapsed")
            if st.button("Save", key="save_header_name"):
                rename_session(st.session_state.current_session_id, new_name)
                st.session_state.current_session_name = new_name
                st.session_state.editing_name = False
                st.rerun()
        else:
            st.markdown(f"""
                <div style='display: flex; align-items: baseline; gap: 12px; margin-bottom: 0;'>
                    <h2 style='margin: 0;'>{st.session_state.current_session_name}</h2>
                    <span style='opacity: 0.4; font-size: 0.85rem;'>({st.session_state.current_pdf})</span>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        mode_label = "Chat" if st.session_state.quiz_mode else "Quiz"
        mode_icon = "💬" if st.session_state.quiz_mode else "📝"
        if st.button(f"{mode_icon} {mode_label}", key="toggle_mode"):
            st.session_state.quiz_mode = not st.session_state.quiz_mode
            if not st.session_state.quiz_mode:
                # Reset quiz state when switching back to chat
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_show_explanation = {}
            st.rerun()

    with col3:
        if st.button("New", key="header_new"):
            st.session_state.current_session_id = None
            st.session_state.current_session_name = None
            st.session_state.current_pdf = None
            st.session_state.messages = []
            st.session_state.editing_name = False
            st.session_state.quiz_mode = False
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_show_explanation = {}
            st.rerun()

    st.divider()

    # ── Quiz Mode ──────────────────────────────────────────────
    if st.session_state.quiz_mode:
        if not st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            # Quiz setup screen
            st.markdown("""
                <div style='text-align: center; padding: 2rem 1rem;'>
                    <h1>Generate a Quiz</h1>
                    <p style='opacity: 0.7;'>Questions will be generated from your PDF content</p>
                </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                num_q = st.selectbox(
                    "Number of questions",
                    options=[10, 15, 20],
                    index=1,
                    help="More questions covers more topics but takes longer to generate"
                )

                if st.button("Generate Quiz", type="primary", use_container_width=True):
                    with st.spinner("Generating questions from your PDF..."):
                        try:
                            questions = generate_quiz(num_q)
                            st.session_state.quiz_questions = questions
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_show_explanation = {}
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate quiz: {e}")

        elif st.session_state.quiz_questions:
            # Display quiz
            progress = len(st.session_state.quiz_answers)
            total = len(st.session_state.quiz_questions)
            st.progress(progress / total, text=f"Answered: {progress}/{total}")

            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"### Question {i+1}: {q['question']}")

                # Determine the state for this question
                selected = st.session_state.quiz_answers.get(i, None)
                is_answered = selected is not None

                # Radio buttons for options
                labels = ["A", "B", "C", "D"]
                chosen = st.radio(
                    f"Select your answer for Q{i+1}",
                    options=q["options"],
                    index=None if not is_answered else selected,
                    format_func=lambda x: x,
                    disabled=is_answered,
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )

                if chosen is not None and not is_answered:
                    st.session_state.quiz_answers[i] = chosen
                    st.rerun()

                # Show correct/incorrect after answering
                if is_answered:
                    is_correct = selected == q["correct_index"]
                    if is_correct:
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect. The correct answer was: {q['options'][q['correct_index']]}")

                    # Show explanation button
                    exp_key = f"exp_{i}"
                    if st.button("Explain this answer", key=exp_key):
                        st.session_state.quiz_show_explanation[i] = not st.session_state.quiz_show_explanation.get(i, False)
                        st.rerun()

                    if st.session_state.quiz_show_explanation.get(i, False):
                        with st.spinner("Generating detailed explanation..."):
                            try:
                                explanation = get_question_explanation(
                                    q["question"],
                                    q["options"][q["correct_index"]]
                                )
                                st.markdown(f"**Explanation:** {explanation}")
                            except Exception as e:
                                # Fallback to built-in explanation
                                st.markdown(f"**Explanation:** {q.get('explanation', 'No explanation available.')}")

                st.divider()

            # Submit button if all answered
            if progress == total and not st.session_state.quiz_submitted:
                if st.button("Submit Quiz and View Summary", type="primary", use_container_width=True):
                    st.session_state.quiz_submitted = True
                    st.rerun()

            # Summary
            if st.session_state.quiz_submitted:
                correct_count = sum(
                    1 for i, q in enumerate(st.session_state.quiz_questions)
                    if st.session_state.quiz_answers.get(i) == q["correct_index"]
                )
                score = (correct_count / total) * 100

                st.markdown("## Quiz Summary")

                # Score display
                if score >= 80:
                    st.success(f"Score: {correct_count}/{total} ({score:.0f}%) - Great job!")
                elif score >= 60:
                    st.warning(f"Score: {correct_count}/{total} ({score:.0f}%) - Good effort!")
                else:
                    st.error(f"Score: {correct_count}/{total} ({score:.0f}%) - Keep studying!")

                # Detailed breakdown
                st.markdown("### Question Breakdown")
                for i, q in enumerate(st.session_state.quiz_questions):
                    is_correct = st.session_state.quiz_answers.get(i) == q["correct_index"]
                    status = "✓" if is_correct else "✗"
                    with st.expander(f"{status} Q{i+1}: {q['question'][:80]}..."):
                        st.markdown(f"**Your answer:** {q['options'][st.session_state.quiz_answers[i]] if i in st.session_state.quiz_answers else 'Not answered'}")
                        st.markdown(f"**Correct answer:** {q['options'][q['correct_index']]}")
                        if "explanation" in q:
                            st.markdown(f"**Why:** {q['explanation']}")

                # Retake button
                if st.button("Generate New Quiz", type="primary", use_container_width=True):
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_show_explanation = {}
                    st.rerun()

    # ── Chat Mode ──────────────────────────────────────────────
    else:
        # Chat messages
        chat_container = st.container()

        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                    <div style='text-align: center; padding: 4rem 1rem; opacity: 0.4;'>
                        <p style='font-size: 1.3rem; margin-bottom: 0.5rem;'>Start the conversation</p>
                        <p style='font-size: 0.9rem;'>Ask anything about your PDF</p>
                    </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Chat input
        question = st.chat_input("Ask a question about your PDF...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            with chat_container:
                with st.chat_message("user"):
                    st.write(question)

                # Build chat history
                chat_history = []
                msgs = st.session_state.messages[:-1]
                for i in range(0, len(msgs) - 1, 2):
                    if i + 1 < len(msgs) and msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                        chat_history.append((msgs[i]["content"], msgs[i + 1]["content"]))

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = ask_question(question, chat_history)
                            st.write(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            # Save to disk after each response
                            save_session(
                                st.session_state.current_session_id,
                                st.session_state.current_pdf,
                                st.session_state.messages
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.session_state.messages.pop()
