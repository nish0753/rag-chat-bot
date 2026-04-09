import streamlit as st
import requests
from datetime import datetime

API_URL = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Force Dark Theme CSS ───────────────────────────────────────
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════
       ENFORCE DARK THEME EVERYWHERE
       ═══════════════════════════════════════════════════════════ */
    
    /* Main app background */
    .stApp {
        background-color: #0f0f1a !important;
    }
    
    /* Main content area */
    .main {
        background-color: #0f0f1a !important;
    }
    
    /* All text */
    .stApp, .main, p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    
    /* Muted text */
    .stCaption, [data-testid="stCaption"] {
        color: #64748b !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid #2d2d4e !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        color: #64748b !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #1e1e30 !important;
        border: 1px solid #2d2d4e !important;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    [data-testid="stChatMessage"] * {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        border: 1px solid #3d3d5a !important;
        border-radius: 16px;
        background: #1a1a2e !important;
    }
    
    [data-testid="stChatInput"] textarea {
        background: #1a1a2e !important;
        color: #e2e8f0 !important;
    }
    
    [data-testid="stChatInput"]::placeholder {
        color: #64748b !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s;
        background: #1e1e30 !important;
        border: 1px solid #3d3d5a !important;
        color: #e2e8f0 !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        border-color: #6366f1 !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #3d3d5a !important;
        border-radius: 12px;
        background: #1a1a2e !important;
        padding: 20px;
    }
    
    [data-testid="stFileUploader"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1 !important;
    }
    
    /* Info/Success/Warning boxes */
    .stSuccess {
        background: #1e3a2f !important;
        border: 1px solid #2d5a45 !important;
        border-radius: 12px;
        color: #6ee7b7 !important;
    }
    
    .stSuccess * {
        color: #6ee7b7 !important;
    }
    
    .stInfo {
        background: #1e2a4a !important;
        border: 1px solid #2d3d6a !important;
        border-radius: 12px;
        color: #93c5fd !important;
    }
    
    .stInfo * {
        color: #93c5fd !important;
    }
    
    .stWarning {
        background: #3a3520 !important;
        border: 1px solid #5a5530 !important;
        border-radius: 12px;
        color: #fcd34d !important;
    }
    
    .stWarning * {
        color: #fcd34d !important;
    }
    
    .stError {
        background: #3a1e1e !important;
        border: 1px solid #5a2d2d !important;
        border-radius: 12px;
    }
    
    /* Dividers */
    hr {
        border-color: #2d2d4e !important;
    }
    
    /* Text input */
    .stTextInput input {
        background: #1a1a2e !important;
        border: 1px solid #3d3d5a !important;
        color: #e2e8f0 !important;
        border-radius: 8px;
    }
    
    .stTextInput input:focus {
        border-color: #6366f1 !important;
    }
    
    .stTextInput label {
        color: #e2e8f0 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #6366f1 transparent transparent transparent !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #1a1a2e !important;
        border-color: #3d3d5a !important;
    }
    
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Sidebar spacing */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Chat avatars */
    [data-testid="stChatMessageAvatar"] {
        background: #6366f1 !important;
    }
    
    /* Hide streamlit footer */
    [data-testid="stSidebarBottom"] {
        display: none;
    }
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

# ── Helper Functions ───────────────────────────────────────────
def load_sessions():
    try:
        response = requests.get(f"{API_URL}/sessions", timeout=10)
        if response.status_code == 200:
            return response.json()["sessions"]
    except:
        pass
    return []

def load_session(session_id):
    try:
        response = requests.get(f"{API_URL}/sessions/{session_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def rename_session(session_id, new_name):
    try:
        response = requests.patch(
            f"{API_URL}/sessions/{session_id}",
            json={"name": new_name},
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

def delete_session(session_id):
    try:
        requests.delete(f"{API_URL}/sessions/{session_id}", timeout=10)
    except:
        pass

def format_time(iso_str):
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

# ── Sidebar: Sessions ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Chat Sessions")
    
    sessions = load_sessions()
    
    if sessions:
        st.caption(f"{len(sessions)} saved conversation{'s' if len(sessions) > 1 else ''}")
        st.divider()
        
        for session in sessions:
            is_active = session["id"] == st.session_state.current_session_id
            
            # Session row
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                btn_label = f"{'▶ ' if is_active else ''}{session['name'][:25]}{'...' if len(session['name']) > 25 else ''}"
                if st.button(
                    btn_label,
                    key=f"load_{session['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    data = load_session(session["id"])
                    if data:
                        st.session_state.current_session_id = session["id"]
                        st.session_state.current_session_name = session["name"]
                        st.session_state.current_pdf = data["pdf_name"]
                        st.session_state.messages = data["messages"]
                        st.session_state.show_rename = None
                        st.rerun()
            
            with col2:
                if st.button("✏️", key=f"edit_{session['id']}", help="Rename session"):
                    st.session_state.show_rename = session["id"]
                    st.rerun()
            
            with col3:
                if st.button("🗑", key=f"del_{session['id']}", help="Delete session"):
                    delete_session(session["id"])
                    if st.session_state.current_session_id == session["id"]:
                        st.session_state.current_session_id = None
                        st.session_state.current_session_name = None
                        st.session_state.current_pdf = None
                        st.session_state.messages = []
                    st.rerun()
            
            # Timestamp
            st.caption(f"{session['message_count']} messages • {format_time(session['updated_at'])}")
            
            # Rename input
            if st.session_state.show_rename == session["id"]:
                new_name = st.text_input(
                    "New name",
                    value=session["name"],
                    key=f"rename_input_{session['id']}",
                    label_visibility="collapsed"
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✓ Save", key=f"save_rename_{session['id']}", use_container_width=True):
                        if rename_session(session["id"], new_name):
                            if st.session_state.current_session_id == session["id"]:
                                st.session_state.current_session_name = new_name
                        st.session_state.show_rename = None
                        st.rerun()
                with c2:
                    if st.button("✗ Cancel", key=f"cancel_rename_{session['id']}", use_container_width=True):
                        st.session_state.show_rename = None
                        st.rerun()
            
            st.divider()
    else:
        st.info("📁 No saved sessions yet.\n\nUpload a PDF to start!")
        st.divider()
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_session_id = None
        st.session_state.current_session_name = None
        st.session_state.current_pdf = None
        st.session_state.messages = []
        st.session_state.show_rename = None
        st.session_state.editing_name = False
        st.rerun()

# ── Main Content ───────────────────────────────────────────────
if not st.session_state.current_session_id:
    # ── Welcome Screen ─────────────────────────────────────────
    st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🤖</h1>
            <h1 style='margin-bottom: 0.5rem;'>RAG Chatbot</h1>
            <p style='font-size: 1.1rem; margin-bottom: 2.5rem; opacity: 0.7;'>
                Upload a PDF and start asking questions
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Maximum file size: 500 MB"
        )
        
        if uploaded_file:
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size > 500 * 1024 * 1024:
                st.error(f"❌ File too large: {file_size_mb:.1f} MB. Maximum: 500 MB")
            else:
                st.info(f"📄 **{uploaded_file.name}** • {file_size_mb:.1f} MB")
                
                if st.button("🚀 Upload & Start Chatting", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {file_size_mb:.1f} MB..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/upload",
                                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                                timeout=300
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.current_session_id = data["session_id"]
                                st.session_state.current_session_name = uploaded_file.name.replace(".pdf", "")
                                st.session_state.current_pdf = uploaded_file.name
                                st.session_state.messages = []
                                st.success(f"✅ Ready! {data['chunks']} chunks indexed")
                                st.rerun()
                            else:
                                st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
        
        st.markdown("""
            <div style='text-align: center; margin-top: 2.5rem; opacity: 0.5; font-size: 0.9rem;'>
                <p>💬 Chat with multiple PDFs • Resume conversations anytime • Rename & organize</p>
            </div>
        """, unsafe_allow_html=True)

else:
    # ── Chat Screen ─────────────────────────────────────────────
    # Header row
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        if st.session_state.editing_name:
            new_name = st.text_input(
                "Session name",
                value=st.session_state.current_session_name,
                key="edit_name_input",
                label_visibility="collapsed"
            )
            if st.button("✓ Save", key="save_header_name"):
                if rename_session(st.session_state.current_session_id, new_name):
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
        if st.button("✏️ Rename", key="header_rename"):
            st.session_state.editing_name = True
            st.rerun()
    
    with col3:
        if st.button("➕ New", key="header_new"):
            st.session_state.current_session_id = None
            st.session_state.current_session_name = None
            st.session_state.current_pdf = None
            st.session_state.messages = []
            st.session_state.editing_name = False
            st.rerun()
    
    st.divider()
    
    # Chat messages
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
                <div style='text-align: center; padding: 4rem 1rem; opacity: 0.4;'>
                    <p style='font-size: 1.3rem; margin-bottom: 0.5rem;'>💬</p>
                    <p style='font-size: 1.1rem; font-weight: 500;'>Start the conversation</p>
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
                    chat_history.append([msgs[i]["content"], msgs[i + 1]["content"]])
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/chat",
                            json={
                                "question": question,
                                "session_id": st.session_state.current_session_id,
                                "chat_history": chat_history
                            },
                            timeout=60
                        )
                        if response.status_code == 200:
                            answer = response.json()["answer"]
                            st.write(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            error = response.json().get("detail", "Unknown error")
                            st.error(f"❌ {error}")
                            st.session_state.messages.pop()
                    except Exception as e:
                        st.error(f"❌ Connection error: {e}")
                        st.session_state.messages.pop()
