import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()

# Get API keys from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Directory where ChromaDB will store vectors locally
CHROMA_DIR = "./chroma_store"

# Local embeddings — free, no API limits, runs offline
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LLM — Groq (fast, free tier available)
# Using llama-3.3-70b-versatile — powerful and fast
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
)


def ingest_pdf(file_path: str) -> int:
    """Load a PDF, split into chunks, embed and store in ChromaDB."""

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return len(chunks)


def ask_question(question: str, chat_history: List[Tuple[str, str]] = None) -> str:
    """
    Retrieve relevant chunks and answer using Groq.
    chat_history: List of (user_question, assistant_answer) tuples
    """

    # Load the existing vector store from disk
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # ConversationalRetrievalChain — remembers chat history
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        rephrase_question=False
    )

    # Invoke with question + past conversation
    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history or []
    })

    return result["answer"]
