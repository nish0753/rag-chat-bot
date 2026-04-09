#!/bin/bash

echo "Starting RAG Chatbot..."
echo ""

# Start FastAPI in background
echo "Starting FastAPI backend on port 8000..."
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!
sleep 2

# Start Streamlit
echo "Starting Streamlit frontend on port 8501..."
streamlit run app.py

# When Streamlit closes, kill FastAPI
kill $FASTAPI_PID 2>/dev/null
