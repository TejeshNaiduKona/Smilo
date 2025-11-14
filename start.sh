#!/bin/bash
# Run Streamlit on port 8501 and FastAPI on port 8000
streamlit run streamlit_app.py &  # runs Streamlit in background
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
 
