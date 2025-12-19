import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    api_key = None

    # Streamlit Cloud
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    # Local fallback
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )
