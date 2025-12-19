import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in Streamlit secrets")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # Most stable for Free Tier
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=1024,
        max_retries=6,              # 🔁 Exponential backoff
        timeout=60                  # ⏳ Wait instead of crashing
    )
