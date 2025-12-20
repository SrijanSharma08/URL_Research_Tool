import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in Streamlit secrets")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # ✅ BEST FREE-TIER MODEL
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=512,
        max_retries=5,                 # handles brief throttling
        timeout=60,
        convert_system_message_to_human=True
    )
