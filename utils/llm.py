import os

# Try loading API key from local config (for local dev)
try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY,
        convert_system_message_to_human=True
    )
