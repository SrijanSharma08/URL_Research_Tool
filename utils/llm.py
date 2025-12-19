import os

# Load API key from local config (dev) or environment (cloud)
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
        convert_system_message_to_human=True,
        safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUAL_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_SELF_HARM": "BLOCK_NONE",
        },
    )
