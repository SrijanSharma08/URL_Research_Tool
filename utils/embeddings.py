import os

try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
