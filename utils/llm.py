from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )
