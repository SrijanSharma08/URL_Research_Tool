import os
import pickle
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.embeddings import get_embeddings
from utils.llm import get_llm


VECTORSTORE_PATH = "vectorstore/faiss_index.pkl"


def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # lower token pressure
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)


def clear_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        os.remove(VECTORSTORE_PATH)


def load_qa_chain():
    if not os.path.exists(VECTORSTORE_PATH):
        return None

    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    # ⬇ keep k small for free tier
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are a research assistant.

Answer the question using ONLY the context below.
If the answer is not present, say "I do not know."

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    )

    # ⏱ local cooldown to prevent Streamlit rerun spam
    last_call_time = {"t": 0}

    def qa_chain(question: str):
        now = time.time()

        # 🚦 hard throttle (minimum 4 seconds between calls)
        if now - last_call_time["t"] < 4:
            return {
                "answer": (
                    "⏳ Please wait a few seconds before asking another question.\n\n"
                    "This helps stay within Gemini Free Tier limits."
                ),
                "sources": []
            }

        last_call_time["t"] = now

        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # ✅ Gemini-safe: render prompt to STRING
        prompt_text = prompt.format(
            context=context,
            question=question
        )

        try:
            # extra buffer for RPM safety
            time.sleep(2)

            response = llm.invoke(prompt_text)

            answer_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

        except Exception as e:
            msg = str(e).lower()

            # 🔴 Rate limit / quota exhausted
            if "429" in msg or "resource_exhausted" in msg or "quota" in msg:
                return {
                    "answer": (
                        "⚠️ **Gemini Free Tier rate limit reached.**\n\n"
                        "Please wait **30–60 seconds** and try again.\n"
                        "This is expected behavior on the free tier."
                    ),
                    "sources": []
                }

            # 🟠 Model / request errors
            if "404" in msg or "400" in msg or "not found" in msg:
                return {
                    "answer": (
                        "⚠️ **Model temporarily unavailable.**\n\n"
                        "Please retry in a moment."
                    ),
                    "sources": []
                }

            # 🔥 Catch-all safety net
            return {
                "answer": (
                    "❌ An unexpected error occurred while contacting the model.\n\n"
                    "Please try again later."
                ),
                "sources": []
            }

        sources = list(
            dict.fromkeys(
                d.metadata.get("source")
                for d in docs
                if d.metadata.get("source")
            )
        )

        return {
            "answer": answer_text,
            "sources": sources
        }

    return qa_chain
