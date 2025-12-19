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
        chunk_size=800,        # smaller chunks = fewer tokens
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are a research assistant.

Answer the question using ONLY the context below.
Use bullet points where helpful.
If the answer is not present in the context, say "I do not know."

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    )

    def qa_chain(question: str):
        docs = retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt_text = prompt.format(
            context=context,
            question=question
        )

        try:
            # ⏳ Prevent Free-Tier burst (15 RPM)
            time.sleep(4)

            response = llm.invoke(prompt_text)

            answer_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

        except Exception as e:
            msg = str(e)

            # 🔴 Rate-limit handling
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                return {
                    "answer": (
                        "⚠️ **Rate limit reached.**\n\n"
                        "Gemini Free Tier allows ~15 requests per minute.\n"
                        "Please wait **30–60 seconds** and try again."
                    ),
                    "sources": []
                }

            # 🟠 Model / request errors
            if "404" in msg or "400" in msg:
                return {
                    "answer": (
                        "⚠️ **Model request failed.**\n\n"
                        "This can happen due to temporary API issues or "
                        "invalid model availability.\n\n"
                        "Please retry shortly."
                    ),
                    "sources": []
                }

            # 🔥 Unknown errors
            return {
                "answer": "❌ An unexpected error occurred. Please try again later.",
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
