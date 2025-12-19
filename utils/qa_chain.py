import os
import pickle
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAIError

from utils.embeddings import get_embeddings
from utils.llm import get_llm


VECTORSTORE_PATH = "vectorstore/faiss_index.pkl"


def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are a research assistant.
Answer the question using ONLY the context below.
Use bullet points where helpful.
If the answer is not present, say you do not know.

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
            # ⏱ Prevent burst RPM (Streamlit reruns)
            time.sleep(2)

            response = llm.invoke(prompt_text)

            answer_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

        except ChatGoogleGenerativeAIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                answer_text = (
                    "⏳ Gemini API rate limit reached.\n\n"
                    "Please wait about **1 minute** and try again."
                )
            else:
                raise e

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
