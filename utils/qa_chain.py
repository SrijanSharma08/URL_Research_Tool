import os
import pickle

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
Use ONLY the context below to answer the question.
Use bullet points where helpful, avoid overuse.
If the answer is not present, say you do not know.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    )

    def qa_chain(question: str):
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": list(set(d.metadata.get("source") for d in docs if "source" in d.metadata))
        }

    return qa_chain
