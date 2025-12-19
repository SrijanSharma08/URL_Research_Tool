import os
import pickle

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
{input}

Answer clearly and concisely.
"""
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    def qa_chain(question: str):
        result = chain.invoke({"input": question})

        sources = list(
            dict.fromkeys(
                doc.metadata.get("source")
                for doc in result["context"]
                if doc.metadata.get("source")
            )
        )

        return {
            "answer": result["answer"],
            "sources": sources
        }

    return qa_chain
