import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from utils.embeddings import get_embeddings
from utils.llm import get_llm

VECTORSTORE_DIR = "vectorstore"


def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)


def clear_vectorstore():
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)


def load_qa_chain():
    if not os.path.exists(VECTORSTORE_DIR):
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the context below.

Guidelines:
- Be concise and accurate.
- Use bullet points where helpful and avoid overuse.
- Keep answers short unless explanation or comparison is required.
- Do not add information not present in the context.

<context>
{context}
</context>

Question: {question}
"""
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return chain, vectorstore
