"""
RAG Utility Script
- Connects to ChromaDB
- Uses a retriever and Gemini model to answer questions based on context

Usage:
from RAG import run_rag
response = run_rag(question, chroma_db_dir, api_key)
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    """Format documents for context injection."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag(
    question,
    chroma_db_dir = "./RAG/chroma_db_",
    api_key = None,
    embedding_model = None,
    k = 5
):
    """
    Run Retrieval-Augmented Generation (RAG) to answer a question using ChromaDB and Gemini.
    Args:
        question (str): The user question.
        chroma_db_dir (str): Directory where ChromaDB is persisted.
        api_key (str, optional): Google API key. If None, uses env var GOOGLE_API_KEY.
        embedding_model (optional): Embedding model to use for Chroma. If None, must be loaded in Chroma.
        k (int): Number of top documents to retrieve.
    Returns:
        str: The generated answer.
    """
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDlVCKsmkbHbQHl49zHkzbBbQ7iTRmdBSM')
    chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash")
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    db_connection = Chroma(persist_directory=chroma_db_dir, embedding_function=embedding_model)
    retriever = db_connection.as_retriever(search_kwargs={"k": k})

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Helpful AI Bot.\nGiven a context and question from user,\nyou should answer based on the given context."""),
        HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.\nContext: {context}\nQuestion: {question}\nAnswer: """)
    ])
    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )
    response = rag_chain.invoke(question)
    return response

if __name__ == "__main__":
    chroma_db_dir = "./RAG/chroma_db_"
    if os.path.exists(chroma_db_dir) and os.path.isdir(chroma_db_dir):
        question = "Please summarize Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
        answer = run_rag(question, chroma_db_dir=chroma_db_dir)
        print(answer)
    else:
        print(f"Chroma DB directory '{chroma_db_dir}' not found. Please provide a reference PDF to build the vector database first.")