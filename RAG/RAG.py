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
    retriever,
    api_key = None,
    k = 5,
):
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDlVCKsmkbHbQHl49zHkzbBbQ7iTRmdBSM')
    chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash")

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
