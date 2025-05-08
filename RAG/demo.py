from vector_db import process_pdf_to_vector_db
from RAG import run_rag
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import sys

process_pdf_to_vector_db(
    pdf_path="./document/Attention is all you need.pdf",
    chroma_db_dir="./RAG/chroma_db_"
)

chroma_db_dir = "./RAG/chroma_db_"
api_key = None
k = 5


chroma_db_dir = "./RAG/chroma_db_"
if os.path.exists(chroma_db_dir) and os.path.isdir(chroma_db_dir):
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDlVCKsmkbHbQHl49zHkzbBbQ7iTRmdBSM')
    chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash")
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    db_connection = Chroma(persist_directory=chroma_db_dir, embedding_function=embedding_model)
    retriever = db_connection.as_retriever(search_kwargs={"k": k})
else:
    print(f"Chroma DB directory '{chroma_db_dir}' not found. Please provide a reference PDF to build the vector database first.")

# CLI loop for interactive Q&A, exit on Ctrl+Q
# print("Enter your question (press Ctrl+Q to quit):")
# while True:
#     try:
#         user_input = input("Q: ")
#         # Detect Ctrl+Q (ASCII 17)
#         if user_input and ord(user_input[0]) == 17:
#             print("Exiting...")
#             break
#         answer = run_rag(user_input, retriever=retriever)
#         print(f"A: {answer}")
#     except KeyboardInterrupt:
#         print("\nExiting...")
#         break