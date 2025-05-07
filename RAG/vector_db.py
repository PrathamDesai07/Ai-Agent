"""
Vector DB Utility Function
- Loads a PDF
- Splits it into chunks
- Embeds chunks
- Stores them in a persistent vector DB

Usage:
from vector_db import process_pdf_to_vector_db
process_pdf_to_vector_db(pdf_path, chroma_db_dir)
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import nltk

def process_pdf_to_vector_db(
    pdf_path,
    chroma_db_dir,
    api_key=None,
    chunk_size=500,
    chunk_overlap=100
):
    """
    Loads a PDF, splits it into chunks, embeds them, and stores in a persistent vector DB.
    Args:
        pdf_path (str): Path to the PDF file.
        chroma_db_dir (str): Directory to persist the Chroma DB.
        api_key (str, optional): Google API key for embeddings. If None, uses env var GOOGLE_API_KEY.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        db: The persisted Chroma vector DB instance.
    """
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDlVCKsmkbHbQHl49zHkzbBbQ7iTRmdBSM')
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    print(f"Loaded {len(pages)} pages from PDF: {pdf_path}")

    # 2. Split into chunks
    nltk.download('punkt', quiet=True)
    splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks. Example chunk type: {type(chunks[0])}")

    # 3. Embed and store
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=chroma_db_dir)
    db.persist()
    print(f"Vector DB persisted at {chroma_db_dir}")
    return db