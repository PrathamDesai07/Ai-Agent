from vector_db import process_pdf_to_vector_db
from RAG import run_rag

process_pdf_to_vector_db(
    pdf_path="./document/Attention is all you need.pdf",
    chroma_db_dir="./RAG/chroma_db_"
)
q = "what is attention?"
o = run_rag(q)
print(o)