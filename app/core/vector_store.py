
# core/vector_store.py
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import get_settings
import os

settings = get_settings()
BLOB_STORAGE_TEXT_FAISS_DIR = os.getenv("BLOB_STORAGE_TEXT_FAISS_DIR")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@asynccontextmanager    
async def faiss_text_db():
    """
    Contextmanager to load and provide the FAISS vector store.
    """
    store = FAISS.load_local(
        BLOB_STORAGE_TEXT_FAISS_DIR,
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        allow_dangerous_deserialization=True
    )
    try:
        yield store
    finally:
        # No explicit teardown needed for FAISS
        pass
