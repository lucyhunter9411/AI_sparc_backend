
# api/deps.py
from fastapi import Depends
from app.core.database import mongo_db
from app.core.vector_store import faiss_text_db

async def get_db():
    async with mongo_db() as db:  # Ensure you're using async with
        yield db  # Yield the database connection


async def get_vectorstore():
    async with faiss_text_db as store:  # Ensure you're using async with
        yield store  # Yield the vector-store connection