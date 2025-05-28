# api/deps.py
from fastapi import Depends
from fastapi.websockets import WebSocket

from app.core.database import mongo_db
from app.core.vector_store import faiss_text_db
from app.websockets.connection_manager import ConnectionManager


async def get_db():
    """Yield an async MongoDB session."""
    async with mongo_db() as db:             # invoke the context-manager
        yield db


async def get_vectorstore():
    """Yield the FAISS vector-store instance."""
    async with faiss_text_db() as store:     # invoke the async context-manager
        yield store

def get_conn_mgr(ws: WebSocket) -> ConnectionManager:
    return ws.app.state.conn_mgr
