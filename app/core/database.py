# core/database.py
from contextlib import asynccontextmanager
from pymongo import MongoClient
from urllib.parse import quote_plus
from app.core.config import get_settings

settings = get_settings()

# Build the MongoDB URI once
MONGO_URI = (
    f"mongodb+srv://{quote_plus(settings.mongo_user)}:{quote_plus(settings.mongo_password)}@"
    f"{settings.mongo_host}/"
    f"{settings.mongo_db}?tls=true&authMechanism=SCRAM-SHA-256"
    "&retrywrites=false&maxIdleTimeMS=120000"
)

@asynccontextmanager
async def mongo_client():
    """
    Async contextmanager to provide a MongoClient on startup and close on shutdown.
    """
    client = MongoClient(MONGO_URI)
    try:
        yield client
    finally:
        client.close()

@asynccontextmanager
async def mongo_db():
    """
    Dependency: yields the configured database instance.
    """
    async with mongo_client() as client:
        yield client[settings.mongo_db]