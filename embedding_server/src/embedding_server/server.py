"""Main entry point for FastAPI application."""

import logging
import os
from http import HTTPStatus
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedding_server.gibson.embedding import AsyncEmbeddingService
from embedding_server.gibson.exceptions import FlakyNetworkException
from embedding_server.search import SearchEmbeddingService
from embedding_server.utils import get_embedding

logging.basicConfig(level=logging.DEBUG if "DEBUG" in os.environ else logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(docs_url="/docs", redoc_url="/redoc")

db_path = Path(__file__).parent.parent / "data" / "embeddings.json"
db = SearchEmbeddingService(db_path)
es = AsyncEmbeddingService()


class EmbeddingRequest(BaseModel):
    """Represents an insert request."""
    text: str
    test_db: str | None = None


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize the services aynchronously on startup."""
    await db.setup()


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Returns a simple health check endpoint to indicate the application is ready."""
    return {"message": "Ready"}


@app.post("/insert")
async def insert_data(request: EmbeddingRequest) -> dict[str, str]:
    """Inserts the provided text and its embeddings into the database.

    Args:
        request: The insert request containing text to embed and store.

    Returns:
        A dictionary with a message indicating successful insertion.

    Raises:
        HTTPException: An error occurred during the embedding or insertion process.
    """
    logger.debug(
        "Received insert request",
        extra={"text": request.text, "test_db": request.test_db},
    )
    global db
    # If a test database is provided, use it instead of the default database.
    # This is used in tests.
    if request.test_db is not None:
        test_db_path = Path(__file__).parent / request.test_db
        Path.unlink(test_db_path, missing_ok=True)
        db = SearchEmbeddingService(test_db_path)
        await db.setup()
        logger.info("Test database setup complete")

    try:
        # Note, we don't handle FlakyNetworkException here yet
        embedding = await get_embedding(es, request.text)

        await db.insert(text=request.text, embeddings=embedding)
        logger.info("Data inserted successfully")
        return {"message": "Data inserted successfully"}
    except ValueError as error:
        logger.error(
            "Error inserting data",
            extra={"error": str(error)},
        )
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=str(error)
        ) from error
    except FlakyNetworkException as error:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(error)
        ) from error
    finally:
        if request.test_db is not None:
            Path.unlink(test_db_path, missing_ok=True)


@app.post("/similarity")
async def get_similarity_embedding(request: EmbeddingRequest) -> list[str]:
    """Endpoint to search for similar embeddings."""
    global db

    if request.test_db is not None:
        test_db_path = Path(__file__).parent / request.test_db
        Path.unlink(test_db_path, missing_ok=True)
        db = SearchEmbeddingService(test_db_path)
        await db.setup()
        logger.info("Test database setup complete")
        embedding = await get_embedding(es, request.text)
        await db.insert(text=request.text, embeddings=embedding)

    try:
        return await db.find_similar_embeddings(request.text)
    except FlakyNetworkException as error:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(error)
        ) from error
    finally:
        if request.test_db is not None:
            Path.unlink(test_db_path, missing_ok=True)


if __name__ == "__main__":
    """Helper to quickly run the FastAPI application for testing."""
    uvicorn.run(app)
