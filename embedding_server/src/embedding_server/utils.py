"""Utility methods."""

import asyncio
import logging
import os

from embedding_server.gibson.embedding import AsyncEmbeddingService
from embedding_server.gibson.exceptions import FlakyNetworkException

logging.basicConfig(level=logging.DEBUG if "DEBUG" in os.environ else logging.INFO)
logger = logging.getLogger(__name__)

RETRIES = 5
DELAY = 0.5


async def get_embedding(es: AsyncEmbeddingService, text: str) -> list[float]:
    """Used to call embed and retry a defined number of times if FlakyNetworkExceptiomn occurs."""
    for attempt in range(RETRIES):
        try:
            return await es.embed(text)
        except FlakyNetworkException:
            logger.info("Caught FlakyNetworkException")
            if attempt < RETRIES - 1:
                await asyncio.sleep(DELAY)
            else:
                raise
    return []
