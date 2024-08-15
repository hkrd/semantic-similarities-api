"""Script to refresh embeddings.json with new data from sentences.txt."""

import asyncio
import logging
from pathlib import Path
from typing import Final

from embedding_server.gibson.database import AsyncEmbeddingDatabase
from embedding_server.gibson.embedding import AsyncEmbeddingService

logger = logging.getLogger(__name__)

root_path: Final[Path] = Path(__file__).parent.parent

embedding_database_file = root_path / "src" / "data" / "embeddings.json"
Path.unlink(embedding_database_file, missing_ok=True)
logger.info(
    "Existing embeddings.json file removed.",
    extra={"file_path": str(embedding_database_file)},
)


async def _process_data(
    text: str,
    embedding_service: AsyncEmbeddingService,
    database: AsyncEmbeddingDatabase,
) -> None:
    """Embed a single text string and insert it into the database.

    Args:
        text: The text to be embedded.
        embedding_service: The embedding service to use for embedding the text.
        database: The database to insert the embedded text into.
    """
    logger.debug(f"Processing text: {text}")
    embedding = await embedding_service.embed(text=text)
    await database.insert(text=text, embeddings=embedding)
    logger.debug("Text inserted into database.", extra={"text": text})


async def main() -> None:
    """Refresh the cached embedding database with new data from `data/sentences.txt`."""
    logger.info("Starting to refresh the embedding database from sentences.txt.")
    embedding_service = AsyncEmbeddingService()
    logger.debug("Embedding service setup completed.")

    database = AsyncEmbeddingDatabase(cache_path=embedding_database_file)
    await database.setup()
    logger.debug("Database setup completed.")

    with Path.open(root_path / "data" / "sentences.txt") as file:
        data = file.read().splitlines()
    logger.info("Data loaded from sentences.txt.", extra={"data_length": len(data)})

    for text in data:
        await _process_data(text, embedding_service, database)

    logger.info("All data processed and inserted into database.")


if __name__ == "__main__":
    asyncio.run(main())
