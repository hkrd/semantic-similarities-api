"""Search Service that implements embeddings search by similarity metric."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from embedding_server.gibson.database import AsyncEmbeddingDatabase
from embedding_server.gibson.embedding import AsyncEmbeddingService
from embedding_server.utils import get_embedding

logging.basicConfig(level=logging.DEBUG if "DEBUG" in os.environ else logging.INFO)
logger = logging.getLogger(__name__)


class SearchEmbeddingService(AsyncEmbeddingDatabase):
    """Provides functionality to search the embeddings database."""


    def __init__(self, cache_path: Path):
        """Initializes the Search Service.

        Args:
            cache_path: The path to the local cache.
        """
        super().__init__(cache_path)

    def _cosine_similarity(
        self, embedding1: list[float], embedding2: npt.NDArray[Any]
    ) -> float:
        return float(
            np.dot(embedding1, embedding2)
            / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        )


    async def find_similar_embeddings(
        self, query_text: str, top_k: int = 5
    ) -> list[str]:
        """Finds the most similar embeddings given a query and returns a list.

        Args:
            query_text: a string with the query to search similarities for.
            top_k: default number of items to return in descending order.

        Returns:
            A list of matching embeddings in descending order.

        Raises:
            FlakyNetworkException after 5 retries expire.
        """
        query_embedding = await get_embedding(AsyncEmbeddingService(), query_text)
        similarities = self.data["Embeddings"].apply(
            lambda emb, query=query_embedding: self._cosine_similarity(query, np.array(emb))
        )
        sorted_indices = similarities.argsort()[::-1][
            :top_k
        ]  # Sort in descending order
        res: pd.Series[str] = self.data.iloc[sorted_indices]["Text"]
        return res.astype(str).to_list()  # type: ignore
