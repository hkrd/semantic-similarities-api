"""Tests FastAPI endpoints."""

import json
from http import HTTPStatus
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from embedding_server.gibson.embedding import AsyncEmbeddingService
from embedding_server.gibson.exceptions import FlakyNetworkException
from embedding_server.server import app
from embedding_server.utils import get_embedding


@pytest.fixture
def client() -> TestClient: # type: ignore
    """Creates TestClient."""
    with TestClient(app) as client:
        yield client


def test_ready(client: TestClient) -> None:
    """Tests /ready endpoint."""
    response = client.get("/ready")
    assert response.status_code == HTTPStatus.OK


def test_insert(client: TestClient) -> None:
    """Tests /insert endpoint."""
    response = client.post(
        "/insert",
        json={
            "text": "This is test",
            "test_db": "unit_test",
        },
    )
    assert response.status_code == HTTPStatus.OK


def test_search_find_expected(client: TestClient) -> None:
    """Tests /similarity endpoint returns expected."""
    expected = "Spam and eggs is a delicious breakfast."
    res1 = client.post(
        "/similarity",
        json={
            "text": expected,
            "test_db": "unit_test",
        },
    )
    body = json.loads(res1.content)
    assert res1.status_code == HTTPStatus.OK
    assert body[0] == expected


def test_search_no_match(client: TestClient) -> None:
    """Tests /similarity endpoint returns different."""
    res2 = client.post(
        "/similarity",
        json={
            "text": "Spam and eggs is a delicious breakfast.",
            "test_db": "unit_test",
        },
    )

    body = json.loads(res2.content)
    assert res2.status_code == HTTPStatus.OK
    assert body[0] != "some text"


@pytest.mark.asyncio
async def test_search_exception(mocker: MockerFixture, client: TestClient) -> None:
    """Tests /similarity endpoint raises exception."""
    expected = "Spam and eggs is a delicious breakfast."
    emb = await get_embedding(AsyncEmbeddingService(), expected)

    async_mock = mocker.patch(
            "embedding_server.gibson.embedding.AsyncEmbeddingService.embed",
            new_callable=AsyncMock,
        )

    async_mock.side_effect = [emb] + [FlakyNetworkException()] * 6

    response = client.post(
            "/similarity",
            json={
                "text": expected,
                "test_db": "unit_test",
            },
        )
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_search_retried(mocker: MockerFixture, client: TestClient) -> None:
    """Tests /similarity endpoint raises exception."""
    expected = "Spam and eggs is a delicious breakfast."
    emb = await AsyncEmbeddingService().embed(expected)

    async_mock = mocker.patch(
            "embedding_server.gibson.embedding.AsyncEmbeddingService.embed",
            new_callable=AsyncMock,
        )

    async_mock.side_effect = [emb, FlakyNetworkException(), emb]
    res3 = client.post(
        "/similarity",
        json={
            "text": expected,
            "test_db": "unit_test",
        },
    )
    body = json.loads(res3.content)
    assert res3.status_code == HTTPStatus.OK
    assert body[0] == expected
