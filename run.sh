#!/usr/bin/env bash
export PYTHONPATH=$(dirname "$0")/embedding_server/src
uvicorn embedding_server.server:app --reload
