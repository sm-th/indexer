"""Client-side embeddings through any OpenAI-compatible `/embeddings` endpoint
(OpenAI, Azure OpenAI, Ollama, vLLM, LiteLLM, TEI, ...)."""

from __future__ import annotations

import json
import time
from typing import Protocol

import httpx


class Embedder(Protocol):
    @property
    def fingerprint(self) -> str:
        """Stable identity of the vector space (model + options)."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingError(RuntimeError):
    pass


class OpenAIEmbeddings:
    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
        batch_size: int = 64,
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(120.0, connect=15.0),
            transport=transport,
        )

    @property
    def fingerprint(self) -> str:
        return json.dumps({"model": self.model, "dimensions": self.dimensions}, sort_keys=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body: dict = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._http.post("/embeddings", json=body)
            except httpx.TransportError as exc:
                if attempt == self.max_attempts:
                    raise EmbeddingError(f"embedding request failed: {type(exc).__name__}") from None
            else:
                if resp.status_code == 200:
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    if len(data) != len(texts):
                        raise EmbeddingError(f"embedding API returned {len(data)} vectors for {len(texts)} inputs")
                    return [d["embedding"] for d in data]
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable or attempt == self.max_attempts:
                    # The body may echo request details but never our key; keep it short.
                    raise EmbeddingError(f"embedding API returned HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")
