"""Text embeddings through Ollama, and the sampling helpers built on them.

Two uses:

* **Coverage-based example selection.** BootstrapFewShot validates which
  training samples the model can reproduce; from that pool we keep the few
  that best span the input space (farthest-point sampling over embeddings),
  so the demos in the returned prompt are representative rather than just
  the first ones that passed. Each chosen example is annotated with the
  inputs it stands in for.
* **Near-duplicate rejection** for synthetic data generation.

Everything degrades: if the embedding model is not installed or Ollama is
down, callers catch ``EmbeddingUnavailable`` and fall back to the plain
behaviour, noting why in the report.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("embeddings")

Vector = list[float]
EmbedFn = Callable[[list[str]], list[Vector]]


class EmbeddingUnavailable(Exception):
    """The embedding model could not be used; features should fall back."""


class EmbeddingService:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self._client = httpx.Client(timeout=timeout)
        self._cache: dict[str, Vector] = {}

    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed texts (cached per text). Raises EmbeddingUnavailable on failure."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            try:
                response = self._client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": missing},
                )
            except httpx.HTTPError as exc:
                raise EmbeddingUnavailable(
                    f"Cannot reach Ollama for embeddings: {exc}"
                ) from exc
            if response.status_code != 200:
                detail = response.text[:200]
                raise EmbeddingUnavailable(
                    f"Embedding model '{self.model}' unavailable "
                    f"(HTTP {response.status_code}: {detail}). "
                    f"Install it with: ollama pull {self.model}"
                )
            vectors = response.json().get("embeddings") or []
            if len(vectors) != len(missing):
                raise EmbeddingUnavailable(
                    f"Expected {len(missing)} embeddings, got {len(vectors)}"
                )
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = [float(x) for x in vector]
            if len(self._cache) > 5000:
                for key in list(self._cache)[:1000]:
                    del self._cache[key]
        return [self._cache[t] for t in texts]

    def close(self) -> None:
        self._client.close()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def centroid(vectors: Sequence[Vector]) -> Vector:
    dims = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dims)]


def farthest_point_selection(
    vectors: Sequence[Vector],
    k: int,
    labels: Sequence[str] | None = None,
) -> list[int]:
    """Pick ``k`` indices that spread over the space.

    Starts from the item nearest the centroid (the most typical one), then
    repeatedly adds the item farthest from everything chosen so far. With
    ``labels``, items whose label is not yet represented are preferred until
    every label has one, so a label dataset's examples show each class.
    """
    n = len(vectors)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    center = centroid(vectors)
    first = max(range(n), key=lambda i: cosine(vectors[i], center))
    chosen = [first]
    # Cosine distance to the nearest chosen item, per candidate.
    min_dist = [1.0 - cosine(vectors[i], vectors[first]) for i in range(n)]
    seen_labels = {labels[first]} if labels else set()

    while len(chosen) < k:
        remaining = [i for i in range(n) if i not in chosen]
        if labels:
            uncovered = [i for i in remaining if labels[i] not in seen_labels]
            if uncovered:
                remaining = uncovered
        nxt = max(remaining, key=lambda i: min_dist[i])
        chosen.append(nxt)
        if labels:
            seen_labels.add(labels[nxt])
        for i in range(n):
            min_dist[i] = min(min_dist[i], 1.0 - cosine(vectors[i], vectors[nxt]))
    return chosen


def coverage_selection(
    pool_texts: list[str],
    population_texts: list[str],
    k: int,
    labels: Sequence[str] | None = None,
    embed: EmbedFn | None = None,
    neighbours: int = 3,
) -> tuple[list[int], dict[int, list[str]]]:
    """Choose ``k`` of ``pool_texts`` that best cover ``population_texts``.

    Returns the chosen pool indices and, per chosen index, the population
    texts it stands closest to (excluding itself), for the UI's
    "covers inputs like" note.
    """
    embed = embed or embedding_service.embed
    if not pool_texts:
        return [], {}
    vectors = embed(pool_texts + population_texts)
    pool_vectors = vectors[: len(pool_texts)]
    population_vectors = vectors[len(pool_texts) :]

    chosen = farthest_point_selection(pool_vectors, k, labels)

    covers: dict[int, list[str]] = {}
    for index in chosen:
        ranked = sorted(
            (
                (cosine(pool_vectors[index], population_vectors[j]), text)
                for j, text in enumerate(population_texts)
                if text != pool_texts[index]
            ),
            key=lambda pair: -pair[0],
        )
        covers[index] = [text for _, text in ranked[:neighbours]]
    return chosen, covers


def filter_near_duplicates(
    new_texts: list[str],
    existing_texts: list[str],
    threshold: float,
    embed: EmbedFn | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Indices of ``new_texts`` to keep, and the rejected ones with what they matched.

    A new text is rejected when its cosine similarity to any existing text,
    or to an earlier kept new text, reaches ``threshold``.
    """
    embed = embed or embedding_service.embed
    if not new_texts:
        return [], []
    vectors = embed(new_texts + existing_texts)
    new_vectors = vectors[: len(new_texts)]
    kept_vectors: list[tuple[str, Vector]] = list(
        zip(existing_texts, vectors[len(new_texts) :], strict=True)
    )

    kept: list[int] = []
    rejected: list[dict[str, Any]] = []
    for index, (text, vector) in enumerate(zip(new_texts, new_vectors, strict=True)):
        best_text, best_sim = None, -1.0
        for other_text, other_vector in kept_vectors:
            sim = cosine(vector, other_vector)
            if sim > best_sim:
                best_text, best_sim = other_text, sim
        if best_text is not None and best_sim >= threshold:
            rejected.append(
                {
                    "input": text,
                    "similar_to": best_text,
                    "similarity": round(best_sim, 3),
                }
            )
            continue
        kept.append(index)
        kept_vectors.append((text, vector))
    return kept, rejected


embedding_service = EmbeddingService()
