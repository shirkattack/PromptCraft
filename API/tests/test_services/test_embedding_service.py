"""Tests for embeddings, coverage selection and de-duplication."""

import httpx
import pytest

from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingUnavailable,
    cosine,
    coverage_selection,
    farthest_point_selection,
    filter_near_duplicates,
)
from tests.conftest import fake_embed


def _service(handler) -> EmbeddingService:
    service = EmbeddingService(base_url="http://ollama.test", model="nomic-embed-text")
    service._client = httpx.Client(transport=httpx.MockTransport(handler))
    return service


class TestEmbeddingService:
    def test_batches_and_caches(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = request.read().decode()
            calls.append(payload)
            import json

            texts = json.loads(payload)["input"]
            return httpx.Response(
                200, json={"embeddings": [[float(len(t)), 1.0] for t in texts]}
            )

        service = _service(handler)
        first = service.embed(["ab", "abc", "ab"])
        assert first == [[2.0, 1.0], [3.0, 1.0], [2.0, 1.0]]
        service.embed(["ab", "abcd"])
        assert len(calls) == 2
        assert '"abcd"' in calls[1] and '"ab"' not in calls[1].replace('"abcd"', "")

    def test_missing_model_is_unavailable_with_install_hint(self):
        service = _service(
            lambda r: httpx.Response(404, json={"error": "model not found"})
        )
        with pytest.raises(EmbeddingUnavailable, match="ollama pull"):
            service.embed(["x"])

    def test_connection_error_is_unavailable(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(EmbeddingUnavailable, match="Cannot reach"):
            _service(handler).embed(["x"])


class TestSelection:
    def test_cosine(self):
        assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine([0, 0], [1, 1]) == 0.0

    def test_farthest_point_spreads_and_starts_typical(self):
        vectors = [[1, 0], [0.9, 0.1], [0, 1], [-1, 0], [0.95, 0.05]]
        chosen = farthest_point_selection(vectors, 3)
        assert len(chosen) == 3 and len(set(chosen)) == 3
        # The three near-identical right-pointing vectors should not all be picked.
        assert not {0, 1, 4} <= set(chosen)
        assert 3 in chosen  # the lone left-pointing vector is maximally far

    def test_labels_are_covered_first(self):
        vectors = [[1, 0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]]
        labels = ["a", "a", "a", "b"]
        chosen = farthest_point_selection(vectors, 2, labels)
        assert 3 in chosen  # the only "b" is taken before a second "a"

    def test_k_at_least_n_returns_all(self):
        assert farthest_point_selection([[1, 0], [0, 1]], 5) == [0, 1]
        assert farthest_point_selection([], 3) == []

    def test_coverage_selection_reports_neighbours(self):
        pool = ["server down outage", "thanks great help", "billing invoice question"]
        population = pool + [
            "database outage tonight",
            "invoice missing",
            "thank you team",
        ]
        chosen, covers = coverage_selection(pool, population, 2, embed=fake_embed)
        assert len(chosen) == 2
        for index in chosen:
            assert pool[index] not in covers[index]
            assert 1 <= len(covers[index]) <= 3


class TestDeduplication:
    def test_rejects_near_duplicates_of_existing_and_of_each_other(self):
        new = [
            "server is down",
            "server is down",
            "totally different topic",
            "SERVER IS DOWN",
        ]
        kept, rejected = filter_near_duplicates(
            new, ["the server is down"], threshold=0.9, embed=fake_embed
        )
        assert 2 in kept
        assert len(kept) + len(rejected) == len(new)
        assert all({"input", "similar_to", "similarity"} <= r.keys() for r in rejected)
        assert any(r["similar_to"] == "the server is down" for r in rejected)

    def test_threshold_one_keeps_everything_distinct(self):
        kept, rejected = filter_near_duplicates(
            ["a b", "c d"], [], threshold=1.01, embed=fake_embed
        )
        assert kept == [0, 1] and rejected == []

    def test_empty(self):
        assert filter_near_duplicates([], ["x"], 0.9, embed=fake_embed) == ([], [])
