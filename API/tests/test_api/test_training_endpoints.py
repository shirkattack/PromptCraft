"""Tests for the training dataset endpoints."""

import csv
import io

import pytest
from fastapi.testclient import TestClient

BASE = "/api/v1/training"


@pytest.fixture
def dataset_id(client: TestClient) -> str:
    response = client.post(
        f"{BASE}/",
        json={"name": "Fixture set", "description": "d", "task_type": "general"},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _add_samples(client: TestClient, dataset_id: str, count: int):
    return client.post(
        f"{BASE}/{dataset_id}/samples/bulk",
        json={
            "samples": [
                {"input_text": f"in {i}", "expected_output": f"out {i}"}
                for i in range(count)
            ]
        },
    )


class TestDatasetSampleCounts:
    """The dataset sample_count must match the rows actually stored."""

    def test_create_with_initial_samples(self, client: TestClient):
        response = client.post(
            f"{BASE}/",
            json={
                "name": "Seeded",
                "task_type": "general",
                "samples": [
                    {"input_text": "a", "expected_output": "b"},
                    {"input_text": "c", "expected_output": "d"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["sample_count"] == 2
        assert response.json()["size"] == "2 samples"

    def test_bulk_create_does_not_double_count(
        self, client: TestClient, dataset_id: str
    ):
        response = _add_samples(client, dataset_id, 3)
        assert response.status_code == 200, response.text
        assert response.json()["created_count"] == 3

        dataset = client.get(f"{BASE}/{dataset_id}").json()
        assert dataset["sample_count"] == 3

        samples = client.get(f"{BASE}/{dataset_id}/samples").json()
        assert len(samples) == 3

    def test_bulk_create_twice_accumulates_correctly(
        self, client: TestClient, dataset_id: str
    ):
        _add_samples(client, dataset_id, 2)
        _add_samples(client, dataset_id, 3)

        assert client.get(f"{BASE}/{dataset_id}").json()["sample_count"] == 5

    def test_single_create_then_delete(self, client: TestClient, dataset_id: str):
        created = client.post(
            f"{BASE}/{dataset_id}/samples",
            json={"input_text": "hello", "expected_output": "world"},
        )
        assert created.status_code == 200, created.text
        assert client.get(f"{BASE}/{dataset_id}").json()["sample_count"] == 1

        deleted = client.delete(f"{BASE}/{dataset_id}/samples/{created.json()['id']}")
        assert deleted.status_code == 200
        assert client.get(f"{BASE}/{dataset_id}").json()["sample_count"] == 0


class TestDatasetLookup:
    def test_missing_dataset_returns_404(self, client: TestClient):
        assert client.get(f"{BASE}/does-not-exist").status_code == 404

    def test_samples_of_missing_dataset_returns_404(self, client: TestClient):
        assert client.get(f"{BASE}/does-not-exist/samples").status_code == 404

    def test_samples_excluded_unless_requested(
        self, client: TestClient, dataset_id: str
    ):
        _add_samples(client, dataset_id, 2)

        assert client.get(f"{BASE}/{dataset_id}").json()["samples"] is None

        with_samples = client.get(f"{BASE}/{dataset_id}?include_samples=true").json()
        assert len(with_samples["samples"]) == 2

    def test_reading_a_dataset_does_not_delete_its_samples(
        self, client: TestClient, dataset_id: str
    ):
        _add_samples(client, dataset_id, 2)

        client.get(f"{BASE}/{dataset_id}")
        client.get(f"{BASE}/{dataset_id}")

        assert len(client.get(f"{BASE}/{dataset_id}/samples").json()) == 2


class TestPaginationGuards:
    def test_limit_is_capped(self, client: TestClient):
        assert client.get(f"{BASE}/?limit=100000").status_code == 422

    def test_negative_skip_rejected(self, client: TestClient):
        assert client.get(f"{BASE}/?skip=-1").status_code == 422


class TestImportExportRoundTrip:
    """Fields containing commas, quotes and newlines must survive a round trip."""

    TRICKY_INPUT = 'Summarize: "one, two", then stop'
    TRICKY_OUTPUT = "line one\nline two, with comma"

    def test_csv_export_is_properly_quoted(self, client: TestClient, dataset_id: str):
        client.post(
            f"{BASE}/{dataset_id}/samples",
            json={
                "input_text": self.TRICKY_INPUT,
                "expected_output": self.TRICKY_OUTPUT,
            },
        )

        exported = client.post(
            f"{BASE}/{dataset_id}/export",
            json={"dataset_id": dataset_id, "format": "csv"},
        )
        assert exported.status_code == 200, exported.text

        rows = list(csv.reader(io.StringIO(exported.json()["data"])))
        assert rows[0] == ["input", "output"]
        assert rows[1] == [self.TRICKY_INPUT, self.TRICKY_OUTPUT]

    def test_csv_import_handles_quoted_commas(self, client: TestClient):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["input", "output"])
        writer.writerow([self.TRICKY_INPUT, self.TRICKY_OUTPUT])

        imported = client.post(
            f"{BASE}/import",
            json={
                "name": "Imported",
                "task_type": "general",
                "file_format": "csv",
                "data": buffer.getvalue(),
            },
        )
        assert imported.status_code == 200, imported.text
        assert imported.json()["sample_count"] == 1

        samples = client.get(f"{BASE}/{imported.json()['id']}/samples").json()
        assert samples[0]["input_text"] == self.TRICKY_INPUT
        assert samples[0]["expected_output"] == self.TRICKY_OUTPUT

    def test_json_import(self, client: TestClient):
        imported = client.post(
            f"{BASE}/import",
            json={
                "name": "Imported JSON",
                "task_type": "general",
                "file_format": "json",
                "data": '[{"input": "a", "output": "b"}, {"input": "c", "output": "d"}]',
            },
        )
        assert imported.status_code == 200, imported.text
        assert imported.json()["sample_count"] == 2

    def test_malformed_import_is_reported(self, client: TestClient):
        response = client.post(
            f"{BASE}/import",
            json={
                "name": "Broken",
                "task_type": "general",
                "file_format": "json",
                "data": "{not json",
            },
        )
        assert response.status_code == 400
        assert response.json()["error_code"] == "IMPORT_PARSE_FAILED"

    def test_unsupported_export_format_rejected(
        self, client: TestClient, dataset_id: str
    ):
        response = client.post(
            f"{BASE}/{dataset_id}/export",
            json={"dataset_id": dataset_id, "format": "parquet"},
        )
        assert response.status_code == 422


class TestDatasetListing:
    def test_list_reports_average_quality(self, client: TestClient, dataset_id: str):
        for score in (0.2, 0.8):
            response = client.post(
                f"{BASE}/{dataset_id}/samples",
                json={
                    "input_text": "in",
                    "expected_output": "out",
                    "quality_score": score,
                },
            )
            assert response.status_code == 200, response.text

        rows = client.get(f"{BASE}/").json()
        row = next(r for r in rows if r["id"] == dataset_id)
        assert row["avg_quality_score"] == pytest.approx(0.5)

    def test_empty_dataset_has_no_average(self, client: TestClient, dataset_id: str):
        rows = client.get(f"{BASE}/").json()
        row = next(r for r in rows if r["id"] == dataset_id)
        assert row["avg_quality_score"] is None
        assert row["sample_count"] == 0

    def test_list_is_newest_first(self, client: TestClient):
        first = client.post(f"{BASE}/", json={"name": "a", "task_type": "t"}).json()
        second = client.post(f"{BASE}/", json={"name": "b", "task_type": "t"}).json()
        # Touching the first dataset moves it back to the top.
        _add_samples(client, first["id"], 1)

        ids = [r["id"] for r in client.get(f"{BASE}/").json()]
        assert ids.index(first["id"]) < ids.index(second["id"])


class TestTrainingStats:
    def test_empty_stats(self, client: TestClient):
        response = client.get(f"{BASE}/stats")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "dataset_count": 0,
            "sample_count": 0,
            "by_task_type": [],
            "recent_datasets": [],
        }

    def test_stats_count_real_rows(self, client: TestClient):
        code = client.post(f"{BASE}/", json={"name": "c", "task_type": "code"}).json()
        qa = client.post(f"{BASE}/", json={"name": "q", "task_type": "qa"}).json()
        client.post(f"{BASE}/", json={"name": "q2", "task_type": "qa"})
        _add_samples(client, code["id"], 3)
        _add_samples(client, qa["id"], 1)

        stats = client.get(f"{BASE}/stats", params={"recent_limit": 2}).json()

        assert stats["dataset_count"] == 3
        assert stats["sample_count"] == 4
        assert stats["by_task_type"][0] == {
            "task_type": "code",
            "dataset_count": 1,
            "sample_count": 3,
        }
        assert {t["task_type"] for t in stats["by_task_type"]} == {"code", "qa"}
        assert len(stats["recent_datasets"]) == 2
        assert "avg_quality_score" in stats["recent_datasets"][0]

    def test_stats_route_is_not_shadowed_by_dataset_lookup(self, client: TestClient):
        # "/stats" must not be treated as a dataset id.
        assert client.get(f"{BASE}/stats").status_code == 200
        assert client.get(f"{BASE}/definitely-missing").status_code == 404


class TestSyntheticDeduplication:
    def _generated(self, *inputs):
        from app.schemas.training import TrainingSampleCreate

        return [
            TrainingSampleCreate(input_text=i, expected_output="high") for i in inputs
        ]

    def test_near_duplicates_are_rejected_and_reported(
        self, client: TestClient, dataset_id: str
    ):
        from unittest.mock import AsyncMock, patch

        client.post(
            f"{BASE}/{dataset_id}/samples",
            json={"input_text": "the server is down", "expected_output": "high"},
        )
        generated = self._generated(
            "server is down", "billing question about invoices", "server is down"
        )
        with patch(
            "app.api.v1.endpoints.training.training_service.generate_synthetic_data",
            new=AsyncMock(return_value=generated),
        ):
            response = client.post(
                f"{BASE}/{dataset_id}/generate",
                json={
                    "dataset_id": dataset_id,
                    "sample_count": 3,
                    "base_prompt": "p",
                    "task_type": "t",
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["rejected_duplicates"] >= 1
        assert body["generated_count"] + body["rejected_duplicates"] == 3
        assert body["dedup_skipped_reason"] is None
        assert all(
            {"input", "similar_to", "similarity"} <= d.keys()
            for d in body["duplicates"]
        )
        assert (
            client.get(f"{BASE}/{dataset_id}").json()["sample_count"]
            == 1 + body["generated_count"]
        )

    def test_generation_still_works_without_embeddings(
        self, client: TestClient, dataset_id: str, monkeypatch
    ):
        from unittest.mock import AsyncMock, patch

        from app.api.v1.endpoints import training as module
        from app.services.embedding_service import EmbeddingUnavailable

        def boom(*args, **kwargs):
            raise EmbeddingUnavailable("no embedding model")

        monkeypatch.setattr(module, "filter_near_duplicates", boom)
        with patch(
            "app.api.v1.endpoints.training.training_service.generate_synthetic_data",
            new=AsyncMock(return_value=self._generated("a", "a")),
        ):
            body = client.post(
                f"{BASE}/{dataset_id}/generate",
                json={
                    "dataset_id": dataset_id,
                    "sample_count": 2,
                    "base_prompt": "p",
                    "task_type": "t",
                },
            ).json()
        assert body["generated_count"] == 2
        assert body["rejected_duplicates"] == 0
        assert "no embedding model" in body["dedup_skipped_reason"]


def test_jsonl_export_round_trips(client: TestClient, dataset_id: str):
    _add_samples(client, dataset_id, 2)
    exported = client.post(
        f"{BASE}/{dataset_id}/export",
        json={"dataset_id": dataset_id, "format": "jsonl"},
    ).json()
    lines = [line for line in exported["data"].splitlines() if line]
    assert len(lines) == 2
    reimported = client.post(
        f"{BASE}/import",
        json={
            "name": "again",
            "task_type": "t",
            "file_format": "jsonl",
            "data": exported["data"],
        },
    )
    assert reimported.status_code == 200, reimported.text
    assert reimported.json()["sample_count"] == 2
