"""Tests for the training data service."""

import pytest

from app.core.exceptions import SyntheticDataGenerationError, TrainingDataError
from app.schemas.training import DatasetImportRequest, TrainingSampleCreate
from app.services.training_service import TrainingDataService


@pytest.fixture
def service() -> TrainingDataService:
    return TrainingDataService()


def _import(data: str, file_format: str = "csv") -> DatasetImportRequest:
    return DatasetImportRequest(
        name="Test", task_type="general", file_format=file_format, data=data
    )


class TestCsvImport:
    def test_quoted_commas_stay_in_one_field(self, service: TrainingDataService):
        data = 'input,output\n"Translate: a, b, c","Traduire : a, b, c"\n'

        samples = service.parse_import_data(_import(data))

        assert len(samples) == 1
        assert samples[0].input_text == "Translate: a, b, c"
        assert samples[0].expected_output == "Traduire : a, b, c"

    def test_embedded_newline_is_preserved(self, service: TrainingDataService):
        data = 'input,output\nquestion,"first line\nsecond line"\n'

        samples = service.parse_import_data(_import(data))

        assert len(samples) == 1
        assert samples[0].expected_output == "first line\nsecond line"

    def test_escaped_quotes(self, service: TrainingDataService):
        data = 'input,output\n"He said ""hi""",greeting\n'

        samples = service.parse_import_data(_import(data))

        assert samples[0].input_text == 'He said "hi"'

    def test_headerless_csv_uses_first_two_columns(self, service: TrainingDataService):
        samples = service.parse_import_data(_import("a,b\nc,d\n"))

        assert [(s.input_text, s.expected_output) for s in samples] == [
            ("a", "b"),
            ("c", "d"),
        ]

    def test_quality_score_is_assigned(self, service: TrainingDataService):
        samples = service.parse_import_data(
            _import("input,output\nquestion,a real answer\n")
        )

        assert 0.0 < samples[0].quality_score <= 1.0

    def test_empty_import_is_rejected(self, service: TrainingDataService):
        with pytest.raises(TrainingDataError) as exc_info:
            service.parse_import_data(_import("input,output\n"))

        assert exc_info.value.error_code == "IMPORT_EMPTY"


class TestJsonImport:
    def test_parses_input_output_pairs(self, service: TrainingDataService):
        samples = service.parse_import_data(
            _import('[{"input": "a", "output": "b"}]', file_format="json")
        )

        assert len(samples) == 1
        assert samples[0].input_text == "a"

    def test_malformed_json_raises(self, service: TrainingDataService):
        with pytest.raises(TrainingDataError) as exc_info:
            service.parse_import_data(_import("{nope", file_format="json"))

        assert exc_info.value.error_code == "IMPORT_PARSE_FAILED"


class TestSyntheticResponseParsing:
    def test_extracts_array_from_surrounding_prose(self, service: TrainingDataService):
        response = (
            'Sure! Here you go:\n[{"input": "a", "output": "b"}]\nHope that helps.'
        )

        samples = service._parse_synthetic_response(response, "general")

        assert len(samples) == 1
        assert samples[0].quality_score == 0.8

    def test_string_extra_data_is_wrapped(self, service: TrainingDataService):
        response = '[{"input": "a", "output": "b", "extra_data": "a note"}]'

        samples = service._parse_synthetic_response(response, "general")

        assert samples[0].extra_data == {"notes": "a note"}

    def test_response_without_json_raises(self, service: TrainingDataService):
        with pytest.raises(SyntheticDataGenerationError) as exc_info:
            service._parse_synthetic_response("I cannot do that.", "general")

        assert exc_info.value.error_code == "SYNTHETIC_DATA_PARSE_FAILED"

    def test_array_without_usable_pairs_raises(self, service: TrainingDataService):
        with pytest.raises(SyntheticDataGenerationError) as exc_info:
            service._parse_synthetic_response('[{"foo": "bar"}]', "general")

        assert exc_info.value.error_code == "SYNTHETIC_DATA_EMPTY"


class TestQualityScoring:
    def test_score_stays_in_range(self, service: TrainingDataService):
        sample = TrainingSampleCreate(
            input_text="What is the capital of France?",
            expected_output="Paris is the capital of France.",
        )

        assert 0.0 <= service.validate_sample_quality(sample) <= 1.0

    def test_echoed_output_scores_lower(self, service: TrainingDataService):
        echoed = TrainingSampleCreate(
            input_text="same text here", expected_output="same text here"
        )
        distinct = TrainingSampleCreate(
            input_text="same text here", expected_output="a genuinely different answer"
        )

        assert service.validate_sample_quality(
            echoed
        ) < service.validate_sample_quality(distinct)
