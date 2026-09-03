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


class TestTolerantSyntheticParsing:
    """Small local models drift from the requested JSON layout in many ways."""

    def _parse(self, text):
        from app.services.training_service import TrainingDataService

        return TrainingDataService()._parse_synthetic_response(text, "general")

    def test_fenced_json_with_sentencepiece_indentation(self):
        # gemma3n through Ollama: code fence plus U+2581 in place of spaces.
        text = '```json\n[\n▁▁{\n▁▁▁▁"input": "My▁website▁is▁down",\n▁▁▁▁"output": "high"\n▁▁}\n]\n```'
        samples = self._parse(text)
        assert len(samples) == 1
        assert samples[0].input_text == "My website is down"
        assert samples[0].expected_output == "high"

    def test_object_wrapping_the_list(self):
        samples = self._parse(
            '{"examples": [{"input": "a", "output": "b"}, {"input": "c", "output": "d"}]}'
        )
        assert [s.input_text for s in samples] == ["a", "c"]

    def test_renamed_keys(self):
        samples = self._parse(
            '[{"prompt": "q1", "response": "r1"}, {"question": "q2", "answer": "r2"}, {"text": "q3", "label": "low"}]'
        )
        assert [(s.input_text, s.expected_output) for s in samples] == [
            ("q1", "r1"),
            ("q2", "r2"),
            ("q3", "low"),
        ]

    def test_json_lines(self):
        samples = self._parse(
            '{"input": "a", "output": "b"}\n{"input": "c", "output": "d"}\n'
        )
        assert len(samples) == 2

    def test_plain_text_pairs_as_last_resort(self):
        text = "Here are examples:\n\n1. Input: Server is down\n   Output: high\n\n2. Input: Thanks!\n   Output: low\n"
        samples = self._parse(text)
        assert [(s.input_text, s.expected_output) for s in samples] == [
            ("Server is down", "high"),
            ("Thanks!", "low"),
        ]

    def test_leftover_fields_become_extra_data(self):
        samples = self._parse('[{"input": "a", "output": "b", "difficulty": "hard"}]')
        assert samples[0].extra_data == {"difficulty": "hard"}

    def test_nothing_usable_raises(self):
        from app.core.exceptions import SyntheticDataGenerationError

        with pytest.raises(SyntheticDataGenerationError):
            self._parse("I cannot help with that request.")

    @pytest.mark.asyncio
    async def test_repair_retry_rescues_prose(self):
        from unittest.mock import patch

        from app.schemas.training import SyntheticDataRequest
        from app.services.training_service import TrainingDataService

        calls = []

        def lm(prompt, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                return ["Sure! Example one: a ticket about an outage is high priority."]
            return ['[{"input": "a ticket about an outage", "output": "high"}]']

        with patch("app.services.lm_manager.LMManager.get_lm", return_value=lm):
            samples = await TrainingDataService().generate_synthetic_data(
                SyntheticDataRequest(
                    dataset_id="d",
                    sample_count=3,
                    base_prompt="Classify tickets",
                    task_type="classification",
                )
            )

        assert len(calls) == 2
        assert "JSON array" in calls[1]
        assert samples[0].expected_output == "high"


def test_normalize_strips_sentencepiece_marker():
    from app.services.eval_service import normalize

    assert normalize("▁high") == "high"
    assert normalize("high▁priority") == "high priority"


class TestFlexibleJsonImport:
    def _import(self, data, fmt="json"):
        return TrainingDataService().parse_import_data(
            DatasetImportRequest(name="n", task_type="t", file_format=fmt, data=data)
        )

    def test_jsonl_and_aliases(self):
        samples = self._import(
            '{"prompt": "a", "response": "b"}\n{"question": "c", "answer": "d"}\n',
            fmt="jsonl",
        )
        assert [(s.input_text, s.expected_output) for s in samples] == [
            ("a", "b"),
            ("c", "d"),
        ]

    def test_jsonl_is_detected_even_when_format_says_json(self):
        samples = self._import(
            '{"input": "a", "output": "b"}\n{"input": "c", "output": "d"}'
        )
        assert len(samples) == 2

    def test_wrapped_list_and_extra_fields(self):
        samples = self._import(
            '{"data": [{"input_text": "a", "expected_output": "b", "source": "manual"}]}'
        )
        assert samples[0].input_text == "a" and samples[0].extra_data == {
            "source": "manual"
        }

    def test_bad_jsonl_line_is_reported(self):
        with pytest.raises(TrainingDataError) as exc_info:
            self._import('{"input": "a", "output": "b"}\nnot json\n', fmt="jsonl")
        assert exc_info.value.details["line"] == 2
