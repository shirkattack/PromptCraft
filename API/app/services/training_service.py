import csv
import io
import json
import logging
import re
from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.core.exceptions import SyntheticDataGenerationError, TrainingDataError
from app.schemas.training import (
    DatasetImportRequest,
    SyntheticDataRequest,
    TrainingSampleCreate,
)
from app.services.lm_manager import LMManager
from app.services.text import clean_model_text, strip_code_fence

logger = logging.getLogger(__name__)


class TrainingDataService:
    """Service for training data management and synthetic data generation."""

    def __init__(self) -> None:
        self.synthetic_prompts = {
            "general": self._generate_general_prompt,
            "creative": self._generate_creative_prompt,
            "code": self._generate_code_prompt,
            "analysis": self._generate_analysis_prompt,
            "translation": self._generate_translation_prompt,
        }

    async def generate_synthetic_data(
        self, request: SyntheticDataRequest
    ) -> list[TrainingSampleCreate]:
        """
        Generate synthetic training data using Promptomatix techniques.
        Adapted from Promptomatix synthetic data generation.

        Raises:
            SyntheticDataGenerationError: if the provider call fails or the
                response cannot be turned into any usable samples.
        """
        try:
            # Get language model
            lm = LMManager.get_lm(
                provider=request.provider,
                model_name=request.model,
                temperature=request.creativity_level,
                max_tokens=2000,
            )

            # Generate synthetic data prompt
            prompt_generator = self.synthetic_prompts.get(
                request.task_type, self._generate_general_prompt
            )

            synthetic_prompt = prompt_generator(
                base_prompt=request.base_prompt,
                sample_count=request.sample_count,
                task_type=request.task_type,
            )

            # The model call is synchronous and can run for minutes against a
            # local model, so it must not block the event loop.
            response = await run_in_threadpool(lm, synthetic_prompt, max_tokens=2000)
            response_text = self._response_text(response)

            try:
                return self._parse_synthetic_response(response_text, request.task_type)
            except SyntheticDataGenerationError as first_error:
                # Small models often answer in prose or a broken layout. One
                # repair pass asks the same model to restate its reply as JSON.
                logger.info(
                    "Synthetic reply unparseable; asking the model to repair it"
                )
                repaired = await run_in_threadpool(
                    lm,
                    self._repair_prompt(response_text, request.sample_count),
                    max_tokens=2000,
                )
                try:
                    return self._parse_synthetic_response(
                        self._response_text(repaired), request.task_type
                    )
                except SyntheticDataGenerationError:
                    raise first_error from None

        except SyntheticDataGenerationError:
            raise
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            raise SyntheticDataGenerationError(
                "Failed to generate synthetic training data",
                error_code="SYNTHETIC_DATA_GENERATION_FAILED",
                details={
                    "provider": request.provider,
                    "model": request.model,
                    "error": str(e),
                },
            ) from e

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, list):
            return str(response[0]) if response else ""
        return str(response)

    def _generate_general_prompt(
        self, base_prompt: str, sample_count: int, task_type: str
    ) -> str:
        """Generate synthetic data prompt for general tasks."""
        return f"""You are an expert data generator. Create {sample_count} diverse, high-quality training examples based on this prompt:

"{base_prompt}"

Generate examples that:
1. Cover different scenarios and contexts
2. Vary in complexity and length
3. Include edge cases and challenging examples
4. Maintain consistency with the task requirements

Format your response as a JSON array where each item has:
- "input": The input text/prompt
- "output": The expected response/completion
- "extra_data": Any relevant context or notes

Example format:
[
  {{"input": "example input", "output": "example output", "extra_data": "context info"}},
  ...
]

Respond with only the JSON array: no code fence, no commentary, plain spaces for indentation.

Generate {sample_count} examples now:"""

    def _generate_creative_prompt(
        self, base_prompt: str, sample_count: int, task_type: str
    ) -> str:
        """Generate synthetic data prompt for creative tasks."""
        return f"""You are a creative writing expert. Generate {sample_count} diverse creative writing examples based on:

"{base_prompt}"

Create examples with varying:
- Writing styles (formal, casual, poetic, technical)
- Tones (serious, humorous, dramatic, informative)
- Lengths (short, medium, long)
- Complexity levels
- Creative approaches

Format as JSON array:
[
  {{"input": "creative prompt", "output": "creative response", "extra_data": "style/tone notes"}},
  ...
]

Respond with only the JSON array: no code fence, no commentary, plain spaces for indentation.

Generate {sample_count} creative examples:"""

    def _generate_code_prompt(
        self, base_prompt: str, sample_count: int, task_type: str
    ) -> str:
        """Generate synthetic data prompt for coding tasks."""
        return f"""You are a programming expert. Generate {sample_count} diverse coding examples based on:

"{base_prompt}"

Create examples covering:
- Different programming languages
- Various difficulty levels (beginner to advanced)
- Different problem types (algorithms, debugging, optimization)
- Multiple approaches to similar problems
- Edge cases and error handling

Format as JSON array:
[
  {{"input": "coding problem/question", "output": "code solution with explanation", "extra_data": "language/difficulty"}},
  ...
]

Respond with only the JSON array: no code fence, no commentary, plain spaces for indentation.

Generate {sample_count} coding examples:"""

    def _generate_analysis_prompt(
        self, base_prompt: str, sample_count: int, task_type: str
    ) -> str:
        """Generate synthetic data prompt for analysis tasks."""
        return f"""You are an analytical expert. Generate {sample_count} diverse analysis examples based on:

"{base_prompt}"

Create examples with:
- Different types of data/content to analyze
- Varying levels of complexity
- Multiple analytical frameworks
- Different conclusion types
- Various reasoning approaches

Format as JSON array:
[
  {{"input": "content to analyze", "output": "detailed analysis", "extra_data": "analysis type/framework"}},
  ...
]

Respond with only the JSON array: no code fence, no commentary, plain spaces for indentation.

Generate {sample_count} analysis examples:"""

    def _generate_translation_prompt(
        self, base_prompt: str, sample_count: int, task_type: str
    ) -> str:
        """Generate synthetic data prompt for translation tasks."""
        return f"""You are a translation expert. Generate {sample_count} diverse translation examples based on:

"{base_prompt}"

Create examples with:
- Different language pairs
- Various text types (formal, casual, technical, literary)
- Different complexity levels
- Cultural context considerations
- Idiomatic expressions and phrases

Format as JSON array:
[
  {{"input": "text to translate + target language", "output": "translated text", "extra_data": "language pair/context"}},
  ...
]

Respond with only the JSON array: no code fence, no commentary, plain spaces for indentation.

Generate {sample_count} translation examples:"""

    INPUT_KEYS = (
        "input",
        "input_text",
        "prompt",
        "question",
        "text",
        "instruction",
        "query",
    )
    OUTPUT_KEYS = (
        "output",
        "expected_output",
        "response",
        "answer",
        "completion",
        "label",
        "target",
    )

    @classmethod
    def _pair_from_item(cls, item: Any) -> tuple[str, str, Any] | None:
        """Extract (input, output, extra) from one generated item, if it has both."""
        if not isinstance(item, dict):
            return None
        lowered = {str(k).strip().lower(): v for k, v in item.items()}
        input_value = next((lowered[k] for k in cls.INPUT_KEYS if lowered.get(k)), None)
        output_value = next(
            (lowered[k] for k in cls.OUTPUT_KEYS if lowered.get(k) not in (None, "")),
            None,
        )
        if input_value is None or output_value is None:
            return None
        used = set(cls.INPUT_KEYS) | set(cls.OUTPUT_KEYS)
        extra = lowered.get("extra_data")
        if extra is None:
            leftovers = {k: v for k, v in lowered.items() if k not in used}
            extra = leftovers or None
        return str(input_value).strip(), str(output_value).strip(), extra

    @staticmethod
    def _find_json_payload(text: str) -> Any:
        """Best-effort JSON extraction: a list, an object wrapping a list, or JSON lines."""
        text = strip_code_fence(clean_model_text(text))
        candidates: list[str] = []
        for opener, closer in (("[", "]"), ("{", "}")):
            start, stop = text.find(opener), text.rfind(closer)
            if start != -1 and stop > start:
                candidates.append(text[start : stop + 1])
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        # JSON lines: one object per line.
        rows = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{") and line.endswith("}"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows or None

    _TEXT_PAIR = re.compile(
        r"(?:^|\n)\s*(?:\d+[.)]\s*)?\**\s*input\s*\**\s*[:\-]\s*(?P<input>.+?)\s*\n\s*\**\s*output\s*\**\s*[:\-]\s*(?P<output>.+?)(?=\n\s*(?:\d+[.)]\s*)?\**\s*input\b|\Z)",
        re.I | re.S,
    )

    def _parse_synthetic_response(
        self, response: str, task_type: str
    ) -> list[TrainingSampleCreate]:
        """Turn a model reply into samples, tolerating the usual formatting drift.

        Accepts a bare JSON array, a fenced code block, an object wrapping the
        list under any key, JSON lines, renamed keys (prompt/response,
        question/answer, ...), sentencepiece whitespace markers, and as a last
        resort "Input: ... / Output: ..." pairs in plain text.
        """
        payload = self._find_json_payload(response)
        items: list[Any] = []
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            nested = next((v for v in payload.values() if isinstance(v, list)), None)
            items = nested if nested is not None else [payload]

        pairs = [pair for pair in (self._pair_from_item(i) for i in items) if pair]

        if not pairs:
            cleaned = clean_model_text(response)
            pairs = [
                (m.group("input").strip(), m.group("output").strip(), None)
                for m in self._TEXT_PAIR.finditer(cleaned)
            ]

        if not pairs:
            logger.warning(
                "Synthetic data reply had no usable pairs; first 300 chars: %r",
                response[:300],
            )
            if items:
                raise SyntheticDataGenerationError(
                    "Model response contained no usable input/output pairs",
                    error_code="SYNTHETIC_DATA_EMPTY",
                    details={"task_type": task_type, "items_returned": len(items)},
                )
            raise SyntheticDataGenerationError(
                "The model's reply contained no input/output pairs. Try again, "
                "a different model, or lower the creativity setting.",
                error_code="SYNTHETIC_DATA_PARSE_FAILED",
                details={"task_type": task_type, "response_preview": response[:500]},
            )

        return [
            TrainingSampleCreate(
                input_text=input_text,
                expected_output=output_text,
                extra_data=self._as_extra_data(extra),
                quality_score=0.8,  # Default quality score for synthetic data
            )
            for input_text, output_text, extra in pairs
            if input_text and output_text
        ]

    @staticmethod
    def _repair_prompt(raw: str, sample_count: int) -> str:
        return (
            "Convert the following text into a JSON array of objects with exactly two "
            'string keys, "input" and "output". Output only the JSON array, with no '
            "code fence and no commentary. Keep at most "
            f"{sample_count} items.\n\nText:\n{raw[:6000]}\n\nJSON array:"
        )

    @staticmethod
    def _as_extra_data(value: Any) -> dict[str, Any]:
        """Normalise the free-form extra_data field the model returns.

        Models routinely answer with a bare string here, which the schema
        rejects, so anything that is not a mapping is wrapped.
        """
        if isinstance(value, dict):
            return value
        if value in (None, ""):
            return {}
        return {"notes": str(value)}

    def validate_sample_quality(self, sample: TrainingSampleCreate) -> float:
        """Calculate quality score for a training sample."""
        score = 0.5  # Base score

        # Length checks
        if len(sample.input_text) > 10:
            score += 0.1
        if len(sample.expected_output) > 20:
            score += 0.1

        # Content quality heuristics
        if sample.input_text.strip() and sample.expected_output.strip():
            score += 0.2

        # Avoid duplicates or very similar content
        if sample.input_text.lower() != sample.expected_output.lower():
            score += 0.1

        return min(score, 1.0)

    def parse_import_data(
        self, request: DatasetImportRequest
    ) -> list[TrainingSampleCreate]:
        """Parse imported data into training samples."""
        try:
            if request.file_format in ("json", "jsonl"):
                samples = self._parse_json_import(request.data)
            elif request.file_format == "csv":
                samples = self._parse_csv_import(request.data)
            else:
                raise TrainingDataError(
                    f"Unsupported import format: {request.file_format}",
                    error_code="UNSUPPORTED_IMPORT_FORMAT",
                    details={"supported": ["json", "csv"]},
                )
        except TrainingDataError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse import data: {e}")
            raise TrainingDataError(
                "Could not parse the imported data",
                error_code="IMPORT_PARSE_FAILED",
                details={"format": request.file_format, "error": str(e)},
            ) from e

        if not samples:
            raise TrainingDataError(
                "The imported data contained no usable samples",
                error_code="IMPORT_EMPTY",
                details={"format": request.file_format},
            )

        return samples

    def _parse_json_import(self, raw: str) -> list[TrainingSampleCreate]:
        """Accept a JSON array, an object wrapping one, or JSON Lines.

        Items may use ``input``/``output`` or the common aliases
        (``prompt``/``response``, ``question``/``answer``, ``text``/``label``,
        ``input_text``/``expected_output``); other fields become extra_data.
        """
        text = raw.strip()
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError:
            # JSON Lines: one object per line.
            data = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise TrainingDataError(
                        f"Line {line_number} is not valid JSON",
                        error_code="IMPORT_PARSE_FAILED",
                        details={"line": line_number, "error": str(exc)},
                    ) from exc

        if isinstance(data, dict):
            nested = next((v for v in data.values() if isinstance(v, list)), None)
            data = nested if nested is not None else [data]
        if not isinstance(data, list):
            raise TrainingDataError(
                "JSON import must be an array of objects (or JSON Lines)",
                error_code="IMPORT_PARSE_FAILED",
            )

        samples = []
        for item in data:
            pair = self._pair_from_item(item)
            if pair is None:
                continue
            input_text, output_text, extra = pair
            samples.append(
                self._build_sample(input_text, output_text, self._as_extra_data(extra))
            )
        return samples

    def _parse_csv_import(self, raw: str) -> list[TrainingSampleCreate]:
        """Parse CSV using the stdlib reader.

        Splitting on "," by hand corrupts any row containing a quoted comma or
        an embedded newline, which is exactly what prompt/response text has.
        """
        reader = csv.reader(io.StringIO(raw))
        rows = [row for row in reader if row]
        if not rows:
            return []

        header = [column.strip().lower() for column in rows[0]]
        if "input" in header and "output" in header:
            input_index, output_index = header.index("input"), header.index("output")
            body = rows[1:]
        else:
            # Headerless file: fall back to the first two columns.
            input_index, output_index = 0, 1
            body = rows

        return [
            self._build_sample(
                row[input_index].strip(),
                row[output_index].strip(),
                {"source": "csv_import"},
            )
            for row in body
            if len(row) > max(input_index, output_index)
        ]

    def _build_sample(
        self, input_text: str, expected_output: str, extra_data: dict[str, Any]
    ) -> TrainingSampleCreate:
        sample = TrainingSampleCreate(
            input_text=input_text,
            expected_output=expected_output,
            extra_data=extra_data,
        )
        sample.quality_score = self.validate_sample_quality(sample)
        return sample


# Global service instance
training_service = TrainingDataService()
