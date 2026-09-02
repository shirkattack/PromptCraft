import csv
import io
import json
import logging
from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.core.exceptions import SyntheticDataGenerationError, TrainingDataError
from app.schemas.training import (
    DatasetImportRequest,
    SyntheticDataRequest,
    TrainingSampleCreate,
)
from app.services.lm_manager import LMManager

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

            # Handle both string and list responses
            if isinstance(response, list):
                response_text = response[0] if response else ""
            else:
                response_text = str(response)

            # Parse the response into training samples
            return self._parse_synthetic_response(response_text, request.task_type)

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

Generate {sample_count} translation examples:"""

    def _parse_synthetic_response(
        self, response: str, task_type: str
    ) -> list[TrainingSampleCreate]:
        """Parse the model response into training samples."""
        json_start = response.find("[")
        json_end = response.rfind("]") + 1

        if json_start == -1 or json_end <= json_start:
            raise SyntheticDataGenerationError(
                "Model response did not contain a JSON array of examples",
                error_code="SYNTHETIC_DATA_PARSE_FAILED",
                details={"task_type": task_type, "response_preview": response[:500]},
            )

        try:
            data = json.loads(response[json_start:json_end])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse synthetic data JSON: {e}")
            raise SyntheticDataGenerationError(
                "Model response was not valid JSON",
                error_code="SYNTHETIC_DATA_PARSE_FAILED",
                details={
                    "task_type": task_type,
                    "error": str(e),
                    "response_preview": response[:500],
                },
            ) from e

        samples = [
            TrainingSampleCreate(
                input_text=str(item["input"]),
                expected_output=str(item["output"]),
                extra_data=self._as_extra_data(item.get("extra_data")),
                quality_score=0.8,  # Default quality score for synthetic data
            )
            for item in data
            if isinstance(item, dict) and "input" in item and "output" in item
        ]

        if not samples:
            raise SyntheticDataGenerationError(
                "Model response contained no usable input/output pairs",
                error_code="SYNTHETIC_DATA_EMPTY",
                details={"task_type": task_type, "items_returned": len(data)},
            )

        return samples

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
            if request.file_format == "json":
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
        data = json.loads(raw)
        if not isinstance(data, list):
            raise TrainingDataError(
                "JSON import must be an array of objects",
                error_code="IMPORT_PARSE_FAILED",
            )

        return [
            self._build_sample(
                str(item.get("input", "")),
                str(item.get("output", "")),
                self._as_extra_data(item.get("extra_data")),
            )
            for item in data
            if isinstance(item, dict)
        ]

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
