import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.lm_manager import LMManager
from app.schemas.training import (
    TrainingSampleCreate, 
    SyntheticDataRequest,
    DatasetImportRequest
)

logger = logging.getLogger(__name__)

class TrainingDataService:
    """Service for training data management and synthetic data generation."""
    
    def __init__(self):
        self.synthetic_prompts = {
            "general": self._generate_general_prompt,
            "creative": self._generate_creative_prompt,
            "code": self._generate_code_prompt,
            "analysis": self._generate_analysis_prompt,
            "translation": self._generate_translation_prompt,
        }
    
    async def generate_synthetic_data(
        self, 
        request: SyntheticDataRequest
    ) -> List[TrainingSampleCreate]:
        """
        Generate synthetic training data using Promptomatix techniques.
        Adapted from Promptomatix synthetic data generation.
        """
        try:
            # Get language model
            lm = LMManager.get_lm(
                provider=request.provider,
                model_name=request.model,
                temperature=request.creativity_level,
                max_tokens=2000
            )
            
            # Generate synthetic data prompt
            prompt_generator = self.synthetic_prompts.get(
                request.task_type, 
                self._generate_general_prompt
            )
            
            synthetic_prompt = prompt_generator(
                base_prompt=request.base_prompt,
                sample_count=request.sample_count,
                task_type=request.task_type
            )
            
            # Generate data using the model
            response = lm(synthetic_prompt, max_tokens=2000)
            
            # Handle both string and list responses
            if isinstance(response, list):
                response_text = response[0] if response else ""
            else:
                response_text = str(response)
            
            # Parse the response into training samples
            samples = self._parse_synthetic_response(response_text, request.task_type)
            
            return samples
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            return []
    
    def _generate_general_prompt(self, base_prompt: str, sample_count: int, task_type: str) -> str:
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

    def _generate_creative_prompt(self, base_prompt: str, sample_count: int, task_type: str) -> str:
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

    def _generate_code_prompt(self, base_prompt: str, sample_count: int, task_type: str) -> str:
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

    def _generate_analysis_prompt(self, base_prompt: str, sample_count: int, task_type: str) -> str:
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

    def _generate_translation_prompt(self, base_prompt: str, sample_count: int, task_type: str) -> str:
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

    def _parse_synthetic_response(self, response: str, task_type: str) -> List[TrainingSampleCreate]:
        """Parse the model response into training samples."""
        samples = []
        
        try:
            # Try to extract JSON from the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                for item in data:
                    if isinstance(item, dict) and "input" in item and "output" in item:
                        sample = TrainingSampleCreate(
                            input_text=str(item["input"]),
                            expected_output=str(item["output"]),
                            extra_data=item.get("extra_data", {}),
                            quality_score=0.8  # Default quality score for synthetic data
                        )
                        samples.append(sample)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse synthetic data JSON: {e}")
            # Fallback: create a single sample from the response
            samples = [TrainingSampleCreate(
                input_text=f"Synthetic data generation prompt ({task_type})",
                expected_output=response[:500] + "..." if len(response) > 500 else response,
                extra_data={"generation_error": str(e), "task_type": task_type},
                quality_score=0.3
            )]
        
        return samples
    
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
    
    def parse_import_data(self, request: DatasetImportRequest) -> List[TrainingSampleCreate]:
        """Parse imported data into training samples."""
        samples = []
        
        try:
            if request.file_format == "json":
                data = json.loads(request.data)
                if isinstance(data, list):
                    for item in data:
                        sample = TrainingSampleCreate(
                            input_text=str(item.get("input", "")),
                            expected_output=str(item.get("output", "")),
                            extra_data=item.get("extra_data", {}),
                            quality_score=self.validate_sample_quality(
                                TrainingSampleCreate(
                                    input_text=str(item.get("input", "")),
                                    expected_output=str(item.get("output", ""))
                                )
                            )
                        )
                        samples.append(sample)
            
            elif request.file_format == "csv":
                # Basic CSV parsing (would need pandas for complex CSV)
                lines = request.data.strip().split('\n')
                headers = lines[0].split(',') if lines else []
                
                for line in lines[1:]:
                    values = line.split(',')
                    if len(values) >= 2:
                        sample = TrainingSampleCreate(
                            input_text=values[0].strip(),
                            expected_output=values[1].strip(),
                            extra_data={"source": "csv_import"},
                            quality_score=self.validate_sample_quality(
                                TrainingSampleCreate(
                                    input_text=values[0].strip(),
                                    expected_output=values[1].strip()
                                )
                            )
                        )
                        samples.append(sample)
        
        except Exception as e:
            logger.error(f"Failed to parse import data: {e}")
        
        return samples

# Global service instance
training_service = TrainingDataService()