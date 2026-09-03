"""Clean-up for text that comes back from local models."""

import re

# Sentencepiece's word-boundary marker. gemma3n through Ollama emits it in
# place of spaces (including JSON indentation), which breaks parsing and
# leaks into prompts and answers if not replaced.
SENTENCEPIECE_SPACE = "▁"

_FENCE = re.compile(r"^\s*```[\w-]*\s*\n?|\n?\s*```\s*$")


def clean_model_text(text: str) -> str:
    """Replace tokenizer artefacts and normalise line endings."""
    return text.replace(SENTENCEPIECE_SPACE, " ").replace("\r\n", "\n")


def strip_code_fence(text: str) -> str:
    """Remove one surrounding ``` fence (with optional language tag)."""
    return _FENCE.sub("", text.strip()).strip()
