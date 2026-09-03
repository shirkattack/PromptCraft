"""Run prompts on a single input, the way a user would after copying them.

A prompt with an ``{input}`` placeholder gets the input substituted; one
without gets the input appended as ``Input: ... / Output:``. The text is sent
as a plain completion, so what is measured here is exactly what the copied
prompt does in someone else's application.
"""

import time
from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.core.logging import get_logger
from app.services.lm_manager import LMManager
from app.services.text import clean_model_text

logger = get_logger("try_service")

MAX_INPUT_CHARS = 4000


def fill_prompt(prompt: str, input_text: str) -> str:
    """Put the input where the prompt expects it."""
    if "{input}" in prompt:
        return prompt.replace("{input}", input_text)
    return f"{prompt.rstrip()}\n\nInput: {input_text}\nOutput:"


def _completion_text(response: Any) -> str:
    if isinstance(response, list):
        response = response[0] if response else ""
    return clean_model_text(str(response)).strip()


async def try_prompts(
    provider: str,
    model: str,
    input_text: str,
    prompts: dict[str, str],
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> list[dict[str, Any]]:
    """Run each named prompt on the input; one entry per prompt, in order.

    A failed call is reported in its entry rather than failing the whole
    request, so the other answer still shows.
    """
    lm = LMManager.get_lm(
        provider=provider,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    results: list[dict[str, Any]] = []
    for label, prompt in prompts.items():
        filled = fill_prompt(prompt, input_text)
        started = time.time()
        try:
            response = await run_in_threadpool(lm, filled, max_tokens=max_tokens)
            output, error = _completion_text(response), None
        except Exception as exc:  # keep the other variant's answer
            logger.warning(f"Try-it call failed for {label}: {exc}")
            output, error = "", str(exc)
        results.append(
            {
                "label": label,
                "prompt_sent": filled,
                "output": output,
                "error": error,
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
    return results
