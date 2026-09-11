"""GEPA: reflective prompt evolution against a dataset.

GEPA (Agrawal et al., 2025) runs the prompt on a few training samples, asks a
metric for *written feedback* on each miss, has a reflection model rewrite the
instructions to address that feedback, and keeps a Pareto front of candidates
that each win on different samples. This module wraps ``dspy.teleprompt.GEPA``
so that:

* the metric produces feedback a small local model can act on (for example
  "the right label is buried in a 60-word answer"),
* every iteration is reported as progress while the job runs, and
* the run returns a timeline: each candidate's instructions, score, parent
  and the feedback that led to it, so the client can show why each edit
  happened.
"""

import logging
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import dspy
from dspy.teleprompt import GEPA
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from app.core.config import settings
from app.services.code_eval_service import evaluate_response
from app.services.eval_service import (
    SHORT_ANSWER_CHARS,
    EvalError,
    EvalMetric,
    Sample,
    choose_metric,
    describe_split,
    example_code_fields,
    normalize,
    render_prompt,
    report_pass_at_1,
    require_code_eval,
    result_identity,
    split_samples,
)
from app.services.progress import ProgressCallback, no_progress

logger = logging.getLogger(__name__)

GEPA_LOGGER_NAME = "dspy.teleprompt.gepa.gepa"

# Feedback metric signature GEPA calls: (gold, pred, trace, pred_name, pred_trace)
FeedbackMetric = Callable[..., dspy.Prediction]


# -- feedback metrics ---------------------------------------------------------


class JudgeWithReason(dspy.Signature):
    """Decide whether a response satisfies the expected output, and say why.

    Judge meaning, not wording. The reason must be specific enough that a
    prompt author could act on it: name what is missing, wrong or extra.
    """

    input = dspy.InputField(desc="The input the response was written for")
    expected_output = dspy.InputField(desc="A reference answer known to be correct")
    actual_output = dspy.InputField(desc="The response being judged")
    verdict = dspy.OutputField(desc="'yes' if the response is correct, otherwise 'no'")
    reason = dspy.OutputField(
        desc="One or two sentences on what made it right or wrong"
    )


def label_feedback_metric(exact: bool) -> FeedbackMetric:
    """Feedback for label-style outputs (classes, short answers).

    Full credit for the bare label, half credit when the label is present but
    wrapped in extra text (with feedback telling the model to stop), none
    otherwise.
    """

    def metric(
        gold: dspy.Example,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        expected = normalize(gold.output)
        raw = str(getattr(pred, "output", "") or "")
        actual = normalize(raw)
        words = len(raw.split())

        if actual == expected:
            return dspy.Prediction(
                score=1.0, feedback=f"Correct: answered exactly '{gold.output}'."
            )
        if not exact and expected and expected in actual:
            return dspy.Prediction(
                score=0.5,
                feedback=(
                    f"The right answer '{gold.output}' is in the response but buried "
                    f"in {words} words. Respond with '{gold.output}' alone, no explanation."
                ),
            )
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"Wrong. Expected '{gold.output}' but got '{raw[:120]}'. "
                f"Input was: '{gold.input[:160]}'."
            ),
        )

    return metric


def judge_feedback_metric() -> FeedbackMetric:
    """Feedback for free-text outputs: a model judge's verdict and reason."""
    judge = dspy.Predict(JudgeWithReason)

    def metric(
        gold: dspy.Example,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        raw = str(getattr(pred, "output", "") or "").strip()
        if not raw:
            return dspy.Prediction(score=0.0, feedback="The response was empty.")
        verdict = judge(
            input=gold.input, expected_output=gold.output, actual_output=raw
        )
        ok = normalize(verdict.verdict).startswith("yes")
        reason = str(getattr(verdict, "reason", "") or "").strip()
        return dspy.Prediction(
            score=1.0 if ok else 0.0,
            feedback=(
                f"{'Correct' if ok else 'Incorrect'}: {reason}"
                if reason
                else (
                    "Correct." if ok else f"Incorrect. Expected: '{gold.output[:160]}'."
                )
            ),
        )

    return metric


def code_feedback_metric() -> FeedbackMetric:
    """Feedback for code outputs: fraction of asserts passed, and why not all.

    ``no_code``, ``syntax_error``, ``wrong_name`` and ``timeout`` score 0.0;
    a partial run scores passed/total; a full pass scores 1.0. The reported
    score of a run stays binary pass@1 (see ``GepaOptimizer._evaluate``); the
    fraction only steers the search.
    """

    def metric(
        gold: dspy.Example,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        response = str(getattr(pred, "output", "") or "")
        result = evaluate_response(response, example_code_fields(gold))
        if result.status == "pass":
            return dspy.Prediction(score=1.0, feedback=result.feedback)
        if result.status in {"no_code", "syntax_error", "wrong_name", "timeout"}:
            return dspy.Prediction(score=0.0, feedback=result.feedback)
        return dspy.Prediction(score=result.fraction, feedback=result.feedback)

    return metric


# Metrics whose partial credit must not leak into the reported score.
BINARY_REPORT_METRICS = {"tests"}


def build_feedback_metric(metric_name: str) -> FeedbackMetric:
    if metric_name == "exact":
        return label_feedback_metric(exact=True)
    if metric_name == "contains":
        return label_feedback_metric(exact=False)
    if metric_name == "llm_judge":
        return judge_feedback_metric()
    if metric_name == "tests":
        require_code_eval()
        return code_feedback_metric()
    raise EvalError(f"Unknown metric: {metric_name}")


# -- instruction clean-up -----------------------------------------------------

_FENCE = re.compile(r"^\s*```[\w-]*\s*\n|\n\s*```\s*$")


def clean_instructions(text: str) -> str:
    """Undo formatting a small reflection model wraps its proposal in.

    llama3.2 tends to return the new instructions as a fenced code block of
    Python comments ("python\\n# Classify ..."). The content is fine; the
    markup would end up in the user's prompt.
    """
    cleaned = _FENCE.sub("", text.strip())
    lines = cleaned.splitlines()
    if lines and lines[0].strip().lower() in {"python", "text", "markdown", "md"}:
        lines = lines[1:]
    stripped = [ln.strip() for ln in lines if ln.strip()]
    if stripped and all(ln.startswith("#") for ln in stripped):
        lines = [re.sub(r"^\s*#\s?", "", ln) for ln in lines]
    return "\n".join(lines).strip()


# -- reflection that stays general ---------------------------------------------

# GEPA's own reflection prompt asks the reflection model to copy "niche and
# domain specific factual information" from the examples into the new
# instruction. That suits a dataset where every input is the same task. When
# the instruction is a template applied to different tasks (code: every sample
# is its own problem), it produces instructions that describe the training
# tasks that failed, which score well on val and do not transfer.
TEMPLATE_REFLECTION_PROMPT = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The instructions are a template: they are sent unchanged with every task, and the task itself is appended after them. Each example below is a different task, and the tasks the assistant will face later are different again; none of these examples will come back.

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs to learn the input format and the kind of tasks involved. Read the responses and the feedback, and find the mistakes that could happen again on a different task: the response format, the function name and signature, missing imports, edge cases, types, efficiency. Write general guidance that prevents those mistakes, and keep any generalizable strategy the assistant used successfully.

Do not mention any function name, variable name, literal value, test case or problem from these examples, and do not explain how to solve any of them: that knowledge does not carry over to other tasks. Do not add placeholders such as {task_input}; the task is appended automatically.

Provide the new instructions within ``` blocks."""

# Metrics whose samples are each a different task, so the instruction is a template.
TEMPLATE_METRICS = {"tests"}

_WORD = re.compile(r"[A-Za-z]{4,}")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PLACEHOLDER = re.compile(r"\{\s*[A-Za-z_][A-Za-z0-9_]*\s*\}")
_QUOTED = re.compile(r"""(['"])(\w{4,40})\1""")
PHRASE_WORDS = 6


def _identifier_like(token: str) -> bool:
    """An underscore, a digit or an inner capital: a name, not an English word."""
    return (
        "_" in token
        or any(ch.isdigit() for ch in token)
        or any(ch.isupper() for ch in token[1:])
    )


def _phrases(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        tuple(words[i : i + PHRASE_WORDS]) for i in range(len(words) - PHRASE_WORDS + 1)
    }


@dataclass
class LeakReport:
    """What a proposed instruction borrowed from the examples it was shown."""

    names: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    literals: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    # Words that occur in exactly one input of the dataset. Recorded for
    # analysis, not enforced: on a small corpus ordinary words qualify too.
    terms: list[str] = field(default_factory=list)

    @property
    def leaked(self) -> list[str]:
        return self.names + self.placeholders + self.literals + self.phrases


class InstructionLeakGuard:
    """Checks a proposed instruction for content specific to the shown examples.

    Enforced for template instructions: a task's function name (any sample's
    entry point, or an identifier an example input calls), a quoted literal
    from an example input, or a run of six words copied from one that
    includes a word rare in the dataset. Enforced for every instruction: an
    invented placeholder such as ``{task_input}``. Anything already in the
    original instruction is allowed.
    """

    def __init__(self, samples: list[Sample], original: str, template: bool) -> None:
        self.template = template
        self.original = original
        self._original_words = {w.lower() for w in _WORD.findall(original)}
        self._original_phrases = _phrases(original)
        self.df: Counter[str] = Counter()
        for sample in samples:
            self.df.update({w.lower() for w in _WORD.findall(sample.input_text)})
        self.entry_points = sorted(
            {
                str((sample.extra_data or {}).get("entry_point") or "")
                for sample in samples
            }
            - {""}
        )

    def _mentions(self, name: str, text: str) -> bool:
        if name in self.original:
            return False
        if _identifier_like(name):
            return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None
        if name[:1].isupper():  # "Split": only as a call or in backticks
            return (
                re.search(rf"(?<!\w){re.escape(name)}\s*\(|`{re.escape(name)}`", text)
                is not None
            )
        return False  # "sum", "find": ordinary words and builtins

    def check(self, text: str, shown_inputs: Sequence[str]) -> LeakReport:
        report = LeakReport()
        report.placeholders = sorted(
            {p for p in _PLACEHOLDER.findall(text) if p not in self.original}
        )
        if not self.template:
            return report

        called = {name for s in shown_inputs for name in _CALL.findall(s)}
        report.names = sorted(
            name
            for name in set(self.entry_points) | called
            if self._mentions(name, text)
        )
        report.literals = sorted(
            {
                literal
                for s in shown_inputs
                for _, literal in _QUOTED.findall(s)
                if (_identifier_like(literal) or len(literal) >= 10)
                and literal in text
                and literal not in self.original
                and literal not in report.names
            }
        )
        proposal_phrases = _phrases(text) - self._original_phrases
        report.phrases = sorted(
            {
                " ".join(phrase)
                for s in shown_inputs
                for phrase in _phrases(s) & proposal_phrases
                if any(
                    len(w) >= 5 and w.isalpha() and 0 < self.df[w] <= 2 for w in phrase
                )
            }
        )[:10]
        proposal_words = {w.lower() for w in _WORD.findall(text)}
        report.terms = sorted(
            {
                word
                for s in shown_inputs
                for word in {w.lower() for w in _WORD.findall(s)} & proposal_words
                if len(word) >= 5
                and self.df[word] == 1
                and word not in self._original_words
            }
        )
        return report


class GuardedInstructionProposer:
    """GEPA's reflection step (``instruction_proposer``) with a leak check.

    Renders the reflection prompt the way GEPA does, with
    ``TEMPLATE_REFLECTION_PROMPT`` for template instructions and GEPA's own
    prompt otherwise, then checks the proposal. A proposal that borrows from
    the examples is regenerated once with the borrowed items named; if it
    still borrows, it is dropped and GEPA skips the iteration without
    spending metric calls on it.
    """

    def __init__(self, guard: InstructionLeakGuard, lm: Any = None) -> None:
        self.guard = guard
        self.lm = lm
        self.prompt_template = TEMPLATE_REFLECTION_PROMPT if guard.template else None
        self.events: list[dict[str, Any]] = []

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        lm = self.lm or dspy.settings.lm
        if lm is None:
            raise EvalError("GEPA's reflection step needs a language model")
        proposals: dict[str, str] = {}
        for name in components_to_update:
            records = list(reflective_dataset.get(name) or [])
            if not records:
                continue
            shown = [
                str(value)
                for record in records
                for value in dict(record.get("Inputs") or {}).values()
            ]
            prompt = InstructionProposalSignature.prompt_renderer(
                {
                    "current_instruction_doc": candidate[name],
                    "dataset_with_feedback": records,
                    "prompt_template": self.prompt_template,
                }
            )
            text = self._ask(lm, prompt)
            report = self.guard.check(text, shown)
            event: dict[str, Any] = {
                "component": name,
                "leaked": report.leaked,
                "terms": report.terms,
                "regenerated": False,
                "rejected": False,
            }
            if report.leaked:
                event["regenerated"] = True
                text = self._ask(lm, f"{prompt}{self._retry_note(report)}")
                again = self.guard.check(text, shown)
                event["leaked_after_retry"] = again.leaked
                event["terms"] = again.terms
                if again.leaked:
                    event["rejected"] = True
                    self.events.append(event)
                    logger.info(
                        "GEPA proposal dropped, still specific to its examples "
                        f"after a retry: {again.leaked[:8]}"
                    )
                    continue
            self.events.append(event)
            proposals[name] = text
        return proposals

    @staticmethod
    def _ask(lm: Any, prompt: Any) -> str:
        outputs = lm(prompt) if isinstance(prompt, str) else lm(messages=prompt)
        first = outputs[0] if outputs else ""
        if isinstance(first, dict):
            first = first.get("text") or ""
        return str(
            InstructionProposalSignature.output_extractor(str(first))["new_instruction"]
        )

    @staticmethod
    def _retry_note(report: LeakReport) -> str:
        parts = []
        borrowed = report.names + report.literals + report.phrases
        if borrowed:
            listed = ", ".join(f"'{item}'" for item in borrowed[:8])
            parts.append(
                f"Your previous instruction mentioned {listed}, which comes from "
                "one of the example tasks."
            )
        if report.placeholders:
            parts.append(
                f"It contained the placeholder {', '.join(report.placeholders)}; "
                "the task is appended automatically, so no placeholder is needed."
            )
        parts.append(
            "Write the instruction again so that it applies equally to every "
            "task, with nothing taken from a specific example. Provide it within "
            "``` blocks."
        )
        return "\n\n" + " ".join(parts)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "template" if self.guard.template else "placeholders",
            "proposals": len(self.events),
            "regenerated": sum(1 for e in self.events if e["regenerated"]),
            "rejected": sum(1 for e in self.events if e["rejected"]),
            "events": self.events[-40:],
        }


# -- iteration tracking --------------------------------------------------------

_ITERATION = re.compile(r"^Iteration (\d+): (.*)$", re.S)


@dataclass
class IterationEvent:
    iteration: int
    kind: str  # selected | proposed | accepted | skipped | scored | other
    message: str
    value: float | None = None


class GepaTracker(logging.Handler):
    """Turns GEPA's log lines into progress updates and a per-iteration record.

    Also collects the feedback strings the metric emits, keyed by the
    iteration in flight, so each proposal can be shown with the feedback that
    triggered it.
    """

    def __init__(self, progress: ProgressCallback) -> None:
        super().__init__(level=logging.INFO)
        self.progress = progress
        self.events: list[IterationEvent] = []
        self.feedback_by_iteration: dict[int, list[str]] = {}
        self.current_iteration = 0
        self.best_score: float | None = None
        self._lock = threading.Lock()

    def record_feedback(self, feedback: str, score: float) -> None:
        if score >= 1.0:
            return  # only misses carry information for the reflection step
        with self._lock:
            bucket = self.feedback_by_iteration.setdefault(self.current_iteration, [])
            if feedback not in bucket and len(bucket) < 12:
                bucket.append(feedback)

    def emit(self, record: logging.LogRecord) -> None:
        match = _ITERATION.match(record.getMessage())
        if not match:
            return
        iteration, rest = int(match.group(1)), match.group(2).strip()
        with self._lock:
            self.current_iteration = iteration

        kind, value = "other", None
        if rest.startswith("Selected program"):
            kind = "selected"
        elif rest.startswith("Proposed new text"):
            kind = "proposed"
            rest = "Reflection proposed new instructions"
        elif "New subsample score is not better" in rest or "worse than both" in rest:
            kind = "skipped"
            rest = "Proposal did not beat its parent on the sample; skipped"
        elif rest.startswith("New program candidate index"):
            kind = "accepted"
        elif rest.startswith("Full valset score for new program"):
            kind = "scored"
            value = _trailing_float(rest)
        elif rest.startswith("Best valset aggregate score so far"):
            value = _trailing_float(rest)
            if value is not None:
                self.best_score = value
        elif rest.startswith("Base program full valset score"):
            value = _trailing_float(rest)
            if value is not None:
                self.best_score = value

        self.events.append(IterationEvent(iteration, kind, rest[:200], value))
        if kind in {"selected", "proposed", "skipped", "accepted", "scored"}:
            self.progress(
                "evolve",
                f"Generation {iteration}: {rest[:120]}",
                current=iteration,
                total=None,
                best_score=(
                    round(self.best_score * 100, 1)
                    if self.best_score is not None
                    else None
                ),
            )

    def accepted_iterations(self) -> list[int]:
        return [e.iteration for e in self.events if e.kind == "accepted"]


def _candidate_instructions(candidate: Any) -> str:
    """The instruction text of a GEPA candidate, whatever shape it comes in."""
    if isinstance(candidate, dict):
        return str(next(iter(candidate.values()), ""))
    if hasattr(candidate, "signature"):  # a compiled dspy program
        return str(getattr(candidate.signature, "instructions", ""))
    if hasattr(candidate, "predictors"):
        preds = list(candidate.predictors())
        return str(getattr(preds[0].signature, "instructions", "")) if preds else ""
    return str(candidate)


def _trailing_float(text: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    return float(match.group(1)) if match else None


# -- the optimizer ---------------------------------------------------------------


@dataclass
class GepaCandidate:
    index: int
    parent: int | None
    generation: int
    instructions: str
    score: float | None
    iteration: int | None = None
    feedback: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "parent": self.parent,
            "generation": self.generation,
            "instructions": self.instructions,
            "score": self.score,
            "iteration": self.iteration,
            "feedback": self.feedback,
        }


class GepaOptimizer:
    """Evolve a prompt's instructions with GEPA and measure the result.

    Must be called inside ``dspy.context(lm=...)`` on the thread that runs it.
    """

    def __init__(
        self,
        samples: list[Sample],
        metric: EvalMetric = "auto",
        budget: int = 60,
        reflection_lm: dspy.LM | None = None,
        train_ratio: float | None = None,
        seed: int = 13,
        progress: ProgressCallback = no_progress,
        user_feedback: list[str] | None = None,
        train: list[Sample] | None = None,
        dev: list[Sample] | None = None,
        guard: bool = True,
        reflection_minibatch_size: int = 3,
    ) -> None:
        """``train``/``dev`` override the internal split (and its caps) when a
        caller such as the benchmark runner brings a fixed split of its own.

        ``guard`` routes GEPA's reflection through ``GuardedInstructionProposer``;
        off, GEPA uses its own reflection prompt with no check, as before.
        """
        if len(samples) < 2:
            raise EvalError("GEPA needs a dataset with at least 2 samples")
        self.samples = samples
        # Notes the user left on earlier versions of this prompt. They are
        # appended to the metric's feedback on every miss, so the reflection
        # step reads them next to the concrete failure.
        self.user_feedback = [
            n.strip()[:300] for n in (user_feedback or []) if n.strip()
        ]
        self.metric_name = choose_metric(metric, samples)
        self.budget = max(10, budget)
        self.reflection_lm = reflection_lm
        self.guard = guard
        self.reflection_minibatch_size = max(1, reflection_minibatch_size)
        self.seed = seed
        self.progress = progress
        if train is not None and dev is not None:
            if not train or not dev:
                raise EvalError("GEPA needs non-empty train and dev sets")
            self.train, self.dev = list(train), list(dev)
        else:
            self.train, self.dev = split_samples(
                samples,
                settings.default_train_ratio if train_ratio is None else train_ratio,
                settings.eval_max_train_samples,
                settings.eval_max_dev_samples,
                seed,
            )
        self.metric = build_feedback_metric(self.metric_name)
        # For code, GEPA searches on the fraction of asserts passed but the
        # number reported for a prompt is pass@1: a sample counts only when
        # every assert passes.
        self.report_binary = self.metric_name in BINARY_REPORT_METRICS

    # -- evaluation helpers

    def _evaluate(self, program: dspy.Module) -> tuple[float, list[dict[str, Any]]]:
        """Score a program on the held-out split; returns (percent, rows)."""
        rows = []
        scores: list[float] = []
        for sample in self.dev:
            example = sample.to_example()
            try:
                pred = program(input=sample.input_text)
            except Exception as exc:  # a single failed call scores zero
                logger.warning(f"Held-out call failed: {exc}")
                pred = dspy.Prediction(output="")
            verdict = self.metric(example, pred)
            score = float(verdict.score)
            scores.append(score)
            rows.append(
                {
                    "input": sample.input_text,
                    "expected": sample.expected_output,
                    "actual": str(getattr(pred, "output", "") or ""),
                    "passed": score >= 1.0,
                    "score": score,
                    "feedback": str(getattr(verdict, "feedback", "") or ""),
                    **result_identity(example),
                }
            )
        if self.report_binary:
            # pass@1: the loop metric's partial credit never reaches the report.
            percent = report_pass_at_1(scores)
        else:
            percent = round(sum(scores) / len(scores) * 100, 2) if scores else 0.0
        return percent, rows

    @staticmethod
    def _program(instructions: str) -> dspy.Predict:
        return dspy.Predict(dspy.Signature("input -> output", instructions.strip()))

    # -- main entry

    def run(self, original: str) -> dict[str, Any]:
        started = time.time()
        logger.info(
            f"GEPA run: metric={self.metric_name} train={len(self.train)} "
            f"dev={len(self.dev)} budget={self.budget} "
            f"report={'pass@1' if self.report_binary else 'mean score'}"
        )
        self.progress(
            "evaluate",
            f"Scoring the original prompt on {len(self.dev)} held-out samples",
            current=0,
            total=None,
        )
        baseline_score, baseline_rows = self._evaluate(self._program(original))
        self.progress(
            "evolve",
            f"Original scored {baseline_score:.0f}%. Evolving with a budget of "
            f"{self.budget} scored calls",
            current=0,
            total=None,
            best_score=baseline_score,
        )

        tracker = GepaTracker(self.progress)
        tracker.best_score = baseline_score / 100

        base_metric = self.metric

        def tracked_metric(
            gold: dspy.Example,
            pred: Any,
            trace: Any = None,
            pred_name: str | None = None,
            pred_trace: Any = None,
        ) -> dspy.Prediction:
            verdict = base_metric(gold, pred, trace, pred_name, pred_trace)
            score = float(verdict.score)
            feedback = str(getattr(verdict, "feedback", "") or "")
            if self.user_feedback and score < 1.0:
                notes = "; ".join(f"'{n}'" for n in self.user_feedback)
                feedback = f"{feedback} The user said about earlier versions: {notes}."
                verdict = dspy.Prediction(score=score, feedback=feedback)
            tracker.record_feedback(feedback, score)
            return verdict

        proposer = (
            GuardedInstructionProposer(
                InstructionLeakGuard(
                    self.train + self.dev,
                    original,
                    template=self.metric_name in TEMPLATE_METRICS,
                )
            )
            if self.guard
            else None
        )
        optimizer = GEPA(
            metric=tracked_metric,
            max_metric_calls=self.budget,
            reflection_minibatch_size=min(
                self.reflection_minibatch_size, len(self.train)
            ),
            reflection_lm=self.reflection_lm,
            instruction_proposer=proposer,
            num_threads=1,  # one local model; parallel calls just queue
            track_stats=True,
            skip_perfect_score=True,
            seed=self.seed,
        )

        gepa_logger = logging.getLogger(GEPA_LOGGER_NAME)
        previous_level = gepa_logger.level
        gepa_logger.addHandler(tracker)
        if gepa_logger.level > logging.INFO or gepa_logger.level == logging.NOTSET:
            gepa_logger.setLevel(logging.INFO)
        try:
            compiled = optimizer.compile(
                self._program(original),
                trainset=[s.to_example() for s in self.train],
                valset=[s.to_example() for s in self.dev],
            )
        finally:
            gepa_logger.removeHandler(tracker)
            gepa_logger.setLevel(previous_level)

        self.progress(
            "evaluate",
            f"Scoring the evolved prompt on {len(self.dev)} held-out samples",
            current=tracker.current_iteration,
            total=None,
        )
        evolved_instructions = clean_instructions(
            str(getattr(compiled.signature, "instructions", "") or original)
        )
        selected_index, selection = self._select_candidate(compiled)
        if selected_index is not None:
            evolved_instructions = (
                clean_instructions(
                    _candidate_instructions(
                        compiled.detailed_results.candidates[selected_index]
                    )
                )
                or evolved_instructions
            )
        evolved_program = self._program(evolved_instructions)
        final_score, final_rows = self._evaluate(evolved_program)

        timeline = self._timeline(compiled, original, tracker)
        improved = final_score > baseline_score
        chosen_instructions = evolved_instructions if improved else original
        chosen_score = final_score if improved else baseline_score

        elapsed = round(time.time() - started, 1)
        detailed = getattr(compiled, "detailed_results", None)
        metric_calls = getattr(detailed, "total_metric_calls", None)

        report = {
            "budget": self.budget,
            "metric_calls": metric_calls,
            "iterations": tracker.current_iteration,
            "reflection_model": getattr(self.reflection_lm, "model", None),
            "user_feedback": self.user_feedback,
            "baseline_score": baseline_score,
            "final_score": final_score,
            "improved": improved,
            "best_index": (
                getattr(detailed, "best_idx", None) if detailed is not None else None
            ),
            "selected_index": selected_index,
            "selection": selection,
            "reflection_minibatch_size": self.reflection_minibatch_size,
            "guard": proposer.summary() if proposer is not None else {"mode": "off"},
            "timeline": [c.as_dict() for c in timeline],
            "instructions": chosen_instructions,
            "elapsed_seconds": elapsed,
        }

        evaluation = {
            "metric": self.metric_name,
            "train_size": len(self.train),
            "dev_size": len(self.dev),
            "total_samples": len(self.samples),
            "max_demos": 0,
            "baseline_score": baseline_score,
            "eval_score": chosen_score,
            "best": "gepa" if improved else "original",
            "improved": improved,
            "candidates": [
                {
                    "name": "original",
                    "score": baseline_score,
                    "demo_count": 0,
                    "bootstrapped_demos": 0,
                    "error": None,
                },
                {
                    "name": "gepa",
                    "score": final_score,
                    "demo_count": 0,
                    "bootstrapped_demos": 0,
                    "error": None,
                },
            ],
            "demos": [],
            "baseline_results": baseline_rows,
            "results": final_rows if improved else baseline_rows,
            "instructions": chosen_instructions,
            "split": describe_split(self.samples, self.train, self.dev, "holdout"),
        }

        return {
            "optimized_prompt": render_prompt(chosen_instructions, []),
            "instructions": chosen_instructions,
            "gepa": report,
            "eval": evaluation,
        }

    def _select_candidate(self, compiled: Any) -> tuple[int | None, str]:
        """Which candidate to return, and by what rule.

        GEPA's own pick (``best_idx``) is the best on the loop metric. When
        the run reports pass@1, the returned candidate is instead the best on
        val pass@1, ties broken by GEPA's aggregate and then the lower index,
        so the reported score belongs to the prompt actually handed back.
        """
        detailed = getattr(compiled, "detailed_results", None)
        subscores = list(getattr(detailed, "val_subscores", None) or [])
        candidates = list(getattr(detailed, "candidates", None) or [])
        if not self.report_binary or not subscores or len(subscores) != len(candidates):
            return None, "gepa aggregate"
        aggregate = list(getattr(detailed, "val_aggregate_scores", None) or [])
        ranked = sorted(
            range(len(subscores)),
            key=lambda i: (
                -report_pass_at_1(dict(subscores[i]).values()),
                -(aggregate[i] if i < len(aggregate) else 0.0),
                i,
            ),
        )
        return ranked[0], "val pass@1"

    def _timeline(
        self, compiled: Any, original: str, tracker: GepaTracker
    ) -> list[GepaCandidate]:
        """Assemble the candidate lineage from GEPA's result object.

        Falls back to a two-entry timeline (original, evolved) when the
        detailed results are missing, so the client always has something.
        """
        detailed = getattr(compiled, "detailed_results", None)
        # GEPAResult exposes these as attributes; to_dict() reshapes them.
        candidates = list(getattr(detailed, "candidates", None) or [])
        parents = list(getattr(detailed, "parents", None) or [])
        scores = list(getattr(detailed, "val_aggregate_scores", None) or [])
        if self.report_binary:
            # GEPA's aggregate is the loop metric (partial credit); report the
            # candidates on the same scale as the run, pass@1 over the dev set.
            subscores = list(getattr(detailed, "val_subscores", None) or [])
            scores = [
                report_pass_at_1(dict(per_instance).values()) / 100
                for per_instance in subscores
            ] or scores
        if not candidates:
            evolved = clean_instructions(
                str(getattr(compiled.signature, "instructions", "") or original)
            )
            return [
                GepaCandidate(0, None, 0, original, None),
                GepaCandidate(1, 0, 1, evolved, None),
            ]

        accepted = tracker.accepted_iterations()
        timeline: list[GepaCandidate] = []
        for index, candidate in enumerate(candidates):
            instructions = _candidate_instructions(candidate)
            parent_list = parents[index] if index < len(parents) else [None]
            parent = next((p for p in (parent_list or [None]) if p is not None), None)
            generation = 0 if parent is None else timeline[parent].generation + 1
            score = scores[index] if index < len(scores) else None
            # Candidate 0 is the seed; candidate k was accepted in the k-th
            # accepted iteration, whose reflection read that iteration's feedback.
            iteration = accepted[index - 1] if 0 < index <= len(accepted) else None
            feedback = (
                tracker.feedback_by_iteration.get(iteration, []) if iteration else []
            )
            timeline.append(
                GepaCandidate(
                    index=index,
                    parent=parent,
                    generation=generation,
                    instructions=clean_instructions(str(instructions)),
                    score=round(float(score) * 100, 2) if score is not None else None,
                    iteration=iteration,
                    feedback=feedback,
                )
            )
        return timeline


__all__ = [
    "SHORT_ANSWER_CHARS",
    "TEMPLATE_REFLECTION_PROMPT",
    "GepaOptimizer",
    "GepaTracker",
    "GuardedInstructionProposer",
    "InstructionLeakGuard",
    "LeakReport",
    "build_feedback_metric",
    "clean_instructions",
]
