## Findings

llama3.2 3B, fixed 198-task test split, three seeds, temperature 0, 512 tokens, Ollama chat API. Reflection model qwen3.6:27b (Q4_K_M, thinking off), budget 500 scored calls. Scores use the corrected scoring described under *Scoring corrections*.

- **Four few-shot examples are the only method that clearly beats `original`.** Random demos: +11.4 pp plus pass@1 on the bare prompt (95% CI +6.2 to +16.5) and +12.8 pp on the fixed prompt (+7.1 to +18.2). Coverage-selected demos: +11.6 and +12.6 pp. Every one of those intervals excludes zero. Random and coverage selection are indistinguishable: 56.4% vs 56.6% (bare) and 56.7% vs 56.6% (fixed).
- **GEPA alone gains little and is seed-sensitive.** Bare prompt: +5.9 pp (CI +0.7 to +11.3), seeds from 46.5% to 54.0%. Fixed prompt: -0.8 pp (CI -6.2 to +4.4), seeds from 40.9% to 46.5%. Its val scores rose more than its test scores (bare 40.0% to 53.9% mean on val; fixed 41.7% to 51.1%): val is the selection set and reads optimistic. Several returned prompts name specific training tasks; see the configuration review.
- **GEPA's instructions on top of demos add nothing.** `gepa_demos` scores 56.6% (bare) and 56.4% (fixed) against 56.6% for the same demos with the original instructions.
- **Seeds moved the GEPA rows by 7.5 pp (bare) and 5.6 pp (fixed)**; random demos moved by at most 3.1 pp; `original`, `one_line` and coverage demos are deterministic, so their three seeds are one run replayed.
- **The one-line format instruction makes no measurable difference**: -1.0 pp (CI -5.6 to +3.5). The -3.5 pp reported before the correction came from answers that extraction broke.
- **Cost.** A demo run is 2 to 5 minutes end to end; a GEPA run is about 24 minutes (median), most of it the 27B reflector.

Two caveats. These GEPA runs returned the candidate that was best on GEPA's fraction-of-asserts metric; since then the service returns the best on val pass@1, which on one seed would have picked 48.3% over 46.7% on val. And llama3.2 has seen MBPP during training; read the deltas, not the levels.

## Model ladder, rung 1: qwen3.5:4b

Same protocol, task model qwen3.5:4b (digest `2a654d98e6fb`, thinking off), bare prompt only. Five of the six methods ran, three seeds each; `gepa_demos` and the fixed-prompt pass were not run, the ladder was stopped after `gepa` seed 3.

- **Nothing beats `original` on this model.** It scores 65.7% plus (79.3% base), 21 points above llama3.2 with the same prompt. Every method's point estimate is below it, and every interval includes zero: one_line -4.0 pp (CI -8.6 to +0.5), random demos -3.2 pp (-7.9 to +2.0), coverage demos -5.1 pp (-10.6 to +1.0), GEPA -2.4 pp (-5.7 to +1.3). Before the correction to the intervals, three of these looked significant.
- **The demo effect does not carry over.** Four examples were worth about +12 pp on llama3.2 and nothing measurable here, with the point estimate negative. The 4B model already writes the function correctly from the bare instruction.
- **GEPA's val gain does not transfer.** Val went from 63.3% to 68.9% (seeds 68.3, 71.7, 66.7) while test went from 65.7% to 63.3% (62.6 to 64.6). All three returned prompts are specific to training tasks the reflector saw fail: seed 1 is a specification of `is_Monotonic` with worked examples, seed 2 discusses `angle_complex(0, 1j)` at length and ends with a `{task_input}` placeholder the reflector invented, seed 3 is a list of "critical domain-specific rules" for the Perrin sequence, string rotation, tuple concatenation and `is_not_prime`.
- **GEPA's prompts make this model hit the token cap.** Answers cut off at 512 tokens: 10 of 198 for `original`, 21, 14 and 33 for the three GEPA seeds. The added instructions ask for docstrings, worked reasoning and the asserts in the answer; a cut-off answer is a syntax error.
- **Cost.** A GEPA run is 78 minutes median here (72 to 84), three times the llama3.2 figure. Evaluating 198 tasks with four demos takes about 11 minutes; the 1-minute rows are cache hits on deterministic methods after their first seed.

## Scoring corrections (2026-09-11)

Three problems in the measurement were found and fixed; every stored run was re-scored.

- **Imports above an unfenced function were dropped.** When an answer had no code fence, extraction started at `def <entry_point>`, so `import math` written above it was lost and the code failed with a NameError. In one llama3.2 GEPA run, 8 of 19 "exception" results were correct code.
- **Junk after the code made working code a syntax error.** A stray closing fence, a garbled DSPy marker (`[/## completed ## ]]`, `[/]`) or a closing brace after a complete function failed to parse. 100 of 183 stored syntax errors were this; one `gepa_demos` seed lost 14 tasks to a trailing fence. Extraction now cuts such lines when that is what makes the code parse, as EvalPlus's sanitizer does; code broken inside the function stays a syntax error.
- **The intervals were too narrow.** The bootstrap resampled (seed, task) pairs as if three seeds were three independent test sets, which shrinks an interval by about √3 and, for deterministic methods, triple-counts one run. Tasks are now the resampling unit, with a task's difference averaged over seeds, and replayed seeds are marked *(identical)*.

Both extraction problems hit unfenced answers, which the original and GEPA prompts produce and demos (whose examples are fenced) do not: llama3.2 `original` gained 2 (bare) and 7 (fixed) tasks, GEPA runs 1 to 6, `gepa_demos` runs 0 to 14, random and coverage demo runs none. `scripts/rescore_results.py` replayed every raw answer from DSPy's cache (temperature 0; no model call, no cache miss in 48 runs), extracted it again and re-ran the sandbox where the code changed; the sandbox gives the same verdict for the same code, which the replayed seeds confirm task for task. Val scores and GEPA's own search were not redone: GEPA searched with the old extraction, so prompts that led to unfenced answers were scored lower than they should have been.

Checked and ruled out: Ollama never truncated a prompt (task models loaded with a 32,768-token context), and the sandbox's verdicts did not vary across replays.

## GEPA configuration review

How the runs used `dspy.GEPA` (DSPy 3.3.1, gepa 0.1.4, both current), and what that means for the results.

- **The reflection prompt caused the task-specific instructions.** GEPA's default reflection prompt tells the reflection model to "identify all niche and domain specific factual information about the task and include it in the instruction". That fits a dataset where every input is the same task. Here every sample is a different problem, the instruction is a template, and the advice produces rules about the training tasks that failed. The GEPA service now passes an `instruction_proposer` for code datasets: the same reflection step with a prompt that says the instruction is a template for unseen tasks, followed by a leak check (a task's function name, a quoted literal from an example, a six-word phrase copied from one with a rare word in it, or an invented `{placeholder}`). A proposal that fails is regenerated once, then dropped, which GEPA skips without spending metric calls. Label datasets keep GEPA's own prompt and are only checked for placeholders. The benchmark runner calls this `gepa_guard`; `gepa` keeps the protocol above.
- **Most of the budget went to the val set.** Every accepted candidate is scored on all 60 val tasks. Across the nine runs that was 70% to 87% of the 500 metric calls, leaving 13 to 27 iterations and 6 to 8 candidates per run. A smaller Pareto set, or a larger budget, buys more search.
- **Reflection sees three examples, often none of them failures.** With `reflection_minibatch_size=3` and `skip_perfect_score=True`, a minibatch of three solved tasks is skipped; on qwen3.5:4b that was 16 of 52 iterations. When it is not skipped the reflector often reads one or two failures, which invites a rule about those tasks. A larger minibatch shows more failures per proposal at the cost of more calls; the runner now takes `--minibatch`.
- **Acceptance is noisy.** A proposal is kept when its score on those three examples beats its parent's (strict improvement of the sum); with a fraction-of-asserts metric, three examples decide whether 60 val calls are spent.
- **Reflection ran at temperature 0.** Combined with DSPy's cache, the same parent and minibatch always yield the same proposal. DSPy's GEPA guidance suggests temperature 1.0 for the reflection model; changing it trades reproducibility for diversity.
- **Merge is on but does nothing.** `dspy.GEPA` defaults to `use_merge=True`; with a single predictor there is nothing to merge, and every run logged "No merge candidates found".
- **The token cap interacts with the prompt.** GEPA's instructions made answers longer; at 512 tokens a longer answer is more often a syntax error, so part of GEPA's loss on qwen3.5:4b is the cap. The runner now records truncated answers per task and takes `--max-tokens`; a result file made with different settings is never reused.
