## Findings

llama3.2 3B, fixed 198-task test split, three seeds, temperature 0, 512 tokens, Ollama chat API. Reflection model qwen3.6:27b (Q4_K_M, thinking off), budget 500 scored calls.

- **Four few-shot examples are the only method that clearly beats `original`.** Random demos: +12.5 pp plus pass@1 on the bare prompt (95% CI +8.9 to +16.0) and +16.3 pp on the fixed prompt (+12.8 to +20.0). Coverage-selected demos: +12.6 and +16.2 pp. Every one of those intervals excludes zero. Random and coverage selection are indistinguishable: 56.4% vs 56.6% (bare) and 56.7% vs 56.6% (fixed).
- **GEPA alone gains little and is seed-sensitive.** Bare prompt: +5.1 pp (CI +1.2 to +8.8), seeds from 46.0% to 51.5%. Fixed prompt: +1.0 pp (CI -2.7 to +4.9), seeds from 38.9% to 45.5%, two of three below `original`. Its val scores rose more than its test scores (bare 40.0% to 53.9% mean on val against +5.1 pp on test; fixed 41.7% to 51.1% against +1.0 pp): val is the selection set and reads optimistic. One run returned a prompt that mentioned a specific training task, which GEPA's reflection does not guard against.
- **GEPA's instructions on top of demos add nothing.** `gepa_demos` scores 53.2% (bare) and 56.4% (fixed) against 56.6% for the same demos with the original instructions.
- **Seeds moved the GEPA rows by 5.5 pp (bare) and 6.6 pp (fixed)**; random demos moved by at most 3.1 pp; coverage demos are deterministic.
- **The one-line format instruction hurts this model**: -3.5 pp (CI -6.4 to -0.8). With demos in front, the instruction line stops mattering: coverage demos land on the same 56.6% under both prompts.
- **Cost.** A demo run is 2 to 5 minutes end to end; a GEPA run is about 24 minutes (median), most of it the 27B reflector.

Two caveats. GEPA returns the candidate that is best on its own fraction-of-asserts metric, which on one seed was not the best candidate on val pass@1 (46.7% returned, 48.3% available). And llama3.2 has seen MBPP during training; read the deltas, not the levels.

## Model ladder, rung 1: qwen3.5:4b

Same protocol, task model qwen3.5:4b (digest `2a654d98e6fb`, thinking off), bare prompt only. Five of the six methods ran, three seeds each; `gepa_demos` and the fixed-prompt pass were not run, the ladder was stopped after `gepa` seed 3.

- **Nothing beats `original` on this model.** It scores 65.7% plus (79.3% base), 22 points above llama3.2 with the same prompt. Every method lands below it: one_line -4.0 pp (CI -6.6 to -1.5), random demos -3.2 pp (-6.7 to +0.0), coverage demos -5.1 pp (-8.4 to -1.9), GEPA -2.7 pp (-5.4 to +0.0). The two demo intervals that touch zero do so at the edge; none crosses into a gain.
- **The demo effect flips sign with the model.** Four examples were worth +12.5 pp on llama3.2 and cost 3 to 5 pp here. The 4B model already writes the function correctly from the bare instruction; the examples appear to pull its output toward the example's shape instead.
- **GEPA's val gain does not transfer.** Val went from 63.3% to 68.9% (seeds 68.3, 71.7, 66.7) while test went from 65.7% to 63.0% (61.6 to 64.6). All three returned prompts are specific to the training tasks the reflector saw fail: seed 1 is a specification of `is_Monotonic` with worked examples, seed 2 discusses `angle_complex(0, 1j)` at length and ends with a literal `{task_input}` placeholder that the reflector invented, seed 3 is a list of "critical domain-specific rules" for the Perrin sequence, string rotation, tuple concatenation and `is_not_prime`. This is the template leak noted on llama3.2, now on every seed; the val set contains some of those tasks, the test set does not.
- **Cost.** A GEPA run is 78 minutes median here (72 to 84), three times the llama3.2 figure, since the 4B model is slower per call and the reflector's prompts get longer. Evaluating 198 tasks with four demos takes about 11 minutes; the 1-minute rows are cache hits on deterministic methods after their first seed.

What this rung changes: the recommendation "add four examples" is model-specific, and the app should measure the original against each candidate on held-out data before choosing, which it does. GEPA's reflection step needs a guard against task-specific instructions before it is worth running on a model that already solves the bare prompt.
