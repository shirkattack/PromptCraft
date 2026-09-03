export type TaskTypeId = "general" | "classification" | "generation" | "summarization" | "qa" | "code" | "translation" | "analysis" | "creative"

export const TASK_TYPE_HELP = {
  effect:
    "The rewrite is framed for that kind of work (the meta-prompt says it is optimizing a code / creative / analysis prompt and applies matching guidelines), synthetic data generation uses a template for that type, and sessions are grouped by it in Analytics. It does not change which model runs or how a dataset is scored.",
  auto: "Guesses the type from words in your prompt (write a function → code, translate → translation, summarize → summarization, classify → classification). Pick one explicitly if the guess is wrong.",
}

const RULES: [TaskTypeId, RegExp][] = [
  ["translation", /\btranslat/i],
  ["code", /\b(code|function|script|program|python|javascript|typescript|sql|regex|bug|refactor|api|class|method|unit test|compile)\b/i],
  ["summarization", /\b(summari[sz]e|summary|tl;?dr|condense|abstract)\b/i],
  ["classification", /\b(classif|categori[sz]e|label|tag|priority|sentiment|spam|triage|detect whether)\b/i],
  ["qa", /\b(answer|question|q&a|faq|what is|explain)\b/i],
  ["analysis", /\b(analy[sz]e|analysis|compare|evaluate|assess|review|critique|pros and cons)\b/i],
  ["creative", /\b(story|poem|song|lyrics|fiction|character|creative|screenplay|joke)\b/i],
  ["generation", /\b(write|rewrite|rephrase|paraphrase|draft|compose|generate|create|email|blog|post|article|essay|letter)\b/i],
]

/** Best-effort task type from the prompt text; "general" when nothing matches. */
export function detectTaskType(prompt: string): TaskTypeId {
  const text = prompt.trim()
  if (!text) return "general"
  for (const [id, pattern] of RULES) {
    if (pattern.test(text)) return id
  }
  return "general"
}
