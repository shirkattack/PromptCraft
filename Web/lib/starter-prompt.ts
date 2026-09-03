import type { TrainingSample } from "@/lib/api/client"

const normalize = (text: string) => text.trim().toLowerCase().replace(/\s+/g, " ").replace(/^[\s.!?:;,"'`*_]+|[\s.!?:;,"'`*_]+$/g, "")

/**
 * Write an instruction that fits a dataset, from its own samples.
 *
 * Label-style data (a small set of short outputs) gets a classification
 * instruction listing the labels; anything else gets a generic instruction
 * with one worked example. The result is a starting point for the
 * optimizer, not a finished prompt.
 */
export function starterPromptFor(samples: TrainingSample[]): string {
  const outputs = samples.map((s) => s.expected_output.trim()).filter(Boolean)
  if (outputs.length === 0) return "Given the input, produce the expected output."

  const distinct = new Map<string, string>()
  for (const output of outputs) {
    const key = normalize(output)
    if (!distinct.has(key)) distinct.set(key, output)
  }
  const short = outputs.every((o) => o.length <= 40)
  const isLabelSet = short && distinct.size > 1 && distinct.size <= Math.max(2, Math.floor(outputs.length / 2))

  if (isLabelSet) {
    const labels = Array.from(distinct.values())
    return `Classify the input as one of: ${labels.join(", ")}. Respond with the label only.`
  }

  const example = samples.find((s) => s.expected_output.trim())!
  const trim = (t: string, n: number) => (t.length > n ? `${t.slice(0, n).trim()}…` : t)
  return [
    "Given the input, write the expected output.",
    `For example, for the input "${trim(example.input_text, 160)}" the expected output is "${trim(example.expected_output, 200)}".`,
  ].join(" ")
}
