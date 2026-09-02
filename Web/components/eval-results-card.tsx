"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { FlaskConical, HelpCircle, Check, X, ArrowRight, Sparkles } from "lucide-react"
import type { EvalReport } from "@/lib/api/client"

const CANDIDATE_LABELS: Record<string, string> = {
  original: "Original prompt",
  rewritten: "Rewrite",
  rewritten_fewshot: "Rewrite + examples",
  original_fewshot: "Original + examples",
  gepa: "Evolved (GEPA)",
}

const METRIC_LABELS: Record<string, string> = {
  exact: "exact match",
  contains: "contains match",
  llm_judge: "model judge",
}

const METRIC_HELP: Record<string, string> = {
  exact: "The answer must equal the expected output after trimming, lower-casing and removing surrounding punctuation.",
  contains: "The expected output must appear inside the answer (or the answer inside it) after normalization. Good for labels and short answers.",
  llm_judge: "The same model is asked whether the answer conveys the expected output. Used for free-text outputs.",
}

export const candidateLabel = (name: string) => CANDIDATE_LABELS[name] ?? name
export const metricLabel = (metric: string) => METRIC_LABELS[metric] ?? metric

const scoreTone = (score: number) => (score >= 80 ? "text-green-500" : score >= 50 ? "text-yellow-500" : "text-red-500")

function ScoreBar({ score, highlight }: { score: number | null; highlight: boolean }) {
  if (score == null) return <span className="text-xs text-muted-foreground">failed</span>
  return (
    <div className="flex items-center gap-2 min-w-[140px]">
      <div className="flex-1 h-1.5 rounded-full bg-muted">
        <div className={`h-1.5 rounded-full ${highlight ? "bg-orange-500" : "bg-muted-foreground/40"}`} style={{ width: `${score}%` }} />
      </div>
      <span className={`font-mono text-xs w-10 text-right ${highlight ? "font-semibold" : ""}`}>{Math.round(score)}%</span>
    </div>
  )
}

interface Props {
  report: EvalReport
}

export function EvalResultsCard({ report }: Props) {
  const [showBaseline, setShowBaseline] = useState(false)
  const baseline = report.baseline_score ?? 0
  const best = report.eval_score ?? 0
  const delta = best - baseline
  const rows = showBaseline ? report.baseline_results : report.results
  const passed = rows.filter((r) => r.passed).length

  return (
    <Card>
      <CardHeader>
        <CardTitle className="font-sans font-bold flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-secondary" />
          Eval Results
          <Badge variant="outline" className="ml-auto font-mono text-[10px] uppercase">
            {metricLabel(report.metric)}
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="w-3 h-3 ml-1 opacity-70" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs text-xs">{METRIC_HELP[report.metric] ?? report.metric}</p>
              </TooltipContent>
            </Tooltip>
          </Badge>
        </CardTitle>
        <CardDescription className="font-serif">
          {report.split?.strategy === "kfold"
            ? `Cross-validated: all ${report.dev_size} samples scored, each held out once across ${report.split.folds} folds. `
            : `${report.dev_size} held-out sample${report.dev_size === 1 ? "" : "s"} scored each candidate. `}
          {report.max_demos > 0
            ? report.split?.strategy === "kfold"
              ? `Few-shot examples were chosen per fold with DSPy's BootstrapFewShot, then refit on all samples for the returned prompt (up to ${report.max_demos}).`
              : `Few-shot examples were chosen from the other ${report.train_size} with DSPy's BootstrapFewShot (up to ${report.max_demos} per prompt).`
            : `The other ${report.train_size} were the training split the evolution learned from.`}
          {report.split?.stratified && report.split.dev_labels && (
            <>
              {" "}
              Split is class-balanced:{" "}
              {Object.entries(report.split.dev_labels)
                .map(([label, count]) => `${label} ×${count}`)
                .join(", ")}{" "}
              held out.
            </>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Before / after */}
        <div className="flex items-center justify-center gap-6 rounded-md border bg-muted/30 px-4 py-4">
          <div className="text-center">
            <div className={`font-sans font-bold text-3xl ${scoreTone(baseline)}`}>{Math.round(baseline)}%</div>
            <div className="text-xs text-muted-foreground">Original prompt</div>
          </div>
          <ArrowRight className="w-5 h-5 text-muted-foreground" />
          <div className="text-center">
            <div className={`font-sans font-bold text-3xl ${scoreTone(best)}`}>{Math.round(best)}%</div>
            <div className="text-xs text-muted-foreground">{candidateLabel(report.best)}</div>
          </div>
          <div className="text-center">
            <div className={`font-sans font-semibold text-xl ${delta > 0 ? "text-green-500" : delta < 0 ? "text-red-500" : "text-muted-foreground"}`}>
              {delta > 0 ? "+" : ""}
              {Math.round(delta)} pts
            </div>
            <div className="text-xs text-muted-foreground">{report.improved ? "measured gain" : "no measured gain"}</div>
          </div>
        </div>

        {!report.improved && (
          <p className="text-xs text-muted-foreground font-serif">
            {baseline >= 100
              ? "The original prompt already passed every held-out sample, so there was nothing to beat. Try a harder dataset or a stricter metric."
              : "No candidate beat the original on these samples. The returned prompt is the best-scoring candidate; more or more varied samples give the optimizer more to work with."}
          </p>
        )}

        {/* Candidate scoreboard */}
        <div className="space-y-2">
          <div className="text-xs font-medium font-sans">Candidates</div>
          <div className="space-y-1.5">
            {report.candidates.map((candidate) => {
              const isBest = candidate.name === report.best
              return (
                <div key={candidate.name} className={`flex items-center gap-3 rounded-md px-2 py-1.5 ${isBest ? "bg-orange-500/10" : ""}`}>
                  <div className="flex-1 min-w-0">
                    <span className="text-sm font-sans">{candidateLabel(candidate.name)}</span>
                    {candidate.name === "original" && <span className="ml-1 text-[10px] uppercase text-muted-foreground">baseline</span>}
                    {isBest && <span className="ml-1 text-[10px] uppercase text-orange-500">selected</span>}
                    {candidate.demo_count > 0 && (
                      <span className="block text-[11px] text-muted-foreground">
                        {candidate.demo_count} example{candidate.demo_count === 1 ? "" : "s"}
                        {candidate.bootstrapped_demos > 0 && `, ${candidate.bootstrapped_demos} verified by the model`}
                      </span>
                    )}
                    {candidate.error && <span className="block text-[11px] text-red-500 truncate">{candidate.error}</span>}
                  </div>
                  <ScoreBar score={candidate.score} highlight={isBest} />
                </div>
              )
            })}
          </div>
        </div>

        {/* Examples kept in the returned prompt */}
        {report.demos.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs font-medium font-sans">
                Examples in the returned prompt · {report.demos.length}
              </div>
              {report.demo_selection && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="outline" className="font-mono text-[10px] cursor-help">
                      {report.demo_selection.method === "coverage" ? "chosen for coverage" : "first to pass"}
                      <HelpCircle className="w-3 h-3 ml-1 opacity-70" />
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs text-xs">
                      {report.demo_selection.method === "coverage"
                        ? `BootstrapFewShot validated ${report.demo_selection.pool} examples the model could reproduce; the ${report.demo_selection.kept} that best span the training inputs (farthest-point sampling over ${report.demo_selection.embedding_model ?? "embedding"} vectors${report.demo_selection.label_balanced ? ", one per class first" : ""}) were kept.`
                        : report.demo_selection.reason
                          ? `Kept the first ${report.demo_selection.kept} of ${report.demo_selection.pool} validated examples. Coverage selection was skipped: ${report.demo_selection.reason}`
                          : `All ${report.demo_selection.pool} validated examples fit within the limit.`}
                    </p>
                  </TooltipContent>
                </Tooltip>
              )}
            </div>
            <div className="space-y-1.5">
              {report.demos.map((demo, index) => (
                <div key={index} className="rounded-md border px-3 py-2 text-xs space-y-1">
                  <div className="flex items-start gap-2">
                    <span className="font-serif flex-1 min-w-0">{demo.input}</span>
                    <span className="font-mono shrink-0 max-w-[40%] truncate" title={demo.output}>
                      → {demo.output}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-1 text-[11px] text-muted-foreground">
                    {demo.bootstrapped && (
                      <Badge variant="secondary" className="text-[10px] h-4 px-1">
                        <Sparkles className="w-2.5 h-2.5 mr-0.5" /> reproduced by the model
                      </Badge>
                    )}
                    {demo.covers && demo.covers.length > 0 && (
                      <span>
                        covers inputs like{" "}
                        {demo.covers.map((text, i) => (
                          <span key={i}>
                            <span className="font-serif text-foreground/80">&ldquo;{text.length > 48 ? `${text.slice(0, 48)}…` : text}&rdquo;</span>
                            {i < demo.covers!.length - 1 ? ", " : ""}
                          </span>
                        ))}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Per-sample results */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-medium font-sans">
              Held-out samples · {passed}/{rows.length} passed
            </div>
            {report.baseline_results.length > 0 && (
              <div className="flex gap-1">
                <Button size="sm" variant={showBaseline ? "ghost" : "secondary"} className="h-6 px-2 text-xs" onClick={() => setShowBaseline(false)}>
                  {candidateLabel(report.best)}
                </Button>
                <Button size="sm" variant={showBaseline ? "secondary" : "ghost"} className="h-6 px-2 text-xs" onClick={() => setShowBaseline(true)}>
                  Original
                </Button>
              </div>
            )}
          </div>
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-xs">
              <thead className="bg-muted/50 text-left">
                <tr>
                  <th className="p-2 font-medium w-8"></th>
                  <th className="p-2 font-medium">Input</th>
                  <th className="p-2 font-medium">Expected</th>
                  <th className="p-2 font-medium">Model answer</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={index} className="border-t align-top">
                    <td className="p-2">
                      {row.passed ? <Check className="w-3.5 h-3.5 text-green-500" /> : <X className="w-3.5 h-3.5 text-red-500" />}
                    </td>
                    <td className="p-2 font-serif whitespace-pre-wrap max-w-[220px]">{row.input}</td>
                    <td className="p-2 font-mono whitespace-pre-wrap max-w-[160px]">{row.expected}</td>
                    <td className="p-2 font-serif whitespace-pre-wrap max-w-[260px] text-muted-foreground">{row.actual || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p className="text-[11px] text-muted-foreground font-serif">
          Scores are for this dataset and model only. The optimized prompt is a plain-text rendering of the measured DSPy program; put your
          data where <code className="font-mono">{"{input}"}</code> appears.
        </p>
      </CardContent>
    </Card>
  )
}
