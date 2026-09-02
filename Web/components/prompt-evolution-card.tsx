"use client"

import { useMemo, useState } from "react"
import { diffWords } from "diff"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Dna, GitBranch, MessageSquareWarning, ChevronDown, ChevronRight, Trophy } from "lucide-react"
import type { GepaCandidate, GepaReport } from "@/lib/api/client"

const scoreTone = (score: number | null) => {
  if (score == null) return "text-muted-foreground"
  return score >= 80 ? "text-green-500" : score >= 50 ? "text-yellow-500" : "text-red-500"
}

function InstructionDiff({ from, to }: { from: string; to: string }) {
  const parts = useMemo(() => diffWords(from, to), [from, to])
  return (
    <div className="rounded-md bg-muted/50 p-3 text-xs font-serif leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto">
      {parts.map((part, index) =>
        part.added ? (
          <mark key={index} className="bg-green-500/20 text-foreground rounded-sm px-0.5">
            {part.value}
          </mark>
        ) : part.removed ? (
          <del key={index} className="bg-red-500/15 text-muted-foreground rounded-sm px-0.5">
            {part.value}
          </del>
        ) : (
          <span key={index}>{part.value}</span>
        ),
      )}
    </div>
  )
}

function CandidateRow({
  candidate,
  parent,
  isBest,
  isSeed,
  defaultOpen,
}: {
  candidate: GepaCandidate
  parent: GepaCandidate | null
  isBest: boolean
  isSeed: boolean
  defaultOpen: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const label = isSeed ? "Original prompt" : `Candidate ${candidate.index}`

  return (
    <div className={`rounded-md border ${isBest ? "border-orange-500/60 bg-orange-500/5" : "border-border"}`}>
      <button type="button" onClick={() => setOpen((o) => !o)} className="w-full flex items-center gap-3 px-3 py-2 text-left">
        {open ? <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" /> : <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />}
        <div className="flex items-center gap-1.5 shrink-0" style={{ marginLeft: `${candidate.generation * 12}px` }}>
          {candidate.generation > 0 && <GitBranch className="w-3.5 h-3.5 text-muted-foreground" />}
          <span className="font-sans text-sm font-medium">{label}</span>
        </div>
        <span className="text-[11px] text-muted-foreground">
          gen {candidate.generation}
          {parent ? ` · from ${parent.index === 0 ? "original" : `candidate ${parent.index}`}` : ""}
          {candidate.iteration != null ? ` · iteration ${candidate.iteration}` : ""}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {isBest && (
            <Badge className="font-sans text-[10px] bg-orange-500 hover:bg-orange-500">
              <Trophy className="w-3 h-3 mr-1" /> selected
            </Badge>
          )}
          <span className={`font-mono text-sm font-semibold ${scoreTone(candidate.score)}`}>
            {candidate.score == null ? "—" : `${Math.round(candidate.score)}%`}
          </span>
        </div>
      </button>

      {open && (
        <div className="px-3 pb-3 space-y-3">
          {candidate.feedback.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center gap-1 text-xs font-medium font-sans">
                <MessageSquareWarning className="w-3.5 h-3.5 text-amber-500" />
                Feedback the reflection read before proposing this
              </div>
              <ul className="space-y-1">
                {candidate.feedback.map((item, index) => (
                  <li key={index} className="text-xs font-serif text-muted-foreground border-l-2 border-amber-500/50 pl-2">
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="space-y-1">
            <div className="text-xs font-medium font-sans">{parent ? "Instructions, changes from parent highlighted" : "Instructions"}</div>
            {parent ? (
              <InstructionDiff from={parent.instructions} to={candidate.instructions} />
            ) : (
              <div className="rounded-md bg-muted/50 p-3 text-xs font-serif whitespace-pre-wrap">{candidate.instructions}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

interface Props {
  report: GepaReport
}

export function PromptEvolutionCard({ report }: Props) {
  const byIndex = useMemo(() => new Map(report.timeline.map((c) => [c.index, c])), [report.timeline])
  const bestIndex = report.improved ? report.best_index : 0
  const accepted = report.timeline.length - 1

  return (
    <Card>
      <CardHeader>
        <CardTitle className="font-sans font-bold flex items-center gap-2">
          <Dna className="w-5 h-5 text-secondary" />
          Prompt Evolution
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="outline" className="ml-auto font-mono text-[10px] uppercase cursor-help">
                GEPA
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              <p className="max-w-xs text-xs">
                Reflective prompt evolution: the prompt runs on training samples, the metric writes feedback for each miss, a reflection
                model rewrites the instructions to address it, and candidates that win on different samples are kept and merged.
              </p>
            </TooltipContent>
          </Tooltip>
        </CardTitle>
        <CardDescription className="font-serif">
          {report.iterations} iteration{report.iterations === 1 ? "" : "s"}, {accepted} accepted candidate{accepted === 1 ? "" : "s"},{" "}
          {report.metric_calls ?? report.budget} scored calls of a {report.budget} budget, {report.elapsed_seconds}s.
          {report.reflection_model && ` Reflection by ${report.reflection_model.replace(/^ollama\//, "")}.`}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-center gap-6 rounded-md border bg-muted/30 px-4 py-3">
          <div className="text-center">
            <div className={`font-sans font-bold text-2xl ${scoreTone(report.baseline_score)}`}>{Math.round(report.baseline_score)}%</div>
            <div className="text-xs text-muted-foreground">Original</div>
          </div>
          <span className="text-muted-foreground">→</span>
          <div className="text-center">
            <div className={`font-sans font-bold text-2xl ${scoreTone(report.final_score)}`}>{Math.round(report.final_score)}%</div>
            <div className="text-xs text-muted-foreground">Evolved</div>
          </div>
          {!report.improved && (
            <p className="text-xs text-muted-foreground font-serif max-w-[220px]">
              The evolved instructions did not beat the original on held-out samples, so the original is kept. A larger budget or more
              samples gives the reflection more to work with.
            </p>
          )}
        </div>

        <div className="space-y-2">
          {report.timeline.map((candidate) => (
            <CandidateRow
              key={candidate.index}
              candidate={candidate}
              parent={candidate.parent == null ? null : (byIndex.get(candidate.parent) ?? null)}
              isBest={candidate.index === bestIndex}
              isSeed={candidate.index === 0}
              defaultOpen={candidate.index === bestIndex && candidate.index !== 0}
            />
          ))}
        </div>

        {report.timeline.length <= 1 && (
          <p className="text-xs text-muted-foreground font-serif">No candidate was accepted within the budget.</p>
        )}

        <p className="text-[11px] text-muted-foreground font-serif">
          Scores are on the held-out split. Candidate scores come from GEPA&apos;s own validation pass; the Evolved figure above is a fresh
          run of the final instructions.{" "}
          <Button variant="link" size="sm" className="h-auto p-0 text-[11px]" asChild>
            <a href="https://arxiv.org/abs/2507.19457" target="_blank" rel="noreferrer">
              GEPA paper
            </a>
          </Button>
        </p>
      </CardContent>
    </Card>
  )
}
