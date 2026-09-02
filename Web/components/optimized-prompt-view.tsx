"use client"

import { useMemo, useState, type ReactNode } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { diffWords } from "diff"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { HelpCircle } from "lucide-react"
import type { OptimizeResponse } from "@/lib/api/client"

type ViewMode = "rendered" | "raw" | "compare"

export interface PromptStats {
  words: number
  chars: number
  tokens: number
  sections: number
  listItems: number
  examples: number
}

// Cheap, model-agnostic estimates. Token count uses the ~4 chars/token rule of
// thumb; the real number depends on the tokenizer and is not worth a request.
export function promptStats(text: string): PromptStats {
  const trimmed = text.trim()
  return {
    words: trimmed ? trimmed.split(/\s+/).length : 0,
    chars: trimmed.length,
    tokens: Math.ceil(trimmed.length / 4),
    sections: (trimmed.match(/^#{1,6}\s+\S/gm) ?? []).length,
    listItems: (trimmed.match(/^\s*(?:[-*+]|\d+[.)])\s+\S/gm) ?? []).length,
    examples: (trimmed.match(/\bexamples?\b/gi) ?? []).length,
  }
}

interface DetectedChange {
  label: string
  detail: string
}

// Observations about what the rewrite added, phrased as facts about the text
// rather than promises about model behaviour.
export function detectChanges(original: string, optimized: string): DetectedChange[] {
  const before = promptStats(original)
  const after = promptStats(optimized)
  const changes: DetectedChange[] = []

  if (after.sections > before.sections) {
    changes.push({ label: "Structured into sections", detail: `${after.sections} heading${after.sections === 1 ? "" : "s"}` })
  }
  if (after.listItems > before.listItems) {
    changes.push({ label: "Itemised instructions", detail: `${after.listItems} list item${after.listItems === 1 ? "" : "s"}` })
  }
  if (after.examples > before.examples) {
    changes.push({ label: "Mentions examples", detail: "gives the model a pattern to follow" })
  }
  const formatWords = /\b(format|output|respond with|return|json|markdown|bullet)/i
  if (formatWords.test(optimized) && !formatWords.test(original)) {
    changes.push({ label: "Specifies output format", detail: "reduces variance between runs" })
  }
  const constraintWords = /\b(must|should|do not|don't|avoid|only|limit|at most|no more than)\b/i
  if (constraintWords.test(optimized) && !constraintWords.test(original)) {
    changes.push({ label: "Adds explicit constraints", detail: "tells the model what not to do" })
  }
  const roleWords = /\b(you are|act as|as an? (expert|senior|professional))/i
  if (roleWords.test(optimized) && !roleWords.test(original)) {
    changes.push({ label: "Assigns a role", detail: "frames the expertise expected" })
  }
  if (before.words > 0) {
    const ratio = after.words / before.words
    if (ratio >= 1.2) changes.push({ label: `${ratio.toFixed(1)}× longer`, detail: `${before.words} → ${after.words} words` })
    else if (ratio <= 0.8) changes.push({ label: `${(1 / ratio).toFixed(1)}× shorter`, detail: `${before.words} → ${after.words} words` })
  }
  return changes
}

const markdownComponents = {
  h1: (p: any) => <h1 className="font-sans text-lg font-bold mt-4 mb-2 first:mt-0" {...p} />,
  h2: (p: any) => <h2 className="font-sans text-base font-bold mt-4 mb-2 first:mt-0" {...p} />,
  h3: (p: any) => <h3 className="font-sans text-sm font-semibold mt-3 mb-1.5 first:mt-0" {...p} />,
  h4: (p: any) => <h4 className="font-sans text-sm font-semibold mt-3 mb-1 first:mt-0" {...p} />,
  p: (p: any) => <p className="font-serif text-sm leading-relaxed mb-3 last:mb-0" {...p} />,
  ul: (p: any) => <ul className="list-disc pl-5 mb-3 space-y-1 font-serif text-sm" {...p} />,
  ol: (p: any) => <ol className="list-decimal pl-5 mb-3 space-y-1 font-serif text-sm" {...p} />,
  li: (p: any) => <li className="leading-relaxed" {...p} />,
  blockquote: (p: any) => <blockquote className="border-l-2 border-orange-500/60 pl-3 italic text-muted-foreground mb-3" {...p} />,
  strong: (p: any) => <strong className="font-semibold text-foreground" {...p} />,
  a: (p: any) => <a className="underline underline-offset-2" target="_blank" rel="noreferrer" {...p} />,
  hr: () => <hr className="my-4 border-border" />,
  table: (p: any) => (
    <div className="overflow-x-auto mb-3">
      <table className="w-full text-xs border-collapse" {...p} />
    </div>
  ),
  th: (p: any) => <th className="border border-border px-2 py-1 text-left font-sans font-semibold bg-muted/50" {...p} />,
  td: (p: any) => <td className="border border-border px-2 py-1 align-top" {...p} />,
  code: ({ className, children, ...p }: any) => {
    const block = typeof className === "string" && className.startsWith("language-")
    return block ? (
      <code className={`block font-mono text-xs ${className}`} {...p}>{children}</code>
    ) : (
      <code className="font-mono text-xs rounded bg-muted px-1 py-0.5" {...p}>{children}</code>
    )
  },
  pre: (p: any) => <pre className="rounded-md bg-muted p-3 mb-3 overflow-x-auto" {...p} />,
}

function Stat({ label, value, delta, hint }: { label: string; value: string | number; delta?: number; hint?: string }) {
  const body = (
    <div className="flex flex-col items-start">
      <span className="text-[11px] uppercase tracking-wide text-muted-foreground font-sans">{label}</span>
      <span className="font-sans font-semibold text-sm">
        {value}
        {delta !== undefined && delta !== 0 && (
          <span className={`ml-1 text-xs font-normal ${delta > 0 ? "text-green-500" : "text-amber-500"}`}>
            {delta > 0 ? "+" : ""}{delta}
          </span>
        )}
      </span>
    </div>
  )
  if (!hint) return body
  return (
    <Tooltip>
      <TooltipTrigger className="cursor-help text-left">{body}</TooltipTrigger>
      <TooltipContent><p className="max-w-xs text-xs">{hint}</p></TooltipContent>
    </Tooltip>
  )
}

interface Props {
  original: string
  optimized: string
  details: OptimizeResponse["optimization_details"] | null
  methodLabel: string
  actions?: ReactNode
}

export function OptimizedPromptView({ original, optimized, details, methodLabel, actions }: Props) {
  const [mode, setMode] = useState<ViewMode>("rendered")
  const before = useMemo(() => promptStats(original), [original])
  const after = useMemo(() => promptStats(optimized), [optimized])
  const changes = useMemo(() => detectChanges(original, optimized), [original, optimized])
  const diff = useMemo(() => (mode === "compare" ? diffWords(original, optimized) : []), [mode, original, optimized])

  const breakdown = Array.isArray(details?.metadata.score_breakdown)
    ? (details!.metadata.score_breakdown as { label: string; points: number; applied: boolean }[])
    : null

  return (
    <div className="space-y-4">
      {/* Stats strip */}
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3 rounded-md border bg-muted/30 px-4 py-3">
        <Stat label="Words" value={after.words} delta={after.words - before.words} />
        <Stat label="≈ Tokens" value={after.tokens} delta={after.tokens - before.tokens} hint="Estimated at ~4 characters per token; the exact count depends on the model's tokenizer." />
        <Stat label="Sections" value={after.sections} delta={after.sections - before.sections} />
        <Stat label="List items" value={after.listItems} delta={after.listItems - before.listItems} />
        {details && (
          <>
            <div className="ml-auto flex items-center gap-2">
              <Badge variant="secondary" className="font-sans text-xs">{methodLabel}</Badge>
              <span className="text-xs text-muted-foreground font-sans">{details.processing_time.toFixed(1)}s</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge className="font-sans cursor-help">
                    {Math.round(details.improvement_score)}/100
                    <HelpCircle className="w-3 h-3 ml-1 opacity-70" />
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="max-w-xs text-xs space-y-1">
                    <p className="font-medium">Heuristic score — structural rubric, not a measured gain.</p>
                    {breakdown ? (
                      <ul className="space-y-0.5">
                        {breakdown.map((item) => (
                          <li key={item.label} className={`flex justify-between gap-3 ${item.applied ? "" : "text-muted-foreground line-through"}`}>
                            <span>{item.label}</span>
                            <span className="font-mono">+{item.points}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>Base 50, then +20 for a 1.2–3× length increase, +10 for markdown formatting, +10 for mentioning examples or a format, +10 for more lines. 0 if nothing changed.</p>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            </div>
          </>
        )}
      </div>

      {/* View switch */}
      <div className="flex items-center gap-1">
        {(["rendered", "raw", "compare"] as ViewMode[]).map((m) => (
          <Button key={m} size="sm" variant={mode === m ? "secondary" : "ghost"} className="font-sans h-7 px-3 text-xs capitalize" onClick={() => setMode(m)}>
            {m}
          </Button>
        ))}
      </div>

      {/* Main pane */}
      {mode === "rendered" && (
        <div className="min-h-[24rem] rounded-md border border-secondary/30 bg-secondary/5 p-5">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{optimized}</ReactMarkdown>
        </div>
      )}
      {mode === "raw" && (
        <pre className="min-h-[24rem] rounded-md border bg-muted/40 p-5 font-mono text-xs whitespace-pre-wrap break-words">{optimized}</pre>
      )}
      {mode === "compare" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <h4 className="font-sans font-semibold text-sm">Original</h4>
            <div className="min-h-[16rem] rounded-md bg-muted p-4 font-serif text-sm whitespace-pre-wrap leading-relaxed">
              {diff.filter((part) => !part.added).map((part, i) => (
                <span key={i} className={part.removed ? "bg-red-500/20 line-through decoration-red-500/60" : ""}>{part.value}</span>
              ))}
            </div>
          </div>
          <div className="space-y-2">
            <h4 className="font-sans font-semibold text-sm">Optimized</h4>
            <div className="min-h-[16rem] rounded-md border border-secondary/30 bg-secondary/5 p-4 font-serif text-sm whitespace-pre-wrap leading-relaxed">
              {diff.filter((part) => !part.removed).map((part, i) => (
                <span key={i} className={part.added ? "bg-green-500/20" : ""}>{part.value}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* What changed */}
      {changes.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium font-sans text-muted-foreground">Detected changes</p>
          <div className="flex flex-wrap gap-2">
            {changes.map((c) => (
              <Tooltip key={c.label}>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="font-sans text-xs cursor-help">{c.label}</Badge>
                </TooltipTrigger>
                <TooltipContent><p className="text-xs">{c.detail}</p></TooltipContent>
              </Tooltip>
            ))}
          </div>
        </div>
      )}

      {actions && <div className="flex gap-2 pt-1">{actions}</div>}
    </div>
  )
}
