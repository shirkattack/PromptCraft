"use client"

import { useMemo, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { toast } from "@/hooks/use-toast"
import apiClient, { type EvalSampleResult, type TryResponse, type TryResult } from "@/lib/api/client"
import { PlayCircle, Loader2, Shuffle, Copy, Check, X, Minus } from "lucide-react"

const normalize = (text: string) => text.trim().toLowerCase().replace(/\s+/g, " ").replace(/^[\s.!?:;,"'`*_]+|[\s.!?:;,"'`*_]+$/g, "")

const matches = (expected: string, actual: string) => {
  const e = normalize(expected)
  const a = normalize(actual)
  return Boolean(e && a && (a === e || a.includes(e) || (a.length >= 3 && e.includes(a))))
}

const wordCount = (text: string) => (text.trim() ? text.trim().split(/\s+/).length : 0)

interface Props {
  sessionId: string
  hasOptimized: boolean
  /** Held-out samples from the last measured run, offered as ready-made inputs. */
  samples?: EvalSampleResult[]
}

function AnswerPane({ title, result, expected }: { title: string; result: TryResult | undefined; expected: string | null }) {
  const [copied, setCopied] = useState(false)
  if (!result) return null
  const ok = expected != null && result.output ? matches(expected, result.output) : null
  return (
    <div className="rounded-md border p-3 space-y-2 min-w-0">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-sans font-medium text-sm">{title}</span>
        <span className="text-xs text-muted-foreground">
          {result.elapsed_seconds}s · {wordCount(result.output)} words
        </span>
        {ok != null && (
          <Badge variant={ok ? "secondary" : "outline"} className={`ml-auto text-[10px] ${ok ? "text-green-600" : "text-red-500"}`}>
            {ok ? <Check className="w-3 h-3 mr-1" /> : <X className="w-3 h-3 mr-1" />}
            {ok ? "matches expected" : "does not match"}
          </Badge>
        )}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={async () => {
                await navigator.clipboard.writeText(result.output)
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              }}
              disabled={!result.output}
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p className="text-xs">Copy answer</p>
          </TooltipContent>
        </Tooltip>
      </div>
      {result.error ? (
        <p className="text-xs text-red-500 font-mono break-words">{result.error}</p>
      ) : (
        <div className="text-sm font-serif whitespace-pre-wrap max-h-72 overflow-y-auto">{result.output || <span className="text-muted-foreground">(empty answer)</span>}</div>
      )}
      <details className="text-[11px] text-muted-foreground">
        <summary className="cursor-pointer">What was sent</summary>
        <pre className="mt-1 whitespace-pre-wrap font-mono max-h-48 overflow-y-auto rounded bg-muted/50 p-2">{result.prompt_sent}</pre>
      </details>
    </div>
  )
}

export function TryItCard({ sessionId, hasOptimized, samples }: Props) {
  const [input, setInput] = useState("")
  const [expected, setExpected] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [response, setResponse] = useState<TryResponse | null>(null)

  const usable = useMemo(() => (samples ?? []).filter((s) => s.input.trim()), [samples])

  const pickSample = () => {
    if (!usable.length) return
    const sample = usable[Math.floor(Math.random() * usable.length)]
    setInput(sample.input)
    setExpected(sample.expected)
    setResponse(null)
  }

  const run = async () => {
    if (!input.trim()) return
    setRunning(true)
    try {
      const result = await apiClient.trySession(sessionId, input.trim())
      setResponse(result)
    } catch (error) {
      toast({ title: "Could not run the prompts", description: error instanceof Error ? error.message : "Unknown error", variant: "destructive", duration: 4000 })
    } finally {
      setRunning(false)
    }
  }

  const original = response?.results.find((r) => r.label === "original")
  const optimized = response?.results.find((r) => r.label === "optimized")
  const verdict = (() => {
    if (!response || expected == null) return null
    const o = original?.output ? matches(expected, original.output) : false
    const p = optimized?.output ? matches(expected, optimized.output) : false
    if (o === p) return o ? "Both answered correctly." : "Neither matched the expected output."
    return p ? "The optimized prompt got it right; the original did not." : "The original got it right; the optimized prompt did not."
  })()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="font-sans font-bold flex items-center gap-2">
          <PlayCircle className="w-5 h-5 text-secondary" />
          Try it
        </CardTitle>
        <CardDescription className="font-serif">
          Run {hasOptimized ? "the original and the optimized prompt" : "the prompt"} on an input of your choosing, with this session&apos;s model.
          {hasOptimized ? " Same input, same model; only the prompt differs." : ""}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            setExpected(null)
          }}
          placeholder="Paste the input the prompt should handle, e.g. a support ticket, a paragraph to rewrite, a question..."
          className="text-sm min-h-[80px]"
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={run} disabled={running || !input.trim()} className="font-sans">
            {running ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <PlayCircle className="w-4 h-4 mr-1" />}
            {running ? "Running..." : hasOptimized ? "Run both" : "Run"}
          </Button>
          {usable.length > 0 && (
            <Button size="sm" variant="outline" onClick={pickSample} className="font-sans">
              <Shuffle className="w-4 h-4 mr-1" />
              Use a held-out sample
            </Button>
          )}
          {expected != null && (
            <span className="text-xs text-muted-foreground">
              expected: <span className="font-mono text-foreground">{expected.length > 80 ? `${expected.slice(0, 80)}…` : expected}</span>
            </span>
          )}
        </div>

        {response && (
          <>
            <div className={`grid gap-3 ${optimized ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
              <AnswerPane title="Original prompt" result={original} expected={expected} />
              {optimized && <AnswerPane title="Optimized prompt" result={optimized} expected={expected} />}
            </div>
            {verdict && (
              <p className="text-xs font-serif text-muted-foreground flex items-center gap-1">
                <Minus className="w-3 h-3" /> {verdict} Match uses the same contains rule as the evaluation.
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
