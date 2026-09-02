"use client"

import { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { useProviders, useOllamaHealth, useSessionActions, usePerformanceMetrics, useOptimizationMethods, useTrainingStats, useTrainingDatasets } from "@/lib/api/hooks"
import type { AIModel, EvalMetric, JobProgress, OptimizationMethod, OptimizationMethodInfo, OptimizeOptions, OptimizeResponse, OutputFormat, TargetLength } from "@/lib/api/client"
import { EvalResultsCard, candidateLabel } from "@/components/eval-results-card"
import { PromptEvolutionCard } from "@/components/prompt-evolution-card"
import { Checkbox } from "@/components/ui/checkbox"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChartTooltip } from "@/components/ui/chart"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { toast } from "@/hooks/use-toast"
import { OptimizedPromptView } from "@/components/optimized-prompt-view"
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import {
  Play,
  Pause,
  RotateCcw,
  Download,
  Settings,
  Zap,
  Info,
  DollarSign,
  Gauge,
  TrendingUp,
  Clock,
  Database,
  CheckCircle,
  XCircle,
  ChevronDown,
  HardDrive,
  Loader2,
  Copy,
  Share2,
  Upload,
  HelpCircle,
  Keyboard,
  Sparkles,
  BarChart3,
  AlertTriangle,
} from "lucide-react"

const taskTypes = [
  { id: "auto", name: "Auto-detect", icon: "🪄" },
  { id: "classification", name: "Classification", icon: "📊" },
  { id: "generation", name: "Generation", icon: "✨" },
  { id: "summarization", name: "Summarization", icon: "📝" },
  { id: "qa", name: "Q&A", icon: "❓" },
  { id: "code", name: "Code", icon: "💻" },
  { id: "translation", name: "Translation", icon: "🌐" },
]

const DRAFT_KEY = "promptcraft:draft"

const CHART_COLORS = ["#f97316", "#fb923c", "#fdba74", "#fed7aa", "#ffedd5"]

const taskTypeLabel = (id: string) => taskTypes.find((t) => t.id === id)?.name ?? id.charAt(0).toUpperCase() + id.slice(1)

type MethodInfo = OptimizationMethodInfo

// Mirrors GET /sessions/optimization-methods so the selector renders before the request lands.
const FALLBACK_METHODS: MethodInfo[] = [
  {
    id: "meta_prompt",
    name: "Meta-Prompt",
    description: "One structured rewrite guided by a prompt-engineering rubric.",
    how_it_works: "A single dspy.Predict call with a meta-prompt asking for clarity, context, structure, examples and constraints.",
    best_for: "Most prompts. The fastest structured option and a good default.",
    returns_reasoning: false,
    relative_speed: "fast",
    recommended_for: [],
  },
  {
    id: "dspy",
    name: "DSPy Chain-of-Thought",
    description: "The model reasons about the prompt first, then rewrites it.",
    how_it_works: "dspy.ChainOfThought over a PromptRewrite signature; the reasoning is shown in Optimization Insights.",
    best_for: "Ambiguous or multi-step asks, or when you want to see why changes were made.",
    returns_reasoning: true,
    relative_speed: "slower",
    recommended_for: [],
  },
  {
    id: "gepa",
    name: "GEPA (evolve from feedback)",
    description: "Evolves the instructions from written feedback on what the prompt gets wrong. Needs a dataset.",
    how_it_works: "dspy.teleprompt.GEPA: the metric writes feedback for each miss, a reflection model rewrites the instructions to address it, and candidates that win on different samples are kept on a Pareto front.",
    best_for: "Prompts with a dataset of expected outputs, when you want a measured gain and a visible reason for every edit.",
    returns_reasoning: true,
    relative_speed: "slowest",
    recommended_for: ["classification", "extraction", "qa"],
    requires_dataset: true,
  },
  {
    id: "simple",
    name: "Simple",
    description: "A plain completion asked to improve the prompt.",
    how_it_works: "A short 'improve this prompt' instruction with no DSPy structure or rubric.",
    best_for: "A quick baseline, or comparing against the structured methods.",
    returns_reasoning: false,
    relative_speed: "fastest",
    recommended_for: [],
  },
]

const methodInfo = (id: string, methods?: MethodInfo[] | null) =>
  (methods ?? FALLBACK_METHODS).find((m) => m.id === id)

function MethodExplainer({ method }: { method?: MethodInfo }) {
  if (!method) return <p className="text-xs">How the rewrite is produced.</p>
  return (
    <div className="max-w-sm text-xs space-y-1.5">
      <p className="font-medium font-sans">{method.name}</p>
      {method.how_it_works && <p>{method.how_it_works}</p>}
      {method.best_for && (
        <p><span className="font-medium">Best for:</span> {method.best_for}</p>
      )}
      <p className="text-muted-foreground">
        {method.relative_speed && <span className="capitalize">{method.relative_speed}</span>}
        {method.relative_speed && " · "}
        {method.returns_reasoning ? "shows the model's reasoning" : "no reasoning trace"}
      </p>
    </div>
  )
}

const formatContext = (tokens: number) => (tokens >= 1000 ? `${Math.round(tokens / 1000)}K tokens` : `${tokens} tokens`)
const formatCost = (model: AIModel) => (model.is_free ? "Free" : `$${model.cost_per_1k_tokens}/1K tokens`)
const formatSpeed = (rating: number) => `${rating}/5 speed`
const formatSize = (bytes: number | null) => {
  if (bytes == null) return null
  const gb = bytes / 1_073_741_824
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(bytes / 1_048_576)} MB`
}
// "3.2B · Q4_K_M · 1.9 GB" — whatever the runtime reported, nothing guessed.
const modelSpecLine = (model: AIModel) =>
  [model.parameter_size, model.quantization, formatSize(model.size_bytes)].filter(Boolean).join(" · ")

const methodLabel = (id: string, methods?: MethodInfo[] | null) =>
  (methods ?? FALLBACK_METHODS).find((m) => m.id === id)?.name ?? id

interface ProviderView {
  id: string
  name: string
  models: AIModel[]
  available: boolean
  unavailableReason: string | null
  icon: string
}

interface ApiStatus {
  connected: boolean
  lastTest: string
}

const getCurrentModel = (providerData: Record<string, ProviderView>, selectedProvider: string, selectedModel: string) => {
  return providerData[selectedProvider]?.models.find((model) => model.id === selectedModel)
}

export function OptimizationDashboard() {
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [optimizationProgress, setOptimizationProgress] = useState(0)
  const [jobProgress, setJobProgress] = useState<JobProgress | null>(null)
  const [jobStartedAt, setJobStartedAt] = useState<number | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [selectedProvider, setSelectedProvider] = useState("Ollama")
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedTaskType, setSelectedTaskType] = useState("auto")
  const [originalPrompt, setOriginalPrompt] = useState("")
  const [optimizedPrompt, setOptimizedPrompt] = useState("")
  const [selectedMethod, setSelectedMethod] = useState<OptimizationMethod>("meta_prompt")
  const [lastResult, setLastResult] = useState<OptimizeResponse["optimization_details"] | null>(null)
  const [advanced, setAdvanced] = useState<Required<Omit<OptimizeOptions, "dataset_id" | "eval_metric" | "max_demos" | "gepa_budget" | "reflection_model">>>({
    temperature: 0.7,
    max_tokens: 2048,
    output_format: "auto",
    target_length: "auto",
    preserve_wording: false,
  })
  // Dataset-driven evaluation. NONE keeps the structural heuristic score.
  const NO_DATASET = "none"
  const [selectedDataset, setSelectedDataset] = useState<string>(NO_DATASET)
  const [evalMetric, setEvalMetric] = useState<EvalMetric>("auto")
  const [maxDemos, setMaxDemos] = useState(4)
  const [gepaBudget, setGepaBudget] = useState(60)
  const [reflectionModel, setReflectionModel] = useState<string>("same")

  // API hooks
  const { data: providers, loading: providersLoading, error: providersError } = useProviders()
  const { data: ollamaHealth } = useOllamaHealth()
  const { createSession, optimizePrompt, loading: sessionLoading } = useSessionActions()
  const { data: performanceMetrics } = usePerformanceMetrics()
  const { data: trainingStats } = useTrainingStats(3)
  const { data: trainingDatasets } = useTrainingDatasets()
  const { data: optimizationMethods } = useOptimizationMethods()

  const activeDataset = trainingDatasets?.find((d) => d.id === selectedDataset) ?? null
  const methodNeedsDataset = selectedMethod === "gepa"
  const datasetMissing = methodNeedsDataset && !activeDataset
  // A deleted dataset must not stay selected.
  useEffect(() => {
    if (selectedDataset !== NO_DATASET && trainingDatasets && !activeDataset) setSelectedDataset(NO_DATASET)
  }, [trainingDatasets, selectedDataset, activeDataset])

  // Transform providers data to match component expectations
  const providerData = useMemo(() => {
    if (!providers) return {} as Record<string, ProviderView>
    return providers.reduce<Record<string, ProviderView>>((acc, provider) => {
      acc[provider.name] = {
        id: provider.id,
        name: provider.name,
        models: provider.models,
        available: provider.available,
        unavailableReason: provider.unavailable_reason,
        icon: provider.id === 'ollama' ? '🦙' : '⚡'
      }
      return acc
    }, {})
  }, [providers])

  // Create API status based on real provider data
  const apiStatus = useMemo(() => {
    if (!providers) return {} as Record<string, ApiStatus>

    return providers.reduce<Record<string, ApiStatus>>((acc, provider) => {
      const status: ApiStatus = {
        connected: provider.available && provider.models.length > 0,
        lastTest: provider.available ? 'Available' : 'Unavailable',
      }

      if (provider.id === 'ollama' && ollamaHealth) {
        status.connected = ollamaHealth.healthy
        status.lastTest = ollamaHealth.healthy ? 'Connected' : 'Disconnected'
      }
      
      acc[provider.name] = status
      return acc
    }, {})
  }, [providers, ollamaHealth])

  // Set default model when providers load
  useEffect(() => {
    if (providerData && !selectedModel) {
      const defaultProviderData = providerData[selectedProvider]
      if (defaultProviderData && defaultProviderData.models.length > 0) {
        setSelectedModel(defaultProviderData.models[0].id)
      }
    }
  }, [providerData, selectedProvider, selectedModel])
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [isOnline, setIsOnline] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  // From the real health probe; assume fine until it answers.
  const ollamaModelsAvailable = ollamaHealth ? ollamaHealth.healthy : true

  useEffect(() => {
    if (jobStartedAt == null) return
    setElapsedSeconds(0)
    const timer = setInterval(() => setElapsedSeconds(Math.round((Date.now() - jobStartedAt) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [jobStartedAt])

  // The prompt draft lives in this browser's localStorage so a reload does
  // not lose it. Nothing is sent anywhere.
  useEffect(() => {
    try {
      const saved = localStorage.getItem(DRAFT_KEY)
      if (saved) setOriginalPrompt((current) => current || saved)
    } catch {
      // Storage unavailable (private mode, blocked); drafts just don't persist.
    }
  }, [])

  const saveDraft = useCallback(
    (announce: boolean) => {
      if (!originalPrompt.trim()) return
      try {
        localStorage.setItem(DRAFT_KEY, originalPrompt)
        setLastSaved(new Date())
        if (announce) toast({ title: "Draft saved", description: "Stored in this browser only.", duration: 2000 })
      } catch {
        if (announce) toast({ title: "Could not save draft", description: "Browser storage is unavailable.", variant: "destructive", duration: 3000 })
      }
    },
    [originalPrompt],
  )

  useEffect(() => {
    const timer = setTimeout(() => saveDraft(false), 1500)
    return () => clearTimeout(timer)
  }, [originalPrompt, saveDraft])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Enter to start optimization
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault()
        if (!isOptimizing && originalPrompt.trim()) {
          handleStartOptimization()
          toast({
            title: "Optimization started",
            description: "Started with Ctrl+Enter shortcut",
            duration: 2000,
          })
        }
      }

      // Ctrl+S to save (prevent default browser save)
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault()
        saveDraft(true)
      }

      // Ctrl+Shift+C to copy optimized result
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "C" && optimizedPrompt) {
        e.preventDefault()
        handleCopyOptimized()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [isOptimizing, originalPrompt, optimizedPrompt, saveDraft])

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true)
      toast({
        title: "Back online",
        description: "Connection restored. Syncing data...",
        duration: 3000,
      })
    }

    const handleOffline = () => {
      setIsOnline(false)
      toast({
        title: "Offline mode",
        description: "You're offline. Ollama models are still available.",
        duration: 3000,
      })
    }

    window.addEventListener("online", handleOnline)
    window.addEventListener("offline", handleOffline)

    return () => {
      window.removeEventListener("online", handleOnline)
      window.removeEventListener("offline", handleOffline)
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLElement>) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLElement>) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLElement>) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    const textFile = files.find((file) => file.type === "text/plain" || file.name.endsWith(".txt"))

    if (textFile) {
      const reader = new FileReader()
      reader.onload = () => {
        if (typeof reader.result === "string") setOriginalPrompt(reader.result)
        toast({
          title: "File imported",
          description: `Loaded prompt from ${textFile.name}`,
          duration: 3000,
        })
      }
      reader.readAsText(textFile)
    } else {
      toast({
        title: "Invalid file type",
        description: "Please drop a .txt file containing your prompt.",
        variant: "destructive",
        duration: 3000,
      })
    }
  }, [])

  const handleStartOptimization = async () => {
    if (!originalPrompt.trim()) {
      toast({
        title: "Error",
        description: "Please enter a prompt to optimize",
        duration: 3000,
      })
      return
    }

    setIsOptimizing(true)
    setOptimizationProgress(0)

    const currentModel = getCurrentModel(providerData, selectedProvider, selectedModel)
    
    toast({
      title: "Optimization started",
      description: `Using ${selectedProvider} ${currentModel?.name || selectedModel}`,
      duration: 3000,
    })

    try {
      // Create a new session
      const providerInfo = providerData[selectedProvider]
      const session = await createSession({
        name: `Optimization ${new Date().toLocaleTimeString()}`,
        original_prompt: originalPrompt,
        provider: providerInfo?.id || selectedProvider.toLowerCase(),
        model: selectedModel,
        task_type: selectedTaskType === 'auto' ? 'general' : selectedTaskType
      })

      // Run as a background job; the API reports each stage while it works.
      setJobStartedAt(Date.now())
      setJobProgress({ stage: "starting", message: "Starting optimization", current: null, total: null, best_score: null, updated_at: "" })
      const options: OptimizeOptions = activeDataset
        ? {
            ...advanced,
            dataset_id: activeDataset.id,
            eval_metric: evalMetric,
            max_demos: maxDemos,
            ...(selectedMethod === "gepa"
              ? { gepa_budget: gepaBudget, reflection_model: reflectionModel === "same" ? null : reflectionModel }
              : {}),
          }
        : advanced
      const result = await optimizePrompt(session.id, selectedMethod, options, (progress) => {
        setJobProgress(progress)
        if (progress.total) setOptimizationProgress(Math.round(((progress.current ?? 0) / progress.total) * 100))
      })

      setOptimizationProgress(100)
      setIsOptimizing(false)
      setJobStartedAt(null)

      if (result.session?.optimized_prompt) {
        setOptimizedPrompt(result.session.optimized_prompt)
        setLastResult(result.optimization_details)
        const evaluation = result.optimization_details.metadata.eval
        toast({
          title: "Optimization complete!",
          description: evaluation
            ? evaluation.improved
              ? `${candidateLabel(evaluation.best)} scored ${Math.round(evaluation.eval_score ?? 0)}% on ${evaluation.dev_size} held-out samples (original: ${Math.round(evaluation.baseline_score ?? 0)}%)`
              : `No candidate beat the original (${Math.round(evaluation.baseline_score ?? 0)}% on ${evaluation.dev_size} held-out samples), so it was kept`
            : `${methodLabel(result.optimization_details.method)} · heuristic score ${Math.round(result.session.performance_score)}/100`,
          duration: 6000,
        })
      } else {
        throw new Error('Optimization result not available')
      }
    } catch (error) {
      setIsOptimizing(false)
      setOptimizationProgress(0)
      setJobStartedAt(null)
      setJobProgress(null)
      toast({
        title: "Optimization failed",
        description: error instanceof Error ? error.message : "An error occurred during optimization",
        duration: 5000,
      })
    }
  }

  const handleProviderChange = (provider: string) => {
    setSelectedProvider(provider)
    // Auto-select first model when provider changes
    const firstModel = providerData[provider]?.models[0]
    if (firstModel) {
      setSelectedModel(firstModel.id)
    }

    toast({
      title: "Provider changed",
      description: `Switched to ${provider}`,
      duration: 2000,
    })
  }

  const handleCopyOptimized = async () => {
    if (optimizedPrompt) {
      try {
        await navigator.clipboard.writeText(optimizedPrompt)
        toast({
          title: "Copied to clipboard",
          description: "Optimized prompt has been copied.",
          duration: 2000,
        })
      } catch (error) {
        toast({
          title: "Copy failed",
          description: "Unable to copy to clipboard.",
          variant: "destructive",
          duration: 2000,
        })
      }
    }
  }

  const handleShareResults = async () => {
    if (navigator.share && optimizedPrompt) {
      try {
        await navigator.share({
          title: "PromptCraft Optimization Results",
          text: `Original: ${originalPrompt}\n\nOptimized: ${optimizedPrompt}`,
        })
        toast({
          title: "Shared successfully",
          description: "Results have been shared.",
          duration: 2000,
        })
      } catch (error) {
        // Fallback to copy
        handleCopyOptimized()
      }
    } else {
      handleCopyOptimized()
    }
  }

  const handleReset = () => {
    setOriginalPrompt("")
    setOptimizedPrompt("")
    setLastResult(null)
    setOptimizationProgress(0)
    setIsOptimizing(false)
  }

  const handleExportResults = () => {
    const data = {
      original: originalPrompt,
      optimized: optimizedPrompt,
      provider: selectedProvider,
      model: selectedModel,
      taskType: selectedTaskType,
      timestamp: new Date().toISOString(),
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `promptcraft-optimization-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    toast({
      title: "Export complete",
      description: "Optimization results have been downloaded.",
      duration: 3000,
    })
  }

  return (
    <TooltipProvider>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
        {/* Input Section */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="font-sans font-bold flex items-center gap-2">
                    Prompt Input
                    <Tooltip>
                      <TooltipTrigger>
                        <Keyboard className="w-4 h-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="text-xs space-y-1">
                          <div>
                            <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Ctrl+Enter</kbd> Start optimization
                          </div>
                          <div>
                            <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Ctrl+S</kbd> Save draft
                          </div>
                          <div>
                            <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Ctrl+Shift+C</kbd> Copy result
                          </div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </CardTitle>
                  <CardDescription className="font-serif">Enter your raw prompt for AI optimization</CardDescription>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  {lastSaved ? (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="flex items-center gap-1 cursor-help">
                          <HardDrive className="w-4 h-4" />
                          <span>Draft saved {lastSaved.toLocaleTimeString()}</span>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="text-xs">Kept in this browser&apos;s local storage. Ctrl+S saves immediately.</p>
                      </TooltipContent>
                    </Tooltip>
                  ) : null}
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-medium font-sans">Original Prompt</label>
                <div
                  className={`relative ${isDragOver ? "ring-2 ring-orange-500 ring-offset-2" : ""}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <Textarea
                    ref={textareaRef}
                    placeholder="Enter your prompt here... (e.g., 'Classify customer emails by urgency') or drag & drop a .txt file"
                    value={originalPrompt}
                    onChange={(e) => setOriginalPrompt(e.target.value)}
                    className="min-h-32 font-serif"
                  />
                  {isDragOver && (
                    <div className="absolute inset-0 bg-orange-500/10 border-2 border-dashed border-orange-500 rounded-md flex items-center justify-center">
                      <div className="text-center">
                        <Upload className="w-8 h-8 mx-auto mb-2 text-orange-500" />
                        <p className="text-sm font-medium text-orange-700">Drop your prompt file here</p>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{originalPrompt.length} characters</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <label className="text-sm font-medium font-sans">AI Provider</label>
                  <Select value={selectedProvider} onValueChange={handleProviderChange}>
                    <SelectTrigger className="w-full">
                      <SelectValue>
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{providerData[selectedProvider]?.icon}</span>
                          <span className="font-sans">{selectedProvider}</span>
                          {selectedProvider === "Ollama" && (
                            <div className="flex items-center gap-1">
                              <Badge variant="secondary" className="text-xs font-sans">
                                FREE
                              </Badge>
                              {!isOnline ? (
                                <Badge variant="outline" className="text-xs text-green-600">
                                  OFFLINE OK
                                </Badge>
                              ) : ollamaModelsAvailable ? (
                                <div className="w-2 h-2 bg-green-500 rounded-full" />
                              ) : (
                                <div className="w-2 h-2 bg-red-500 rounded-full" />
                              )}
                            </div>
                          )}
                          {selectedProvider !== "Ollama" && !isOnline && (
                            <Badge variant="destructive" className="text-xs">
                              OFFLINE
                            </Badge>
                          )}
                        </div>
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {providersLoading && (
                        <SelectItem value="loading" disabled>
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Loading providers...</span>
                          </div>
                        </SelectItem>
                      )}
                      {providersError && (
                        <SelectItem value="error" disabled>
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-red-500">Error loading providers</span>
                          </div>
                        </SelectItem>
                      )}
                      {!providersLoading && !providersError && Object.entries(providerData).map(([provider, data]) => (
                        <SelectItem key={provider} value={provider} disabled={!data.available || (provider !== "Ollama" && !isOnline)}>
                          <div className="flex items-center gap-2">
                            <span className="text-lg">{data.icon}</span>
                            <span className="font-sans">{provider}</span>
                            {provider === "Ollama" && (
                              <Badge variant="secondary" className="text-xs font-sans">
                                FREE
                              </Badge>
                            )}
                            {!data.available && (
                              <Badge variant="outline" className="text-xs text-muted-foreground" title={data.unavailableReason ?? undefined}>
                                UNAVAILABLE
                              </Badge>
                            )}
                            {data.available && provider !== "Ollama" && !isOnline && (
                              <Badge variant="outline" className="text-xs text-muted-foreground">
                                OFFLINE
                              </Badge>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-3">
                  <label className="text-sm font-medium font-sans">Model</label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="w-full">
                      <SelectValue>
                        <span className="font-sans">
                          {getCurrentModel(providerData, selectedProvider, selectedModel)?.name}
                        </span>
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {providerData[selectedProvider]?.models.map((model) => (
                        <SelectItem
                          key={model.id}
                          value={model.id}
                          disabled={selectedProvider === "Ollama" && !ollamaModelsAvailable}
                        >
                          <div className="flex flex-col items-start">
                            <span className="font-sans font-medium">{model.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {[modelSpecLine(model), formatContext(model.context_window), model.is_free ? null : formatCost(model)]
                                .filter(Boolean)
                                .join(" • ")}
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {getCurrentModel(providerData, selectedProvider, selectedModel) && (
                <Card className="bg-muted/30">
                  <CardContent className="pt-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-2 cursor-help">
                          <Info className="w-4 h-4 text-muted-foreground" />
                          <div>
                            <div className="font-sans font-medium">Context Window</div>
                            <div className="text-muted-foreground">
                              {(() => { const m = getCurrentModel(providerData, selectedProvider, selectedModel); return m ? formatContext(m.context_window) : null })()}
                            </div>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">Context length reported by the model runtime (prompt + response)</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-2 cursor-help">
                          <Database className="w-4 h-4 text-muted-foreground" />
                          <div>
                            <div className="font-sans font-medium">Model</div>
                            <div className="text-muted-foreground">
                              {(() => { const m = getCurrentModel(providerData, selectedProvider, selectedModel); return m ? (modelSpecLine(m) || formatCost(m)) : null })()}
                            </div>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">Parameter count, quantization and on-disk size as reported by Ollama</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-2 cursor-help">
                          <Gauge className="w-4 h-4 text-muted-foreground" />
                          <div>
                            <div className="font-sans font-medium">Speed</div>
                            <div className="text-muted-foreground">
                              {(() => { const m = getCurrentModel(providerData, selectedProvider, selectedModel); return m ? formatSpeed(m.speed_rating) : null })()}
                            </div>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">Relative speed from parameter count: smaller models answer faster</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-2 cursor-help">
                          <Zap className="w-4 h-4 text-muted-foreground" />
                          <div>
                            <div className="font-sans font-medium">Best for</div>
                            <div className="text-muted-foreground text-xs">
                              {getCurrentModel(providerData, selectedProvider, selectedModel)?.best_use_case}
                            </div>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">Recommended use cases for optimal performance</p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="space-y-3">
                <label className="text-sm font-medium font-sans">Task Type</label>
                <Select value={selectedTaskType} onValueChange={setSelectedTaskType}>
                  <SelectTrigger className="w-full md:w-64">
                    <SelectValue>
                      <div className="flex items-center gap-2">
                        <span>{taskTypes.find((t) => t.id === selectedTaskType)?.icon}</span>
                        <span className="font-sans">{taskTypes.find((t) => t.id === selectedTaskType)?.name}</span>
                      </div>
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {taskTypes.map((taskType) => (
                      <SelectItem key={taskType.id} value={taskType.id}>
                        <div className="flex items-center gap-2">
                          <span>{taskType.icon}</span>
                          <span className="font-sans">{taskType.name}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <label className="text-sm font-medium font-sans flex items-center gap-2">
                  Optimization Method
                  <Tooltip>
                    <TooltipTrigger>
                      <HelpCircle className="w-3 h-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <MethodExplainer method={methodInfo(selectedMethod, optimizationMethods)} />
                    </TooltipContent>
                  </Tooltip>
                </label>
                <Select value={selectedMethod} onValueChange={(value) => setSelectedMethod(value as OptimizationMethod)}>
                  <SelectTrigger className="w-full md:w-64">
                    <SelectValue>
                      <span className="font-sans">{methodLabel(selectedMethod, optimizationMethods)}</span>
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {(optimizationMethods ?? FALLBACK_METHODS).map((method) => (
                      <SelectItem key={method.id} value={method.id}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex flex-col text-left">
                              <span className="font-sans">
                                {method.name}
                                {method.relative_speed && (
                                  <span className="ml-2 text-[10px] uppercase tracking-wide text-muted-foreground">{method.relative_speed}</span>
                                )}
                              </span>
                              <span className="text-xs text-muted-foreground">{method.best_for ?? method.description}</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent side="right" align="start">
                            <MethodExplainer method={method} />
                          </TooltipContent>
                        </Tooltip>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <label className="text-sm font-medium font-sans flex items-center gap-2">
                  Measure against a dataset
                  <span className="text-xs font-normal text-muted-foreground">optional</span>
                  <Tooltip>
                    <TooltipTrigger>
                      <HelpCircle className="w-3 h-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <div className="max-w-xs text-xs space-y-1">
                        <p className="font-medium">Replaces the heuristic score with a measured one.</p>
                        <p>
                          The dataset is split into train and held-out samples. The original prompt, the rewrite and few-shot versions of each
                          (examples picked by DSPy&apos;s BootstrapFewShot on the train split) are all scored on the held-out samples. You get the
                          best one, with the full scoreboard.
                        </p>
                        <p>Takes longer: roughly one model call per sample per candidate.</p>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </label>
                <div className="flex flex-col md:flex-row gap-2">
                  <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                    <SelectTrigger className="w-full md:w-64">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NO_DATASET}>
                        <span className="font-sans">None</span>
                        <span className="ml-2 text-xs text-muted-foreground">heuristic score only</span>
                      </SelectItem>
                      {(trainingDatasets ?? []).map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id} disabled={dataset.sample_count < 2}>
                          <span className="font-sans">{dataset.name}</span>
                          <span className="ml-2 text-xs text-muted-foreground">
                            {dataset.sample_count} sample{dataset.sample_count === 1 ? "" : "s"}
                            {dataset.sample_count < 2 && " · needs 2+"}
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {activeDataset && (
                    <>
                      <Select value={evalMetric} onValueChange={(v) => setEvalMetric(v as EvalMetric)}>
                        <SelectTrigger className="w-full md:w-44">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto metric</SelectItem>
                          <SelectItem value="contains">Contains match</SelectItem>
                          <SelectItem value="exact">Exact match</SelectItem>
                          <SelectItem value="llm_judge">Model judge</SelectItem>
                        </SelectContent>
                      </Select>
                      <div className="flex items-center gap-2">
                        <Input
                          type="number"
                          min={1}
                          max={8}
                          value={maxDemos}
                          onChange={(e) => setMaxDemos(Math.min(8, Math.max(1, Number(e.target.value) || 1)))}
                          className="w-20"
                        />
                        <span className="text-xs text-muted-foreground whitespace-nowrap">max examples</span>
                      </div>
                    </>
                  )}
                </div>
                {activeDataset && (
                  <p className="text-xs text-muted-foreground font-serif">
                    {activeDataset.sample_count} samples: about {Math.max(1, Math.round(activeDataset.sample_count * 0.2))} held out for scoring, the rest
                    available as examples.{" "}
                    {evalMetric === "auto" && "Auto picks contains-match for short expected outputs and the model judge for longer ones."}
                  </p>
                )}
                {!trainingDatasets?.length && (
                  <p className="text-xs text-muted-foreground font-serif">No datasets yet. Create one in the sidebar&apos;s Training Data tab.</p>
                )}
                {datasetMissing && (
                  <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs">
                    <AlertTriangle className="w-3.5 h-3.5 mt-0.5 text-amber-500 shrink-0" />
                    <span>GEPA evolves the prompt from feedback on a dataset. Pick one above to enable Start Optimization.</span>
                  </div>
                )}
                {selectedMethod === "gepa" && activeDataset && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 rounded-md border bg-muted/30 p-3">
                    <div className="space-y-1">
                      <label className="text-xs font-medium font-sans flex items-center gap-1">
                        Evolution budget
                        <Tooltip>
                          <TooltipTrigger>
                            <HelpCircle className="w-3 h-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs text-xs">
                              Scored model calls GEPA may spend proposing and testing new instructions. About 60 calls took a minute on a 3B
                              model with 16 samples; more budget means more generations.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </label>
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min={10}
                          max={300}
                          step={10}
                          value={gepaBudget}
                          onChange={(e) => setGepaBudget(Number(e.target.value))}
                          className="flex-1 accent-orange-500"
                        />
                        <span className="text-sm text-muted-foreground w-10 font-mono">{gepaBudget}</span>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs font-medium font-sans flex items-center gap-1">
                        Reflection model
                        <Tooltip>
                          <TooltipTrigger>
                            <HelpCircle className="w-3 h-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs text-xs">
                              Writes the new instructions after reading the feedback. A larger model reflects better while the task model
                              keeps doing the fast scoring calls.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </label>
                      <Select value={reflectionModel} onValueChange={setReflectionModel}>
                        <SelectTrigger className="h-9">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="same">Same as task model</SelectItem>
                          {(providerData[selectedProvider]?.models ?? []).map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              {model.name}
                              {model.parameter_size ? ` · ${model.parameter_size}` : ""}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                )}
              </div>

              <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
                <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground">
                  <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                  Advanced Settings
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium font-sans flex items-center gap-2">
                        Temperature
                        <Tooltip>
                          <TooltipTrigger><HelpCircle className="w-3 h-3 text-muted-foreground" /></TooltipTrigger>
                          <TooltipContent><p className="max-w-xs text-xs">Sampling temperature for the rewrite. Lower is more deterministic; 0.7 is the default.</p></TooltipContent>
                        </Tooltip>
                      </label>
                      <div className="flex items-center gap-2">
                        <input
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={advanced.temperature}
                          onChange={(e) => setAdvanced((a) => ({ ...a, temperature: Number(e.target.value) }))}
                          className="flex-1 accent-orange-500"
                        />
                        <span className="text-sm text-muted-foreground w-8 font-mono">{advanced.temperature.toFixed(1)}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium font-sans flex items-center gap-2">
                        Max Tokens
                        <Tooltip>
                          <TooltipTrigger><HelpCircle className="w-3 h-3 text-muted-foreground" /></TooltipTrigger>
                          <TooltipContent><p className="max-w-xs text-xs">Upper bound on the rewritten prompt's length, in model tokens (64–8192).</p></TooltipContent>
                        </Tooltip>
                      </label>
                      <Input
                        type="number"
                        min={64}
                        max={8192}
                        step={64}
                        value={advanced.max_tokens}
                        onChange={(e) => setAdvanced((a) => ({ ...a, max_tokens: Math.min(8192, Math.max(64, Number(e.target.value) || 64)) }))}
                        className="text-sm"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium font-sans">Output Format</label>
                      <Select value={advanced.output_format} onValueChange={(v) => setAdvanced((a) => ({ ...a, output_format: v as OutputFormat }))}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Let the model decide</SelectItem>
                          <SelectItem value="markdown">Ask for Markdown</SelectItem>
                          <SelectItem value="plain">Ask for plain text</SelectItem>
                          <SelectItem value="json">Ask for JSON (with schema)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium font-sans">Target Length</label>
                      <Select value={advanced.target_length} onValueChange={(v) => setAdvanced((a) => ({ ...a, target_length: v as TargetLength }))}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">No preference</SelectItem>
                          <SelectItem value="concise">Concise — close to the original</SelectItem>
                          <SelectItem value="balanced">Balanced — about 1.5–2×</SelectItem>
                          <SelectItem value="detailed">Detailed — context, constraints, example</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <label className="flex items-start gap-2 text-sm font-sans cursor-pointer">
                    <Checkbox
                      checked={advanced.preserve_wording}
                      onCheckedChange={(checked) => setAdvanced((a) => ({ ...a, preserve_wording: checked === true }))}
                      className="mt-0.5"
                    />
                    <span>
                      Preserve my wording
                      <span className="block text-xs text-muted-foreground">Improve structure and add instructions around the request instead of rephrasing it.</span>
                    </span>
                  </label>
                </CollapsibleContent>
              </Collapsible>

              <div className="flex gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={handleStartOptimization}
                      disabled={isOptimizing || !originalPrompt || datasetMissing || (selectedProvider !== "Ollama" && !isOnline)}
                      className="font-sans font-semibold bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700"
                    >
                      {isOptimizing ? (
                        <>
                          <Pause className="w-4 h-4" />
                          Optimizing...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          Start Optimization
                        </>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-xs">Press Ctrl+Enter to start optimization</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" onClick={handleReset}>
                      <RotateCcw className="w-4 h-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-xs">Reset form</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" onClick={() => setShowAdvanced(!showAdvanced)}>
                      <Settings className="w-4 h-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-xs">Advanced settings</p>
                  </TooltipContent>
                </Tooltip>
              </div>

              {isOptimizing && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm font-sans">
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-orange-500" />
                      {jobProgress?.message ?? "Working..."}
                    </span>
                    <span className="text-muted-foreground tabular-nums">
                      {jobProgress?.total ? `${jobProgress.current ?? 0}/${jobProgress.total} · ` : ""}
                      {elapsedSeconds}s
                    </span>
                  </div>
                  <Progress value={jobProgress?.total ? optimizationProgress : undefined} className={`w-full ${jobProgress?.total ? "" : "animate-pulse"}`} />
                  <div className="flex items-center gap-2 text-xs text-muted-foreground font-serif">
                    <span className="capitalize">{jobProgress?.stage ?? "starting"}</span>
                    {jobProgress?.best_score != null && (
                      <Badge variant="outline" className="font-mono text-[10px]">best so far {Math.round(jobProgress.best_score)}%</Badge>
                    )}
                    {activeDataset && <span>Measured runs make one model call per sample per candidate.</span>}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          {optimizedPrompt && (
            <Card>
              <CardHeader>
                <CardTitle className="font-sans font-bold flex items-center gap-2">
                  <Zap className="w-5 h-5 text-secondary" />
                  Optimized Prompt
                </CardTitle>
                <CardDescription className="font-serif">
                  Rendered as the model will read it. Switch to Raw to copy exact text, or Compare to see what changed.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <OptimizedPromptView
                  original={originalPrompt}
                  optimized={optimizedPrompt}
                  details={lastResult}
                  methodLabel={methodLabel(lastResult?.method ?? selectedMethod, optimizationMethods)}
                  actions={
                    <>
                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <Button size="sm" className="font-sans" onClick={handleExportResults}>
                                            <Download className="w-4 h-4" />
                                            Export
                                          </Button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="text-xs">Download results as JSON file</p>
                                        </TooltipContent>
                                      </Tooltip>

                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            className="font-sans bg-transparent"
                                            onClick={handleCopyOptimized}
                                          >
                                            <Copy className="w-4 h-4" />
                                            Copy Optimized
                                          </Button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="text-xs">Copy optimized prompt to clipboard (Ctrl+Shift+C)</p>
                                        </TooltipContent>
                                      </Tooltip>

                                      <Tooltip>
                                        <TooltipTrigger asChild>
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            className="font-sans bg-transparent"
                                            onClick={handleShareResults}
                                          >
                                            <Share2 className="w-4 h-4" />
                                            Share
                                          </Button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p className="text-xs">Share optimization results</p>
                                        </TooltipContent>
                                      </Tooltip>
                    </>
                  }
                />
              </CardContent>
            </Card>
          )}

          {lastResult?.metadata.gepa && <PromptEvolutionCard report={lastResult.metadata.gepa} />}
          {lastResult?.metadata.eval && <EvalResultsCard report={lastResult.metadata.eval} />}
        </div>

        {/* Enhanced Right Sidebar */}
        <div className="space-y-6">
          {/* Optimization Insights: what the last run actually did */}
          <Card>
            <CardHeader>
              <CardTitle className="font-sans font-bold text-sm flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Optimization Insights
              </CardTitle>
              <CardDescription className="text-xs">How the last rewrite was produced</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!lastResult ? (
                <div className="space-y-2 text-sm text-muted-foreground font-serif">
                  <p>Run an optimization to see the method, timing and reasoning behind the rewrite.</p>
                  <div className="rounded-md border bg-muted/30 p-3 text-foreground">
                    <MethodExplainer method={methodInfo(selectedMethod, optimizationMethods)} />
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary" className="font-sans text-xs">
                      {methodLabel(lastResult.method, optimizationMethods)}
                    </Badge>
                    {typeof lastResult.metadata.predictor === "string" && (
                      <Badge variant="outline" className="font-mono text-xs">{lastResult.metadata.predictor}</Badge>
                    )}
                    <span className="text-xs text-muted-foreground font-sans ml-auto">
                      {lastResult.processing_time.toFixed(1)}s
                    </span>
                  </div>

                  {(() => {
                    const st = lastResult.metadata.settings as Partial<Required<OptimizeOptions>> | undefined
                    if (!st) return null
                    const chips = [
                      `temp ${st.temperature}`,
                      `${st.max_tokens} tokens max`,
                      st.output_format && st.output_format !== "auto" ? `${st.output_format} output` : null,
                      st.target_length && st.target_length !== "auto" ? `${st.target_length} length` : null,
                      st.preserve_wording ? "wording preserved" : null,
                    ].filter(Boolean) as string[]
                    return (
                      <div className="flex flex-wrap gap-1">
                        {chips.map((c) => (
                          <Badge key={c} variant="outline" className="font-mono text-[10px]">{c}</Badge>
                        ))}
                      </div>
                    )
                  })()}

                  {(lastResult.metadata.predictor === "template_fallback" || lastResult.metadata.fallback === true) && (
                    <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs">
                      <AlertTriangle className="w-3.5 h-3.5 mt-0.5 text-amber-500 shrink-0" />
                      <div>
                        <p className="font-medium font-sans">Fell back to a simpler strategy</p>
                        {typeof lastResult.metadata.fallback_reason === "string" && (
                          <p className="text-muted-foreground font-mono break-words">{lastResult.metadata.fallback_reason}</p>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="flex justify-between items-center text-sm">
                    <span className="font-serif text-muted-foreground flex items-center gap-1">
                      {lastResult.score_type === "measured" ? "Measured score" : "Heuristic score"}
                      <Tooltip>
                        <TooltipTrigger>
                          <HelpCircle className="w-3 h-3" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <div className="max-w-xs text-xs space-y-1">
                            {lastResult.score_type === "measured" ? (
                              <p>Share of held-out dataset samples the metric accepted. See the Eval Results card for every candidate.</p>
                            ) : (
                              <p>A structural rubric, not a measured gain. Pick a dataset above to measure the prompt instead.</p>
                            )}
                            {lastResult.score_type !== "measured" && Array.isArray(lastResult.metadata.score_breakdown) && (
                              <ul className="space-y-0.5">
                                {(lastResult.metadata.score_breakdown as { label: string; points: number; applied: boolean }[]).map((item) => (
                                  <li key={item.label} className={`flex justify-between gap-3 ${item.applied ? "" : "text-muted-foreground line-through"}`}>
                                    <span>{item.label}</span>
                                    <span className="font-mono">+{item.points}</span>
                                  </li>
                                ))}
                              </ul>
                            )}
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    </span>
                    <span className="font-sans font-semibold">
                      {lastResult.score_type === "measured" ? `${Math.round(lastResult.improvement_score)}%` : `${Math.round(lastResult.improvement_score)}/100`}
                    </span>
                  </div>

                  {typeof lastResult.metadata.rewrite === "string" && lastResult.metadata.eval && lastResult.metadata.eval.best !== "rewritten" && (
                    <div className="space-y-1">
                      <p className="text-xs font-medium font-sans">Rewrite (not selected: {candidateLabel(lastResult.metadata.eval.best)} scored higher)</p>
                      <div className="max-h-32 overflow-y-auto rounded-md bg-muted/50 p-3 text-xs font-serif whitespace-pre-wrap">
                        {lastResult.metadata.rewrite}
                      </div>
                    </div>
                  )}

                  {typeof lastResult.metadata.reasoning === "string" && lastResult.metadata.reasoning.trim() && (
                    <div className="space-y-1">
                      <p className="text-xs font-medium font-sans">Model reasoning</p>
                      <div className="max-h-48 overflow-y-auto rounded-md bg-muted/50 p-3 text-xs font-serif leading-relaxed whitespace-pre-wrap">
                        {lastResult.metadata.reasoning}
                      </div>
                    </div>
                  )}

                  {typeof lastResult.metadata.meta_prompt_used === "string" && (
                    <div className="space-y-1">
                      <p className="text-xs font-medium font-sans">Meta-prompt (excerpt)</p>
                      <div className="max-h-32 overflow-y-auto rounded-md bg-muted/50 p-3 text-xs font-mono whitespace-pre-wrap">
                        {lastResult.metadata.meta_prompt_used}
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* Session Stats: from /sessions/analytics/performance */}
          <Card>
            <CardHeader>
              <CardTitle className="font-sans font-bold text-sm flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                Session Stats
              </CardTitle>
              <CardDescription className="text-xs">Across all stored sessions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Total Optimizations</span>
                <span className="font-sans font-semibold">{performanceMetrics?.total_optimizations ?? "—"}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Avg. Score</span>
                <span className="font-sans font-bold text-2xl text-green-500">
                  {performanceMetrics ? Math.round(performanceMetrics.average_improvement) : "—"}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Completed</span>
                <div className="flex items-center gap-1">
                  <span className="font-sans font-semibold text-green-500">
                    {performanceMetrics ? `${Math.round(performanceMetrics.success_rate)}%` : "—"}
                  </span>
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Processing Time (all runs)</span>
                <span className="font-sans font-semibold">
                  {performanceMetrics?.total_processing_time != null ? `${performanceMetrics.total_processing_time.toFixed(1)}s` : "—"}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Training Data Overview */}
          <Card>
            <CardHeader>
              <CardTitle className="font-sans font-bold text-sm flex items-center gap-2">
                <Database className="w-4 h-4" />
                Training Data Overview
              </CardTitle>
              <CardDescription className="text-xs">Datasets stored locally, managed from the sidebar.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="space-y-1">
                  <div className="font-sans font-bold text-2xl">{trainingStats?.dataset_count ?? "—"}</div>
                  <div className="text-xs text-muted-foreground">Datasets</div>
                </div>
                <div className="space-y-1">
                  <div className="font-sans font-bold text-2xl">
                    {trainingStats ? trainingStats.sample_count.toLocaleString() : "—"}
                  </div>
                  <div className="text-xs text-muted-foreground">Total Samples</div>
                </div>
              </div>

              {trainingStats && trainingStats.sample_count > 0 && (
                <div className="space-y-2">
                  <div className="text-xs font-medium font-sans">Samples by Task Type</div>
                  <div className="h-32">
                    <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 300, height: 128 }}>
                      <PieChart>
                        <Pie
                          data={trainingStats.by_task_type.map((entry) => ({ name: taskTypeLabel(entry.task_type), value: entry.sample_count }))}
                          cx="50%"
                          cy="50%"
                          innerRadius={20}
                          outerRadius={50}
                          dataKey="value"
                          nameKey="name"
                        >
                          {trainingStats.by_task_type.map((entry, index) => (
                            <Cell key={entry.task_type} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                          ))}
                        </Pie>
                        <ChartTooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <div className="space-y-2">
                <div className="text-xs font-medium font-sans">Recent Datasets</div>
                {trainingStats && trainingStats.recent_datasets.length > 0 ? (
                  <div className="space-y-1">
                    {trainingStats.recent_datasets.map((dataset) => (
                      <div key={dataset.id} className="flex justify-between gap-2 text-xs">
                        <span className="font-serif truncate" title={dataset.name}>{dataset.name}</span>
                        <span className="text-muted-foreground shrink-0">{dataset.sample_count} samples</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">
                    No datasets yet. Generate or import one from the Training Data tab in the sidebar.
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Enhanced API Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="font-sans font-bold text-sm flex items-center gap-2">
                <Settings className="w-4 h-4" />
                API Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {providersLoading && (
                <div className="flex items-center justify-center py-8">
                  <div className="text-sm text-muted-foreground">Loading API providers...</div>
                </div>
              )}
              {providersError && (
                <div className="flex items-center justify-center py-8">
                  <div className="text-sm text-red-500">Error loading providers: {providersError}</div>
                </div>
              )}
              {!providersLoading && !providersError && Object.entries(providerData).map(([provider, data]) => (
                <Collapsible key={provider}>
                  <CollapsibleTrigger className="flex items-center justify-between w-full p-2 hover:bg-muted/50 rounded-md">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{data.icon}</span>
                      <span className="font-sans text-sm font-medium">{provider}</span>
                      <div className="flex items-center gap-1">
                        {apiStatus[provider]?.connected ? (
                          <CheckCircle className="w-3 h-3 text-green-500" />
                        ) : (
                          <XCircle className="w-3 h-3 text-red-500" />
                        )}
                      </div>
                    </div>
                    <ChevronDown className="w-4 h-4" />
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-3 pt-2">
                    {!data.available ? (
                      <p className="text-xs text-muted-foreground font-serif">
                        {data.unavailableReason ?? "Not available in this build."}
                      </p>
                    ) : (
                      <>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Connection Status:</span>
                          <div className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                            <span className="font-medium text-green-500">Connected</span>
                          </div>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Models Available:</span>
                          <span className="font-medium">{data.models.length}</span>
                        </div>
                      </>
                    )}
                  </CollapsibleContent>
                </Collapsible>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </TooltipProvider>
  )
}
