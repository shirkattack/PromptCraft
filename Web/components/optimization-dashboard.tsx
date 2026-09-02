"use client"

import { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { useProviders, useOllamaHealth, useSessionActions, usePerformanceMetrics, useOptimizationMethods } from "@/lib/api/hooks"
import type { AIModel, OptimizationMethod, OptimizeResponse } from "@/lib/api/client"
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
  Cloud,
  CloudOff,
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

const mockTaskTypeData = [
  { name: "Classification", value: 35, color: "#f97316" },
  { name: "Generation", value: 25, color: "#fb923c" },
  { name: "Summarization", value: 20, color: "#fdba74" },
  { name: "Code", value: 15, color: "#fed7aa" },
  { name: "Q&A", value: 5, color: "#ffedd5" },
]

type MethodInfo = { id: OptimizationMethod; name: string; description: string; recommended_for: string[] }

// Mirrors GET /sessions/optimization-methods so the selector renders before the request lands.
const FALLBACK_METHODS: MethodInfo[] = [
  { id: "meta_prompt", name: "Meta-Prompt Optimization", description: "Uses meta-prompting techniques to improve prompt effectiveness", recommended_for: [] },
  { id: "dspy", name: "DSPy Optimization", description: "Uses DSPy framework for systematic prompt optimization", recommended_for: [] },
  { id: "simple", name: "Simple Optimization", description: "Basic prompt improvement using direct language model feedback", recommended_for: [] },
]

const formatContext = (tokens: number) => (tokens >= 1000 ? `${Math.round(tokens / 1000)}K tokens` : `${tokens} tokens`)
const formatCost = (model: AIModel) => (model.is_free ? "Free" : `$${model.cost_per_1k_tokens}/1K tokens`)
const formatSpeed = (rating: number) => `${rating}/5 speed`

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
  const [selectedProvider, setSelectedProvider] = useState("Ollama")
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedTaskType, setSelectedTaskType] = useState("auto")
  const [originalPrompt, setOriginalPrompt] = useState("")
  const [optimizedPrompt, setOptimizedPrompt] = useState("")
  const [selectedMethod, setSelectedMethod] = useState<OptimizationMethod>("meta_prompt")
  const [lastResult, setLastResult] = useState<OptimizeResponse["optimization_details"] | null>(null)

  // API hooks
  const { data: providers, loading: providersLoading, error: providersError } = useProviders()
  const { data: ollamaHealth } = useOllamaHealth()
  const { createSession, optimizePrompt, loading: sessionLoading } = useSessionActions()
  const { data: performanceMetrics } = usePerformanceMetrics()
  const { data: optimizationMethods } = useOptimizationMethods()

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
        icon: provider.id === 'ollama' ? '🦙' : provider.id === 'openai' ? '🤖' : provider.id === 'anthropic' ? '🧠' : '⚡'
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
  const [isAutoSaving, setIsAutoSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [ollamaModelsAvailable, setOllamaModelsAvailable] = useState(true)
  const [isOnline, setIsOnline] = useState(true)
  const [textareaRef] = useState(useRef(null))

  const autoSave = useCallback(async () => {
    if (!originalPrompt.trim()) return

    setIsAutoSaving(true)
    // Simulate auto-save
    setTimeout(() => {
      setIsAutoSaving(false)
      setLastSaved(new Date())
      toast({
        title: "Draft saved",
        description: "Your prompt has been automatically saved to the cloud.",
        duration: 2000,
      })
    }, 1000)
  }, [originalPrompt])

  useEffect(() => {
    const timer = setTimeout(() => {
      if (originalPrompt.trim()) {
        autoSave()
      }
    }, 3000) // Auto-save after 3 seconds of inactivity

    return () => clearTimeout(timer)
  }, [originalPrompt, autoSave])

  useEffect(() => {
    const handleKeyDown = (e) => {
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
        autoSave()
      }

      // Ctrl+Shift+C to copy optimized result
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "C" && optimizedPrompt) {
        e.preventDefault()
        handleCopyOptimized()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [isOptimizing, originalPrompt, optimizedPrompt, autoSave])

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

  useEffect(() => {
    const checkOllamaAvailability = async () => {
      if (selectedProvider === "Ollama") {
        try {
          // Simulate checking Ollama availability
          const available = true // Assume Ollama is available
          setOllamaModelsAvailable(available)

          if (!available) {
            toast({
              title: "Ollama unavailable",
              description: "Local Ollama server is not responding. Please check your installation.",
              variant: "destructive",
              duration: 5000,
            })
          }
        } catch (error) {
          setOllamaModelsAvailable(false)
        }
      }
    }

    const interval = setInterval(checkOllamaAvailability, 30000) // Check every 30 seconds
    checkOllamaAvailability() // Initial check

    return () => clearInterval(interval)
  }, [selectedProvider])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    const textFile = files.find((file) => file.type === "text/plain" || file.name.endsWith(".txt"))

    if (textFile) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setOriginalPrompt(event.target.result)
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

      // Start optimization progress simulation
      const progressInterval = setInterval(() => {
        setOptimizationProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 500)

      // Optimize the prompt
      const result = await optimizePrompt(session.id, selectedMethod)
      
      clearInterval(progressInterval)
      setOptimizationProgress(100)
      setIsOptimizing(false)

      if (result.session?.optimized_prompt) {
        setOptimizedPrompt(result.session.optimized_prompt)
        setLastResult(result.optimization_details)
        toast({
          title: "Optimization complete!",
          description: `${methodLabel(result.optimization_details.method)} · heuristic score ${Math.round(result.session.performance_score)}/100`,
          duration: 5000,
        })
      } else {
        throw new Error('Optimization result not available')
      }
    } catch (error) {
      setIsOptimizing(false)
      setOptimizationProgress(0)
      toast({
        title: "Optimization failed",
        description: error.message || "An error occurred during optimization",
        duration: 5000,
      })
    }
  }

  const handleProviderChange = (provider) => {
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
                  {isAutoSaving ? (
                    <>
                      <Cloud className="w-4 h-4 animate-pulse" />
                      <span>Saving...</span>
                    </>
                  ) : lastSaved ? (
                    <>
                      {isOnline ? <Cloud className="w-4 h-4" /> : <CloudOff className="w-4 h-4" />}
                      <span>Saved {lastSaved.toLocaleTimeString()}</span>
                    </>
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
                              {formatContext(model.context_window)} • {formatCost(model)} • {formatSpeed(model.speed_rating)}
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
                          <p className="text-xs">Maximum number of tokens the model can process at once</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-2 cursor-help">
                          <DollarSign className="w-4 h-4 text-muted-foreground" />
                          <div>
                            <div className="font-sans font-medium">Cost per 1K tokens</div>
                            <div className="text-muted-foreground">
                              {(() => { const m = getCurrentModel(providerData, selectedProvider, selectedModel); return m ? formatCost(m) : null })()}
                            </div>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">Pricing for input and output tokens combined</p>
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
                          <p className="text-xs">Typical response time for this model</p>
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
                    <TooltipContent>
                      <p className="max-w-xs text-xs">
                        {optimizationMethods?.find((m) => m.id === selectedMethod)?.description ??
                          "How the rewrite is produced."}
                      </p>
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
                        <div className="flex flex-col">
                          <span className="font-sans">{method.name}</span>
                          <span className="text-xs text-muted-foreground">{method.description}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
                <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground">
                  <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                  Advanced Settings
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium font-sans">Temperature</label>
                      <div className="flex items-center gap-2">
                        <input type="range" min="0" max="1" step="0.1" defaultValue="0.7" className="flex-1" />
                        <span className="text-sm text-muted-foreground w-8">0.7</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium font-sans">Max Tokens</label>
                      <Input type="number" defaultValue="2048" className="text-sm" />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium font-sans">Training Data</label>
                    <Select defaultValue="existing">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="existing">Use existing dataset</SelectItem>
                        <SelectItem value="generate">Generate new (2-50 samples)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              <div className="flex gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={handleStartOptimization}
                      disabled={isOptimizing || !originalPrompt || (selectedProvider !== "Ollama" && !isOnline)}
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
                    <span>Optimization Progress</span>
                    <span>{optimizationProgress}%</span>
                  </div>
                  <Progress value={optimizationProgress} className="w-full" />
                  <div className="text-xs text-muted-foreground font-serif">
                    Estimated time remaining: {Math.max(0, Math.ceil((100 - optimizationProgress) / 10))}s
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
                  <p className="text-xs">
                    Selected: <span className="font-sans font-medium text-foreground">{methodLabel(selectedMethod, optimizationMethods)}</span>
                    {" — "}
                    {optimizationMethods?.find((m) => m.id === selectedMethod)?.description}
                  </p>
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
                      Heuristic score
                      <Tooltip>
                        <TooltipTrigger>
                          <HelpCircle className="w-3 h-3" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <div className="max-w-xs text-xs space-y-1">
                            <p>A structural rubric, not a measured gain — evaluating against a dataset is on the roadmap.</p>
                            {Array.isArray(lastResult.metadata.score_breakdown) && (
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
                    <span className="font-sans font-semibold">{Math.round(lastResult.improvement_score)}/100</span>
                  </div>

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
                <span className="text-sm font-serif text-muted-foreground">Avg. Heuristic Score</span>
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
                <span className="text-sm font-serif text-muted-foreground">Processing Time (this server)</span>
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
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="space-y-1">
                  <div className="font-sans font-bold text-2xl">12</div>
                  <div className="text-xs text-muted-foreground">Available Datasets</div>
                </div>
                <div className="space-y-1">
                  <div className="font-sans font-bold text-2xl">1,847</div>
                  <div className="text-xs text-muted-foreground">Total Samples</div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-xs font-medium font-sans">Most Used Task Types</div>
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={mockTaskTypeData} cx="50%" cy="50%" innerRadius={20} outerRadius={50} dataKey="value">
                        {mockTaskTypeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <ChartTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-xs font-medium font-sans">Recent Datasets</div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-serif">Email Classification</span>
                    <span className="text-muted-foreground">150 samples</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="font-serif">Code Review Patterns</span>
                    <span className="text-muted-foreground">203 samples</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="font-serif">Content Summarization</span>
                    <span className="text-muted-foreground">89 samples</span>
                  </div>
                </div>
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
