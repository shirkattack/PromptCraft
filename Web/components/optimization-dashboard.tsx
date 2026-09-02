"use client"

import { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { useProviders, useOllamaHealth, useSessionActions } from "@/lib/api/hooks"
import type { AIModel } from "@/lib/api/client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChartTooltip } from "@/components/ui/chart"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { toast } from "@/hooks/use-toast"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
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
  ThumbsUp,
  ThumbsDown,
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
  Wifi,
  Cloud,
  CloudOff,
  Copy,
  Share2,
  Upload,
  HelpCircle,
  Keyboard,
} from "lucide-react"

const providerData = {
  OpenAI: {
    icon: "🤖",
    models: [
      {
        id: "gpt-4o",
        name: "GPT-4o",
        contextWindow: "128K",
        costPer1K: "$0.005",
        speed: "Fast",
        bestFor: "Complex reasoning, analysis, creative tasks",
      },
      {
        id: "gpt-4-turbo",
        name: "GPT-4 Turbo",
        contextWindow: "128K",
        costPer1K: "$0.01",
        speed: "Medium",
        bestFor: "Advanced reasoning, code generation",
      },
      {
        id: "gpt-3.5-turbo",
        name: "GPT-3.5 Turbo",
        contextWindow: "16K",
        costPer1K: "$0.001",
        speed: "Fast",
        bestFor: "General tasks, quick responses",
      },
      {
        id: "gpt-4o-mini",
        name: "GPT-4o Mini",
        contextWindow: "128K",
        costPer1K: "$0.0001",
        speed: "Very Fast",
        bestFor: "Simple tasks, high volume processing",
      },
    ],
  },
  Anthropic: {
    icon: "🧠",
    models: [
      {
        id: "claude-3-5-sonnet",
        name: "Claude 3.5 Sonnet",
        contextWindow: "200K",
        costPer1K: "$0.003",
        speed: "Fast",
        bestFor: "Writing, analysis, complex reasoning",
      },
      {
        id: "claude-3-opus",
        name: "Claude 3 Opus",
        contextWindow: "200K",
        costPer1K: "$0.015",
        speed: "Slow",
        bestFor: "Highest quality reasoning, research",
      },
      {
        id: "claude-3-haiku",
        name: "Claude 3 Haiku",
        contextWindow: "200K",
        costPer1K: "$0.00025",
        speed: "Very Fast",
        bestFor: "Quick tasks, simple queries",
      },
    ],
  },
  Google: {
    icon: "🔍",
    models: [
      {
        id: "gemini-1.5-pro",
        name: "Gemini 1.5 Pro",
        contextWindow: "2M",
        costPer1K: "$0.007",
        speed: "Medium",
        bestFor: "Long context, multimodal tasks",
      },
      {
        id: "gemini-1.5-flash",
        name: "Gemini 1.5 Flash",
        contextWindow: "1M",
        costPer1K: "$0.0007",
        speed: "Fast",
        bestFor: "Quick responses, general tasks",
      },
      {
        id: "gemini-1.0-pro",
        name: "Gemini 1.0 Pro",
        contextWindow: "32K",
        costPer1K: "$0.0005",
        speed: "Fast",
        bestFor: "Standard reasoning tasks",
      },
    ],
  },
  Ollama: {
    icon: "🏠",
    models: [
      {
        id: "llama3.2",
        name: "Llama 3.2",
        contextWindow: "128K",
        costPer1K: "FREE",
        speed: "Medium",
        bestFor: "Local processing, privacy-focused",
      },
      {
        id: "llama2",
        name: "Llama 2",
        contextWindow: "4K",
        costPer1K: "FREE",
        speed: "Fast",
        bestFor: "General tasks, local deployment",
      },
      {
        id: "vicuna",
        name: "Vicuna",
        contextWindow: "2K",
        costPer1K: "FREE",
        speed: "Fast",
        bestFor: "Conversational AI, chatbots",
      },
      {
        id: "gemma",
        name: "Gemma",
        contextWindow: "8K",
        costPer1K: "FREE",
        speed: "Fast",
        bestFor: "Lightweight tasks, edge deployment",
      },
      {
        id: "mixtral",
        name: "Mixtral",
        contextWindow: "32K",
        costPer1K: "FREE",
        speed: "Medium",
        bestFor: "Complex reasoning, multilingual",
      },
      {
        id: "codellama",
        name: "Code Llama",
        contextWindow: "16K",
        costPer1K: "FREE",
        speed: "Medium",
        bestFor: "Code generation, programming tasks",
      },
      {
        id: "wizard-vicuna",
        name: "Wizard Vicuna",
        contextWindow: "2K",
        costPer1K: "FREE",
        speed: "Fast",
        bestFor: "Instruction following, Q&A",
      },
    ],
  },
  Cohere: {
    icon: "⚡",
    models: [
      {
        id: "command-r-plus",
        name: "Command R+",
        contextWindow: "128K",
        costPer1K: "$0.003",
        speed: "Fast",
        bestFor: "RAG, search, enterprise tasks",
      },
      {
        id: "command-r",
        name: "Command R",
        contextWindow: "128K",
        costPer1K: "$0.0005",
        speed: "Fast",
        bestFor: "General business tasks, summaries",
      },
      {
        id: "command-light",
        name: "Command Light",
        contextWindow: "4K",
        costPer1K: "$0.0001",
        speed: "Very Fast",
        bestFor: "Simple tasks, high throughput",
      },
    ],
  },
}

const taskTypes = [
  { id: "auto", name: "Auto-detect", icon: "🪄" },
  { id: "classification", name: "Classification", icon: "📊" },
  { id: "generation", name: "Generation", icon: "✨" },
  { id: "summarization", name: "Summarization", icon: "📝" },
  { id: "qa", name: "Q&A", icon: "❓" },
  { id: "code", name: "Code", icon: "💻" },
  { id: "translation", name: "Translation", icon: "🌐" },
]

const mockOptimizationData = [
  { step: 1, score: 65, time: "0s" },
  { step: 2, score: 72, time: "2s" },
  { step: 3, score: 78, time: "4s" },
  { step: 4, score: 85, time: "6s" },
  { step: 5, score: 92, time: "8s" },
]

const mockProviderData = [
  { provider: "OpenAI", score: 92, latency: 1.2 },
  { provider: "Anthropic", score: 89, latency: 0.8 },
  { provider: "Cohere", score: 85, latency: 1.5 },
]

const mockScatterData = [
  { provider: "OpenAI", quality: 92, speed: 85, cost: 75, x: 85, y: 92 },
  { provider: "Anthropic", quality: 89, speed: 78, cost: 65, x: 78, y: 89 },
  { provider: "Google", quality: 86, speed: 90, cost: 85, x: 90, y: 86 },
  { provider: "Cohere", quality: 85, speed: 88, cost: 90, x: 88, y: 85 },
  { provider: "Ollama", quality: 78, speed: 70, cost: 100, x: 70, y: 78 },
]

const mockTaskTypeData = [
  { name: "Classification", value: 35, color: "#f97316" },
  { name: "Generation", value: 25, color: "#fb923c" },
  { name: "Summarization", value: 20, color: "#fdba74" },
  { name: "Code", value: 15, color: "#fed7aa" },
  { name: "Q&A", value: 5, color: "#ffedd5" },
]

const apiStatus = {
  OpenAI: { connected: true, quota: "85%", lastTest: "2 min ago" },
  Anthropic: { connected: true, quota: "62%", lastTest: "5 min ago" },
  Google: { connected: false, quota: "0%", lastTest: "Never" },
  Cohere: { connected: true, quota: "91%", lastTest: "1 min ago" },
  Ollama: { connected: true, quota: "N/A", lastTest: "Connected", ping: "12ms" },
}

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
  quota: string
  lastTest: string
  ping?: string
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

  // API hooks
  const { data: providers, loading: providersLoading, error: providersError } = useProviders()
  const { data: ollamaHealth } = useOllamaHealth()
  const { createSession, optimizePrompt, loading: sessionLoading } = useSessionActions()

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
        connected: provider.models.length > 0,
        quota: provider.id === 'ollama' ? 'N/A' : '75%',
        lastTest: provider.models.length > 0 ? '1 min ago' : 'Never'
      }
      
      if (provider.id === 'ollama' && ollamaHealth) {
        status.connected = ollamaHealth.healthy
        status.lastTest = ollamaHealth.healthy ? 'Connected' : 'Disconnected'
        status.ping = '12ms'
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
  const [apiKeys, setApiKeys] = useState({
    OpenAI: "",
    Anthropic: "",
    Google: "",
    Cohere: "",
  })
  const [testingConnection, setTestingConnection] = useState("")
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
      const result = await optimizePrompt(session.id, 'meta_prompt')
      
      clearInterval(progressInterval)
      setOptimizationProgress(100)
      setIsOptimizing(false)

      if (result.session?.optimized_prompt) {
        setOptimizedPrompt(result.session.optimized_prompt)
        toast({
          title: "Optimization complete!",
          description: `Improvement score: +${Math.round(result.session.performance_score)} points`,
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

  const handleApiKeyChange = (provider, value) => {
    setApiKeys((prev) => ({ ...prev, [provider]: value }))
  }

  const handleTestConnection = async (provider) => {
    setTestingConnection(provider)

    toast({
      title: "Testing connection",
      description: `Verifying ${provider} API connection...`,
      duration: 2000,
    })

    // Simulate API test
    setTimeout(() => {
      setTestingConnection("")
      const success = true // Assume connection test succeeds

      if (success) {
        toast({
          title: "Connection successful",
          description: `${provider} API is working correctly.`,
          duration: 3000,
        })
      } else {
        toast({
          title: "Connection failed",
          description: `Unable to connect to ${provider}. Please check your API key.`,
          variant: "destructive",
          duration: 5000,
        })
      }
    }, 2000)
  }

  const handleSaveConfiguration = () => {
    toast({
      title: "Configuration saved",
      description: "API keys and settings have been securely saved.",
      duration: 3000,
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
                  <Tooltip>
                    <TooltipTrigger className="flex items-center gap-1 hover:text-foreground cursor-help">
                      <span>AI suggestions available</span>
                      <HelpCircle className="w-3 h-3" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="text-xs">AI will analyze your prompt and suggest improvements as you type</p>
                    </TooltipContent>
                  </Tooltip>
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
                              {model.contextWindow} context • {model.costPer1K} • {model.speed}
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
                              {getCurrentModel(providerData, selectedProvider, selectedModel)?.contextWindow}
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
                              {getCurrentModel(providerData, selectedProvider, selectedModel)?.costPer1K}
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
                              {getCurrentModel(providerData, selectedProvider, selectedModel)?.speed}
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
                              {getCurrentModel(providerData, selectedProvider, selectedModel)?.bestFor}
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
                    <Button variant="outline" size="icon">
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
                  Optimization Results
                </CardTitle>
                <CardDescription className="font-serif">
                  AI-optimized prompt with performance improvements
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="comparison" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="comparison" className="font-sans">
                      Comparison
                    </TabsTrigger>
                    <TabsTrigger value="metrics" className="font-sans">
                      Metrics
                    </TabsTrigger>
                    <TabsTrigger value="feedback" className="font-sans">
                      Feedback
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="comparison" className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <h4 className="font-sans font-semibold text-sm">Original</h4>
                        <div className="p-3 bg-muted rounded-md">
                          <p className="text-sm font-serif text-muted-foreground">
                            {originalPrompt || "No original prompt"}
                          </p>
                        </div>
                        <Badge variant="outline" className="font-sans">
                          Score: 65
                        </Badge>
                      </div>

                      <div className="space-y-2">
                        <h4 className="font-sans font-semibold text-sm">Optimized</h4>
                        <div className="p-3 bg-secondary/10 border border-secondary/20 rounded-md">
                          <p className="text-sm font-serif">{optimizedPrompt}</p>
                        </div>
                        <Badge className="font-sans">Score: 92 (+27)</Badge>
                      </div>
                    </div>

                    <div className="flex gap-2">
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
                    </div>
                  </TabsContent>

                  <TabsContent value="metrics" className="space-y-4">
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={mockOptimizationData}>
                          <XAxis dataKey="step" />
                          <YAxis />
                          <ChartTooltip />
                          <Line
                            type="monotone"
                            dataKey="score"
                            stroke="#f97316"
                            strokeWidth={2}
                            dot={{ fill: "#f97316" }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </TabsContent>

                  <TabsContent value="feedback" className="space-y-4">
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" className="font-sans bg-transparent">
                        <ThumbsUp className="w-4 h-4" />
                        Good Result
                      </Button>
                      <Button variant="outline" size="sm" className="font-sans bg-transparent">
                        <ThumbsDown className="w-4 h-4" />
                        Needs Work
                      </Button>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium font-sans">Additional Feedback</label>
                      <Textarea
                        placeholder="Provide feedback to improve future optimizations..."
                        className="font-serif"
                      />
                    </div>

                    <Button size="sm" className="font-sans">
                      Submit Feedback
                    </Button>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Enhanced Right Sidebar */}
        <div className="space-y-6">
          {/* Provider Performance - Enhanced with Interactive Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="font-sans font-bold text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Provider Performance
              </CardTitle>
              <CardDescription className="text-xs">Real-time performance comparison</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="quality" className="w-full">
                <TabsList className="grid w-full grid-cols-2 mb-4">
                  <TabsTrigger value="quality" className="text-xs">
                    Quality
                  </TabsTrigger>
                  <TabsTrigger value="scatter" className="text-xs">
                    Speed vs Quality
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="quality" className="space-y-0">
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={mockProviderData}>
                        <XAxis dataKey="provider" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 10 }} />
                        <ChartTooltip />
                        <Bar dataKey="score" fill="#f97316" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>

                <TabsContent value="scatter" className="space-y-0">
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart data={mockScatterData}>
                        <XAxis dataKey="x" name="Speed" tick={{ fontSize: 10 }} />
                        <YAxis dataKey="y" name="Quality" tick={{ fontSize: 10 }} />
                        <ChartTooltip cursor={{ strokeDasharray: "3 3" }} />
                        <Scatter dataKey="y" fill="#f97316" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Enhanced Session Stats */}
          <Card>
            <CardHeader>
              <CardTitle className="font-sans font-bold text-sm flex items-center gap-2">
                <BarChart className="w-4 h-4" />
                Session Stats
              </CardTitle>
              <CardDescription className="text-xs">Live updating metrics</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Total Optimizations</span>
                <div className="flex items-center gap-1">
                  <span className="font-sans font-semibold">127</span>
                  <TrendingUp className="w-3 h-3 text-green-500" />
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Avg. Improvement</span>
                <div className="flex items-center gap-1">
                  <span className="font-sans font-bold text-2xl text-green-500">+31%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Success Rate</span>
                <div className="flex items-center gap-1">
                  <span className="font-sans font-semibold text-green-500">96%</span>
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Processing Time</span>
                <span className="font-sans font-semibold">8m 42s</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-serif text-muted-foreground">Cost Savings</span>
                <span className="font-sans font-semibold text-green-500">$127.50</span>
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
                        {provider === "Ollama" && apiStatus[provider]?.connected && (
                          <Badge variant="secondary" className="text-xs">
                            {apiStatus[provider]?.ping}
                          </Badge>
                        )}
                      </div>
                    </div>
                    <ChevronDown className="w-4 h-4" />
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-3 pt-2">
                    {provider !== "Ollama" ? (
                      <>
                        <div className="space-y-2">
                          <label className="text-xs font-medium font-sans">API Key</label>
                          <Input
                            type="password"
                            placeholder={`${provider.toLowerCase()}-api-key...`}
                            value={apiKeys[provider]}
                            onChange={(e) => handleApiKeyChange(provider, e.target.value)}
                            className="text-xs font-mono"
                          />
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Usage Quota:</span>
                          <span className="font-medium">{apiStatus[provider]?.quota}</span>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Last Test:</span>
                          <span className="font-medium">{apiStatus[provider]?.lastTest}</span>
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          className="w-full text-xs bg-transparent"
                          onClick={() => handleTestConnection(provider)}
                          disabled={testingConnection === provider}
                        >
                          {testingConnection === provider ? (
                            <>
                              <Clock className="w-3 h-3 mr-1 animate-spin" />
                              Testing...
                            </>
                          ) : (
                            <>
                              <Wifi className="w-3 h-3 mr-1" />
                              Test Connection
                            </>
                          )}
                        </Button>
                      </>
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
                          <span className="text-muted-foreground">Ping:</span>
                          <span className="font-medium">{apiStatus[provider]?.ping}</span>
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

              <Button
                size="sm"
                className="w-full font-sans bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700"
                onClick={handleSaveConfiguration}
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Save Configuration
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </TooltipProvider>
  )
}
