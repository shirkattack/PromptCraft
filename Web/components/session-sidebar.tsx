"use client"

import { useState, useMemo, useEffect, useRef, type ChangeEvent } from "react"
import { useSessions, usePerformanceMetrics, useTrainingDatasets, useProviders, notifyTrainingChanged, requestLoadPrompt, requestNewOptimization, requestOpenSession, OPEN_SESSION } from "@/lib/api/hooks"
import apiClient, { type OptimizationSession, type TrainingDatasetSummary, type TrainingSample, type DatasetFileFormat } from "@/lib/api/client"
import { getRelativeTime, parseApiDate } from "@/lib/utils"
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarFooter,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from "recharts"
import { ChartTooltip } from "@/components/ui/chart"
import { toast } from "@/hooks/use-toast"
import {
  Plus,
  Search,
  FileText,
  Clock,
  Settings,
  BarChart3,
  Database,
  Zap,
  Eye,
  Download,
  Upload,
  Trash2,
  MoreHorizontal,
  Loader2,
  Sparkles,
  HelpCircle,
  ExternalLink,
} from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

// Mirrors the task types offered by the dashboard; the API stores any string.
const TASK_TYPES = [
  "general",
  "classification",
  "generation",
  "summarization",
  "qa",
  "code",
  "translation",
  "analysis",
  "creative",
]

const CHART_COLORS = ["#f97316", "#fb923c", "#fdba74", "#fed7aa", "#ffedd5"]

const NEW_DATASET = "__new__"

const titleCase = (value: string) => (value === "qa" ? "Q&A" : value.charAt(0).toUpperCase() + value.slice(1))

const percent = (value: number | null | undefined) => (value == null ? null : Math.round(value * 100))

const getScoreColor = (score: number) => {
  if (score >= 90) return "bg-green-500"
  if (score >= 70) return "bg-yellow-500"
  return "bg-red-500"
}

/** Trigger a browser download of text content. */
const downloadText = (filename: string, content: string, mime: string) => {
  const url = URL.createObjectURL(new Blob([content], { type: mime }))
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

const safeFilename = (name: string) => name.replace(/[^a-z0-9-_]+/gi, "_").toLowerCase() || "dataset"

const errorMessage = (err: unknown, fallback: string) => (err instanceof Error ? err.message : fallback)

interface ImportForm {
  name: string
  task_type: string
  file_format: DatasetFileFormat
  data: string
}

interface GenerateForm {
  target: string // NEW_DATASET or an existing dataset id
  name: string
  task_type: string
  base_prompt: string
  sample_count: number
  model: string // "<providerId>::<modelId>"
}

const EMPTY_IMPORT: ImportForm = { name: "", task_type: "general", file_format: "json", data: "" }
const EMPTY_GENERATE: GenerateForm = {
  target: NEW_DATASET,
  name: "",
  task_type: "general",
  base_prompt: "",
  sample_count: 10,
  model: "",
}

/** Everything the Analytics tab shows, derived from real sessions only. */
function buildAnalytics(sessions: OptimizationSession[]) {
  const completed = sessions.filter((s) => s.status === "completed")

  const byModel = new Map<string, { runs: number; completed: number; scoreTotal: number }>()
  for (const session of sessions) {
    const entry = byModel.get(session.model) ?? { runs: 0, completed: 0, scoreTotal: 0 }
    entry.runs += 1
    if (session.status === "completed") {
      entry.completed += 1
      entry.scoreTotal += session.performance_score
    }
    byModel.set(session.model, entry)
  }
  const modelPerformance = Array.from(byModel.entries())
    .map(([model, stats]) => ({
      model,
      runs: stats.runs,
      avgScore: stats.completed ? Math.round(stats.scoreTotal / stats.completed) : null,
    }))
    .sort((a, b) => b.runs - a.runs)

  const byTask = new Map<string, number>()
  for (const session of sessions) byTask.set(session.task_type, (byTask.get(session.task_type) ?? 0) + 1)
  const taskTypeDistribution = Array.from(byTask.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([name, count], index) => ({
      name: titleCase(name),
      count,
      value: sessions.length ? Math.round((count / sessions.length) * 100) : 0,
      color: CHART_COLORS[index % CHART_COLORS.length],
    }))

  // Success rate per calendar day, oldest first, capped to the last 14 days with data.
  const byDay = new Map<string, { total: number; completed: number }>()
  for (const session of sessions) {
    const day = session.created_at.slice(0, 10)
    const entry = byDay.get(day) ?? { total: 0, completed: 0 }
    entry.total += 1
    if (session.status === "completed") entry.completed += 1
    byDay.set(day, entry)
  }
  const successRateOverTime = Array.from(byDay.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-14)
    .map(([day, stats]) => ({
      date: day.slice(5), // MM-DD
      rate: Math.round((stats.completed / stats.total) * 100),
      runs: stats.total,
    }))

  const recentActivity = [...sessions]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .slice(0, 5)

  return { completedCount: completed.length, modelPerformance, taskTypeDistribution, successRateOverTime, recentActivity }
}

const activityLabel = (session: OptimizationSession) => {
  if (session.status === "completed") return "Optimization completed"
  if (session.status === "running") return "Optimization running"
  return "Optimization failed"
}

export function SessionSidebar() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState("optimization")
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([])
  const [mounted, setMounted] = useState(false)
  const { open } = useSidebar()

  // Fetch data from API
  const { data: sessions, loading: sessionsLoading, error: sessionsError, refetch: refetchSessions } = useSessions()
  const { data: performanceMetrics } = usePerformanceMetrics()
  const { data: datasets, loading: datasetsLoading, error: datasetsError, refetch: refetchDatasets } = useTrainingDatasets()
  const { data: providers } = useProviders()

  // Dataset dialogs are lifted to the component root; nesting them inside the
  // per-row dropdown menu unmounts them when the menu closes.
  const [previewDataset, setPreviewDataset] = useState<TrainingDatasetSummary | null>(null)
  const [previewSamples, setPreviewSamples] = useState<TrainingSample[] | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<TrainingDatasetSummary | null>(null)
  const [importOpen, setImportOpen] = useState(false)
  const [importForm, setImportForm] = useState<ImportForm>(EMPTY_IMPORT)
  const [generateOpen, setGenerateOpen] = useState(false)
  const [generateForm, setGenerateForm] = useState<GenerateForm>(EMPTY_GENERATE)
  const [busy, setBusy] = useState(false)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const promptFileInput = useRef<HTMLInputElement>(null)

  // Keep the highlighted session in sync when the dashboard opens one.
  useEffect(() => {
    const onOpen = (event: Event) => setActiveSessionId((event as CustomEvent<{ sessionId: string }>).detail.sessionId)
    window.addEventListener(OPEN_SESSION, onOpen)
    return () => window.removeEventListener(OPEN_SESSION, onOpen)
  }, [])

  const handleImportPromptFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return
    file.text().then((text) => {
      requestLoadPrompt(text.trim(), file.name)
      toast({ title: "Prompt loaded", description: `${file.name} is in the prompt box. Edit it, then optimize.`, duration: 3000 })
    })
  }

  // Prevent hydration mismatch by only rendering after mount
  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!previewDataset) {
      setPreviewSamples(null)
      setPreviewError(null)
      return
    }
    let cancelled = false
    apiClient
      .getSamples(previewDataset.id, 25)
      .then((samples) => {
        if (!cancelled) setPreviewSamples(samples)
      })
      .catch((err) => {
        if (!cancelled) setPreviewError(errorMessage(err, "Failed to load samples"))
      })
    return () => {
      cancelled = true
    }
  }, [previewDataset])

  const modelOptions = useMemo(
    () =>
      (providers ?? [])
        .filter((provider) => provider.available)
        .flatMap((provider) =>
          provider.models.map((model) => ({
            value: `${provider.id}::${model.id}`,
            label: provider.models.length > 1 || (providers ?? []).length > 1 ? `${model.name} (${provider.name})` : model.name,
          })),
        ),
    [providers],
  )

  useEffect(() => {
    if (!generateForm.model && modelOptions.length > 0) {
      setGenerateForm((form) => ({ ...form, model: modelOptions[0].value }))
    }
  }, [modelOptions, generateForm.model])

  // Transform API sessions to match component expectations
  const transformedSessions = useMemo(() => {
    if (!sessions) return []
    return [...sessions]
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .map((session) => ({
        id: session.id,
        name: session.name,
        date: session.created_at.slice(0, 10),
        score: Math.round(session.performance_score),
        provider: session.provider,
        model: session.model,
        status: session.status === "completed" ? "completed" : session.status === "running" ? "in-progress" : "failed",
      }))
  }, [sessions])

  const filteredSessions = mounted
    ? transformedSessions.filter((session) => session.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : []

  const filteredDatasets = mounted
    ? (datasets ?? []).filter((dataset) => dataset.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : []

  const analytics = useMemo(() => buildAnalytics(mounted && sessions ? sessions : []), [sessions, mounted])

  const getProviderIcon = (provider: string) => provider.charAt(0).toUpperCase()

  const handleDatasetSelect = (datasetId: string, checked: boolean | "indeterminate") => {
    setSelectedDatasets((current) =>
      checked === true ? Array.from(new Set([...current, datasetId])) : current.filter((id) => id !== datasetId),
    )
  }

  const handleSelectAll = (checked: boolean | "indeterminate") => {
    setSelectedDatasets(checked === true ? filteredDatasets.map((d) => d.id) : [])
  }

  const deleteDatasets = async (ids: string[]) => {
    setBusy(true)
    const failures: string[] = []
    for (const id of ids) {
      try {
        await apiClient.deleteDataset(id)
      } catch (err) {
        failures.push(errorMessage(err, id))
      }
    }
    setBusy(false)
    setSelectedDatasets((current) => current.filter((id) => !ids.includes(id) || failures.length > 0))
    notifyTrainingChanged()
    if (failures.length) {
      toast({ title: "Some datasets were not deleted", description: failures.join("; "), variant: "destructive" })
    } else {
      toast({ title: ids.length === 1 ? "Dataset deleted" : `${ids.length} datasets deleted` })
    }
  }

  const handleExportDataset = async (dataset: TrainingDatasetSummary, format: DatasetFileFormat) => {
    try {
      const exported = await apiClient.exportDataset(dataset.id, format)
      downloadText(
        `${safeFilename(dataset.name)}.${format}`,
        exported.data,
        format === "json" ? "application/json" : "text/csv",
      )
      toast({ title: "Export ready", description: `${exported.sample_count} samples as ${format.toUpperCase()}` })
    } catch (err) {
      toast({ title: "Export failed", description: errorMessage(err, "Unknown error"), variant: "destructive" })
    }
  }

  const handleImportFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    const extension = file.name.split(".").pop()?.toLowerCase()
    file.text().then((text) =>
      setImportForm((form) => ({
        ...form,
        data: text,
        file_format: extension === "csv" ? "csv" : extension === "jsonl" ? "jsonl" : extension === "json" ? "json" : form.file_format,
        name: form.name || file.name.replace(/\.[^.]+$/, ""),
      })),
    )
  }

  const handleImport = async () => {
    if (!importForm.name.trim() || !importForm.data.trim()) {
      toast({ title: "Name and data are required", variant: "destructive" })
      return
    }
    setBusy(true)
    try {
      const created = await apiClient.importDataset({
        name: importForm.name.trim(),
        task_type: importForm.task_type,
        file_format: importForm.file_format,
        data: importForm.data,
      })
      notifyTrainingChanged()
      toast({ title: "Dataset imported", description: `${created.sample_count} samples in "${created.name}"` })
      setImportOpen(false)
      setImportForm(EMPTY_IMPORT)
    } catch (err) {
      toast({ title: "Import failed", description: errorMessage(err, "Unknown error"), variant: "destructive" })
    } finally {
      setBusy(false)
    }
  }

  const handleGenerate = async () => {
    const isNew = generateForm.target === NEW_DATASET
    if (!generateForm.base_prompt.trim() || (isNew && !generateForm.name.trim())) {
      toast({ title: "A base prompt and a dataset name are required", variant: "destructive" })
      return
    }
    const [provider, model] = generateForm.model.split("::")
    setBusy(true)
    let datasetId = generateForm.target
    try {
      if (isNew) {
        const created = await apiClient.createDataset({
          name: generateForm.name.trim(),
          task_type: generateForm.task_type,
          description: `Generated from: ${generateForm.base_prompt.trim().slice(0, 120)}`,
        })
        datasetId = created.id
        notifyTrainingChanged()
      }
      const result = await apiClient.generateSamples(datasetId, {
        sample_count: generateForm.sample_count,
        base_prompt: generateForm.base_prompt.trim(),
        task_type: generateForm.task_type,
        ...(provider && model ? { provider, model } : {}),
      })
      notifyTrainingChanged()
      toast({
        title: "Samples generated",
        description: [
          `${result.generated_count} added`,
          result.rejected_duplicates ? `${result.rejected_duplicates} near-duplicate${result.rejected_duplicates === 1 ? "" : "s"} rejected` : null,
          result.failed_count ? `${result.failed_count} failed` : null,
          `in ${result.processing_time.toFixed(1)}s`,
        ]
          .filter(Boolean)
          .join(", "),
      })
      setGenerateOpen(false)
      setGenerateForm({ ...EMPTY_GENERATE, model: generateForm.model })
    } catch (err) {
      notifyTrainingChanged()
      toast({ title: "Generation failed", description: errorMessage(err, "Unknown error"), variant: "destructive" })
    } finally {
      setBusy(false)
    }
  }

  const renderDatasetMeta = (dataset: TrainingDatasetSummary) => {
    const quality = percent(dataset.avg_quality_score)
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground font-serif">
        <span>{dataset.sample_count} samples</span>
        {quality != null && (
          <>
            <span>•</span>
            <span className={`${getScoreColor(quality)} text-white px-1 rounded`}>{quality}%</span>
          </>
        )}
      </div>
    )
  }

  return (
    <Sidebar className="w-80" collapsible="icon">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-col h-full" suppressHydrationWarning>
        <SidebarHeader className="border-b">
          <div className="flex items-center justify-between">
            {open && (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg flex items-center justify-center animate-pulse">
                    <Zap className="w-4 h-4 text-white" />
                  </div>
                  <h2 className="font-sans font-bold text-lg bg-gradient-to-r from-orange-500 to-orange-600 bg-clip-text text-transparent">
                    PromptCraft Studio
                  </h2>
                </div>
              </div>
            )}
            <SidebarTrigger />
          </div>

          {open && (
            <div className="pt-4">
              <TabsList className="grid w-full grid-cols-3 bg-muted/50">
                <TabsTrigger value="optimization" className="text-xs font-sans">
                  <Zap className="w-3 h-3 mr-1" />
                  Optimization
                </TabsTrigger>
                <TabsTrigger value="training" className="text-xs font-sans">
                  <Database className="w-3 h-3 mr-1" />
                  Training Data
                </TabsTrigger>
                <TabsTrigger value="analytics" className="text-xs font-sans">
                  <BarChart3 className="w-3 h-3 mr-1" />
                  Analytics
                </TabsTrigger>
              </TabsList>
            </div>
          )}
        </SidebarHeader>

        <SidebarContent>
          {open ? (
            <>
              <TabsContent value="optimization" className="mt-0 space-y-4">
                <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Quick Actions</SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    <SidebarMenuItem>
                      <SidebarMenuButton
                        className="font-sans bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-600 hover:to-orange-700"
                        onClick={() => {
                          setActiveSessionId(null)
                          requestNewOptimization()
                        }}
                      >
                        <Plus className="w-4 h-4" />
                        New Optimization
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans" onClick={() => promptFileInput.current?.click()}>
                        <FileText className="w-4 h-4" />
                        Import Prompt
                      </SidebarMenuButton>
                      <input ref={promptFileInput} type="file" accept=".txt,.md,.prompt,text/plain,text/markdown" className="hidden" onChange={handleImportPromptFile} />
                    </SidebarMenuItem>
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Recent Sessions</SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="px-2 pb-2">
                    <div className="relative">
                      <Search className="absolute left-2 top-2.5 w-4 h-4 text-muted-foreground" />
                      <Input
                        placeholder="Search sessions..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-8 h-8 text-sm font-serif"
                      />
                    </div>
                  </div>

                  <SidebarMenu suppressHydrationWarning>
                    {!mounted || sessionsLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <div className="text-sm text-muted-foreground">Loading sessions...</div>
                      </div>
                    ) : sessionsError ? (
                      <div className="flex flex-col items-center justify-center gap-2 py-8 px-3 text-center">
                        <div className="text-sm text-red-500">Could not load sessions: {sessionsError}</div>
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => refetchSessions()}>
                          Retry
                        </Button>
                      </div>
                    ) : filteredSessions.length === 0 ? (
                      <div className="flex items-center justify-center py-8">
                        <div className="text-sm text-muted-foreground">No sessions found</div>
                      </div>
                    ) : filteredSessions.map((session) => (
                      <SidebarMenuItem key={session.id}>
                        <SidebarMenuButton
                          className={`flex-col items-start h-auto p-3 hover:bg-accent/50 group ${activeSessionId === session.id ? "bg-accent" : ""}`}
                          isActive={activeSessionId === session.id}
                          title="Open this session's prompt and result"
                          onClick={() => {
                            setActiveSessionId(session.id)
                            requestOpenSession(session.id)
                          }}
                        >
                          <div className="flex items-center justify-between w-full">
                            <span className="font-sans font-medium text-sm truncate">{session.name}</span>
                            <Badge className={`text-xs font-sans text-white ${getScoreColor(session.score)}`}>
                              {session.score}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground font-serif">
                            <Clock className="w-3 h-3" />
                            {session.date}
                            <span>•</span>
                            <div className="flex items-center gap-1">
                              <div className="w-4 h-4 bg-muted rounded-full flex items-center justify-center text-xs">
                                {getProviderIcon(session.provider)}
                              </div>
                              <span>{session.model}</span>
                            </div>
                          </div>
                          <div className="opacity-0 group-hover:opacity-100 transition-opacity text-xs text-muted-foreground mt-1">
                            {session.status === "completed" ? "Click to reopen" : `Status: ${session.status}`}
                          </div>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>
                </SidebarGroupContent>
                </SidebarGroup>
              </TabsContent>

              <TabsContent value="training" className="mt-0 space-y-4">
                <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    Training Datasets
                    <Badge variant="secondary" className="text-xs">
                      {datasets?.length ?? 0}
                    </Badge>
                  </div>
                  {selectedDatasets.length > 0 && (
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-red-600 hover:text-red-700" disabled={busy}>
                          <Trash2 className="w-3 h-3 mr-1" />
                          {selectedDatasets.length}
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete Selected Datasets</AlertDialogTitle>
                          <AlertDialogDescription>
                            Delete {selectedDatasets.length} selected dataset{selectedDatasets.length === 1 ? "" : "s"} and all
                            of their samples? This cannot be undone.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction onClick={() => deleteDatasets(selectedDatasets)} className="bg-red-600 hover:bg-red-700">
                            Delete
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  )}
                </SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="px-2 pb-2">
                    <div className="relative">
                      <Search className="absolute left-2 top-2.5 w-4 h-4 text-muted-foreground" />
                      <Input
                        placeholder="Search datasets..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-8 h-8 text-sm font-serif"
                      />
                    </div>
                  </div>

                  {filteredDatasets.length > 0 && (
                    <div className="px-2 pb-2">
                      <div className="flex items-center gap-2">
                        <Checkbox
                          checked={selectedDatasets.length > 0 && selectedDatasets.length === filteredDatasets.length}
                          onCheckedChange={handleSelectAll}
                          className="h-4 w-4"
                        />
                        <span className="text-xs text-muted-foreground font-sans">
                          Select all ({selectedDatasets.length} selected)
                        </span>
                      </div>
                    </div>
                  )}

                  <SidebarMenu>
                    {!mounted || datasetsLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <div className="text-sm text-muted-foreground">Loading datasets...</div>
                      </div>
                    ) : datasetsError ? (
                      <div className="flex flex-col items-center justify-center gap-2 py-8 px-3 text-center">
                        <div className="text-sm text-red-500">Could not load datasets: {datasetsError}</div>
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => refetchDatasets()}>
                          Retry
                        </Button>
                      </div>
                    ) : filteredDatasets.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-8 px-3 text-center gap-1">
                        <div className="text-sm text-muted-foreground">
                          {datasets?.length ? "No datasets match your search" : "No datasets yet"}
                        </div>
                        {!datasets?.length && (
                          <div className="text-xs text-muted-foreground">Generate one from a prompt or import JSON/CSV below.</div>
                        )}
                      </div>
                    ) : filteredDatasets.map((dataset) => (
                      <SidebarMenuItem key={dataset.id}>
                        <div className="flex items-start gap-2 p-3 hover:bg-accent/50 rounded-md">
                          <Checkbox
                            checked={selectedDatasets.includes(dataset.id)}
                            onCheckedChange={(checked) => handleDatasetSelect(dataset.id, checked)}
                            className="h-4 w-4 mt-1"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between w-full gap-1">
                              <span className="font-sans font-medium text-sm truncate" title={dataset.name}>{dataset.name}</span>
                              <div className="flex items-center gap-1 shrink-0">
                                <Badge variant="outline" className="text-xs">
                                  {titleCase(dataset.task_type)}
                                </Badge>
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                                      <MoreHorizontal className="w-3 h-3" />
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent align="end">
                                    <DropdownMenuItem onClick={() => setPreviewDataset(dataset)}>
                                      <Eye className="w-4 h-4 mr-2" />
                                      Preview samples
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onClick={() => {
                                        setGenerateForm((form) => ({ ...form, target: dataset.id, task_type: dataset.task_type }))
                                        setGenerateOpen(true)
                                      }}
                                    >
                                      <Sparkles className="w-4 h-4 mr-2" />
                                      Generate more samples
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem onClick={() => handleExportDataset(dataset, "json")}>
                                      <Download className="w-4 h-4 mr-2" />
                                      Export JSON
                                    </DropdownMenuItem>
                                    <DropdownMenuItem onClick={() => handleExportDataset(dataset, "csv")}>
                                      <Download className="w-4 h-4 mr-2" />
                                      Export CSV
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem onClick={() => setDeleteTarget(dataset)} className="text-red-600">
                                      <Trash2 className="w-4 h-4 mr-2" />
                                      Delete
                                    </DropdownMenuItem>
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              </div>
                            </div>
                            {renderDatasetMeta(dataset)}
                            <div className="text-xs text-muted-foreground">
                              Updated {getRelativeTime(parseApiDate(dataset.last_modified))}
                            </div>
                          </div>
                        </div>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>

                  <div className="px-2 pt-2 space-y-2">
                    <SidebarMenuButton
                      className="w-full font-sans bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-600 hover:to-orange-700"
                      onClick={() => {
                        setGenerateForm((form) => ({ ...form, target: NEW_DATASET }))
                        setGenerateOpen(true)
                      }}
                    >
                      <Sparkles className="w-4 h-4" />
                      Generate Dataset
                    </SidebarMenuButton>
                    <SidebarMenuButton className="w-full font-sans" variant="outline" onClick={() => setImportOpen(true)}>
                      <Upload className="w-4 h-4" />
                      Import Dataset
                    </SidebarMenuButton>
                  </div>
                </SidebarGroupContent>
                </SidebarGroup>
              </TabsContent>

              <TabsContent value="analytics" className="mt-0 space-y-4">
                <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold flex items-center justify-between">
                  Performance Overview
                  <Badge variant="secondary" className="text-xs">
                    all time
                  </Badge>
                </SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-4" suppressHydrationWarning>
                    <div className="grid grid-cols-2 gap-3 text-center">
                      <div className="space-y-1">
                        <div className="font-sans font-bold text-xl text-green-500">
                          {performanceMetrics ? `${Math.round(performanceMetrics.success_rate)}%` : "—"}
                        </div>
                        <div className="text-xs text-muted-foreground">Success Rate</div>
                      </div>
                      <div className="space-y-1">
                        <div className="font-sans font-bold text-xl text-orange-500">
                          {performanceMetrics ? `+${Math.round(performanceMetrics.average_improvement)}%` : "—"}
                        </div>
                        <div className="text-xs text-muted-foreground">Avg Improvement</div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-xs font-medium font-sans">Success Rate by Day</div>
                      {analytics.successRateOverTime.length >= 2 ? (
                        <div className="h-24">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={analytics.successRateOverTime}>
                              <Line type="monotone" dataKey="rate" stroke="#f97316" strokeWidth={2} dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <div className="text-xs text-muted-foreground">Runs on at least two different days are needed for a trend.</div>
                      )}
                    </div>
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Model Performance</SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-3">
                    {analytics.modelPerformance.length === 0 && (
                      <div className="text-xs text-muted-foreground px-2">No optimizations yet.</div>
                    )}
                    {analytics.modelPerformance.slice(0, 3).map((entry) => (
                      <div key={entry.model} className="flex items-center justify-between p-2 rounded-md hover:bg-accent/50">
                        <div className="flex items-center gap-2 min-w-0">
                          <div className="w-6 h-6 bg-muted rounded-full flex items-center justify-center text-xs shrink-0">
                            {entry.model.charAt(0).toUpperCase()}
                          </div>
                          <div className="min-w-0">
                            <div className="font-sans text-sm font-medium truncate" title={entry.model}>{entry.model}</div>
                            <div className="text-xs text-muted-foreground">{entry.runs} run{entry.runs === 1 ? "" : "s"}</div>
                          </div>
                        </div>
                        <div className="text-right shrink-0">
                          <div className="font-sans text-sm font-semibold">{entry.avgScore ?? "—"}</div>
                          <div className="text-xs text-muted-foreground">avg score</div>
                        </div>
                      </div>
                    ))}

                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm" className="w-full text-xs font-sans bg-transparent">
                          <BarChart3 className="w-3 h-3 mr-1" />
                          View Detailed Analytics
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-4xl">
                        <DialogHeader>
                          <DialogTitle className="font-sans">Detailed Analytics</DialogTitle>
                          <DialogDescription>Computed from every session stored in the local database.</DialogDescription>
                        </DialogHeader>
                        <div className="space-y-6">
                          <div className="grid grid-cols-3 gap-4">
                            <div className="text-center space-y-1">
                              <div className="font-sans font-bold text-2xl">{performanceMetrics?.total_optimizations ?? 0}</div>
                              <div className="text-xs text-muted-foreground">Total Optimizations</div>
                            </div>
                            <div className="text-center space-y-1">
                              <div className="font-sans font-bold text-2xl text-green-500">{analytics.completedCount}</div>
                              <div className="text-xs text-muted-foreground">Successful</div>
                            </div>
                            <div className="text-center space-y-1">
                              <div className="font-sans font-bold text-2xl text-orange-500">
                                +{Math.round(performanceMetrics?.average_improvement ?? 0)}%
                              </div>
                              <div className="text-xs text-muted-foreground">Avg Improvement</div>
                            </div>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-3">
                              <h4 className="font-sans font-semibold text-sm">Average Score by Model</h4>
                              <div className="h-48">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart data={analytics.modelPerformance}>
                                    <XAxis dataKey="model" tick={{ fontSize: 10 }} />
                                    <YAxis tick={{ fontSize: 10 }} domain={[0, 100]} />
                                    <ChartTooltip />
                                    <Bar dataKey="avgScore" fill="#f97316" radius={[2, 2, 0, 0]} />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </div>

                            <div className="space-y-3">
                              <h4 className="font-sans font-semibold text-sm">Sessions by Task Type</h4>
                              <div className="h-48">
                                <ResponsiveContainer width="100%" height="100%">
                                  <PieChart>
                                    <Pie
                                      data={analytics.taskTypeDistribution}
                                      cx="50%"
                                      cy="50%"
                                      innerRadius={30}
                                      outerRadius={70}
                                      dataKey="count"
                                      nameKey="name"
                                    >
                                      {analytics.taskTypeDistribution.map((entry) => (
                                        <Cell key={entry.name} fill={entry.color} />
                                      ))}
                                    </Pie>
                                    <ChartTooltip />
                                  </PieChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          </div>

                          <div className="space-y-3">
                            <h4 className="font-sans font-semibold text-sm">Success Rate by Day</h4>
                            {analytics.successRateOverTime.length >= 2 ? (
                              <div className="h-32">
                                <ResponsiveContainer width="100%" height="100%">
                                  <LineChart data={analytics.successRateOverTime}>
                                    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                                    <YAxis tick={{ fontSize: 10 }} domain={[0, 100]} />
                                    <ChartTooltip />
                                    <Line type="monotone" dataKey="rate" stroke="#f97316" strokeWidth={2} dot={{ fill: "#f97316" }} />
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                            ) : (
                              <div className="text-xs text-muted-foreground">Runs on at least two different days are needed for a trend.</div>
                            )}
                          </div>
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Task Types</SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-2">
                    {analytics.taskTypeDistribution.length === 0 && (
                      <div className="text-xs text-muted-foreground px-2">No optimizations yet.</div>
                    )}
                    {analytics.taskTypeDistribution.slice(0, 4).map((task) => (
                      <div key={task.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: task.color }} />
                          <span className="text-xs font-serif">{task.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-muted rounded-full h-1">
                            <div className="h-1 rounded-full" style={{ backgroundColor: task.color, width: `${task.value}%` }} />
                          </div>
                          <span className="text-xs text-muted-foreground w-8">{task.value}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Recent Activity</SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-2">
                    {analytics.recentActivity.length === 0 && (
                      <div className="text-xs text-muted-foreground px-2">Nothing yet. Run an optimization to see it here.</div>
                    )}
                    {analytics.recentActivity.map((session) => (
                      <div key={session.id} className="flex items-start gap-2 p-2 rounded-md hover:bg-accent/50">
                        <div
                          className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                            session.status === "completed" ? "bg-green-500" : session.status === "running" ? "bg-orange-500" : "bg-red-500"
                          }`}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium font-sans truncate">{activityLabel(session)}</div>
                          <div className="text-xs text-muted-foreground font-serif truncate">{session.name}</div>
                          <div className="text-xs text-muted-foreground">{getRelativeTime(parseApiDate(session.created_at))}</div>
                        </div>
                        {session.status === "completed" && (
                          <Badge variant="secondary" className="text-xs">
                            {Math.round(session.performance_score)}
                          </Badge>
                        )}
                      </div>
                    ))}
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>
              </TabsContent>
            </>
          ) : (
            <div className="space-y-2">
              <SidebarGroup>
                <SidebarGroupContent>
                  <SidebarMenu>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans">
                        <Zap className="w-4 h-4" />
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans">
                        <Database className="w-4 h-4" />
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans">
                        <BarChart3 className="w-4 h-4" />
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            </div>
          )}
        </SidebarContent>
      </Tabs>

      <SidebarFooter className="border-t">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton className="font-sans">
              <Settings className="w-4 h-4" />
              {open && "Settings"}
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>

      {/* Preview samples */}
      <Dialog open={previewDataset !== null} onOpenChange={(isOpen) => !isOpen && setPreviewDataset(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="font-sans">{previewDataset?.name}</DialogTitle>
            <DialogDescription>
              {previewDataset && (
                <>
                  {previewDataset.sample_count} samples • {titleCase(previewDataset.task_type)}
                  {percent(previewDataset.avg_quality_score) != null && ` • Quality: ${percent(previewDataset.avg_quality_score)}%`}
                  {previewDataset.description && ` • ${previewDataset.description}`}
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {previewError ? (
              <div className="text-sm text-red-500">{previewError}</div>
            ) : previewSamples === null ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading samples...
              </div>
            ) : previewSamples.length === 0 ? (
              <div className="text-sm text-muted-foreground">This dataset has no samples yet.</div>
            ) : (
              previewSamples.map((sample) => (
                <div key={sample.id} className="border rounded-lg p-3 space-y-2">
                  <div>
                    <div className="text-xs font-medium text-muted-foreground">Input</div>
                    <p className="text-sm font-serif whitespace-pre-wrap">{sample.input_text}</p>
                  </div>
                  <div>
                    <div className="text-xs font-medium text-muted-foreground">Expected output</div>
                    <p className="text-sm font-serif whitespace-pre-wrap">{sample.expected_output}</p>
                  </div>
                  {sample.quality_score != null && sample.quality_score > 0 && (
                    <div className="text-xs text-muted-foreground">Quality {percent(sample.quality_score)}%</div>
                  )}
                </div>
              ))
            )}
            {previewDataset && previewSamples && previewSamples.length < previewDataset.sample_count && (
              <div className="text-xs text-muted-foreground">
                Showing the first {previewSamples.length} of {previewDataset.sample_count}. Export to see them all.
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => previewDataset && handleExportDataset(previewDataset, "json")}>
              <Download className="w-4 h-4 mr-2" />
              Export JSON
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete one dataset */}
      <AlertDialog open={deleteTarget !== null} onOpenChange={(isOpen) => !isOpen && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Dataset</AlertDialogTitle>
            <AlertDialogDescription>
              Delete &quot;{deleteTarget?.name}&quot; and its {deleteTarget?.sample_count ?? 0} samples? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteTarget) deleteDatasets([deleteTarget.id])
                setDeleteTarget(null)
              }}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Import */}
      <Dialog open={importOpen} onOpenChange={(isOpen) => !busy && setImportOpen(isOpen)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="font-sans flex items-center gap-2">
              Import Dataset
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="w-4 h-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="bottom" align="start">
                  <div className="max-w-sm text-xs space-y-2">
                    <p className="font-medium">What a dataset is</p>
                    <p>
                      A list of <b>inputs</b> and the <b>output</b> you expect for each. Your prompt is the instruction that should turn an
                      input into its output; the optimizer scores candidates on samples it did not see.
                    </p>
                    <p className="font-medium">Accepted layouts</p>
                    <pre className="rounded bg-muted p-2 whitespace-pre-wrap font-mono">{`JSON   [{"input": "…", "output": "…"}, …]
JSONL  {"input": "…", "output": "…"}  (one per line)
CSV    input,output
       "Server is down",high`}</pre>
                    <p>
                      Key aliases work too: <code>prompt/response</code>, <code>question/answer</code>, <code>text/label</code>. Any other
                      fields are kept as extra data. JSON Lines is handy for large or appended files; otherwise the formats are equivalent.
                    </p>
                    <p>10 to 20 varied samples are enough to start; label datasets should include every label.</p>
                  </div>
                </TooltipContent>
              </Tooltip>
            </DialogTitle>
            <DialogDescription>
              Inputs and the outputs you expect. Hover the question mark for the formats, or start from a template.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-wrap gap-2 text-xs">
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              onClick={() =>
                downloadText(
                  "dataset-template.json",
                  JSON.stringify(
                    [
                      { input: "Production database is down, all customers affected", output: "high" },
                      { input: "Question about how billing cycles work", output: "medium" },
                      { input: "Thanks for the quick help yesterday!", output: "low" },
                    ],
                    null,
                    2,
                  ),
                  "application/json",
                )
              }
            >
              <Download className="w-3 h-3 mr-1" /> JSON template
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              onClick={() =>
                downloadText(
                  "dataset-template.csv",
                  'input,output\n"Production database is down, all customers affected",high\n"Question about how billing cycles work",medium\n"Thanks for the quick help yesterday!",low\n',
                  "text/csv",
                )
              }
            >
              <Download className="w-3 h-3 mr-1" /> CSV template
            </Button>
            <Button variant="ghost" size="sm" className="h-7 text-xs" asChild>
              <a href="https://github.com/shirkattack/PromptCraft/blob/main/docs/examples/support-tickets.csv" target="_blank" rel="noreferrer">
                <ExternalLink className="w-3 h-3 mr-1" /> Example dataset
              </a>
            </Button>
          </div>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-medium font-sans">Name</label>
                <Input
                  value={importForm.name}
                  onChange={(e) => setImportForm((form) => ({ ...form, name: e.target.value }))}
                  placeholder="Support ticket triage"
                  className="h-8 text-sm"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium font-sans">Task type</label>
                <Select value={importForm.task_type} onValueChange={(value) => setImportForm((form) => ({ ...form, task_type: value }))}>
                  <SelectTrigger className="h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TASK_TYPES.map((type) => (
                      <SelectItem key={type} value={type}>
                        {titleCase(type)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-medium font-sans">Format</label>
                <Select
                  value={importForm.file_format}
                  onValueChange={(value) => setImportForm((form) => ({ ...form, file_format: value as DatasetFileFormat }))}
                >
                  <SelectTrigger className="h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="json">JSON</SelectItem>
                    <SelectItem value="jsonl">JSON Lines</SelectItem>
                    <SelectItem value="csv">CSV</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium font-sans">From file</label>
                <Input type="file" accept=".json,.jsonl,.csv,.txt" onChange={handleImportFile} className="h-8 text-xs" />
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium font-sans">Data</label>
              <Textarea
                value={importForm.data}
                onChange={(e) => setImportForm((form) => ({ ...form, data: e.target.value }))}
                placeholder={
                  importForm.file_format === "csv"
                    ? "input,output\n..., ..."
                    : importForm.file_format === "jsonl"
                      ? '{"input": "...", "output": "..."}\n{"input": "...", "output": "..."}'
                      : '[{"input": "...", "output": "..."}]'
                }
                className="font-mono text-xs min-h-[160px]"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setImportOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button onClick={handleImport} disabled={busy}>
              {busy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Upload className="w-4 h-4 mr-2" />}
              Import
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Generate synthetic samples */}
      <Dialog open={generateOpen} onOpenChange={(isOpen) => !busy && setGenerateOpen(isOpen)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="font-sans">Generate Synthetic Samples</DialogTitle>
            <DialogDescription>
              The selected model writes input/output pairs in the style of your base prompt. Each sample is one model call, so
              large counts take a while.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="space-y-1">
              <label className="text-xs font-medium font-sans">Dataset</label>
              <Select
                value={generateForm.target}
                onValueChange={(value) => {
                  const existing = datasets?.find((d) => d.id === value)
                  setGenerateForm((form) => ({ ...form, target: value, task_type: existing?.task_type ?? form.task_type }))
                }}
              >
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NEW_DATASET}>Create a new dataset</SelectItem>
                  {(datasets ?? []).map((dataset) => (
                    <SelectItem key={dataset.id} value={dataset.id}>
                      Add to: {dataset.name} ({dataset.sample_count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {generateForm.target === NEW_DATASET && (
                <div className="space-y-1">
                  <label className="text-xs font-medium font-sans">Name</label>
                  <Input
                    value={generateForm.name}
                    onChange={(e) => setGenerateForm((form) => ({ ...form, name: e.target.value }))}
                    placeholder="Email classification"
                    className="h-8 text-sm"
                  />
                </div>
              )}
              <div className="space-y-1">
                <label className="text-xs font-medium font-sans">Task type</label>
                <Select
                  value={generateForm.task_type}
                  onValueChange={(value) => setGenerateForm((form) => ({ ...form, task_type: value }))}
                  disabled={generateForm.target !== NEW_DATASET}
                >
                  <SelectTrigger className="h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TASK_TYPES.map((type) => (
                      <SelectItem key={type} value={type}>
                        {titleCase(type)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium font-sans">Base prompt</label>
              <Textarea
                value={generateForm.base_prompt}
                onChange={(e) => setGenerateForm((form) => ({ ...form, base_prompt: e.target.value }))}
                placeholder="Classify the priority of a customer support email as high, medium or low."
                className="text-sm min-h-[100px]"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-medium font-sans">Samples</label>
                <Input
                  type="number"
                  min={1}
                  max={200}
                  value={generateForm.sample_count}
                  onChange={(e) =>
                    setGenerateForm((form) => ({ ...form, sample_count: Math.min(200, Math.max(1, Number(e.target.value) || 1)) }))
                  }
                  className="h-8 text-sm"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium font-sans">Model</label>
                <Select value={generateForm.model} onValueChange={(value) => setGenerateForm((form) => ({ ...form, model: value }))}>
                  <SelectTrigger className="h-8 text-sm">
                    <SelectValue placeholder={modelOptions.length ? "Choose a model" : "No models available"} />
                  </SelectTrigger>
                  <SelectContent>
                    {modelOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setGenerateOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button onClick={handleGenerate} disabled={busy || modelOptions.length === 0}>
              {busy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Sparkles className="w-4 h-4 mr-2" />}
              {busy ? "Generating..." : "Generate"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Sidebar>
  )
}
