"use client"

import { useState, useMemo, useEffect } from "react"
import { useSessions, usePerformanceMetrics } from "@/lib/api/hooks"
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
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
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
import {
  Plus,
  Search,
  FileText,
  Clock,
  TrendingUp,
  Settings,
  BarChart3,
  Database,
  Zap,
  Eye,
  Download,
  Upload,
  Trash2,
  Merge,
  MoreHorizontal,
  Calendar,
} from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

// Mock sessions removed - now using API data

const mockTrainingData = [
  {
    id: 1,
    name: "Email Classification Dataset",
    taskType: "Classification",
    sampleCount: 150,
    qualityScore: 94,
    dateCreated: "2024-01-10",
    lastUsed: "2024-01-15",
    examples: [
      { input: "Urgent: Server down in production", output: "High Priority" },
      { input: "Question about billing cycle", output: "Medium Priority" },
      { input: "Thank you for the great service", output: "Low Priority" },
    ],
  },
  {
    id: 2,
    name: "Content Summarization Examples",
    taskType: "Summarization",
    sampleCount: 89,
    qualityScore: 87,
    dateCreated: "2024-01-08",
    lastUsed: "2024-01-14",
    examples: [
      { input: "Long article about AI developments...", output: "AI technology advances rapidly with new models..." },
      { input: "Research paper on climate change...", output: "Study shows significant environmental impacts..." },
    ],
  },
  {
    id: 3,
    name: "Code Review Patterns",
    taskType: "Code",
    sampleCount: 203,
    qualityScore: 91,
    dateCreated: "2024-01-05",
    lastUsed: "2024-01-13",
    examples: [
      { input: "function calculateTotal() { return a + b }", output: "Add parameter validation and error handling" },
      { input: "const data = await fetch(url)", output: "Add try-catch block and response validation" },
    ],
  },
]

const analyticsData = {
  successRateOverTime: [
    { date: "Jan", rate: 78 },
    { date: "Feb", rate: 82 },
    { date: "Mar", rate: 85 },
    { date: "Apr", rate: 88 },
    { date: "May", rate: 91 },
    { date: "Jun", rate: 94 },
    { date: "Jul", rate: 96 },
  ],
  providerPerformance: [
    { provider: "OpenAI", avgScore: 92, totalOptimizations: 45, avgCost: 0.12 },
    { provider: "Anthropic", avgScore: 89, totalOptimizations: 38, avgCost: 0.08 },
    { provider: "Google", avgScore: 86, totalOptimizations: 22, avgCost: 0.05 },
    { provider: "Cohere", avgScore: 85, totalOptimizations: 31, avgCost: 0.04 },
    { provider: "Ollama", avgScore: 78, totalOptimizations: 67, avgCost: 0.0 },
  ],
  taskTypeDistribution: [
    { name: "Classification", value: 35, color: "#f97316" },
    { name: "Generation", value: 25, color: "#fb923c" },
    { name: "Summarization", value: 20, color: "#fdba74" },
    { name: "Code", value: 15, color: "#fed7aa" },
    { name: "Q&A", value: 5, color: "#ffedd5" },
  ],
  recentActivity: [
    { id: 1, action: "Optimization completed", session: "Email Classification", score: 92, time: "2 min ago" },
    { id: 2, action: "Dataset created", name: "Customer Support Patterns", samples: 150, time: "15 min ago" },
    { id: 3, action: "Provider switched", from: "OpenAI", to: "Anthropic", time: "1 hour ago" },
    { id: 4, action: "API key updated", provider: "Google", time: "2 hours ago" },
    { id: 5, action: "Optimization failed", session: "Code Review", error: "Rate limit", time: "3 hours ago" },
  ],
  monthlyStats: {
    totalOptimizations: 203,
    successfulOptimizations: 195,
    avgImprovement: 31,
    totalCostSaved: 127.5,
    topProvider: "OpenAI",
    topTaskType: "Classification",
  },
}

export function SessionSidebar() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState("optimization")
  const [selectedDatasets, setSelectedDatasets] = useState([])
  const [previewDataset, setPreviewDataset] = useState(null)
  const [analyticsTimeRange, setAnalyticsTimeRange] = useState("7d")
  const [mounted, setMounted] = useState(false)
  const { open } = useSidebar()

  // Fetch data from API
  const { data: sessions, loading: sessionsLoading, error: sessionsError } = useSessions()
  const { data: performanceMetrics, loading: metricsLoading, error: metricsError } = usePerformanceMetrics()

  // Prevent hydration mismatch by only rendering after mount
  useEffect(() => {
    setMounted(true)
  }, [])

  // Transform API sessions to match component expectations
  const transformedSessions = useMemo(() => {
    if (!sessions) return []
    return sessions.map(session => ({
      id: session.id,
      name: session.name,
      date: new Date(session.created_at).toISOString().split('T')[0],
      score: Math.round(session.performance_score),
      provider: session.provider,
      model: session.model,
      status: session.status === 'completed' ? 'completed' : session.status === 'running' ? 'in-progress' : 'failed'
    }))
  }, [sessions])

  const filteredSessions = mounted ? transformedSessions.filter((session) =>
    session.name.toLowerCase().includes(searchQuery.toLowerCase()),
  ) : []

  const filteredTrainingData = mounted ? mockTrainingData.filter((dataset) =>
    dataset.name.toLowerCase().includes(searchQuery.toLowerCase()),
  ) : []

  // Transform performance metrics to analytics data format
  const analyticsData = useMemo(() => {
    if (!mounted || !performanceMetrics || !sessions) {
      return {
        successRateOverTime: [],
        providerPerformance: [],
        taskTypeDistribution: [],
        monthlyStats: {
          totalOptimizations: 0,
          successfulOptimizations: 0,
          avgImprovement: 0,
          totalCostSaved: 0
        },
        recentActivity: []
      }
    }

    // Calculate provider performance from sessions
    const providerStats = sessions.reduce((acc: any, session) => {
      if (!acc[session.provider]) {
        acc[session.provider] = { scores: [], count: 0, totalCost: 0 }
      }
      acc[session.provider].scores.push(session.performance_score)
      acc[session.provider].count++
      return acc
    }, {})

    const providerPerformance = Object.entries(providerStats).map(([provider, stats]: [string, any]) => ({
      provider,
      avgScore: Math.round(stats.scores.reduce((a: number, b: number) => a + b, 0) / stats.scores.length),
      totalOptimizations: stats.count,
      avgCost: provider === 'Ollama' ? 0 : 0.08 // Mock cost for now
    }))

    // Calculate task type distribution from sessions
    const taskTypes = sessions.reduce((acc: any, session) => {
      acc[session.task_type] = (acc[session.task_type] || 0) + 1
      return acc
    }, {})

    const taskTypeDistribution = Object.entries(taskTypes).map(([name, value], index) => ({
      name,
      value: value as number,
      color: ["#f97316", "#fb923c", "#fdba74", "#fed7aa", "#ffedd5"][index % 5]
    }))

    const successfulCount = sessions.filter(s => s.status === 'completed').length

    return {
      successRateOverTime: [
        { date: "Jan", rate: 78 }, { date: "Feb", rate: 82 }, { date: "Mar", rate: 85 },
        { date: "Apr", rate: 88 }, { date: "May", rate: 91 }, { date: "Jun", rate: 94 }, { date: "Jul", rate: Math.round(performanceMetrics.success_rate) }
      ],
      providerPerformance,
      taskTypeDistribution,
      monthlyStats: {
        totalOptimizations: performanceMetrics.total_optimizations,
        successfulOptimizations: successfulCount,
        avgImprovement: Math.round(performanceMetrics.average_improvement),
        totalCostSaved: Math.round(performanceMetrics.cost_savings)
      },
      recentActivity: sessions.slice(0, 5).map((session, index) => ({
        id: index + 1,
        action: session.status === 'completed' ? 'Optimization completed' : 'Optimization in progress',
        session: session.name,
        score: Math.round(session.performance_score),
        time: '15 min ago'
      }))
    }
  }, [performanceMetrics, sessions, mounted])

  const getScoreColor = (score: number) => {
    if (score >= 90) return "bg-green-500"
    if (score >= 70) return "bg-yellow-500"
    return "bg-red-500"
  }

  const getProviderIcon = (provider: string) => {
    // Simple text representation for now
    return provider.charAt(0)
  }

  const handleDatasetSelect = (datasetId, checked) => {
    if (checked) {
      setSelectedDatasets([...selectedDatasets, datasetId])
    } else {
      setSelectedDatasets(selectedDatasets.filter((id) => id !== datasetId))
    }
  }

  const handleSelectAll = (checked) => {
    if (checked) {
      setSelectedDatasets(filteredTrainingData.map((d) => d.id))
    } else {
      setSelectedDatasets([])
    }
  }

  const handleBulkDelete = () => {
    // Implement bulk delete logic
    setSelectedDatasets([])
  }

  const handleBulkMerge = () => {
    // Implement bulk merge logic
    setSelectedDatasets([])
  }

  const handleExportDataset = (datasetId) => {
    // Implement export logic
    console.log("Exporting dataset:", datasetId)
  }

  const handleDeleteDataset = (datasetId) => {
    // Implement delete logic
    console.log("Deleting dataset:", datasetId)
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
                      <SidebarMenuButton className="font-sans bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-600 hover:to-orange-700">
                        <Plus className="w-4 h-4" />
                        New Optimization
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans">
                        <FileText className="w-4 h-4" />
                        Import Prompt
                      </SidebarMenuButton>
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
                      <div className="flex items-center justify-center py-8">
                        <div className="text-sm text-red-500">Error loading sessions: {sessionsError}</div>
                      </div>
                    ) : filteredSessions.length === 0 ? (
                      <div className="flex items-center justify-center py-8">
                        <div className="text-sm text-muted-foreground">No sessions found</div>
                      </div>
                    ) : filteredSessions.map((session) => (
                      <SidebarMenuItem key={session.id}>
                        <SidebarMenuButton className="flex-col items-start h-auto p-3 hover:bg-accent/50 group">
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
                            Status: {session.status}
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
                    Saved Training Data
                    <Badge variant="secondary" className="text-xs">
                      {mockTrainingData.length}
                    </Badge>
                  </div>
                  {selectedDatasets.length > 0 && (
                    <div className="flex items-center gap-1">
                      <Button size="sm" variant="ghost" onClick={handleBulkMerge} className="h-6 px-2 text-xs">
                        <Merge className="w-3 h-3" />
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-2 text-xs text-red-600 hover:text-red-700"
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete Selected Datasets</AlertDialogTitle>
                            <AlertDialogDescription>
                              Are you sure you want to delete {selectedDatasets.length} selected datasets? This action
                              cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction onClick={handleBulkDelete} className="bg-red-600 hover:bg-red-700">
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
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

                  {filteredTrainingData.length > 0 && (
                    <div className="px-2 pb-2">
                      <div className="flex items-center gap-2">
                        <Checkbox
                          checked={selectedDatasets.length === filteredTrainingData.length}
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
                    {filteredTrainingData.map((dataset) => (
                      <SidebarMenuItem key={dataset.id}>
                        <div className="flex items-start gap-2 p-3 hover:bg-accent/50 rounded-md">
                          <Checkbox
                            checked={selectedDatasets.includes(dataset.id)}
                            onCheckedChange={(checked) => handleDatasetSelect(dataset.id, checked)}
                            className="h-4 w-4 mt-1"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between w-full">
                              <span className="font-sans font-medium text-sm truncate">{dataset.name}</span>
                              <div className="flex items-center gap-1">
                                <Badge variant="outline" className="text-xs">
                                  {dataset.taskType}
                                </Badge>
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                                      <MoreHorizontal className="w-3 h-3" />
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent align="end">
                                    <Dialog>
                                      <DialogTrigger asChild>
                                        <DropdownMenuItem onSelect={(e) => e.preventDefault()}>
                                          <Eye className="w-4 h-4 mr-2" />
                                          Preview
                                        </DropdownMenuItem>
                                      </DialogTrigger>
                                      <DialogContent className="max-w-2xl">
                                        <DialogHeader>
                                          <DialogTitle className="font-sans">{dataset.name}</DialogTitle>
                                          <DialogDescription>
                                            {dataset.sampleCount} samples • {dataset.taskType} • Quality:{" "}
                                            {dataset.qualityScore}%
                                          </DialogDescription>
                                        </DialogHeader>
                                        <div className="space-y-4 max-h-96 overflow-y-auto">
                                          {dataset.examples.map((example, idx) => (
                                            <div key={idx} className="border rounded-lg p-3 space-y-2">
                                              <div>
                                                <label className="text-xs font-medium text-muted-foreground">
                                                  Input:
                                                </label>
                                                <p className="text-sm font-serif">{example.input}</p>
                                              </div>
                                              <div>
                                                <label className="text-xs font-medium text-muted-foreground">
                                                  Output:
                                                </label>
                                                <p className="text-sm font-serif">{example.output}</p>
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                        <DialogFooter>
                                          <Button variant="outline" onClick={() => handleExportDataset(dataset.id)}>
                                            <Download className="w-4 h-4 mr-2" />
                                            Export
                                          </Button>
                                        </DialogFooter>
                                      </DialogContent>
                                    </Dialog>
                                    <DropdownMenuItem onClick={() => handleExportDataset(dataset.id)}>
                                      <Download className="w-4 h-4 mr-2" />
                                      Export
                                    </DropdownMenuItem>
                                    <DropdownMenuItem>
                                      <Upload className="w-4 h-4 mr-2" />
                                      Import
                                    </DropdownMenuItem>
                                    <AlertDialog>
                                      <AlertDialogTrigger asChild>
                                        <DropdownMenuItem onSelect={(e) => e.preventDefault()} className="text-red-600">
                                          <Trash2 className="w-4 h-4 mr-2" />
                                          Delete
                                        </DropdownMenuItem>
                                      </AlertDialogTrigger>
                                      <AlertDialogContent>
                                        <AlertDialogHeader>
                                          <AlertDialogTitle>Delete Dataset</AlertDialogTitle>
                                          <AlertDialogDescription>
                                            Are you sure you want to delete "{dataset.name}"? This action cannot be
                                            undone.
                                          </AlertDialogDescription>
                                        </AlertDialogHeader>
                                        <AlertDialogFooter>
                                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                                          <AlertDialogAction
                                            onClick={() => handleDeleteDataset(dataset.id)}
                                            className="bg-red-600 hover:bg-red-700"
                                          >
                                            Delete
                                          </AlertDialogAction>
                                        </AlertDialogFooter>
                                      </AlertDialogContent>
                                    </AlertDialog>
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              </div>
                            </div>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground font-serif">
                              <span>{dataset.sampleCount} samples</span>
                              <span>•</span>
                              <span className={`${getScoreColor(dataset.qualityScore)} text-white px-1 rounded`}>
                                {dataset.qualityScore}%
                              </span>
                            </div>
                            <div className="text-xs text-muted-foreground">Last used: {dataset.lastUsed}</div>
                          </div>
                        </div>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>

                  <div className="px-2 pt-2 space-y-2">
                    <SidebarMenuButton className="w-full font-sans bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-600 hover:to-orange-700">
                      <Plus className="w-4 h-4" />
                      Generate New Dataset
                    </SidebarMenuButton>
                    <SidebarMenuButton className="w-full font-sans" variant="outline">
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
                    {analyticsTimeRange}
                  </Badge>
                </SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-4" suppressHydrationWarning>
                    <div className="grid grid-cols-2 gap-3 text-center">
                      <div className="space-y-1">
                        <div className="font-sans font-bold text-xl text-green-500">96%</div>
                        <div className="text-xs text-muted-foreground">Success Rate</div>
                        <div className="flex items-center justify-center gap-1">
                          <TrendingUp className="w-3 h-3 text-green-500" />
                          <span className="text-xs text-green-500">+4%</span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="font-sans font-bold text-xl text-orange-500">+31%</div>
                        <div className="text-xs text-muted-foreground">Avg Improvement</div>
                        <div className="flex items-center justify-center gap-1">
                          <TrendingUp className="w-3 h-3 text-green-500" />
                          <span className="text-xs text-green-500">+2%</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-xs font-medium font-sans">Success Rate Trend</div>
                      <div className="h-24">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={analyticsData.successRateOverTime}>
                            <Line type="monotone" dataKey="rate" stroke="#f97316" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Provider Analytics</SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-3">
                    {analyticsData.providerPerformance.slice(0, 3).map((provider) => (
                      <div
                        key={provider.provider}
                        className="flex items-center justify-between p-2 rounded-md hover:bg-accent/50"
                      >
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 bg-muted rounded-full flex items-center justify-center text-xs">
                            {provider.provider.charAt(0)}
                          </div>
                          <div>
                            <div className="font-sans text-sm font-medium">{provider.provider}</div>
                            <div className="text-xs text-muted-foreground">{provider.totalOptimizations} runs</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-sans text-sm font-semibold">{provider.avgScore}</div>
                          <div className="text-xs text-muted-foreground">${provider.avgCost.toFixed(3)}</div>
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
                          <DialogTitle className="font-sans">Detailed Analytics Dashboard</DialogTitle>
                          <DialogDescription>Comprehensive performance metrics and insights</DialogDescription>
                        </DialogHeader>
                        <div className="space-y-6">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="text-center space-y-1">
                              <div className="font-sans font-bold text-2xl">
                                {analyticsData.monthlyStats.totalOptimizations}
                              </div>
                              <div className="text-xs text-muted-foreground">Total Optimizations</div>
                            </div>
                            <div className="text-center space-y-1">
                              <div className="font-sans font-bold text-2xl text-green-500">
                                {analyticsData.monthlyStats.successfulOptimizations}
                              </div>
                              <div className="text-xs text-muted-foreground">Successful</div>
                            </div>
                            <div className="text-center space-y-1">
                              <div className="font-sans font-bold text-2xl text-orange-500">
                                +{analyticsData.monthlyStats.avgImprovement}%
                              </div>
                              <div className="text-xs text-muted-foreground">Avg Improvement</div>
                            </div>
                            <div className="text-center space-y-1">
                              <div className="font-sans font-bold text-2xl text-green-500">
                                ${analyticsData.monthlyStats.totalCostSaved}
                              </div>
                              <div className="text-xs text-muted-foreground">Cost Saved</div>
                            </div>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-3">
                              <h4 className="font-sans font-semibold text-sm">Provider Performance</h4>
                              <div className="h-48">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart data={analyticsData.providerPerformance}>
                                    <XAxis dataKey="provider" tick={{ fontSize: 10 }} />
                                    <YAxis tick={{ fontSize: 10 }} />
                                    <ChartTooltip />
                                    <Bar dataKey="avgScore" fill="#f97316" radius={[2, 2, 0, 0]} />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </div>

                            <div className="space-y-3">
                              <h4 className="font-sans font-semibold text-sm">Task Type Distribution</h4>
                              <div className="h-48">
                                <ResponsiveContainer width="100%" height="100%">
                                  <PieChart>
                                    <Pie
                                      data={analyticsData.taskTypeDistribution}
                                      cx="50%"
                                      cy="50%"
                                      innerRadius={30}
                                      outerRadius={70}
                                      dataKey="value"
                                    >
                                      {analyticsData.taskTypeDistribution.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                      ))}
                                    </Pie>
                                    <ChartTooltip />
                                  </PieChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          </div>

                          <div className="space-y-3">
                            <h4 className="font-sans font-semibold text-sm">Success Rate Over Time</h4>
                            <div className="h-32">
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={analyticsData.successRateOverTime}>
                                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                                  <YAxis tick={{ fontSize: 10 }} />
                                  <ChartTooltip />
                                  <Line
                                    type="monotone"
                                    dataKey="rate"
                                    stroke="#f97316"
                                    strokeWidth={2}
                                    dot={{ fill: "#f97316" }}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        </div>
                        <DialogFooter>
                          <Button variant="outline">
                            <Download className="w-4 h-4 mr-2" />
                            Export Report
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Task Performance</SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-2">
                    <div className="text-xs font-medium font-sans">Most Used Task Types</div>
                    <div className="space-y-2">
                      {analyticsData.taskTypeDistribution.slice(0, 3).map((task) => (
                        <div key={task.name} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: task.color }} />
                            <span className="text-xs font-serif">{task.name}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-muted rounded-full h-1">
                              <div
                                className="h-1 rounded-full"
                                style={{
                                  backgroundColor: task.color,
                                  width: `${task.value}%`,
                                }}
                              />
                            </div>
                            <span className="text-xs text-muted-foreground w-8">{task.value}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Recent Activity</SidebarGroupLabel>
                <SidebarGroupContent>
                  <div className="space-y-2">
                    {analyticsData.recentActivity.slice(0, 4).map((activity) => (
                      <div key={activity.id} className="flex items-start gap-2 p-2 rounded-md hover:bg-accent/50">
                        <div className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium font-sans truncate">{activity.action}</div>
                          <div className="text-xs text-muted-foreground font-serif">
                            {activity.session ||
                              activity.name ||
                              `${activity.from} → ${activity.to}` ||
                              activity.provider}
                          </div>
                          <div className="text-xs text-muted-foreground">{activity.time}</div>
                        </div>
                        {activity.score && (
                          <Badge variant="secondary" className="text-xs">
                            {activity.score}
                          </Badge>
                        )}
                      </div>
                    ))}
                  </div>
                </SidebarGroupContent>
              </SidebarGroup>

              <SidebarGroup>
                <SidebarGroupLabel className="font-sans font-semibold">Quick Actions</SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans">
                        <Download className="w-4 h-4" />
                        Export Analytics
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans">
                        <Calendar className="w-4 h-4" />
                        Schedule Report
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton className="font-sans">
                        <Settings className="w-4 h-4" />
                        Analytics Settings
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  </SidebarMenu>
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
    </Sidebar>
  )
}
