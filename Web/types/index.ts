export interface AIProvider {
  id: string
  name: string
  logo: string
  models: AIModel[]
}

export interface AIModel {
  id: string
  name: string
  contextWindow: number
  costPer1kTokens: number
  speedRating: number
  bestUseCase: string
  isFree?: boolean
}

export interface OptimizationSession {
  id: string
  name: string
  originalPrompt: string
  optimizedPrompt: string
  provider: string
  model: string
  taskType: string
  performanceScore: number
  createdAt: Date
  status: "completed" | "running" | "failed"
}

export interface TrainingDataset {
  id: string
  name: string
  description: string
  sampleCount: number
  taskType: string
  createdAt: Date
  lastModified: Date
  size: string
}

export interface PerformanceMetrics {
  totalOptimizations: number
  averageImprovement: number
  successRate: number
  totalProcessingTime: number
  costSavings: number
}

export interface ProviderPerformance {
  provider: string
  speed: number
  quality: number
  cost: number
  reliability: number
}
