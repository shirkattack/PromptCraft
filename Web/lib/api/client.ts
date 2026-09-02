// API Client for PromptCraft Backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api/v1'

export type SessionStatus = 'completed' | 'running' | 'failed'
export type OptimizationMethod = 'meta_prompt' | 'dspy' | 'simple' | 'gepa'

export interface OptimizationSession {
  id: string
  name: string
  original_prompt: string
  optimized_prompt: string | null
  provider: string
  model: string
  task_type: string
  performance_score: number
  created_at: string
  status: SessionStatus
  optimization_method: string | null
  processing_time: number | null
  /** Set when the run was measured against a training dataset. */
  dataset_id: string | null
  baseline_score: number | null
  eval_score: number | null
  eval_metric: string | null
  eval_sample_count: number | null
}

export interface AIProvider {
  id: string
  name: string
  logo: string
  models: AIModel[]
  /** False when the backend has no adapter for this provider or it is unreachable. */
  available: boolean
  unavailable_reason: string | null
}

export interface AIModel {
  id: string
  name: string
  context_window: number
  cost_per_1k_tokens: number
  speed_rating: number
  best_use_case: string
  is_free: boolean
  /** Reported by the runtime (Ollama /api/tags); null when unknown. */
  parameter_size: string | null
  quantization: string | null
  family: string | null
  size_bytes: number | null
  capabilities: string[]
}

export interface PerformanceMetrics {
  total_optimizations: number
  average_improvement: number
  success_rate: number
  /** Only known for optimizations run by the current API process. */
  total_processing_time: number | null
  /** Not tracked; local Ollama runs have no billed cost. */
  cost_savings: number | null
}

export interface CreateSessionRequest {
  name: string
  original_prompt: string
  provider: string
  model: string
  task_type: string
}

/** One entry of GET /sessions/optimization-methods. */
export interface OptimizationMethodInfo {
  id: OptimizationMethod
  name: string
  description: string
  how_it_works?: string
  best_for?: string
  returns_reasoning?: boolean
  relative_speed?: string
  recommended_for: string[]
  /** GEPA needs a dataset to evolve against. */
  requires_dataset?: boolean
}

export type OutputFormat = 'auto' | 'markdown' | 'plain' | 'json'
export type TargetLength = 'auto' | 'concise' | 'balanced' | 'detailed'
export type EvalMetric = 'auto' | 'exact' | 'contains' | 'llm_judge'
export type ScoreType = 'measured' | 'heuristic'

/** Advanced settings for an optimization run; mirrors OptimizeRequest on the API. */
export interface OptimizeOptions {
  temperature?: number
  max_tokens?: number
  output_format?: OutputFormat
  target_length?: TargetLength
  preserve_wording?: boolean
  /** Measure against this dataset and return the best-scoring candidate. */
  dataset_id?: string | null
  eval_metric?: EvalMetric
  max_demos?: number
  /** GEPA only: scored model calls the evolution may spend (10-500). */
  gepa_budget?: number
  /** GEPA only: model that writes new instructions; defaults to the task model. */
  reflection_model?: string | null
}

/** One entry of the GEPA lineage (metadata.gepa.timeline). */
export interface GepaCandidate {
  index: number
  parent: number | null
  generation: number
  instructions: string
  score: number | null
  iteration: number | null
  feedback: string[]
}

/** metadata.gepa on a GEPA run. */
export interface GepaReport {
  budget: number
  metric_calls: number | null
  iterations: number
  reflection_model: string | null
  baseline_score: number
  final_score: number
  improved: boolean
  best_index: number | null
  timeline: GepaCandidate[]
  instructions: string
  elapsed_seconds: number
}

export interface EvalCandidate {
  name: string
  score: number | null
  demo_count: number
  bootstrapped_demos: number
  error: string | null
}

export interface EvalSampleResult {
  input: string
  expected: string
  actual: string
  passed: boolean
}

/** metadata.eval on a measured run. */
export interface EvalReport {
  metric: string
  train_size: number
  dev_size: number
  total_samples: number
  max_demos: number
  baseline_score: number | null
  eval_score: number | null
  best: string
  improved: boolean
  candidates: EvalCandidate[]
  demos: { input: string; output: string; bootstrapped?: boolean }[]
  baseline_results: EvalSampleResult[]
  results: EvalSampleResult[]
  instructions: string
}

export interface OptimizeResponse {
  message: string
  session: OptimizationSession
  optimization_details: {
    method: OptimizationMethod
    improvement_score: number
    /** "measured" when a dataset was used, otherwise the structural heuristic. */
    score_type: ScoreType
    processing_time: number
    metadata: Record<string, unknown> & { eval?: EvalReport; gepa?: GepaReport }
  }
}

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface JobProgress {
  stage: string
  message: string
  current: number | null
  total: number | null
  best_score: number | null
  updated_at: string
}

/** Snapshot of a background optimization (GET /sessions/{id}/optimize/status). */
export interface OptimizationJob {
  session_id: string
  status: JobStatus
  progress: JobProgress
  history: { stage: string; message: string; current: number | null; total: number | null; best_score: number | null; at: string }[]
  result: OptimizeResponse | null
  error: string | null
  error_status: number | null
  started_at: string
  finished_at: string | null
  elapsed_seconds: number
}

export interface OllamaHealth {
  status: string
  healthy: boolean
}

// Training data (GET/POST /training)

export interface TrainingDatasetSummary {
  id: string
  name: string
  description: string | null
  task_type: string
  sample_count: number
  created_at: string
  last_modified: string
  size: string | null
  /** Mean sample quality (0-1); null when the dataset has no samples. */
  avg_quality_score: number | null
}

export interface TrainingSample {
  id: string
  dataset_id: string
  input_text: string
  expected_output: string
  extra_data: Record<string, unknown> | null
  quality_score: number | null
  created_at: string
}

export interface TrainingDataset extends Omit<TrainingDatasetSummary, 'avg_quality_score'> {
  samples: TrainingSample[] | null
}

export interface TaskTypeStats {
  task_type: string
  dataset_count: number
  sample_count: number
}

export interface TrainingStats {
  dataset_count: number
  sample_count: number
  by_task_type: TaskTypeStats[]
  recent_datasets: TrainingDatasetSummary[]
}

export type DatasetFileFormat = 'json' | 'csv'

export interface CreateDatasetRequest {
  name: string
  description?: string | null
  task_type: string
}

export interface ImportDatasetRequest extends CreateDatasetRequest {
  file_format: DatasetFileFormat
  /** Raw JSON or CSV text. */
  data: string
}

export interface ExportDatasetResponse {
  dataset_name: string
  format: DatasetFileFormat
  data: string
  sample_count: number
  export_timestamp: string
}

export interface GenerateSamplesRequest {
  sample_count: number
  base_prompt: string
  task_type: string
  provider?: string
  model?: string
  creativity_level?: number
}

export interface GenerateSamplesResponse {
  dataset_id: string
  generated_count: number
  failed_count: number
  samples: TrainingSample[]
  processing_time: number
}

export class APIError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'APIError'
  }
}

class APIClient {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`

    // Merge caller headers *under* ours so the API key and content type
    // cannot be dropped by a caller that passes its own headers.
    const headers = new Headers(options.headers)
    headers.set('Content-Type', 'application/json')

    const apiKey = process.env.NEXT_PUBLIC_API_KEY
    if (apiKey) {
      headers.set('X-API-Key', apiKey)
    }

    const response = await fetch(url, { ...options, headers })

    if (!response.ok) {
      // The backend returns {"error": "..."} for handled failures; surface
      // that instead of a bare status line when it is present.
      let detail = `${response.status} ${response.statusText}`
      try {
        const body = await response.json()
        if (body?.error) detail = String(body.error)
      } catch {
        // Non-JSON body; keep the status line.
      }
      throw new APIError(response.status, `API Error: ${detail}`)
    }

    return response.json()
  }

  // Session endpoints
  async getSessions(): Promise<OptimizationSession[]> {
    return this.request<OptimizationSession[]>('/sessions/')
  }

  async getSession(sessionId: string): Promise<OptimizationSession> {
    return this.request<OptimizationSession>(`/sessions/${sessionId}`)
  }

  async createSession(data: CreateSessionRequest): Promise<OptimizationSession> {
    return this.request<OptimizationSession>('/sessions/', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async updateSession(
    sessionId: string,
    data: Partial<Pick<OptimizationSession, 'optimized_prompt' | 'performance_score' | 'status'>>,
  ): Promise<OptimizationSession> {
    return this.request<OptimizationSession>(`/sessions/${sessionId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
  }

  async deleteSession(sessionId: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/sessions/${sessionId}`, {
      method: 'DELETE',
    })
  }

  async optimizePrompt(
    sessionId: string,
    optimizationMethod: OptimizationMethod = 'meta_prompt',
    options: OptimizeOptions = {},
  ): Promise<OptimizeResponse> {
    return this.request<OptimizeResponse>(`/sessions/${sessionId}/optimize`, {
      method: 'POST',
      body: JSON.stringify({ optimization_method: optimizationMethod, ...options }),
    })
  }

  /** Start an optimization in the background; poll getOptimizationStatus. */
  async startOptimization(
    sessionId: string,
    optimizationMethod: OptimizationMethod = 'meta_prompt',
    options: OptimizeOptions = {},
  ): Promise<OptimizationJob> {
    return this.request<OptimizationJob>(`/sessions/${sessionId}/optimize/start`, {
      method: 'POST',
      body: JSON.stringify({ optimization_method: optimizationMethod, ...options }),
    })
  }

  async getOptimizationStatus(sessionId: string): Promise<OptimizationJob> {
    return this.request<OptimizationJob>(`/sessions/${sessionId}/optimize/status`)
  }

  // Provider endpoints
  async getProviders(): Promise<AIProvider[]> {
    return this.request<AIProvider[]>('/providers/')
  }

  async checkOllamaHealth(): Promise<OllamaHealth> {
    return this.request<OllamaHealth>('/providers/ollama/health')
  }

  async getOllamaModels(): Promise<AIModel[]> {
    return this.request<AIModel[]>('/providers/ollama/models')
  }

  // Training data endpoints
  async getDatasets(): Promise<TrainingDatasetSummary[]> {
    return this.request<TrainingDatasetSummary[]>('/training/')
  }

  async getTrainingStats(recentLimit = 5): Promise<TrainingStats> {
    return this.request<TrainingStats>(`/training/stats?recent_limit=${recentLimit}`)
  }

  async createDataset(data: CreateDatasetRequest): Promise<TrainingDataset> {
    return this.request<TrainingDataset>('/training/', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async deleteDataset(datasetId: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/training/${datasetId}`, {
      method: 'DELETE',
    })
  }

  async getSamples(datasetId: string, limit = 50): Promise<TrainingSample[]> {
    return this.request<TrainingSample[]>(`/training/${datasetId}/samples?limit=${limit}`)
  }

  async importDataset(data: ImportDatasetRequest): Promise<TrainingDataset> {
    return this.request<TrainingDataset>('/training/import', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async exportDataset(datasetId: string, format: DatasetFileFormat = 'json'): Promise<ExportDatasetResponse> {
    return this.request<ExportDatasetResponse>(`/training/${datasetId}/export`, {
      method: 'POST',
      body: JSON.stringify({ dataset_id: datasetId, format, include_metadata: true }),
    })
  }

  async generateSamples(datasetId: string, data: GenerateSamplesRequest): Promise<GenerateSamplesResponse> {
    return this.request<GenerateSamplesResponse>(`/training/${datasetId}/generate`, {
      method: 'POST',
      body: JSON.stringify({ dataset_id: datasetId, ...data }),
    })
  }

  // Analytics endpoints
  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    return this.request<PerformanceMetrics>('/sessions/analytics/performance')
  }

  async getOptimizationMethods(): Promise<{ methods: OptimizationMethodInfo[] }> {
    return this.request('/sessions/optimization-methods')
  }
}

// Export singleton instance
export const apiClient = new APIClient()
export default apiClient
