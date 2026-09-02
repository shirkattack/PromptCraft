// API Client for PromptCraft Backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api/v1'

export type SessionStatus = 'completed' | 'running' | 'failed'
export type OptimizationMethod = 'meta_prompt' | 'dspy' | 'simple'

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

export type OutputFormat = 'auto' | 'markdown' | 'plain' | 'json'
export type TargetLength = 'auto' | 'concise' | 'balanced' | 'detailed'

/** Advanced settings for an optimization run; mirrors OptimizeRequest on the API. */
export interface OptimizeOptions {
  temperature?: number
  max_tokens?: number
  output_format?: OutputFormat
  target_length?: TargetLength
  preserve_wording?: boolean
}

export interface OptimizeResponse {
  message: string
  session: OptimizationSession
  optimization_details: {
    method: OptimizationMethod
    improvement_score: number
    processing_time: number
    metadata: Record<string, unknown>
  }
}

export interface OllamaHealth {
  status: string
  healthy: boolean
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

  // Analytics endpoints
  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    return this.request<PerformanceMetrics>('/sessions/analytics/performance')
  }

  async getOptimizationMethods(): Promise<{
    methods: { id: OptimizationMethod; name: string; description: string; recommended_for: string[] }[]
  }> {
    return this.request('/sessions/optimization-methods')
  }
}

// Export singleton instance
export const apiClient = new APIClient()
export default apiClient
