"use client"

import { useState, useEffect, useCallback } from "react"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8765"

interface AIModel {
  id: string
  name: string
  context_window: number
  cost_per_1k_tokens: number
  speed_rating: number
  best_use_case: string
  is_free?: boolean
  // Frontend-friendly aliases
  contextWindow?: string
  costPer1K?: string
  speed?: string
  bestFor?: string
}

interface AIProvider {
  id: string
  name: string
  logo: string
  models: AIModel[]
}

interface OllamaHealth {
  status: string
  healthy: boolean
}

interface OptimizationSession {
  id: string
  name: string
  original_prompt: string
  optimized_prompt?: string
  provider: string
  model: string
  task_type: string
  performance_score?: number
  status: string
  created_at?: string
}

interface CreateSessionParams {
  name: string
  original_prompt: string
  provider: string
  model: string
  task_type: string
}

interface UseDataResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refetch: () => void
}

function transformModel(model: AIModel): AIModel {
  return {
    ...model,
    contextWindow: model.context_window ? `${Math.round(model.context_window / 1000)}K` : "N/A",
    costPer1K: model.is_free || model.cost_per_1k_tokens === 0
      ? "FREE"
      : `$${model.cost_per_1k_tokens?.toFixed(4) || "N/A"}`,
    speed: model.speed_rating >= 4 ? "Fast" : model.speed_rating >= 2 ? "Medium" : "Slow",
    bestFor: model.best_use_case || "General tasks"
  }
}

export function useProviders(): UseDataResult<AIProvider[]> {
  const [data, setData] = useState<AIProvider[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchProviders = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/providers/`)
      if (!response.ok) {
        throw new Error(`Failed to fetch providers: ${response.statusText}`)
      }
      const providers: AIProvider[] = await response.json()

      // Transform models to have frontend-friendly properties
      const transformedProviders = providers.map(provider => ({
        ...provider,
        models: provider.models.map(transformModel)
      }))

      setData(transformedProviders)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch providers")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchProviders()
  }, [fetchProviders])

  return { data, loading, error, refetch: fetchProviders }
}

export function useOllamaHealth(): UseDataResult<OllamaHealth> {
  const [data, setData] = useState<OllamaHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchHealth = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/providers/ollama/health`)
      if (!response.ok) {
        throw new Error(`Failed to check Ollama health: ${response.statusText}`)
      }
      const health: OllamaHealth = await response.json()
      setData(health)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to check Ollama health")
      setData({ status: "unavailable", healthy: false })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    // Poll health every 30 seconds
    const interval = setInterval(fetchHealth, 30000)
    return () => clearInterval(interval)
  }, [fetchHealth])

  return { data, loading, error, refetch: fetchHealth }
}

interface SessionActionsResult {
  createSession: (params: CreateSessionParams) => Promise<OptimizationSession>
  optimizePrompt: (sessionId: string, method?: string) => Promise<{ session: OptimizationSession; optimization_details: any }>
  loading: boolean
  error: string | null
}

export function useSessionActions(): SessionActionsResult {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const createSession = useCallback(async (params: CreateSessionParams): Promise<OptimizationSession> => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      })
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || errorData.error || `Failed to create session: ${response.statusText}`)
      }
      return await response.json()
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to create session"
      setError(message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const optimizePrompt = useCallback(async (
    sessionId: string,
    method: string = "meta_prompt"
  ): Promise<{ session: OptimizationSession; optimization_details: any }> => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/sessions/${sessionId}/optimize?optimization_method=${method}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      )
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || errorData.error || `Failed to optimize prompt: ${response.statusText}`)
      }
      return await response.json()
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to optimize prompt"
      setError(message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  return { createSession, optimizePrompt, loading, error }
}

export function useSessions(): UseDataResult<OptimizationSession[]> {
  const [data, setData] = useState<OptimizationSession[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSessions = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/`)
      if (!response.ok) {
        throw new Error(`Failed to fetch sessions: ${response.statusText}`)
      }
      const sessions: OptimizationSession[] = await response.json()
      setData(sessions)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch sessions")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchSessions()
  }, [fetchSessions])

  return { data, loading, error, refetch: fetchSessions }
}

interface PerformanceMetrics {
  total_optimizations: number
  average_improvement: number
  success_rate: number
  total_processing_time: number
  cost_savings: number
}

export function usePerformanceMetrics(): UseDataResult<PerformanceMetrics> {
  const [data, setData] = useState<PerformanceMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/analytics/performance`)
      if (!response.ok) {
        throw new Error(`Failed to fetch performance metrics: ${response.statusText}`)
      }
      const metrics: PerformanceMetrics = await response.json()
      setData(metrics)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch performance metrics")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchMetrics()
  }, [fetchMetrics])

  return { data, loading, error, refetch: fetchMetrics }
}
