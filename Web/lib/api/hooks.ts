// React hooks for API data fetching
import { useState, useEffect, useCallback } from 'react'
import apiClient, { APIError, CreateSessionRequest, JobProgress, OptimizationMethod, OptimizeOptions } from './client'

// Components that fetch the same resource each hold their own copy, so a
// mutation in one place (the dashboard finishing an optimization, the sidebar
// deleting a dataset) notifies every hook watching that resource to refetch.
export const SESSIONS_CHANGED = 'promptcraft:sessions-changed'
export const TRAINING_CHANGED = 'promptcraft:training-changed'

export function notifySessionsChanged() {
  window.dispatchEvent(new Event(SESSIONS_CHANGED))
}

export function notifyTrainingChanged() {
  window.dispatchEvent(new Event(TRAINING_CHANGED))
}

// Generic hook for async data fetching
function useAsyncData<T>(
  fetchFunction: () => Promise<T>,
  dependencies: unknown[] = [],
  refreshOn?: string,
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const result = await fetchFunction()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      setData(null)
    } finally {
      setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies)

  useEffect(() => {
    fetchData()
  }, [fetchData])

  useEffect(() => {
    if (!refreshOn) return
    window.addEventListener(refreshOn, fetchData)
    return () => window.removeEventListener(refreshOn, fetchData)
  }, [refreshOn, fetchData])

  return { data, loading, error, refetch: fetchData }
}

// Sessions hooks
export function useSessions() {
  return useAsyncData(() => apiClient.getSessions(), [], SESSIONS_CHANGED)
}

// Providers hooks
export function useProviders() {
  return useAsyncData(() => apiClient.getProviders())
}

export function useOllamaHealth() {
  return useAsyncData(() => apiClient.checkOllamaHealth())
}

export function useOptimizationMethods() {
  return useAsyncData(() => apiClient.getOptimizationMethods().then((r) => r.methods))
}

// Training data hooks
export function useTrainingDatasets() {
  return useAsyncData(() => apiClient.getDatasets(), [], TRAINING_CHANGED)
}

export function useTrainingStats(recentLimit = 5) {
  return useAsyncData(() => apiClient.getTrainingStats(recentLimit), [recentLimit], TRAINING_CHANGED)
}

// Analytics hooks
export function usePerformanceMetrics() {
  return useAsyncData(() => apiClient.getPerformanceMetrics(), [], SESSIONS_CHANGED)
}

// Session management hooks
export function useSessionActions() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const createSession = async (data: CreateSessionRequest) => {
    try {
      setLoading(true)
      setError(null)
      const result = await apiClient.createSession(data)
      notifySessionsChanged()
      return result
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session')
      throw err
    } finally {
      setLoading(false)
    }
  }

  /**
   * Run an optimization as a background job and resolve with its result.
   * `onProgress` receives every progress snapshot while the job runs.
   */
  const optimizePrompt = async (
    sessionId: string,
    method?: OptimizationMethod,
    options?: OptimizeOptions,
    onProgress?: (progress: JobProgress) => void,
    pollIntervalMs = 1000,
  ) => {
    try {
      setLoading(true)
      setError(null)
      let job = await apiClient.startOptimization(sessionId, method, options)
      onProgress?.(job.progress)
      while (job.status === 'queued' || job.status === 'running') {
        await new Promise((resolve) => setTimeout(resolve, pollIntervalMs))
        job = await apiClient.getOptimizationStatus(sessionId)
        onProgress?.(job.progress)
      }
      notifySessionsChanged()
      if (job.status === 'failed' || !job.result) {
        throw new APIError(job.error_status ?? 500, job.error ?? 'Optimization failed')
      }
      return job.result
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to optimize prompt')
      throw err
    } finally {
      setLoading(false)
    }
  }

  const deleteSession = async (sessionId: string) => {
    try {
      setLoading(true)
      setError(null)
      await apiClient.deleteSession(sessionId)
      notifySessionsChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete session')
      throw err
    } finally {
      setLoading(false)
    }
  }

  return {
    createSession,
    optimizePrompt,
    deleteSession,
    loading,
    error
  }
}