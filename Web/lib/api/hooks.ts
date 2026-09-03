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

// Sidebar -> dashboard requests. The two components share no React state,
// so the sidebar asks and the dashboard listens.
export const LOAD_PROMPT = 'promptcraft:load-prompt'
export const NEW_OPTIMIZATION = 'promptcraft:new-optimization'
export const OPEN_SESSION = 'promptcraft:open-session'

export function requestLoadPrompt(text: string, source?: string) {
  window.dispatchEvent(new CustomEvent(LOAD_PROMPT, { detail: { text, source } }))
}

export function requestNewOptimization() {
  window.dispatchEvent(new Event(NEW_OPTIMIZATION))
}

export function requestOpenSession(sessionId: string) {
  window.dispatchEvent(new CustomEvent(OPEN_SESSION, { detail: { sessionId } }))
}

/** True for failures where the request never got an answer (API restarting, not up yet). */
export function isTransientNetworkError(err: unknown): boolean {
  if (!(err instanceof Error)) return false
  if (err.name === 'TypeError') return true // fetch(): "Failed to fetch" / "NetworkError when attempting to fetch resource"
  return /NetworkError|Failed to fetch|Load failed|ECONNREFUSED/i.test(err.message)
}

/** Retry a request a few times on transient network failure (dev-server reloads, startup races). */
export async function withRetry<T>(fn: () => Promise<T>, attempts = 4, baseDelayMs = 500): Promise<T> {
  let lastError: unknown
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await fn()
    } catch (err) {
      lastError = err
      if (!isTransientNetworkError(err) || attempt === attempts - 1) throw err
      await new Promise((resolve) => setTimeout(resolve, baseDelayMs * 2 ** attempt))
    }
  }
  throw lastError
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
      const result = await withRetry(fetchFunction)
      setData(result)
    } catch (err) {
      setError(
        isTransientNetworkError(err)
          ? 'The API did not answer. It may be starting or restarting; retry in a moment.'
          : err instanceof Error
            ? err.message
            : 'An error occurred',
      )
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