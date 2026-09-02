// React hooks for API data fetching
import { useState, useEffect, useCallback } from 'react'
import apiClient, { CreateSessionRequest, OptimizationMethod } from './client'

// Generic hook for async data fetching
function useAsyncData<T>(
  fetchFunction: () => Promise<T>,
  dependencies: any[] = []
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
  }, dependencies)

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return { data, loading, error, refetch: fetchData }
}

// Sessions hooks
export function useSessions() {
  return useAsyncData(() => apiClient.getSessions())
}

export function useSession(sessionId: string) {
  return useAsyncData(() => apiClient.getSession(sessionId), [sessionId])
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

// Analytics hooks
export function usePerformanceMetrics() {
  return useAsyncData(() => apiClient.getPerformanceMetrics())
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
      return result
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session')
      throw err
    } finally {
      setLoading(false)
    }
  }

  const optimizePrompt = async (sessionId: string, method?: OptimizationMethod) => {
    try {
      setLoading(true)
      setError(null)
      const result = await apiClient.optimizePrompt(sessionId, method)
      return result
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