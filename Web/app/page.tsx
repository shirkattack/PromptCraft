"use client"

import dynamic from "next/dynamic"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { OptimizationDashboard } from "@/components/optimization-dashboard"
import { ThemeToggle } from "@/components/theme-toggle"

// The sidebar is client-only. It already waits for mount and loads everything
// from the API, so server-rendering it adds nothing, and skipping it means
// browser extensions that edit SVGs and inputs before React attaches (a common
// source of dev-only hydration warnings) have nothing to mismatch.
const SessionSidebar = dynamic(
  () => import("@/components/session-sidebar").then((m) => m.SessionSidebar),
  { ssr: false, loading: () => <div className="w-80 shrink-0 border-r bg-muted/20" aria-hidden /> },
)

export default function HomePage() {
  return (
    <SidebarProvider defaultOpen={true}>
      <div className="flex min-h-screen w-full">
        <SessionSidebar />
        <SidebarInset className="flex-1">
          <OptimizationDashboard />
        </SidebarInset>
        <ThemeToggle />
      </div>
    </SidebarProvider>
  )
}
