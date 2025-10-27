"use client"

import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { SessionSidebar } from "@/components/session-sidebar"
import { OptimizationDashboard } from "@/components/optimization-dashboard"
import { ThemeToggle } from "@/components/theme-toggle"

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
