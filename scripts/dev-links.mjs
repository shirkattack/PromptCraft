#!/usr/bin/env node
// Waits for the API and the web app to come up, then prints one block of
// clickable links. Terminals that support OSC 8 hyperlinks (iTerm2, VS Code,
// Kitty, WezTerm, Windows Terminal) render them as links; others show the
// plain URL, which most still auto-detect with Cmd/Ctrl+click.

const WEB = process.env.WEB_URL ?? "http://localhost:3000"
const API = process.env.API_URL ?? "http://127.0.0.1:8000"
const TIMEOUT_MS = 90_000

const ESC = "\x1b"
const link = (url, label = url) => `${ESC}]8;;${url}${ESC}\\${label}${ESC}]8;;${ESC}\\`
const dim = (text) => `${ESC}[2m${text}${ESC}[0m`
const bold = (text) => `${ESC}[1m${text}${ESC}[0m`

async function isUp(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(2000) })
    return response.ok
  } catch {
    return false
  }
}

async function waitFor(url) {
  const deadline = Date.now() + TIMEOUT_MS
  while (Date.now() < deadline) {
    if (await isUp(url)) return true
    await new Promise((resolve) => setTimeout(resolve, 1000))
  }
  return false
}

const [apiUp, webUp] = await Promise.all([waitFor(`${API}/health`), waitFor(WEB)])

const rows = [
  ["Web app", WEB, webUp],
  ["API", `${API}/health`, apiUp],
  ["API docs", `${API}/docs`, apiUp],
]

console.log("")
console.log(bold("PromptCraft is ready"))
for (const [label, url, up] of rows) {
  const status = up ? "" : dim("  (not responding yet)")
  console.log(`  ${label.padEnd(9)} ${link(url)}${status}`)
}
console.log(dim("  Cmd+click (macOS) or Ctrl+click to open."))
console.log("")
