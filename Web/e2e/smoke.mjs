// End-to-end smoke test against a running PromptCraft (web + API + Ollama).
//
//   npm run e2e                 # from the repository root, with `npm run dev` up
//   WEB_URL=... API_URL=... node e2e/smoke.mjs
//
// One-time: npx playwright install chromium   (downloads the browser)
//
// Imports a dataset, previews and exports it, runs a measured optimization
// with the default model, checks analytics, deletes the dataset and reloads.
// Exits non-zero if any step fails or the page logs an error. Needs Ollama
// with at least one model, so it is a local check, not a CI job.

import { chromium } from "playwright"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

const WEB = process.env.WEB_URL ?? "http://localhost:3000"
const API = process.env.API_URL ?? "http://127.0.0.1:8000"
const OPTIMIZE_TIMEOUT_MS = Number(process.env.E2E_OPTIMIZE_TIMEOUT_MS ?? 240_000)
const DATASET_NAME = `E2E tickets ${Date.now()}`
const OUT = process.env.E2E_OUT_DIR ?? fs.mkdtempSync(path.join(os.tmpdir(), "promptcraft-e2e-"))

const CSV = [
  "input,output",
  '"Production database is down, all customers affected",high',
  '"Cannot log in since this morning, blocking my work",high',
  '"Security alert: suspicious logins from unknown IPs",high',
  '"Question about how billing cycles work",medium',
  '"Export to CSV is missing one column",medium',
  '"Feature request: dark mode for the dashboard",low',
  '"Thanks for the quick help yesterday!",low',
  '"Typo on the pricing page footer",low',
  '"Payment failed and the order is stuck, customer waiting",high',
  '"Would like to change the email on my account",medium',
].join("\n")

const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a)
const results = []
async function step(name, fn) {
  try {
    await fn()
    results.push({ name, ok: true })
    log("PASS", name)
  } catch (err) {
    const message = String(err?.message ?? err).split("\n")[0]
    results.push({ name, ok: false, error: message })
    log("FAIL", name, "->", message)
  }
}

// Preflight: both servers answer.
for (const [label, url] of [["web", WEB], ["api", `${API}/health`]]) {
  try {
    const r = await fetch(url)
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
  } catch (err) {
    console.error(`${label} not reachable at ${url}: ${err.message}. Start it with: npm run dev`)
    process.exit(2)
  }
}

const browser = await chromium.launch()
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true })
const page = await context.newPage()
const consoleMessages = []
page.on("console", (m) => {
  if (["error", "warning"].includes(m.type())) consoleMessages.push({ type: m.type(), text: m.text().slice(0, 400) })
})
page.on("pageerror", (e) => consoleMessages.push({ type: "pageerror", text: String(e).slice(0, 400) }))
const shot = (name) => page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: true })
const selectOption = async (trigger, name) => {
  await trigger.click()
  await page.getByRole("option", { name }).first().click()
}
const datasetRow = () => page.locator('[data-sidebar="menu-item"]', { hasText: DATASET_NAME }).first()
const openRowMenu = async () => {
  await datasetRow().getByRole("button").last().click()
  await page.getByRole("menuitem", { name: /Preview samples/ }).waitFor({ timeout: 3000 })
}

await step("home loads", async () => {
  await page.goto(WEB, { waitUntil: "networkidle" })
  await page.getByRole("button", { name: /Start Optimization/ }).waitFor({ timeout: 15000 })
  await shot("01-home")
})

await step("import dataset from CSV", async () => {
  await page.getByRole("tab", { name: /Training Data/ }).click()
  await page.getByRole("button", { name: /Import Dataset/ }).click()
  const dialog = page.getByRole("dialog")
  await dialog.getByPlaceholder("Support ticket triage").fill(DATASET_NAME)
  await selectOption(dialog.getByRole("combobox").nth(0), "Classification")
  await selectOption(dialog.getByRole("combobox").nth(1), "CSV")
  await dialog.locator("textarea").fill(CSV)
  await dialog.getByRole("button", { name: /^Import$/ }).click()
  await page.getByText("Dataset imported").first().waitFor({ timeout: 10000 })
  await datasetRow().waitFor({ timeout: 5000 })
})

await step("preview samples", async () => {
  await openRowMenu()
  await page.getByRole("menuitem", { name: /Preview samples/ }).click()
  const dialog = page.getByRole("dialog")
  await dialog.getByText("Expected output").first().waitFor({ timeout: 5000 })
  await shot("02-preview")
  await page.keyboard.press("Escape")
  await dialog.waitFor({ state: "hidden", timeout: 3000 })
})

await step("export JSON downloads a file", async () => {
  await openRowMenu()
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 10000 }),
    page.getByRole("menuitem", { name: /Export JSON/ }).click(),
  ])
  const target = path.join(OUT, download.suggestedFilename())
  await download.saveAs(target)
  const data = JSON.parse(fs.readFileSync(target, "utf8"))
  if (!Array.isArray(data) || data.length !== 10) throw new Error(`expected 10 rows, got ${data.length}`)
})

await step("measured optimization", async () => {
  await page.locator("textarea").first().fill("Classify the priority of this support ticket as high, medium or low.")
  await selectOption(page.getByRole("combobox").filter({ hasText: "None" }).first(), new RegExp(DATASET_NAME))
  await page.getByRole("button", { name: /Start Optimization/ }).click()
  await page.getByText("Eval Results").waitFor({ timeout: OPTIMIZE_TIMEOUT_MS })
  await page.getByText("Try it").first().waitFor({ timeout: 5000 })
  await shot("03-results")
})

await step("try it runs both prompts", async () => {
  await page.getByPlaceholder(/Paste the input the prompt should handle/).fill("Checkout page shows a 500 error for every customer")
  await page.getByRole("button", { name: /Run both/ }).click()
  await page.getByText("Optimized prompt", { exact: true }).first().waitFor({ timeout: OPTIMIZE_TIMEOUT_MS })
})

await step("analytics tab", async () => {
  await page.getByRole("tab", { name: /Analytics/ }).click()
  await page.getByText("Performance Overview").waitFor({ timeout: 5000 })
  await page.getByRole("button", { name: /View Detailed Analytics/ }).click()
  await page.getByRole("dialog").waitFor({ timeout: 3000 })
  await page.keyboard.press("Escape")
})

await step("delete dataset with confirmation", async () => {
  await page.getByRole("tab", { name: /Training Data/ }).click()
  await openRowMenu()
  await page.getByRole("menuitem", { name: /Delete/ }).click()
  await page.getByRole("alertdialog").getByRole("button", { name: /^Delete$/ }).click()
  await page.getByText("Dataset deleted").first().waitFor({ timeout: 10000 })
})

await step("reload is clean", async () => {
  consoleMessages.length = 0
  await page.reload({ waitUntil: "networkidle" })
  await page.getByRole("button", { name: /Start Optimization/ }).waitFor({ timeout: 15000 })
})

await browser.close()

// Remove the sessions this run created so the local history stays tidy.
try {
  const sessions = await (await fetch(`${API}/api/v1/sessions/?limit=50`)).json()
  for (const s of sessions) {
    if (s.original_prompt === "Classify the priority of this support ticket as high, medium or low." && s.name.startsWith("Optimization ")) {
      const created = new Date(s.created_at.endsWith("Z") ? s.created_at : `${s.created_at}Z`)
      if (Date.now() - created.getTime() < 30 * 60 * 1000) await fetch(`${API}/api/v1/sessions/${s.id}`, { method: "DELETE" })
    }
  }
} catch {
  // best effort
}

const failed = results.filter((r) => !r.ok)
console.log("\nResults:")
for (const r of results) console.log(` ${r.ok ? "PASS" : "FAIL"} ${r.name}${r.error ? ` (${r.error})` : ""}`)
console.log(`\nConsole errors/warnings: ${consoleMessages.length ? "" : "none"}`)
for (const m of consoleMessages) console.log(` [${m.type}] ${m.text}`)
console.log(`\nScreenshots: ${OUT}`)
process.exit(failed.length || consoleMessages.some((m) => m.type !== "warning") ? 1 : 0)
