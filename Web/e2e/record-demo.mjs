// Records the README demo: a GEPA run against the bundled ticket dataset.
//
//   node e2e/record-demo.mjs            # writes e2e/video/*.webm
//   ../scripts/record-demo.sh           # runs this, then converts to ../docs/demo.gif
//
// Expects `npm run dev` to be up. Imports the dataset if it is missing. The
// prompt text is varied per run so DSPy's cache does not replay an earlier
// identical run instantly (which would leave no progress to show).

import { chromium } from "playwright"
import fs from "node:fs"
import path from "node:path"

const WEB = process.env.WEB_URL ?? "http://localhost:3000"
const API = process.env.API_URL ?? "http://127.0.0.1:8000"
const OUT = path.resolve("e2e/video")
const DATASET = "Support ticket priority"
const PROMPT = `Classify the priority of this support ticket as high, medium, or low. Ticket ${Date.now() % 1000}:`
const BUDGET = Number(process.env.DEMO_GEPA_BUDGET ?? 60)

fs.rmSync(OUT, { recursive: true, force: true })
fs.mkdirSync(OUT, { recursive: true })

// Make sure the dataset exists.
const datasets = await (await fetch(`${API}/api/v1/training/`)).json()
if (!datasets.some((d) => d.name === DATASET)) {
  const csv = fs.readFileSync(path.resolve("../docs/examples/support-tickets.csv"), "utf8")
  await fetch(`${API}/api/v1/training/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: DATASET, task_type: "classification", file_format: "csv", data: csv }),
  })
}

const browser = await chromium.launch()
const context = await browser.newContext({
  viewport: { width: 1280, height: 800 },
  recordVideo: { dir: OUT, size: { width: 1280, height: 800 } },
  colorScheme: "light",
})
const page = await context.newPage()
const pause = (ms) => page.waitForTimeout(ms)
const selectOption = async (trigger, name) => {
  await trigger.click()
  await pause(500)
  await page.getByRole("option", { name }).first().click()
}
const scrollTo = async (locator) => {
  await locator.scrollIntoViewIfNeeded()
  await pause(200)
}

await page.goto(WEB, { waitUntil: "networkidle" })
await pause(1200)

const textarea = page.locator("textarea").first()
await textarea.click()
await textarea.pressSequentially(PROMPT, { delay: 25 })
await pause(600)

await selectOption(page.getByRole("combobox").filter({ hasText: /Meta-Prompt/ }).first(), /GEPA/)
await pause(900)
await selectOption(page.getByRole("combobox").filter({ hasText: "None" }).first(), new RegExp(DATASET))
await pause(1200)
if (BUDGET !== 60) {
  await page.locator('input[type="range"]').first().fill(String(BUDGET))
  await pause(400)
}

await page.getByRole("button", { name: /Start Optimization/ }).click()
await page.getByText(/Generation \d+/).first().waitFor({ timeout: 90_000 }).catch(() => {})
await pause(2500)
await page.getByText("Prompt Evolution").waitFor({ timeout: 600_000 })
await pause(1500)

await scrollTo(page.getByText("Optimized Prompt").first())
await pause(1800)
await scrollTo(page.getByText("Prompt Evolution").first())
await pause(1500)
const feedback = page.getByText("Feedback the reflection read before proposing this").first()
if (await feedback.count()) {
  await scrollTo(feedback)
  await pause(2200)
}
await scrollTo(page.getByText("Eval Results").first())
await pause(1500)
await scrollTo(page.getByText("Held-out samples").first())
await pause(2500)

await context.close()
await browser.close()
const video = fs.readdirSync(OUT).find((f) => f.endsWith(".webm"))
console.log(path.join(OUT, video))
