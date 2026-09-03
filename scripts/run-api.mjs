#!/usr/bin/env node
// Runs a command inside the API's Python environment from the repo root.
//
//   node scripts/run-api.mjs uvicorn app.main:app --reload   (npm run dev:api)
//   node scripts/run-api.mjs pytest tests -v                   (npm run test:api)
//
// Uses `uv run` when uv is installed (it manages API/.venv), otherwise the
// interpreter in API/.venv created by scripts/setup-api.mjs, otherwise
// whatever `python` is on PATH.

import { execSync, spawnSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

const root = path.resolve(new URL(".", import.meta.url).pathname, "..")
const api = path.join(root, "API")
const [cmd, ...args] = process.argv.slice(2)
if (!cmd) {
  console.error("usage: run-api.mjs <command> [args...]")
  process.exit(2)
}

const hasUv = (() => {
  try {
    execSync("uv --version", { stdio: "ignore" })
    return true
  } catch {
    return false
  }
})()

let exe, argv
if (hasUv) {
  exe = "uv"
  argv = ["run", cmd, ...args]
} else {
  const bin = path.join(api, ".venv", process.platform === "win32" ? "Scripts" : "bin")
  const local = path.join(bin, cmd)
  exe = fs.existsSync(local) ? local : cmd
  argv = args
}

const result = spawnSync(exe, argv, { cwd: api, stdio: "inherit" })
process.exit(result.status ?? 1)
