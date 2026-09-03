#!/usr/bin/env node
// Runs after `npm install` at the repo root: sets up the Python API with uv
// (or a plain venv + pip when uv is missing), installs the web app's
// dependencies, and creates API/.env from the example if it does not exist.
//
// Skip with: SKIP_SETUP=1 npm install

import { execSync, spawnSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

if (process.env.SKIP_SETUP) process.exit(0)

const root = path.resolve(new URL(".", import.meta.url).pathname, "..")
const api = path.join(root, "API")
const web = path.join(root, "Web")

const run = (cmd, args, cwd) => {
  console.log(`\n> ${cmd} ${args.join(" ")}  (${path.relative(root, cwd) || "."})`)
  const result = spawnSync(cmd, args, { cwd, stdio: "inherit" })
  if (result.status !== 0) {
    console.error(`\n${cmd} failed with exit code ${result.status}`)
    process.exit(result.status ?? 1)
  }
}

const has = (cmd) => {
  try {
    execSync(`${cmd} --version`, { stdio: "ignore" })
    return true
  } catch {
    return false
  }
}

// 1. Python API
if (has("uv")) {
  run("uv", ["sync", "--all-extras"], api)
} else {
  console.log("\nuv not found; falling back to python -m venv + pip. Install uv for faster, locked installs: https://docs.astral.sh/uv/")
  const python = has("python3") ? "python3" : "python"
  if (!fs.existsSync(path.join(api, ".venv"))) run(python, ["-m", "venv", ".venv"], api)
  const pip = path.join(api, ".venv", process.platform === "win32" ? "Scripts" : "bin", "pip")
  run(pip, ["install", "-r", "requirements.txt"], api)
}

// 2. Environment file
const envFile = path.join(api, ".env")
if (!fs.existsSync(envFile)) {
  fs.copyFileSync(path.join(api, ".env.example"), envFile)
  console.log("\nCreated API/.env from API/.env.example")
}

// 3. Web app
run("npm", ["install", "--legacy-peer-deps"], web)

console.log("\nSetup complete. Start everything with: npm run dev")
