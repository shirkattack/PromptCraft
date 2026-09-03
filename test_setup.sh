#!/bin/bash

# PromptCraft setup check: verifies the toolchain, Ollama, the API environment
# and the health endpoints. Run from the repository root after `npm install`.

set -u
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASSED=0; FAILED=0

run_test() {
    local name="$1" cmd="$2"
    echo -n "Testing $name... "
    if eval "$cmd" &>/dev/null; then
        echo -e "${GREEN}PASS${NC}"; ((PASSED++)); return 0
    else
        echo -e "${RED}FAIL${NC}"; ((FAILED++)); return 1
    fi
}

echo "PromptCraft setup check"
echo "======================="
echo
echo "1. Toolchain"
run_test "Node.js 18+" "node --version | grep -E 'v(1[8-9]|[2-9][0-9])'"
if command -v uv &>/dev/null; then
    run_test "uv" "uv --version"
    PY_RUN="uv run --project API"
else
    echo -e "uv: ${YELLOW}not installed${NC} (falling back to API/.venv; see https://docs.astral.sh/uv/)"
    PY_RUN="API/.venv/bin/python -c"
fi
run_test "Root dependencies (concurrently)" "[ -d node_modules/concurrently ]"
run_test "Web dependencies" "[ -d Web/node_modules ]"

echo
echo "2. Ollama"
run_test "Ollama installed" "command -v ollama"
run_test "Ollama service reachable" "curl -s http://localhost:11434/api/tags"
TAGS=$(curl -s http://localhost:11434/api/tags 2>/dev/null)
MODELS=$(echo "$TAGS" | grep -o '"name":"[^"]*"' | wc -l | tr -d ' ')
if [ "${MODELS:-0}" -gt 0 ]; then
    echo -e "Models available: ${GREEN}PASS${NC} ($MODELS found)"; ((PASSED++))
else
    echo -e "Models available: ${RED}FAIL${NC} (run: ollama pull llama3.2)"; ((FAILED++))
fi
if echo "$TAGS" | grep -q "nomic-embed-text"; then
    echo -e "Embedding model: ${GREEN}PASS${NC} (coverage selection and de-duplication enabled)"; ((PASSED++))
else
    echo -e "Embedding model: ${YELLOW}optional${NC} (ollama pull nomic-embed-text enables coverage selection and de-duplication)"
fi

echo
echo "3. API environment"
run_test "API/.env exists" "[ -f API/.env ]"
if command -v uv &>/dev/null; then
    run_test "API dependencies (uv)" "uv run --project API python -c 'import app.main'"
else
    run_test "API dependencies (venv)" "API/.venv/bin/python -c 'import app.main'"
fi

echo
echo "4. API endpoints"
echo "Starting a temporary API on port 8001..."
node scripts/run-api.mjs uvicorn app.main:app --host 127.0.0.1 --port 8001 &>/dev/null &
API_PID=$!
for _ in $(seq 1 20); do
    curl -s http://127.0.0.1:8001/health &>/dev/null && break
    sleep 1
done
run_test "Health endpoint" "curl -s http://127.0.0.1:8001/health | grep -q healthy"
run_test "Database migrated" "curl -s http://127.0.0.1:8001/health | grep -q connected"
run_test "Ollama health via API" "curl -s http://127.0.0.1:8001/api/v1/providers/ollama/health | grep -q true"
run_test "Model list via API" "curl -s http://127.0.0.1:8001/api/v1/providers/ollama/models | grep -q context_window"
kill $API_PID 2>/dev/null; wait $API_PID 2>/dev/null

echo
echo "Results: ${PASSED} passed, ${FAILED} failed"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}Setup looks good.${NC} Start everything with: npm run dev"
else
    echo -e "${YELLOW}Fix the failures above.${NC} Common causes:"
    echo "  - npm install            (installs the API with uv and the web app, creates API/.env)"
    echo "  - ollama serve           (start Ollama)"
    echo "  - ollama pull llama3.2   (at least one model)"
    exit 1
fi
echo "Manual walkthrough: TESTING_GUIDE.md"
