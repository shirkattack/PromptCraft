#!/bin/bash

# PromptCraft Quick Test Script
# Runs basic tests to verify the setup is working

echo "🧪 PromptCraft Quick Test Suite"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to run a test with output
run_test_with_output() {
    local test_name="$1"
    local test_command="$2"
    
    echo "Testing $test_name..."
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "1. Environment Setup Tests"
echo "-------------------------"

# Test Node.js
run_test "Node.js version" "node --version | grep -E 'v1[8-9]|v[2-9][0-9]'"

# Test Python
run_test "Python version" "python --version | grep -E 'Python 3\.(1[1-9]|[2-9][0-9])'"

# Test Ollama installation
run_test "Ollama installation" "command -v ollama"

echo ""
echo "2. Ollama Service Tests"
echo "----------------------"

# Test Ollama service
run_test "Ollama service running" "curl -s http://localhost:11434/api/tags"

# Test models available
echo -n "Checking available models... "
MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | wc -l)
if [ "$MODELS" -gt 0 ]; then
    echo -e "${GREEN}✅ PASS${NC} ($MODELS models found)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC} (No models found)"
    ((TESTS_FAILED++))
fi

echo ""
echo "3. API Backend Tests"
echo "-------------------"

# Check if API directory exists
if [ ! -d "API" ]; then
    echo -e "${RED}❌ API directory not found. Please run from project root.${NC}"
    exit 1
fi

cd API

# Test environment file
run_test "Environment file exists" "[ -f .env ]"

# Test if we can import the main app
run_test "Python dependencies" "python -c 'from app.main import app; print(\"OK\")'"

echo ""
echo "4. Quick API Test"
echo "----------------"

# Start API in background and test it
echo "Starting API server for testing..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 &
API_PID=$!

# Wait for server to start
sleep 3

# Test API endpoints
run_test "API health endpoint" "curl -s http://127.0.0.1:8001/health | grep -q 'healthy'"
run_test "API root endpoint" "curl -s http://127.0.0.1:8001/ | grep -q 'PromptCraft'"
run_test "Ollama health via API" "curl -s http://127.0.0.1:8001/api/v1/providers/ollama/health | grep -q 'true'"

# Kill the API server
kill $API_PID 2>/dev/null
wait $API_PID 2>/dev/null

cd ..

echo ""
echo "5. Frontend Tests"
echo "----------------"

if [ -d "Web" ]; then
    cd Web
    
    # Test package.json exists
    run_test "Frontend package.json exists" "[ -f package.json ]"
    
    # Test if node_modules exists (dependencies installed)
    if [ -d "node_modules" ]; then
        echo -e "Frontend dependencies: ${GREEN}✅ INSTALLED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "Frontend dependencies: ${YELLOW}⚠️  NOT INSTALLED${NC} (run 'npm install --legacy-peer-deps')"
        ((TESTS_FAILED++))
    fi
    
    cd ..
else
    echo -e "${YELLOW}⚠️  Web directory not found${NC}"
fi

echo ""
echo "📊 Test Results Summary"
echo "======================"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n🎉 ${GREEN}All tests passed! Your PromptCraft setup looks good.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start the API: cd API && make dev"
    echo "2. Start the frontend: cd Web && npm run dev"
    echo "3. Visit http://localhost:3000 to use PromptCraft"
    echo "4. Run the full test suite: see TESTING_GUIDE.md"
else
    echo -e "\n⚠️  ${YELLOW}Some tests failed. Please check the issues above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "- Install missing dependencies"
    echo "- Start Ollama service: ollama serve"
    echo "- Pull models: ollama pull llama3.2:latest"
    echo "- Copy environment file: cp API/.env.example API/.env"
fi

echo ""
echo "For detailed testing, see: TESTING_GUIDE.md"
