# PromptCraft Testing Guide

This guide provides a comprehensive series of tests to validate your PromptCraft Ollama-only setup. Follow these tests in order to ensure everything is working correctly.

## 🧪 Test Categories

1. [Environment Setup Tests](#1-environment-setup-tests)
2. [Ollama Integration Tests](#2-ollama-integration-tests)
3. [API Backend Tests](#3-api-backend-tests)
4. [Frontend Integration Tests](#4-frontend-integration-tests)
5. [End-to-End Workflow Tests](#5-end-to-end-workflow-tests)
6. [Performance Tests](#6-performance-tests)
7. [Error Handling Tests](#7-error-handling-tests)

---

## 1. Environment Setup Tests

### Test 1.1: Verify Prerequisites
```bash
# Check Node.js version (should be 18.0.0+)
node --version

# Check Python version (should be 3.11+)
python --version

# Check npm/yarn
npm --version
```

**Expected Result:** All versions meet minimum requirements.

### Test 1.2: Verify Ollama Installation
```bash
# Check if Ollama is installed
ollama --version

# Check if Ollama service is running
curl -s http://localhost:11434/api/tags
```

**Expected Result:** Ollama version displayed and API responds with model list.

### Test 1.3: Verify Models Available
```bash
# List installed models
ollama list

# Test a specific model
ollama run llama3.2:latest "Hello, world!"
```

**Expected Result:** At least one model is installed and responds to prompts.

---

## 2. Ollama Integration Tests

### Test 2.1: Direct Ollama API Tests
```bash
# Test Ollama health endpoint
curl http://localhost:11434/api/tags

# Test model generation
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:latest",
    "prompt": "Write a haiku about coding",
    "stream": false
  }'
```

**Expected Result:** JSON responses with model information and generated text.

### Test 2.2: Model Performance Test
```bash
# Test different models (if available)
ollama run llama3.2:latest "Explain Python in one sentence"
ollama run mistral:7b "What is machine learning?"
```

**Expected Result:** Different models respond with appropriate answers.

---

## 3. API Backend Tests

### Test 3.1: Setup Backend Environment
```bash
cd API

# Install dependencies
make install-dev

# Copy environment file
cp .env.example .env

# Verify configuration
cat .env
```

**Expected Result:** Dependencies installed, .env file created with Ollama configuration.

### Test 3.2: Run Unit Tests
```bash
# Run the test suite
make test

# Run with coverage
make test-cov

# Run specific test files
pytest tests/test_main.py -v
pytest tests/test_services/test_ollama_service.py -v
```

**Expected Result:** All tests pass with good coverage.

### Test 3.3: Start Backend Server
```bash
# Start development server
make dev

# Or manually
uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

**Expected Result:** Server starts without errors, shows startup logs.

### Test 3.4: API Health Checks
```bash
# Test root endpoint
curl http://127.0.0.1:8765/

# Test health endpoint
curl http://127.0.0.1:8765/health

# Test API documentation
open http://127.0.0.1:8765/docs
```

**Expected Result:** 
- Root returns API info
- Health returns "healthy" status
- Docs page loads with interactive API documentation

### Test 3.5: Ollama Service Integration
```bash
# Test Ollama health through API
curl http://127.0.0.1:8765/api/v1/providers/ollama/health

# Test model listing
curl http://127.0.0.1:8765/api/v1/providers/ollama/models

# Test providers endpoint
curl http://127.0.0.1:8765/api/v1/providers/
```

**Expected Result:** 
- Health check returns true
- Models list shows available Ollama models
- Providers shows Ollama as available

---

## 4. Frontend Integration Tests

### Test 4.1: Setup Frontend
```bash
cd ../Web

# Install dependencies
npm install --legacy-peer-deps

# Start development server
npm run dev
```

**Expected Result:** Frontend server starts on http://localhost:3000

### Test 4.2: Frontend Connectivity
```bash
# Test frontend loads
curl -I http://localhost:3000

# Check in browser
open http://localhost:3000
```

**Expected Result:** Frontend loads without errors, shows PromptCraft interface.

### Test 4.3: API Connection from Frontend
Open browser developer tools and check:
1. Network tab shows successful API calls
2. Console shows no errors
3. Providers dropdown populates with Ollama models

---

## 5. End-to-End Workflow Tests

### Test 5.1: Create Optimization Session
Using the API directly:
```bash
# Create a new session
curl -X POST http://127.0.0.1:8765/api/v1/sessions/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Session",
    "original_prompt": "Write a hello world program",
    "provider": "ollama",
    "model": "llama3.2:latest",
    "task_type": "code"
  }'
```

**Expected Result:** Returns session object with ID.

### Test 5.2: Optimize a Prompt
```bash
# Use the session ID from previous test
curl -X POST http://127.0.0.1:8765/api/v1/sessions/{SESSION_ID}/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "optimization_method": "meta_prompt"
  }'
```

**Expected Result:** Returns optimized prompt with improvement score.

### Test 5.3: Frontend Workflow Test
Using the web interface:

1. **Create Session:**
   - Enter prompt: "Explain quantum computing"
   - Select Ollama provider
   - Choose available model
   - Set task type to "general"

2. **Run Optimization:**
   - Click "Start Optimization"
   - Wait for results
   - Verify before/after comparison

3. **Check Analytics:**
   - View session history
   - Check performance metrics
   - Verify data persistence

**Expected Result:** Complete workflow works without errors.

---

## 6. Performance Tests

### Test 6.1: Response Time Test
```bash
# Time API responses
time curl http://127.0.0.1:8765/health
time curl http://127.0.0.1:8765/api/v1/providers/ollama/models
```

**Expected Result:** Responses under 1 second for health, under 5 seconds for models.

### Test 6.2: Concurrent Requests Test
```bash
# Test multiple simultaneous requests
for i in {1..5}; do
  curl http://127.0.0.1:8765/health &
done
wait
```

**Expected Result:** All requests complete successfully.

### Test 6.3: Large Prompt Test
Test with a long prompt (500+ words) through the frontend or API.

**Expected Result:** System handles large prompts without timeout.

---

## 7. Error Handling Tests

### Test 7.1: Ollama Service Down Test
```bash
# Stop Ollama service
pkill ollama

# Test API responses
curl http://127.0.0.1:8765/api/v1/providers/ollama/health
curl http://127.0.0.1:8765/api/v1/providers/ollama/models

# Restart Ollama
ollama serve &
```

**Expected Result:** API returns appropriate error messages, recovers when Ollama restarts.

### Test 7.2: Invalid Model Test
```bash
# Try to use non-existent model
curl -X POST http://127.0.0.1:8765/api/v1/sessions/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Error Test",
    "original_prompt": "Test prompt",
    "provider": "ollama",
    "model": "nonexistent:model",
    "task_type": "general"
  }'
```

**Expected Result:** Returns appropriate error message.

### Test 7.3: Malformed Request Test
```bash
# Send invalid JSON
curl -X POST http://127.0.0.1:8765/api/v1/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"invalid": json}'
```

**Expected Result:** Returns validation error with details.

---

## 🎯 Success Criteria

Your PromptCraft installation is working correctly if:

- ✅ All environment setup tests pass
- ✅ Ollama integration works smoothly
- ✅ API backend responds correctly to all endpoints
- ✅ Frontend loads and connects to backend
- ✅ End-to-end prompt optimization workflow completes
- ✅ Performance is acceptable for your use case
- ✅ Error handling works as expected

## 🐛 Troubleshooting

### Common Issues:

1. **Ollama not responding:**
   ```bash
   ollama serve
   # Wait a few seconds, then test again
   ```

2. **Port conflicts:**
   ```bash
   # Check what's using port 8765
   lsof -i :8765
   # Kill process or change port in config
   ```

3. **Model not found:**
   ```bash
   # Pull the model
   ollama pull llama3.2:latest
   ```

4. **Frontend can't connect to API:**
   - Check CORS settings in API config
   - Verify API is running on correct port
   - Check browser console for errors

## 📊 Test Results Template

Create a checklist to track your testing:

```
[ ] Environment Setup Tests (1.1-1.3)
[ ] Ollama Integration Tests (2.1-2.2)
[ ] API Backend Tests (3.1-3.5)
[ ] Frontend Integration Tests (4.1-4.3)
[ ] End-to-End Workflow Tests (5.1-5.3)
[ ] Performance Tests (6.1-6.3)
[ ] Error Handling Tests (7.1-7.3)

Notes:
- Test Date: ___________
- Environment: ___________
- Issues Found: ___________
- Overall Status: ___________
```

## 🚀 Next Steps

After all tests pass:
1. Try different optimization methods (meta_prompt, dspy, simple)
2. Test with various model types (code, creative, analysis)
3. Experiment with different prompt types and lengths
4. Set up monitoring for production use
5. Consider adding more Ollama models for specific use cases
