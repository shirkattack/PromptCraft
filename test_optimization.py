#!/usr/bin/env python3
"""
PromptCraft Optimization Test Script

This script tests the core prompt optimization functionality
by making direct API calls to verify everything works.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
API_BASE = "http://127.0.0.1:8765"
TEST_PROMPTS = [
    {
        "name": "Simple Coding Task",
        "prompt": "Write a hello world program",
        "task_type": "code"
    },
    {
        "name": "Creative Writing",
        "prompt": "Write a short story about a robot",
        "task_type": "creative"
    },
    {
        "name": "Analysis Task",
        "prompt": "Explain the benefits of renewable energy",
        "task_type": "analysis"
    }
]

class PromptCraftTester:
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_api_health(self) -> bool:
        """Test if the API is responding"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log(f"API Health: {data.get('status', 'unknown')}")
                return data.get('status') == 'healthy'
            else:
                self.log(f"Health check failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"Health check error: {e}", "ERROR")
            return False
    
    def test_ollama_connection(self) -> bool:
        """Test Ollama connection through API"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/providers/ollama/health", timeout=10)
            if response.status_code == 200:
                is_healthy = response.json()
                self.log(f"Ollama Health: {'OK' if is_healthy else 'Failed'}")
                return is_healthy
            else:
                self.log(f"Ollama health check failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"Ollama health check error: {e}", "ERROR")
            return False
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/providers/ollama/models", timeout=15)
            if response.status_code == 200:
                models = response.json()
                model_names = [model.get('id', 'unknown') for model in models]
                self.log(f"Available models: {', '.join(model_names)}")
                return model_names
            else:
                self.log(f"Failed to get models: {response.status_code}", "ERROR")
                return []
        except Exception as e:
            self.log(f"Error getting models: {e}", "ERROR")
            return []
    
    def create_session(self, prompt_data: Dict[str, str], model: str) -> str:
        """Create an optimization session"""
        try:
            session_data = {
                "name": f"Test: {prompt_data['name']}",
                "original_prompt": prompt_data["prompt"],
                "provider": "ollama",
                "model": model,
                "task_type": prompt_data["task_type"]
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/sessions/",
                json=session_data,
                timeout=10
            )
            
            if response.status_code == 200:
                session = response.json()
                session_id = session.get('id')
                self.log(f"Created session: {session_id}")
                return session_id
            else:
                self.log(f"Failed to create session: {response.status_code} - {response.text}", "ERROR")
                return None
        except Exception as e:
            self.log(f"Error creating session: {e}", "ERROR")
            return None
    
    def optimize_prompt(self, session_id: str, method: str = "simple") -> Dict[str, Any]:
        """Optimize a prompt using the specified method"""
        try:
            optimization_data = {
                "optimization_method": method
            }
            
            self.log(f"Starting optimization with method: {method}")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/api/v1/sessions/{session_id}/optimize",
                json=optimization_data,
                timeout=60  # Longer timeout for optimization
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get('success', False)
                improvement_score = result.get('improvement_score', 0)
                
                self.log(f"Optimization completed in {duration:.2f}s")
                self.log(f"Success: {success}, Improvement Score: {improvement_score}")
                
                if success:
                    original = result.get('original_prompt', '')[:50] + "..."
                    optimized = result.get('optimized_prompt', '')[:50] + "..."
                    self.log(f"Original: {original}")
                    self.log(f"Optimized: {optimized}")
                
                return result
            else:
                self.log(f"Optimization failed: {response.status_code} - {response.text}", "ERROR")
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            self.log(f"Optimization error: {e}", "ERROR")
            return {"success": False, "error": str(e)}
    
    def run_full_test(self):
        """Run the complete test suite"""
        self.log("🧪 Starting PromptCraft Optimization Tests")
        self.log("=" * 50)
        
        # Test 1: API Health
        self.log("Test 1: API Health Check")
        if not self.test_api_health():
            self.log("❌ API health check failed. Is the server running?", "ERROR")
            return False
        self.log("✅ API is healthy")
        
        # Test 2: Ollama Connection
        self.log("\nTest 2: Ollama Connection")
        if not self.test_ollama_connection():
            self.log("❌ Ollama connection failed. Is Ollama running?", "ERROR")
            return False
        self.log("✅ Ollama is connected")
        
        # Test 3: Get Models
        self.log("\nTest 3: Available Models")
        models = self.get_available_models()
        if not models:
            self.log("❌ No models available. Please pull some models with 'ollama pull'", "ERROR")
            return False
        
        # Use the first available model
        test_model = models[0]
        self.log(f"✅ Using model: {test_model}")
        
        # Test 4: Optimization Tests
        self.log("\nTest 4: Prompt Optimization")
        optimization_methods = ["simple", "meta_prompt", "dspy"]
        
        for i, prompt_data in enumerate(TEST_PROMPTS, 1):
            self.log(f"\n--- Test 4.{i}: {prompt_data['name']} ---")
            
            # Create session
            session_id = self.create_session(prompt_data, test_model)
            if not session_id:
                self.log("❌ Failed to create session", "ERROR")
                continue
            
            # Test different optimization methods
            for method in optimization_methods:
                self.log(f"\nTesting {method} optimization...")
                result = self.optimize_prompt(session_id, method)
                
                if result.get('success'):
                    self.log(f"✅ {method} optimization successful")
                    self.test_results.append({
                        "prompt": prompt_data['name'],
                        "method": method,
                        "success": True,
                        "score": result.get('improvement_score', 0)
                    })
                else:
                    self.log(f"❌ {method} optimization failed: {result.get('error', 'Unknown error')}", "ERROR")
                    self.test_results.append({
                        "prompt": prompt_data['name'],
                        "method": method,
                        "success": False,
                        "error": result.get('error', 'Unknown error')
                    })
                
                # Small delay between tests
                time.sleep(1)
        
        # Summary
        self.log("\n" + "=" * 50)
        self.log("📊 Test Results Summary")
        self.log("=" * 50)
        
        successful_tests = sum(1 for r in self.test_results if r['success'])
        total_tests = len(self.test_results)
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Successful: {successful_tests}")
        self.log(f"Failed: {total_tests - successful_tests}")
        
        if successful_tests == total_tests:
            self.log("🎉 All optimization tests passed!")
        else:
            self.log("⚠️  Some tests failed. Check the logs above.")
        
        # Detailed results
        self.log("\nDetailed Results:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            if result['success']:
                self.log(f"{status} {result['prompt']} - {result['method']}: Score {result['score']}")
            else:
                self.log(f"{status} {result['prompt']} - {result['method']}: {result['error']}")
        
        return successful_tests == total_tests

def main():
    """Main function"""
    print("PromptCraft Optimization Test Suite")
    print("====================================")
    
    # Check if API server is likely running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code != 200:
            print("❌ API server doesn't seem to be running.")
            print("Please start it with: cd API && make dev")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API server.")
        print("Please start it with: cd API && make dev")
        sys.exit(1)
    
    # Run tests
    tester = PromptCraftTester()
    success = tester.run_full_test()

    if success:
        print("\n🎉 All tests completed successfully!")
        print("Your PromptCraft optimization functionality is working correctly.")
    else:
        print("\n⚠️  Some tests failed.")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
