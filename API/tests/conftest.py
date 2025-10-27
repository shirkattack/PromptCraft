"""
Pytest configuration and fixtures for PromptCraft API tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import httpx

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings


# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_db():
    """Create a test database for each test function."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield TestingSessionLocal()
    
    # Clean up
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(test_db) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(scope="function")
async def async_client(test_db) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_optimization_session():
    """Sample optimization session data for testing."""
    return {
        "name": "Test Session",
        "original_prompt": "Write a hello world program",
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "task_type": "code"
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return {
        "input_text": "What is Python?",
        "expected_output": "Python is a high-level programming language.",
        "extra_data": {"category": "programming"},
        "quality_score": 0.9
    }


@pytest.fixture
def mock_ollama_response():
    """Mock response from Ollama API."""
    return {
        "models": [
            {
                "name": "llama3.2:latest",
                "size": 4661224676,
                "digest": "sha256:abc123",
                "modified_at": "2024-01-01T00:00:00Z"
            }
        ]
    }


class MockLLM:
    """Mock language model for testing."""
    
    def __init__(self, response: str = "Mock response"):
        self.response = response
    
    def __call__(self, prompt: str, **kwargs) -> str:
        return self.response
    
    async def acall(self, prompt: str, **kwargs) -> str:
        return self.response


@pytest.fixture
def mock_llm():
    """Mock language model instance."""
    return MockLLM()


@pytest.fixture
def mock_successful_llm():
    """Mock LLM that returns successful optimization."""
    return MockLLM("This is an improved version of your prompt that is more specific and clear.")


@pytest.fixture
def mock_failing_llm():
    """Mock LLM that raises an exception."""
    class FailingLLM:
        def __call__(self, prompt: str, **kwargs):
            raise Exception("Mock LLM failure")
        
        async def acall(self, prompt: str, **kwargs):
            raise Exception("Mock LLM failure")
    
    return FailingLLM()
