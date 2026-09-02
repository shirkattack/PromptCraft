"""
Pytest configuration and fixtures for PromptCraft API tests.

This module provides common fixtures and configuration for all tests.
"""

from collections.abc import AsyncGenerator

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app

# In-memory so tests never leave a test.db behind or inherit state from one.
TEST_DATABASE_URL = "sqlite://"


@pytest.fixture(scope="function")
def test_db():
    """Create a test database for each test function."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one shared in-memory database across connections
    )
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
    # httpx >= 0.28 dropped the `app=` shortcut in favour of an explicit transport.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_optimization_session():
    """Sample optimization session data for testing."""
    return {
        "name": "Test Session",
        "original_prompt": "Write a hello world program",
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "task_type": "code",
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return {
        "input_text": "What is Python?",
        "expected_output": "Python is a high-level programming language.",
        "extra_data": {"category": "programming"},
        "quality_score": 0.9,
    }


@pytest.fixture
def mock_ollama_response():
    """Mock response from Ollama /api/tags, in the shape current Ollama emits."""
    return {
        "models": [
            {
                "name": "llama3.2:latest",
                "size": 2019393189,
                "digest": "sha256:abc123",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {
                    "family": "llama",
                    "parameter_size": "3.2B",
                    "quantization_level": "Q4_K_M",
                    "context_length": 131072,
                },
                "capabilities": ["completion", "tools"],
            },
            {
                "name": "nomic-embed-text:latest",
                "size": 274302450,
                "digest": "sha256:def456",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"family": "nomic-bert", "parameter_size": "137M"},
                "capabilities": ["embedding"],
            },
        ]
    }


class MockLLM:
    """Mock language model for testing.

    Deliberately not a `dspy.BaseLM`: DSPy rejects it, which exercises the
    services' fallback paths without needing a running model.
    """

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
    return MockLLM(
        "This is an improved version of your prompt that is more specific and clear."
    )


@pytest.fixture
def mock_failing_llm():
    """Mock LLM that raises an exception."""

    class FailingLLM:
        def __call__(self, prompt: str, **kwargs):
            raise Exception("Mock LLM failure")

        async def acall(self, prompt: str, **kwargs):
            raise Exception("Mock LLM failure")

    return FailingLLM()
