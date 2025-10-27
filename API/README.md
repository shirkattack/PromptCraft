# PromptCraft API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional, production-ready FastAPI backend for PromptCraft - an AI-powered prompt optimization platform featuring real-time optimization, comprehensive analytics, and **local AI support via Ollama**. No external API keys required!

## 🚀 Features

### Core Functionality
- **Local AI with Ollama**: Complete privacy with local model execution - no external API keys needed
- **Advanced Prompt Optimization**: DSPy framework integration with meta-prompting techniques
- **Multiple Local Models**: Support for Llama, Mistral, CodeLlama, and other Ollama models
- **Real-time Analytics**: Performance tracking, success rates, and improvement scoring
- **Session Management**: Persistent optimization history with SQLite database
- **Async Architecture**: High-performance async/await throughout the application

### Technical Excellence
- **Modern Python 3.11+**: Leveraging latest Python features and type hints
- **Comprehensive Testing**: Full test suite with pytest, coverage reporting
- **Production-Ready Logging**: Structured logging with rotation and levels
- **Custom Exception Handling**: Detailed error reporting and debugging
- **Security Best Practices**: Input validation, CORS, trusted hosts
- **Developer Experience**: Pre-commit hooks, linting, formatting, and documentation

## 📋 Requirements

- **Python**: 3.11 or higher
- **Ollama**: Required for AI functionality - [Install Ollama](https://ollama.ai/)
- **Dependencies**: Managed via `pyproject.toml` with optional development extras
- **Database**: SQLite (default) or PostgreSQL for production

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd PromptCraft/API

# Install with development dependencies
make install-dev

# Set up environment variables
cp .env.example .env
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve  # In one terminal
ollama pull llama3.2:latest  # In another terminal

# Run the application
make dev
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install

# Set up environment
cp .env.example .env
```

### Environment Configuration

Edit `.env` file with your configuration:

```env
# Ollama Configuration (Required)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=120
DEFAULT_MODEL_NAME=llama3.2:latest

# Application Settings
DATABASE_URL=sqlite:///./app.db
LOG_LEVEL=INFO
DEBUG=false
CORS_ORIGINS=["http://localhost:3000"]

# Security
API_HOST=127.0.0.1
API_PORT=8000
```

## 🚀 Usage

### Development Server

```bash
# Start development server with auto-reload
make dev

# Or manually
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Production Deployment

```bash
# Install production dependencies only
pip install -e .

# Run with production settings
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

- **Interactive Docs**: http://127.0.0.1:8000/docs (Swagger UI)
- **Alternative Docs**: http://127.0.0.1:8000/redoc (ReDoc)
- **OpenAPI Schema**: http://127.0.0.1:8000/openapi.json

## 🧪 Testing

### Run Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_main.py -v

# Run with markers
pytest -m "not slow" tests/
```

### Test Coverage

The project maintains high test coverage with comprehensive unit and integration tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Service Tests**: Business logic validation
- **Mock Testing**: External service simulation

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check
```

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast Python linting
- **MyPy**: Static type checking
- **Various**: Trailing whitespace, YAML validation, etc.

### Project Structure

```
API/
├── app/                          # Application package
│   ├── api/                      # API routes
│   │   └── v1/                   # API version 1
│   │       ├── endpoints/        # Route handlers
│   │       └── router.py         # Route configuration
│   ├── core/                     # Core functionality
│   │   ├── config.py            # Settings and configuration
│   │   ├── database.py          # Database setup
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── logging.py           # Logging configuration
│   ├── models/                   # SQLAlchemy models
│   ├── schemas/                  # Pydantic schemas
│   ├── services/                 # Business logic
│   │   ├── optimization_service.py  # Core optimization
│   │   ├── ollama_service.py        # Ollama integration
│   │   ├── lm_manager.py            # Language model management
│   │   └── training_service.py      # Training data management
│   └── main.py                   # FastAPI application
├── tests/                        # Test suite
│   ├── conftest.py              # Test configuration
│   ├── test_main.py             # Main app tests
│   └── test_services/           # Service tests
├── logs/                         # Application logs
├── .env.example                  # Environment template
├── .pre-commit-config.yaml      # Pre-commit configuration
├── Makefile                     # Development commands
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## 📊 API Endpoints

### System Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive documentation

### Optimization Sessions
- `GET /api/v1/sessions/` - List sessions
- `POST /api/v1/sessions/` - Create session
- `GET /api/v1/sessions/{id}` - Get session
- `POST /api/v1/sessions/{id}/optimize` - Optimize prompt
- `GET /api/v1/sessions/analytics/performance` - Analytics

### AI Providers
- `GET /api/v1/providers/` - List providers and models
- `GET /api/v1/providers/ollama/health` - Ollama status
- `GET /api/v1/providers/ollama/models` - Ollama models

### Training Data
- `POST /api/v1/training/synthetic` - Generate synthetic data
- `POST /api/v1/training/import` - Import training data
- `GET /api/v1/training/datasets` - List datasets

## 🔒 Security Features

- **Input Validation**: Pydantic schemas with comprehensive validation
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Trusted Hosts**: Host validation middleware
- **API Key Management**: Secure environment-based key storage
- **Error Handling**: Sanitized error responses
- **Logging**: Security event logging and monitoring

## 📈 Performance

- **Async Architecture**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Caching**: Response caching for improved performance
- **Monitoring**: Health checks and performance metrics
- **Scalability**: Horizontal scaling support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`make install-dev`)
4. Make your changes following the code style
5. Run tests and linting (`make check`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 and use Black for formatting
- Add type hints to all functions and methods
- Write comprehensive docstrings
- Include tests for new functionality
- Update documentation as needed
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues (configure your repository URL)
- **Documentation**: [API Documentation](http://127.0.0.1:8000/docs)
- **Discussions**: GitHub Discussions (configure your repository URL)

## 🙏 Acknowledgments

- **FastAPI**: Modern, fast web framework for building APIs
- **DSPy**: Framework for algorithmic prompt optimization
- **Pydantic**: Data validation using Python type annotations
- **SQLAlchemy**: Python SQL toolkit and ORM
- **pytest**: Testing framework for Python

---

**Built with ❤️ using modern Python and FastAPI best practices**
