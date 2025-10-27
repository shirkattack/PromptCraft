.PHONY: help install install-dev test test-cov lint format clean run dev docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,test,docs]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

lint: ## Run linting
	ruff check app/ tests/
	mypy app/

format: ## Format code
	black app/ tests/
	isort app/ tests/
	ruff check app/ tests/ --fix

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

run: ## Run the application
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev: ## Run in development mode with auto-reload
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --log-level debug

docs: ## Generate documentation
	mkdocs serve

build: ## Build the package
	python -m build

check: lint test ## Run all checks (lint + test)

ci: install-dev lint test ## Run CI pipeline locally
