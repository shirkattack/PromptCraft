#!/bin/bash

# Installs Ollama if needed and pulls the models PromptCraft uses by default:
# llama3.2 for optimization and nomic-embed-text for coverage-based example
# selection and synthetic de-duplication (optional but recommended).

set -u

echo "PromptCraft Ollama setup"
echo "========================"

if ! command -v ollama &>/dev/null; then
    echo "Ollama is not installed. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is installed."
fi

if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Ollama is not running. Start it with: ollama serve"
    echo "Then run this script again."
    exit 1
fi
echo "Ollama is running."

models=(
    "llama3.2:latest"        # default task model (3B, fast)
    "nomic-embed-text:latest" # embeddings for example selection and de-duplication
)

for model in "${models[@]}"; do
    echo
    echo "Pulling $model..."
    if ollama pull "$model"; then
        echo "Pulled $model"
    else
        echo "Failed to pull $model"
    fi
done

echo
echo "Installed models:"
ollama list

echo
echo "Next: from the repository root run"
echo "  npm install    # sets up the API and web app, creates API/.env"
echo "  npm run dev    # starts both; links are printed when they are up"
echo
echo "Any other model from https://ollama.com/library works too: ollama pull <name>"
echo "Set DEFAULT_MODEL_NAME in API/.env to make it the default."
