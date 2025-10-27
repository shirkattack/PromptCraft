#!/bin/bash

# PromptCraft Ollama Setup Script
# This script helps you set up Ollama with recommended models for PromptCraft

echo "🚀 PromptCraft Ollama Setup"
echo "=========================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Installing now..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed successfully!"
else
    echo "✅ Ollama is already installed"
fi

# Check if Ollama service is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama service is not running. Please start it with: ollama serve"
    echo "   Then run this script again."
    exit 1
else
    echo "✅ Ollama service is running"
fi

echo ""
echo "📦 Pulling recommended models for PromptCraft..."

# Array of recommended models
models=(
    "llama3.2:latest"
    "mistral:7b"
    "codellama:7b"
    "llama3.2:1b"
)

for model in "${models[@]}"; do
    echo "⬇️  Pulling $model..."
    if ollama pull "$model"; then
        echo "✅ Successfully pulled $model"
    else
        echo "❌ Failed to pull $model"
    fi
    echo ""
done

echo "🎉 Setup complete!"
echo ""
echo "Available models:"
ollama list

echo ""
echo "🔧 Next steps:"
echo "1. Copy the environment file: cp .env.example .env"
echo "2. Start the PromptCraft API: make dev"
echo "3. Visit http://localhost:8000/docs to see the API documentation"
echo ""
echo "💡 Tip: You can pull additional models with: ollama pull <model-name>"
echo "   Popular options: gemma:7b, phi3:mini, qwen2:7b"
