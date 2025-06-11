#!/bin/bash

# Empathetic Tutor AI - Environment Setup Script
# For macOS development

echo "🚀 Setting up Empathetic Tutor AI Environment..."

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.9+ is required. Please install it first."
    exit 1
fi

# Create project directory structure
echo "📁 Creating directory structure..."
mkdir -p empathetic-tutor-ai/{backend,frontend,tests,logs,data,models,scripts,docs}
cd empathetic-tutor-ai

# Backend structure
mkdir -p backend/{app,tests,migrations}
mkdir -p backend/app/{api,core,models,services,utils,ml,db}
mkdir -p backend/app/api/{routes,middleware,dependencies}
mkdir -p backend/app/services/{nlp,ml,feedback,intervention}
mkdir -p backend/app/ml/{features,models,training,evaluation}

# Data directories
mkdir -p data/{raw,processed,embeddings,knowledge_graphs}
mkdir -p logs/{api,ml,student_interactions,system}
mkdir -p models/{checkpoints,production,experiments}

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Setup ChromaDB directory
mkdir -p .chromadb

# Create .env file
echo "🔐 Creating .env file..."
cat > .env << EOL
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database
DATABASE_URL=sqlite:///./empathetic_tutor.db
CHROMA_PERSIST_DIR=./.chromadb

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# ML Settings
MODEL_DIR=./models
LEARNING_RATE=0.001
BATCH_SIZE=32

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000"]

# Feature Weights (initial)
WEIGHT_SENTIMENT=-0.8
WEIGHT_KEYWORDS=1.0
WEIGHT_PUNCTUATION=0.5
WEIGHT_GRAMMAR=0.7
WEIGHT_LATENCY=1.2
WEIGHT_REPETITION=1.0
WEIGHT_HELP_SEEKING=-0.3

# Thresholds
FRUSTRATION_TAU1=0.3
FRUSTRATION_TAU2=0.7
PROFANITY_THRESHOLD=0.6

# Learning Parameters
ALPHA_STUDENT=0.1
ALPHA_GLOBAL=0.01
ALPHA_EMPATHY=0.05
EOL

# Initialize SQLite database
echo "🗄️ Initializing database..."
python scripts/init_db.py

# Create initial knowledge graph structure
echo "🕸️ Creating knowledge graph templates..."
python scripts/init_knowledge_graph.py

# Run initial tests
echo "🧪 Running initial tests..."
pytest tests/ -v

echo "✅ Setup complete! Activate the environment with: source venv/bin/activate"
echo "🚀 Start the API with: uvicorn backend.app.main:app --reload"