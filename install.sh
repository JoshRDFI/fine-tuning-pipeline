#!/bin/bash

# Multi-Modal RAG and Fine-tuning Pipeline Installation Script
# This script installs all system dependencies and Python packages required for the pipeline

set -e  # Exit on any error

echo "🚀 Starting installation of Multi-Modal RAG and Fine-tuning Pipeline..."
echo "================================================================"

# Check if running as root for apt commands
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  Some commands require sudo privileges. You may be prompted for your password."
fi

# 1. Create and activate virtual environment FIRST
echo ""
echo "🐍 Setting up Python virtual environment..."
echo "----------------------------------------"

# Check if ft-rag exists, create if not
if [ ! -d "ft-rag" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ft-rag
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ft-rag/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 2. Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
echo "--------------------------------"

# Tesseract OCR
echo "Installing Tesseract OCR..."
sudo apt update
sudo apt install -y tesseract-ocr

# Poppler utilities
echo "Installing Poppler utilities..."
sudo apt install -y poppler-utils

# PostgreSQL and pgvector
echo "Installing PostgreSQL and pgvector..."
sudo apt install -y postgresql postgresql-contrib

# Install pgvector extension
echo "Setting up pgvector extension..."
# Add pgvector repository and install the extension
sudo add-apt-repository ppa:pgvector/stable -y
sudo apt update
# Detect PostgreSQL version and install corresponding pgvector package
PG_VERSION=$(psql --version | grep -oP '\d+' | head -1)
echo "Detected PostgreSQL version: $PG_VERSION"
sudo apt install -y postgresql-$PG_VERSION-pgvector

echo "✅ System dependencies installed successfully!"

# 3. Install PyTorch nightly build for CUDA 12
echo ""
echo "🔥 Installing PyTorch nightly build for CUDA 12..."
echo "------------------------------------------------"
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu12*

# 4. Install project requirements
echo ""
echo "📚 Installing project requirements..."
echo "-----------------------------------"
pip install -r requirements.txt

# 5. Verify installation
echo ""
echo "🔍 Verifying installation..."
echo "---------------------------"

# Check Python packages
echo "Checking installed packages..."
python -c "
import torch
import transformers
import sentence_transformers
import psycopg2
import streamlit
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import camelot
import tabula
import llama_index
import langchain
print('✅ All Python packages installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"

# Check system dependencies
echo ""
echo "Checking system dependencies..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract OCR installed: $(tesseract --version | head -n1)"
else
    echo "❌ Tesseract OCR not found"
fi

if command -v pdftoppm &> /dev/null; then
    echo "✅ Poppler utilities installed"
else
    echo "❌ Poppler utilities not found"
fi

if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL installed: $(psql --version)"
else
    echo "❌ PostgreSQL not found"
fi

# Check pgvector extension
echo "Checking pgvector extension..."
if psql -U postgres -c "SELECT * FROM pg_extension WHERE extname = 'vector';" 2>/dev/null | grep -q vector; then
    echo "✅ pgvector extension installed"
else
    echo "⚠️  pgvector extension not found - you may need to enable it manually"
    echo "   Run: psql -U postgres -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
fi

# 6. Create necessary directories
echo ""
echo "📁 Creating project directories..."
echo "--------------------------------"
mkdir -p scripts tests data extracted_assets logs

# 7. Set up environment file
echo ""
echo "⚙️  Setting up environment configuration..."
echo "----------------------------------------"
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp env_template.txt .env
    echo "⚠️  Please edit .env file with your database credentials and other settings"
else
    echo ".env file already exists"
fi

# 8. Run interactive setup
echo ""
echo "🔧 Running interactive setup..."
echo "-------------------------------"
python setup_interactive.py

# 9. Final instructions
echo ""
echo "🎉 Installation and setup completed successfully!"
echo "==============================================="
echo ""
echo "📋 Next steps:"
echo "1. Place your PDF files in the 'data' directory"
echo "2. Run the pipeline: python run_pipeline.py"
echo "3. Or run individual phases: python run_pipeline.py --phase 1"
echo "4. Launch Streamlit UI: streamlit run scripts/streamlit_app.py"
echo ""
echo "📖 For detailed instructions, see README.md"
echo "🐛 For troubleshooting, check the logs directory"
echo ""
echo "Happy coding! 🚀" 