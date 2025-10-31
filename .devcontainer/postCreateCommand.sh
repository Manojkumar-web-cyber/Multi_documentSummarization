#!/bin/bash

echo "🚀 Setting up environment..."

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "✅ Setup complete!"
