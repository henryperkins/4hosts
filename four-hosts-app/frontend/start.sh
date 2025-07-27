#!/bin/bash

echo "🚀 Four Hosts Frontend Setup"
echo "=========================="

# Check if .env exists, if not create from example
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
npm install

# Start development server
echo ""
echo "🌟 Starting development server..."
echo "Frontend will be available at: http://localhost:5173"
echo "Make sure backend is running at: http://localhost:8000"
echo ""

npm run dev