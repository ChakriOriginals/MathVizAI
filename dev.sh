#!/usr/bin/env bash
# scripts/dev.sh — start backend + frontend for local development

set -e

# Check .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env not found. Copying .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env and add your OPENAI_API_KEY, then re-run this script."
    exit 1
fi

# Check required tools
command -v python3 >/dev/null 2>&1 || { echo "python3 is required."; exit 1; }
command -v manim   >/dev/null 2>&1 || { echo "manim not found. Run: pip install manim"; exit 1; }

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Create directories
mkdir -p outputs tmp

echo ""
echo "🚀 Starting backend on http://localhost:8000 ..."
uvicorn backend.api.main:app --reload --port 8000 &
BACKEND_PID=$!

sleep 2

echo "🎨 Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501 &
FRONTEND_PID=$!

echo ""
echo "✅ MathVizAI is running!"
echo "   Backend:  http://localhost:8000/docs"
echo "   Frontend: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both servers."

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Servers stopped.'" INT TERM

wait
